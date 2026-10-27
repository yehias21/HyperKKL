r"""Real-data battery SOC experiment, end to end (design: experiments/real/DESIGN.md).

    python -I -m hyperkkl.realdata_run prepare                  # load, identify, episodes, scales
    CUDA_VISIBLE_DEVICES=5 python -I -m hyperkkl.realdata_run train --seed 0
    python -I -m hyperkkl.realdata_run tune --filter ekf        # Q/R on the validation cycle (CPU)
    CUDA_VISIBLE_DEVICES=5 python -I -m hyperkkl.realdata_run evaluate
    python -I -m hyperkkl.realdata_run report                   # results.json, figures, report.md

Reuses the paper's code unchanged: data.integrate / run_latent (simulation, latent filter),
train.phase1 / train.phase2 (with realdata.masked_scales: the masked loss), evaluate.
build_observers (learned KKL observers), observers.EKF / UKF / run. Every artefact is written
under experiments/real/ (the shared experiments/cache is not touched).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from scipy.stats import qmc

from . import design as D
from . import realdata as R
from .config import load
from .data import Design, EpisodeSet, integrate

OUT = R.REAL
CACHE = OUT / "cache"
LOGS = OUT / "logs"
C20 = R.RAW / "25degC/C20_OCV_and_1C_discharge_tests_start_of_tests/05-08-17_13.26_C20_OCV_Test_C20_25dC.mat"
METHODS = ["autonomous", "latent", "concat", "lora", "hyper"]
SEEDS = (0, 1, 2)
N_STEPS, DT_S = 1800, 1.0
ALPHA, NZ = 1.3, 7
STRIDE_TRAIN, STRIDE_EVAL = 120, 300
INIT_ERR = 0.2
SETTLE = N_STEPS // 2
FILTER_GRID = {"q_soc": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], "c": [0.01, 0.1, 1.0],
               "r_vm": [0.003, 0.01, 0.03, 0.1]}
R_DT = 0.3


def _log(name):
    LOGS.mkdir(parents=True, exist_ok=True)
    f = open(LOGS / f"{name}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        f.write(line + "\n"); f.flush()
    return log


def design() -> Design:
    A, B = D.butterworth_AB(NZ, ALPHA, 2)
    dt = DT_S / R.TIME_UNIT
    return Design(A, B, dt, N_STEPS, ALPHA, D.latent_step(A, B, dt))


def system_cfg(params, ic_box, Q) -> dict:
    return {"name": "battery_pf", "plant": "battery_pf", "role": "real", "params": params,
            "sensors": [3, 4], "ic_box": ic_box, "T": N_STEPS * DT_S / R.TIME_UNIT, "nz": NZ,
            "alpha": ALPHA, "dt": DT_S / R.TIME_UNIT, "u_scale": float(Q), "t_ref": 1,
            "u_bounds": [], "actuator": {"kind": "affine"}, "mismatch": 0.0, "knowledge": "full"}


def full_cfg(prep) -> dict:
    cfg = load("battery")
    cfg["system"] = prep["scfg"]
    cfg["baselines"] = []
    cfg["methods"] = list(METHODS)
    return cfg


def cold_plant(params, t_amb_c):
    """The 25 degC ECM moved to another ambient: R scaled by the Arrhenius factor (the
    plant's R(T) is relative to T_amb), C unchanged, T_amb = the new ambient."""
    p = dict(params)
    T_new = 273.15 + t_amb_c
    fac = math.exp(p["Ea_R"] * (1.0 / T_new - 1.0 / p["T_amb"]))
    for k in ("R0", "R1", "R2"):
        p[k] *= fac
    p["T_amb"] = T_new
    return R.Pan18650PF(sensors=(3, 4), **p), fac


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def _sim_windows(plant, des, recs, replays, n, rng, device="cpu"):
    """Training windows' measured currents replayed on the identified model from perturbed
    initial states (SOC +-0.1 kept inside [0, 1] over the window, V1, V2 x U(0.5, 1.5),
    dT +- 1 K, Vm measured). Returns x, z (float64) and u."""
    starts = [(i, s) for i, r in enumerate(recs) for s in R.window_starts(len(r), N_STEPS, STRIDE_TRAIN)]
    pick = rng.randint(0, len(starts), size=n)
    X0, sigs, U = [], [], []
    for j in pick:
        i, s = starts[j]
        r, rp = recs[i], replays[i]
        u = r.u[s:s + N_STEPS + 1].copy()
        c = np.concatenate([[0.0], -np.cumsum(u[:-1]) * DT_S / (3600.0 * r.Q)])
        lo, hi = max(0.0, -c.min()), min(1.0, 1.0 - c.max())
        soc0 = float(np.clip(r.soc[s] + rng.uniform(-0.1, 0.1), lo, hi))
        X0.append([soc0, rp[s, 1] * rng.uniform(0.5, 1.5), rp[s, 2] * rng.uniform(0.5, 1.5),
                   r.dT[s] + rng.uniform(-1, 1), r.vm[s]])
        sigs.append(lambda t, u=u: u[np.clip(np.round(t / des.dt).astype(int), 0, N_STEPS)])
        U.append(u)
    x, z, ug = integrate(plant, des, np.asarray(X0), sigs, device=device)
    return x.cpu(), z.cpu(), ug.cpu()


def _sobol(box, n, seed):
    m = int(math.ceil(math.log2(max(n, 2))))
    X = qmc.Sobol(d=len(box), scramble=True, seed=seed).random_base2(m)[:n]
    return box[:, 0] + X * (box[:, 1] - box[:, 0])


def prepare(log=print):
    torch.set_num_threads(8)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    oc = R.c20_capacity_and_ocv(C20)
    Q = oc["Q"]
    f25 = R.drive_cycle_files(25.0)
    recs = {c: R.load_recording(p, Q) for c, p in f25.items()}
    off = R.rest_offset([recs[c] for c in R.SPLIT["train"]])
    for r in recs.values():
        r.t_amb_eff = r.temp + off
    R.check_split(R.SPLIT, recs)
    f0 = R.drive_cycle_files(0.0)
    recs0 = {c: R.load_recording(p, Q, t_amb_eff=0.0 + off) for c, p in f0.items()}
    log(f"loaded {len(recs)} recordings at 25 degC, {len(recs0)} at 0 degC; Q = {Q:.4f} Ah, "
        f"thermocouple offset {off:.3f} K; max |own CC - lab Ah| = "
        f"{max(r.cc_check for r in list(recs.values()) + list(recs0.values())):.4f} Ah")
    train = [recs[c] for c in R.SPLIT["train"]]
    idn = R.identify(train, C20)
    params = idn["params"]
    log(f"identified: {json.dumps({k: round(v, 6) for k, v in idn['ecm'].items()})}  "
        f"thermal {json.dumps({k: round(v, 4) for k, v in idn['thermal'].items()})}  "
        f"OCV fit rms {1e3 * idn['ocv_fit']['rms']:.1f} mV")
    plant = R.Pan18650PF(sensors=(3, 4), **params)
    plant0, fac0 = cold_plant(params, 0.0 + off)
    names = list(recs)
    reps = dict(zip(names, R.replay_many(plant, [recs[c] for c in names])))
    names0 = list(recs0)
    reps0 = dict(zip(names0, R.replay_many(plant0, [recs0[c] for c in names0])))
    model_fit = {}
    for tag, rr, rp in (("25", recs, reps), ("0", recs0, reps0)):
        for c, r in rr.items():
            e = rp[c][:, 4] - r.vm
            model_fit[f"{tag}/{c}"] = {"vm_rmse_mV": 1e3 * float(np.sqrt(np.mean(e ** 2))),
                                       "dT_rmse_K": float(np.sqrt(np.mean((rp[c][:, 3] - r.dT) ** 2)))}
    log("open-loop model fit (Vm RMSE, mV): " + ", ".join(f"{k} {v['vm_rmse_mV']:.0f}" for k, v in model_fit.items()))
    log(f"replays done ({time.time() - t0:.0f}s)")

    des = design()
    rng = np.random.RandomState(0)
    trp = [reps[c] for c in R.SPLIT["train"]]
    xs, zs, us = _sim_windows(plant, des, train, trp, 600, rng)
    allx = np.concatenate([reps[c] for c in R.SPLIT["train"]])
    ic_box = [[0.0, 1.0]] + [[float(allx[:, j].min()), float(allx[:, j].max())] for j in (1, 2)] + \
             [[float(np.concatenate([r.dT for r in train]).min()), float(np.concatenate([r.dT for r in train]).max())],
              [float(np.concatenate([r.vm for r in train]).min()), float(np.concatenate([r.vm for r in train]).max())]]
    box = np.asarray(ic_box)
    xf = xs.reshape(-1, 5).numpy()
    lo, hi = xf.min(0), xf.max(0)
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    dom = np.stack([np.minimum(ctr - 1.15 * half, box[:, 0]), np.maximum(ctr + 1.15 * half, box[:, 1])], 1)
    kappa, alpha = D.spectral_bounds(des.A)
    y_bar = float(plant.h(xs.reshape(-1, 5)).norm(dim=-1).max())
    z_scale = float(zs[:, zs.shape[1] // 2:].pow(2).sum(-1).mean().sqrt())
    Bn = float(np.linalg.norm(des.B, 2))
    t_burn = D.washout_time(kappa, alpha, Bn, y_bar, z_scale, 1e-3)
    burn = int(math.ceil(t_burn / des.dt))
    xr = xs[:, burn:].reshape(-1, 5)
    scales = {"domain": dom.tolist(), "y_bar": y_bar, "z_scale": z_scale, "t_burn": t_burn,
              "memory_s": D.memory_length(kappa, alpha, Bn, y_bar, z_scale, 1e-3),
              "x_std": xr.std(0).clamp(min=1e-9).tolist(), "y_std": plant.h(xr).std(0).tolist(),
              "x_mean": xr.mean(0).tolist(), "dt": des.dt, "N": des.N,
              "design_check": D.validate(des.A, des.B, des.dt, des.alpha)}
    log(f"design: nz {NZ} alpha {ALPHA}/min dt {des.dt:.5f} min N {des.N}  gate {scales['design_check']['ok']} "
        f"(max|lam|dt {scales['design_check']['max_lam_dt']:.4f})  t_burn {t_burn:.2f} min (burn {burn})  "
        f"x_std {np.round(scales['x_std'], 4).tolist()}")

    def es_sim(x, z, u):
        y = plant.h(x.reshape(-1, 5)).reshape(x.shape[0], x.shape[1], 2)
        from .data import run_latent
        zo = run_latent(des, y)
        f = lambda a: a.float()
        return EpisodeSet(f(x), f(u), f(y), f(y), f(z), f(zo), ["replay"] * len(x), burn, des.dt, [], 0)

    train_sim = es_sim(xs, zs, us)
    X0 = _sobol(dom, 400, 101)
    xp, zp, up = integrate(plant, des, X0, [lambda t: np.zeros_like(t)] * len(X0))
    p1 = es_sim(xp.cpu(), zp.cpu(), up.cpu())
    Xc = torch.as_tensor(_sobol(dom, 32768, 104), dtype=torch.float64)
    coll = {"x": Xc.float(), "f0": plant.f(Xc, torch.zeros(len(Xc), 1, dtype=Xc.dtype)).float(),
            "y": plant.h(Xc).float()}
    log(f"phase-1 sims: p1 {len(p1)}  replayed train {len(train_sim)}  coll {len(Xc)} ({time.time() - t0:.0f}s)")

    def real(rlist, rp, stride):
        return R.episodes(rlist, rp, N_STEPS, stride, des, burn)

    sets = {"train": real(train, trp, STRIDE_TRAIN),
            "val": real([recs[c] for c in R.SPLIT["val"]], [reps[c] for c in R.SPLIT["val"]], STRIDE_EVAL)}
    for c in R.SPLIT["test"]:
        sets[f"test25/{c}"] = real([recs[c]], [reps[c]], STRIDE_EVAL)
    for c in names0:
        sets[f"test0/{c}"] = real([recs0[c]], [reps0[c]], STRIDE_EVAL)
    log("real episode sets: " + ", ".join(f"{k} {len(v)}" for k, v in sets.items()))
    scfg = system_cfg(params, ic_box, Q)
    prep = {"scfg": scfg, "scales": scales, "identification": idn, "offset_K": off, "Q": Q,
            "arrhenius_factor_0C": fac0, "model_fit": model_fit, "burn": burn,
            "sim": {"p1": p1, "train": train_sim, "coll": coll}, "sets": sets,
            "recs": {**{f"25/{c}": r for c, r in recs.items()}, **{f"0/{c}": r for c, r in recs0.items()}},
            "replays": {**{f"25/{c}": v for c, v in reps.items()}, **{f"0/{c}": v for c, v in reps0.items()}}}
    torch.save(prep, CACHE / "prep.pt")
    log(f"prepare done ({time.time() - t0:.0f}s)")
    return prep


def load_prep():
    return torch.load(CACHE / "prep.pt", weights_only=False)


# ---------------------------------------------------------------------------
# priors and episodes for evaluation
# ---------------------------------------------------------------------------

def priors(es: EpisodeSet, err=INIT_ERR):
    """Two priors per window (SOC0 +- err), kept when inside [0, 1]. Returns the episode
    index, the sign and x0_hat (E', 5): (SOC0 + e, 0, 0, dT_0, Vm_0)."""
    idx, sgn, X0 = [], [], []
    for i in range(len(es)):
        s0 = float(es.x[i, 0, 0])
        for e in (+err, -err):
            if 0.0 <= s0 + e <= 1.0:
                idx.append(i); sgn.append(e)
                X0.append([s0 + e, 0.0, 0.0, float(es.y[i, 0, 0]), float(es.y[i, 0, 1])])
    return np.asarray(idx), np.asarray(sgn), torch.as_tensor(np.asarray(X0), dtype=torch.float32)


def expand(es: EpisodeSet, err=INIT_ERR):
    idx, sgn, X0 = priors(es, err)
    ti = torch.as_tensor(idx, dtype=torch.long)
    return es.y[ti], es.u[ti], es.x[ti, :, 0], X0, idx, sgn


def soc_metrics(soc_hat, soc, settle=SETTLE, dt_min=DT_S / 60.0, band=0.05) -> dict:
    """Per episode, % SOC: settled RMSE, whole RMSE, settled max, convergence time [min]."""
    e = (np.asarray(soc_hat, float) - np.asarray(soc, float)) * 100.0
    e = np.nan_to_num(e, nan=1e4, posinf=1e4, neginf=-1e4)
    rs = np.sqrt(np.mean(e[:, settle:] ** 2, 1))
    ra = np.sqrt(np.mean(e ** 2, 1))
    mx = np.abs(e[:, settle:]).max(1)
    bad = np.abs(e) > 100 * band
    T = e.shape[1]
    last = np.where(bad.any(1), T - 1 - np.argmax(bad[:, ::-1], 1), -1)    # last bad index
    tc = (last + 1) * dt_min
    return {"rmse_settled": rs, "rmse_all": ra, "max_settled": mx, "t_conv_min": tc,
            "conv_censored": (last == T - 1)}


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(seed: int, device: str, log=print):
    from .train import phase1, phase2
    prep = load_prep()
    cfg = full_cfg(prep)
    des = design()
    scales = prep["scales"]
    torch.manual_seed(seed); np.random.seed(seed)
    out = CACHE / f"models_seed{seed}.pt"
    t0 = time.time()
    base = phase1(cfg, prep["sim"], scales, des, device, seed, log)
    log(f"phase1 done ({time.time() - t0:.0f}s): {json.dumps({k: round(v, 5) for k, v in base['diag'].items()})}")
    ms = R.masked_scales(scales)
    data2 = {"train": prep["sets"]["train"], "val": prep["sets"]["val"]}
    trained = {}
    for m in METHODS:
        kind = "none" if m == "autonomous" else m
        r = phase2(kind, cfg, data2, ms, des, base, device, seed, log)
        trained[m] = r
        log(f"phase2 {m} done: val {r['val']:.4e} params {r['n_params']} ({time.time() - t0:.0f}s)")
    torch.save({"base": base, "trained": trained, "seed": seed}, out)
    log(f"saved {out} ({time.time() - t0:.0f}s)")


def learned_observers(prep, seed, device):
    from .evaluate import build_observers
    from .job import _to
    m = torch.load(CACHE / f"models_seed{seed}.pt", weights_only=False, map_location="cpu")
    cfg = full_cfg(prep)
    return build_observers(cfg, design(), prep["scales"], _to(m["base"], device),
                           _to(m["trained"], device), None, device)


@torch.no_grad()
def run_soc(ob, Y, U, X0, device, chunk=256, dtype=torch.float32) -> np.ndarray:
    from .observers import run
    out = []
    for i in range(0, len(Y), chunk):
        y, u, x0 = (a[i:i + chunk].to(device=device, dtype=dtype) for a in (Y, U, X0))
        out.append(run(ob, y, u, x0)[:, :, 0].double().cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------

def _filter_noise(scales, g):
    xs = np.asarray(scales["x_std"], float)
    q = g["c"] * xs
    q[0] = g["q_soc"]
    return q, np.array([R_DT, g["r_vm"]]), np.array([INIT_ERR, xs[1], xs[2], 0.5, 0.01])


def make_filter(kind, plant, scales, g):
    """observers.EKF / observers.UKF (unchanged) on the identified ECM."""
    from .observers import EKF, UKF
    q, r, p0 = _filter_noise(scales, g)
    f = (EKF if kind == "ekf" else UKF)(plant, design().dt, q, r, p0)
    f.name = kind
    return f


def grid_filter(kind, plant, scales, grid, n_rep):
    """ONE filter object running every grid point at once: after the standard reset, Q and R
    are replaced by per-episode (batched) matrices, episodes ordered grid-point-major (n_rep
    episodes per point). The plain Kalman update and both predictions broadcast them, so each
    episode is exactly the filter built with its own grid point (tests/test_realdata.py)."""
    f = make_filter(kind, plant, scales, grid[0])
    dt = design().dt
    qs, rs = zip(*[_filter_noise(scales, g)[:2] for g in grid])
    Q = torch.as_tensor(np.stack([np.diag(np.maximum(q ** 2 * dt, 1e-12)) for q in qs]))
    Rm = torch.as_tensor(np.stack([np.diag(np.maximum(r ** 2, 1e-12)) for r in rs]))
    reset0 = f.reset

    def reset(B, x0_hat, u0):
        reset0(B, x0_hat, u0)
        f.Q = Q.to(x0_hat).repeat_interleave(n_rep, 0)
        f.R = Rm.to(x0_hat).repeat_interleave(n_rep, 0)
    f.reset = reset
    return f


def filter_grid():
    return [{"q_soc": a, "c": c, "r_vm": r} for a in FILTER_GRID["q_soc"] for c in FILTER_GRID["c"]
            for r in FILTER_GRID["r_vm"]]


def tune(kind: str, device, log=print):
    torch.set_num_threads(2)
    prep = load_prep()
    plant = R.Pan18650PF(sensors=(3, 4), **prep["scfg"]["params"])
    Y, U, S, X0, _, _ = expand(prep["sets"]["val"])
    grid = filter_grid()
    n = len(Y)
    rep = lambda a: a.repeat(len(grid), *([1] * (a.dim() - 1)))
    t0 = time.time()
    f = grid_filter(kind, plant, prep["scales"], grid, n)
    sh = run_soc(f, rep(Y), rep(U), rep(X0), device, chunk=10 ** 9, dtype=torch.float64)
    scores = []
    for i, g in enumerate(grid):
        m = soc_metrics(sh[i * n:(i + 1) * n], S.numpy())
        scores.append({**g, "val_rmse_settled": float(np.mean(m["rmse_settled"])),
                       "val_rmse_all": float(np.mean(m["rmse_all"]))})
        log(f"  {kind} {g}: val settled RMSE {scores[-1]['val_rmse_settled']:.3f} %  all {scores[-1]['val_rmse_all']:.3f} %")
    best = min(scores, key=lambda s: s["val_rmse_settled"])
    res = {"choice": {k: best[k] for k in ("q_soc", "c", "r_vm")}, "scores": scores, "n_val": n,
           "wall_s": time.time() - t0}
    (CACHE / f"tune_{kind}.json").write_text(json.dumps(res, indent=1))
    log(f"{kind} choice {res['choice']} -> val {best['val_rmse_settled']:.3f} % ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def _pad_full(recs):
    """Whole recordings as one padded batch: after its end a recording holds its last y, gets
    u = 0 (padding is cut off before any metric)."""
    n = max(len(r) for r in recs)
    Y = np.zeros((len(recs), n, 2)); U = np.zeros((len(recs), n, 1))
    for i, r in enumerate(recs):
        Y[i, :len(r)] = r.y; Y[i, len(r):] = r.y[-1]
        U[i, :len(r), 0] = r.u
    X0 = np.array([[r.soc[0] - INIT_ERR, 0.0, 0.0, r.dT[0], r.vm[0]] for r in recs])
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    return t(Y), t(U), t(X0)


def evaluate(device, log=print):
    torch.set_num_threads(2)
    prep = load_prep()
    params = prep["scfg"]["params"]
    plants = {"25": R.Pan18650PF(sensors=(3, 4), **params), "0": cold_plant(params, 0.0 + prep["offset_K"])[0]}
    tuned = {k: json.loads((CACHE / f"tune_{k}.json").read_text())["choice"] for k in ("ekf", "ukf")}
    sets = {k: v for k, v in prep["sets"].items() if k != "train"}
    temp_of = lambda name: "0" if name.startswith("test0/") else "25"
    pred, truth, meta, batch = {}, {}, {}, {"25": [], "0": []}
    for name, es in sets.items():
        Y, U, S, X0, idx, sgn = expand(es)
        truth[name] = S.numpy().astype(np.float64)
        meta[name] = {"idx": idx, "sgn": sgn, "start": np.array([es.meta[i]["start"] for i in idx]),
                      "soc0": np.array([es.meta[i]["soc0"] for i in idx])}
        batch[temp_of(name)].append((name, Y, U, X0))
        pred[(name, "cc")] = (X0[:, 0].double().numpy()[:, None] - np.concatenate(
            [np.zeros((len(U), 1)), np.cumsum(U[:, :-1, 0].double().numpy(), 1)], 1) * DT_S / (3600 * prep["Q"]))
    # whole recordings from full charge, prior SOC 0.8 (test cycles at 25 degC, every 0 degC file)
    fkeys = {"25": [f"25/{c}" for c in R.SPLIT["test"]], "0": [k for k in prep["recs"] if k.startswith("0/")]}
    full = {}
    for tmp, keys in fkeys.items():
        for k in keys:
            r = prep["recs"][k]
            full[k] = {"soc": r.soc.copy(), "t_min": r.t / 60.0, "u": r.u.copy(),
                       "cc": r.soc[0] - INIT_ERR - np.concatenate([[0.0], np.cumsum(r.u[:-1])]) * DT_S / (3600 * r.Q)}
    fb = {tmp: _pad_full([prep["recs"][k] for k in keys]) for tmp, keys in fkeys.items()}

    def run_all(make, tag, dtype):
        for tmp in ("25", "0"):
            ob = make(tmp)
            Y = torch.cat([b[1] for b in batch[tmp]]); U = torch.cat([b[2] for b in batch[tmp]])
            X0 = torch.cat([b[3] for b in batch[tmp]])
            t0 = time.time()
            sh = run_soc(ob, Y, U, X0, device, chunk=10 ** 9, dtype=dtype)
            o = 0
            for name, y, _, _ in batch[tmp]:
                pred[(name,) + tag] = sh[o:o + len(y)]; o += len(y)
            sf = run_soc(ob, *fb[tmp], device, chunk=10 ** 9, dtype=dtype)
            for i, k in enumerate(fkeys[tmp]):
                full[k]["/".join(map(str, tag))] = sf[i, :len(prep["recs"][k])]
            log(f"  {'/'.join(map(str, tag)):14s} {tmp} degC: " + "  ".join(
                f"{n.split('/')[-1]} {np.mean(soc_metrics(pred[(n,) + tag], truth[n])['rmse_settled']):.2f}"
                for n, *_ in batch[tmp]) + f"  ({time.time() - t0:.0f}s)")

    for k in ("ekf", "ukf"):
        run_all(lambda tmp: make_filter(k, plants[tmp], prep["scales"], tuned[k]), (k,), torch.float64)
    for seed in SEEDS:
        obs = learned_observers(prep, seed, device)
        for m, ob in obs.items():
            run_all(lambda tmp: ob, (m, seed), torch.float32)
    torch.save({"pred": pred, "truth": truth, "meta": meta, "full": full, "tuned": tuned},
               CACHE / "predictions.pt")
    log("evaluate done")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "train", "tune", "evaluate", "baselines",
                                      "evaluate_baselines", "report"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--filter", default="ekf")
    ap.add_argument("--methods", default=None, help="comma list (evaluate_baselines)")
    ap.add_argument("--temps", default="25,0")
    a = ap.parse_args(argv)
    if a.stage == "prepare":
        prepare(_log("prepare"))
    elif a.stage == "train":
        torch.set_num_threads(4)
        train(a.seed, a.device, _log(f"train_seed{a.seed}"))
    elif a.stage == "tune":
        tune(a.filter, a.device, _log(f"tune_{a.filter}"))
    elif a.stage == "evaluate":
        evaluate(a.device, _log("evaluate"))
    elif a.stage == "baselines":
        from .realdata_baselines import fit
        torch.set_num_threads(2)
        fit(a.seed, a.device, _log(f"baselines_seed{a.seed}"))
    elif a.stage == "evaluate_baselines":
        from .realdata_baselines import evaluate as evb
        torch.set_num_threads(2)
        ms = a.methods.split(",") if a.methods else None
        tg = None if ms is None and a.temps == "25,0" else \
            f"_seed{a.seed}_{'+'.join(ms or ['all'])}_{a.temps.replace(',', '+')}"
        evb(a.device, _log(f"evaluate_baselines_seed{a.seed}"), seeds=(a.seed,), methods=ms,
            temps=tuple(a.temps.split(",")), tag=tg)
    else:
        from .realdata_report import report
        report(_log("report"))


if __name__ == "__main__":
    main()
