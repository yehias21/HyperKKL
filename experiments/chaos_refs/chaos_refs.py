"""Chaotic-plant reference baselines (reviewer_d.md E2/E3; SYNTHESIS.md section 4 rank 5).

(a) Pecora-Carroll (PC) synchronisation observer on lorenz (dev) and rossler (test).
    The measured coordinate y = x2 replaces x2 in the model, and the UNMEASURED equations
    (x1, x3) are integrated forward, driven by y:
        lorenz : x1' = sigma (y - x1),          x3' = x1 y - beta x3
        rossler: x1' = -(y + x3),               x3' = b + x3 (x1 - c)
    No learning, no gain, no tuning. x2_hat = y_k (the raw measurement). RK4 at the run's
    sample time with y linear between y_k and y_{k+1} and u held (ZOH), the same causal
    information an EKF step gets (observers.run: advance(y_k, u_k, y_{k+1}, u_{k+1})).
    The response uses the MODELLER's model of the node (data.model_plant: nominal parameters,
    system.knowledge full | partial); "pc_true" (reference only) uses the real plant of the
    block. Episodes are EXACTLY those of the runs: evaluate._blocks on each run's own
    config, read from experiments/cache (read-only: a cache miss raises), checked against
    the true trajectories saved in each run's traj.npz. Metric: metrics.episode_metrics
    (settled NRMSE, second half). Paired comparisons with metrics.paired_relative over the
    run's training seeds (PC is deterministic: the same array is paired with every seed).

(b) Largest Lyapunov exponent lambda_max (two-trajectory renormalisation, Benettin) of the
    real plant (with its saturating actuator, without process noise) under the exact
    evaluation inputs of each tier (protocol v2; every family, bounded box), the input of
    every episode extended cyclically to K episode lengths, first episode length discarded.
    The finite-time exponent over the evaluation window itself and the conditional
    Lyapunov exponent (CLE) of the PC response (two response copies driven by the same x2)
    are computed in the same pass.

CPU only. Writes RESULTS-side data to experiments/chaos_refs/results.json.
Usage: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/chaos_refs/chaos_refs.py
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)   # tiny batched ops: 1 thread is ~20x faster than 8 here

import hyperkkl.data as Dm                                    # noqa: E402
from hyperkkl import evaluate as EVm                          # noqa: E402
from hyperkkl.metrics import episode_metrics, paired_relative  # noqa: E402
from hyperkkl.systems import build_plant                       # noqa: E402

OUT = Path(__file__).resolve().parent
TREE = ROOT / "experiments" / "tree"
assert not torch.cuda.is_available(), "CPU only"


# ---------------------------------------------------------------------------
# read-only access to experiments/cache
# ---------------------------------------------------------------------------

def _ro_cached(path, build, *a, **k):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss (read-only mode, not building): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


Dm._cached = _ro_cached
_SCALES = {}
Dm.system_scales = lambda cfg, device="cpu": _SCALES["cur"]   # the run's own scales (results.json)


# ---------------------------------------------------------------------------
# Pecora-Carroll observer
# ---------------------------------------------------------------------------

@torch.no_grad()
def pecora_carroll(model, Y, U, X0, dt, meas=(1,)) -> torch.Tensor:
    """Y (E, N1, ny), U (E, N1, nu), X0 (E, nx) -> xhat (E, N1, nx), float64."""
    Y, U, X = Y.double(), U.double(), X0.double().clone()
    E, N1, _ = Y.shape
    meas = list(meas)
    unm = [j for j in range(X.shape[1]) if j not in meas]

    def rhs(x, yv, u):
        xf = x.clone()
        xf[:, meas] = yv
        d = model.f(xf, u)
        d[:, meas] = 0.0
        return d

    X[:, meas] = Y[:, 0]
    out = torch.empty(E, N1, X.shape[1], dtype=torch.float64)
    out[:, 0] = X
    h = dt
    for k in range(N1 - 1):
        y0, y1, u = Y[:, k], Y[:, k + 1], U[:, k]
        ym = 0.5 * (y0 + y1)
        k1 = rhs(X, y0, u)
        k2 = rhs(X + h / 2 * k1, ym, u)
        k3 = rhs(X + h / 2 * k2, ym, u)
        k4 = rhs(X + h * k3, y1, u)
        X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X[:, meas] = y1
        out[:, k + 1] = X
    return torch.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)


# ---------------------------------------------------------------------------
# Lyapunov exponents (Benettin, two trajectories) + CLE of the PC response
# ---------------------------------------------------------------------------

@torch.no_grad()
def lyapunov(plant, resp_models: dict, x0, U, dt, K: int, meas=(1,), d0=1e-7, burn_frac=0.2):
    """Two-trajectory (Benettin) lambda_max, renormalised every step, float64 RK4.
    x0 (E, nx) true initial state; U (E, N1, 1) the evaluation input (ZOH), repeated
    cyclically K times. Returns per episode: lam_long (cycles 2..K), lam_ep (first episode,
    t in [burn_frac T, T]) and, per response model, the largest conditional Lyapunov
    exponent (CLE) of the PC response (x1, x3) driven by the reference x2 (cycles 2..K).
    Reference, perturbed and response copies are integrated as ONE augmented ODE."""
    x = x0.double().clone()
    E, nx = x.shape
    N = U.shape[1] - 1
    Ud = U.double()
    meas = list(meas)
    unm = [j for j in range(nx) if j not in meas]
    names = list(resp_models)
    R = len(names)
    g = torch.Generator().manual_seed(0)
    v = torch.randn(E, nx, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=1, keepdim=True)
    X = torch.cat([x, x + d0 * v])                                   # (2E, nx)
    w = torch.randn(R, E, len(unm), generator=g, dtype=torch.float64)
    w = w / w.norm(dim=2, keepdim=True)
    ra = x[:, unm].expand(R, E, len(unm)).clone()
    Rr = torch.cat([ra, ra + d0 * w], 1)                             # (R, 2E, nu): copies a | b
    S = torch.zeros(E, dtype=torch.float64)
    Cs = torch.zeros(R, E, dtype=torch.float64)
    k_burn = int(burn_frac * N)
    h = dt

    def F(Xs, Rs, u2):
        dX = plant.f(Xs, u2)
        xm = Xs[:E, meas]                                            # the reference drives
        dR = torch.empty_like(Rs)
        for i, n in enumerate(names):
            z = torch.empty(2 * E, nx, dtype=torch.float64)
            z[:, unm] = Rs[i]
            z[:, meas] = torch.cat([xm, xm])
            dR[i] = resp_models[n].f(z, u2)[:, unm]
        return dX, dR

    S_burn = S_T = None
    C_T = None
    for c in range(K):
        for k in range(N):
            u2 = torch.cat([Ud[:, k], Ud[:, k]])
            a1 = F(X, Rr, u2)
            a2 = F(X + h / 2 * a1[0], Rr + h / 2 * a1[1], u2)
            a3 = F(X + h / 2 * a2[0], Rr + h / 2 * a2[1], u2)
            a4 = F(X + h * a3[0], Rr + h * a3[1], u2)
            X = X + h / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
            Rr = Rr + h / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
            dv = X[E:] - X[:E]
            d = dv.norm(dim=1)
            S += torch.log(d / d0)
            X[E:] = X[:E] + d0 * dv / d[:, None]
            dr = Rr[:, E:] - Rr[:, :E]
            dn = dr.norm(dim=2).clamp(min=1e-300)
            Cs += torch.log(dn / d0)
            Rr[:, E:] = Rr[:, :E] + d0 * dr / dn[..., None]
            if c == 0 and k + 1 == k_burn:
                S_burn = S.clone()
        if c == 0:
            S_T, C_T = S.clone(), Cs.clone()
    T = N * dt
    lam_ep = ((S_T - S_burn) / ((N - k_burn) * dt)).numpy()
    if K > 1:
        lam_long = ((S - S_T) / ((K - 1) * T)).numpy()
        cle = {n: ((Cs[i] - C_T[i]) / ((K - 1) * T)).numpy() for i, n in enumerate(names)}
    else:
        lam_long, cle = lam_ep, {n: (C_T[i] / T).numpy() for i, n in enumerate(names)}
    return {"lam_ep": lam_ep, "lam_long": lam_long, "cle": cle, "finite": torch.isfinite(X[:E]).all(1).numpy()}


def lyap_system(system: str, node: str, K: int, variant: str, log=print) -> dict:
    """lambda_max per tier and family on the evaluation inputs of ``node`` for the plant
    ``variant`` (nominal, or the node's mismatched real plant); CLE of the PC response with
    the true f, the partial model and (mismatch) the nominal model."""
    runs = load_runs(node, system)
    r0 = runs[sorted(runs)[0]]
    cfg, scales = r0["config"], r0["scales"]
    _SCALES["cur"] = scales
    sc = dict(cfg["system"], knowledge="full")
    des = Dm.make_design(sc)
    mm = float(sc.get("mismatch", 0.0)) if variant == "mismatch" else 0.0
    plant = build_plant(sc, perturb=mm)
    resp = {"true_f": plant, "partial_f": build_plant(sc, perturb=0.0, knowledge="partial")}
    if mm:
        resp["nominal_f"] = build_plant(sc, perturb=0.0)
    ess = {t: Dm.eval_data(cfg, t, device="cuda") for t in cfg["eval"]["tiers"]}   # 'cuda': cache KEY only
    U = torch.cat([e.u for e in ess.values()])
    x0 = torch.cat([e.x[:, 0] for e in ess.values()])
    tiers = np.concatenate([[t] * len(e) for t, e in ess.items()])
    fam = np.concatenate([e.families for e in ess.values()])
    # unforced reference (u = 0) on the id initial states, in the same batch
    nz = min(16, len(ess["id"]))
    U = torch.cat([U, torch.zeros_like(ess["id"].u[:nz])])
    x0 = torch.cat([x0, ess["id"].x[:nz, 0]])
    tiers = np.concatenate([tiers, ["u=0"] * nz])
    fam = np.concatenate([fam, ["zero_input"] * nz])
    t0 = time.time()
    L = lyapunov(plant, resp, x0, U, des.dt, K)
    out = {"node": node, "variant": variant, "mismatch": mm, "params": plant.params(), "K": K,
           "dt": des.dt, "T": des.T, "wall_s": time.time() - t0, "tiers": {}}
    for t in list(ess) + ["u=0"]:
        it = tiers == t
        d = {"n": int(it.sum()), "lam_long": summarise_metric(L["lam_long"][it]),
             "lam_long_min": float(L["lam_long"][it].min()),
             "lam_ep": summarise_metric(L["lam_ep"][it]), "lam_ep_min": float(L["lam_ep"][it].min()),
             "frac_positive_long": float((L["lam_long"][it] > 0).mean()),
             "cle": {n: summarise_metric(v[it]) for n, v in L["cle"].items()},
             "finite": float(L["finite"][it].mean()), "families": {}}
        for f in sorted(set(fam[it])):
            i = it & (fam == f)
            d["families"][f] = {"n": int(i.sum()), "lam_long_mean": float(L["lam_long"][i].mean()),
                                "lam_long_min": float(L["lam_long"][i].min()),
                                "lam_long_max": float(L["lam_long"][i].max()),
                                "lam_ep_mean": float(L["lam_ep"][i].mean()),
                                "cle_true_f_max": float(L["cle"]["true_f"][i].max())}
        out["tiers"][t] = d
        log(f"  lyap {system} {variant} {t}: lam_long mean {d['lam_long']['mean']:.4f} min {d['lam_long_min']:.4f}; "
            f"lam_ep mean {d['lam_ep']['mean']:.4f}; CLE " +
            " ".join(f"{n} {d['cle'][n]['mean']:.4f}/max {d['cle'][n]['max']:.4f}" for n in d["cle"]))
    log(f"  ({out['wall_s']:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

NODES = [  # (node, system, label)
    ("C3-faults", "lorenz", "exact model"),
    ("C3-partial", "lorenz", "partial model"),
    ("C3-realistic", "lorenz", "realistic (partial + 30% mismatch + coloured-t noise)"),
    ("C3-regimes", "lorenz", "partial model, stress conditions"),
    ("TEST-C1", "rossler", "exact model"),
    ("TEST-C3", "rossler", "realistic (partial + 30% mismatch + coloured-t noise)"),
]
COMPARE = ["lora", "hyper", "latent", "nlox", "koopman_kf", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle",
           "ekf", "ukf", "pf", "kkl_mb", "ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle"]
FILTERS = ("ekf", "ukf", "pf", "kkl_mb", "ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle")
GREYBOX = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
THR = (0.1, 0.5)          # burst thresholds on the normalised error ne(t)


def load_runs(node, system):
    out = {}
    for p in sorted((TREE / node / "runs" / system).glob("seed*/results.json")):
        out[int(p.parent.name[4:])] = json.loads(p.read_text())
    return out


def bursts(ne: np.ndarray, settle: int, dt: float, lam: float, thr: float) -> dict:
    """ne (E, N1). Up-crossings of thr in the settled window per Lyapunov time, and the
    fraction of settled time above thr."""
    s = ne[:, settle:]
    above = s > thr
    up = (above[:, 1:] & ~above[:, :-1]).sum(1) + above[:, 0]
    T_L = (s.shape[1] * dt) * lam                       # settled window in Lyapunov times
    return {"rate_per_TL": float(up.sum() / (len(s) * T_L)) if lam > 0 else float("nan"),
            "frac_time_above": float(above.mean())}


def summarise_metric(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max())}


def run_node(node, system, label, lam_ref, log=print) -> dict:
    runs = load_runs(node, system)
    seeds = sorted(runs)
    r0 = runs[seeds[0]]
    cfg, scales = r0["config"], r0["scales"]
    _SCALES["cur"] = scales
    sc = cfg["system"]
    des = Dm.make_design(sc)
    assert abs(des.dt - scales["dt"]) < 1e-12 and des.N == scales["N"]
    blocks = []
    for tier in cfg["eval"]["tiers"]:
        for cname in cfg["eval"]["conditions"]:
            sub = dict(cfg, eval=dict(cfg["eval"], tiers=[tier], conditions=[cname]))
            try:
                blocks += EVm._blocks(sub, des, scales, "cuda")   # 'cuda' only for the cache KEY
            except FileNotFoundError as e:
                log(f"  {node}/{tier}/{cname}: skipped ({e})")
    model = Dm.model_plant(sc)
    mism = float(sc.get("mismatch", 0.0))
    sigma = scales["x_std"]
    sensors = list(build_plant(sc).sensors)
    res = {"node": node, "system": system, "label": label, "seeds": seeds, "knowledge": sc.get("knowledge", "full"),
           "mismatch": mism, "sensor": cfg["data"]["sensor"], "dt": des.dt, "T": des.T, "blocks": {}, "check": {}}
    # episode identity check against the saved true trajectories (first 4 clean episodes)
    for tier in ("shape_ood", "sequence_ood"):
        for k in seeds:
            p = TREE / node / "runs" / system / f"seed{k}" / "traj.npz"
            if not p.exists():
                continue
            z = np.load(p)
            nm = f"{tier}/clean/true"
            b = [b for b in blocks if b["tier"] == tier and b["cname"] == "clean"]
            if nm in z.files and b:
                d = float(np.abs(z[nm] - b[0]["es"].x[:4].numpy()).max())
                res["check"][f"{tier}/seed{k}"] = d
    for b in blocks:
        tier, cname, es = b["tier"], b["cname"], b["es"]
        t0 = time.time()
        N1 = es.x.shape[1]
        settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (N1 - 1))
        true_plant = build_plant(sc, perturb=mism + b["shift"])
        xh = {"pc": pecora_carroll(model, es.y, b["u"], b["x0"], des.dt, sensors),
              "pc_true": pecora_carroll(true_plant, es.y, b["u"], b["x0"], des.dt, sensors)}
        blk = {"n_episodes": len(es), "families": es.families, "methods": {}, "compare": {}, "bursts": {}}
        for name, x in xh.items():
            m = episode_metrics(es.x, x.float(), sigma, settle, sensors)
            blk["methods"][name] = {k: np.asarray(v).astype(float).tolist() for k, v in m.items()
                                    if k in ("nrmse", "nrmse_unmeasured", "nrmse_x1", "nrmse_x2", "nrmse_x3",
                                             "max_nerr", "t_conv_steps", "diverged", "nrmse_transient")}
            s_ = torch.as_tensor(np.asarray(sigma), dtype=torch.float64)
            ne = (((x - es.x.double()) / s_).clamp(-100, 100).pow(2).mean(-1).sqrt()).numpy()
            blk["bursts"][name] = {str(t): bursts(ne, settle, des.dt, lam_ref.get(tier, float("nan")), t) for t in THR}
            blk["bursts"][name]["ne_p999_settled"] = float(np.percentile(ne[:, settle:], 99.9))
        # paired comparisons vs the run's methods (same episodes; PC repeated over seeds)
        for name in ("pc", "pc_true"):
            for key in ("nrmse", "nrmse_unmeasured"):
                a = np.asarray(blk["methods"][name][key])
                for m in COMPARE:
                    per = []
                    for k in seeds:
                        bb = ((runs[k]["results"].get(tier) or {}).get(cname) or {}).get("methods", {})
                        if m not in bb or key not in bb[m]:
                            per = None
                            break
                        per.append(np.asarray(bb[m][key], float))
                    if not per:
                        continue
                    p = paired_relative([a] * len(per), per, n_boot=1000)
                    ent = {"mean_other": float(np.mean([q.mean() for q in per])), **p}
                    if key == "nrmse":
                        mx = np.concatenate([np.asarray(runs[k]["results"][tier][cname]["methods"][m]["max_nerr"])
                                             for k in seeds])
                        ent.update(max_nerr_other_median=float(np.median(mx)), max_nerr_other_max=float(mx.max()))
                    blk["compare"].setdefault(name if key == "nrmse" else name + ":unmeasured", {})[m] = ent
        # bursts of the other methods from the saved trajectories (4 clean episodes per seed)
        if cname == "clean" and tier in ("shape_ood", "sequence_ood"):
            ob = {}
            for m in COMPARE:
                nes = []
                for k in seeds:
                    p = TREE / node / "runs" / system / f"seed{k}" / "traj.npz"
                    if not p.exists():
                        continue
                    z = np.load(p)
                    if f"{tier}/clean/{m}" not in z.files:
                        continue
                    xt, xm = z[f"{tier}/clean/true"].astype(np.float64), z[f"{tier}/clean/{m}"].astype(np.float64)
                    nes.append(np.sqrt(np.mean(np.clip((xm - xt) / np.asarray(sigma), -100, 100) ** 2, -1)))
                if nes:
                    ne = np.concatenate(nes)
                    ob[m] = {str(t): bursts(ne, settle, des.dt, lam_ref.get(tier, float("nan")), t) for t in THR}
                    ob[m]["ne_p999_settled"] = float(np.percentile(ne[:, settle:], 99.9))
                    ob[m]["n_episodes"] = int(len(ne))
            ne_pc = {}
            for name, x in xh.items():
                s_ = torch.as_tensor(np.asarray(sigma), dtype=torch.float64)
                ne = (((x[:4] - es.x[:4].double()) / s_).clamp(-100, 100).pow(2).mean(-1).sqrt()).numpy()
                ne_pc[name] = {str(t): bursts(ne, settle, des.dt, lam_ref.get(tier, float("nan")), t) for t in THR}
                ne_pc[name]["ne_p999_settled"] = float(np.percentile(ne[:, settle:], 99.9))
            blk["bursts_traj4"] = {"others": ob, "pc_same4": ne_pc}
        res["blocks"][f"{tier}/{cname}"] = blk
        log(f"  {node} {system} {tier}/{cname}: PC {np.mean(blk['methods']['pc']['nrmse']):.4f}  "
            f"PC(true) {np.mean(blk['methods']['pc_true']['nrmse']):.4f}  ({time.time() - t0:.0f}s)")
    return res


def main():
    t0 = time.time()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "script": "experiments/chaos_refs/chaos_refs.py",
           "lyapunov": {}, "pc": {}}
    K = {"lorenz": int(os.environ.get("K_LORENZ", 6)), "rossler": int(os.environ.get("K_ROSSLER", 6))}
    LY = {"lorenz": [("C3-faults", "nominal"), ("C3-realistic", "mismatch")],
          "rossler": [("TEST-C1", "nominal"), ("TEST-C3", "mismatch")]}
    for system, lst in LY.items():
        for node, var in lst:
            print(f"[lyapunov] {system} {var} plant on {node} inputs, K={K[system]}", flush=True)
            res["lyapunov"][f"{system}/{var}"] = lyap_system(system, node, K[system], var)
    for node, system, label in NODES:
        var = "mismatch" if node in ("C3-realistic", "TEST-C3") else "nominal"
        lv = res["lyapunov"][f"{system}/{var}"]
        lam_ref = {t: lv["tiers"][t]["lam_long"]["mean"] for t in lv["tiers"] if t != "u=0"}
        print(f"[pc] {node} {system} ({label}); lambda_ref {lam_ref}", flush=True)
        res["pc"][f"{node}/{system}"] = run_node(node, system, label, lam_ref)
        res["pc"][f"{node}/{system}"]["lambda_ref"] = lam_ref
    res["wall_s"] = time.time() - t0
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"done in {res['wall_s']:.0f}s -> {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
