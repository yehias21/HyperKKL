"""One structural-campaign job = (system, tier in {ref, struct}, seed).

    python run_struct.py --system vdp --tier struct --seed 0 [--device cuda:0] [--smoke]

ref    = the C3-realistic / TEST-C3 setting exactly (partial model f_known at nominal parameters,
         real plant 30% off nominal, coloured Student-t sensor noise), config from the stored run
struct = the same + the hidden structural dynamics of structural.py in the REAL plant only.
Package = experiments/structural/code/pkg (snapshot of the beat_greybox scratch copy: live package
+ SOAP + beat.py). Caches under experiments/structural/cache; reference-tier data files that already
exist in the live cache are read from there READ-ONLY (identical keys), never written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
sys.path.insert(0, str(HERE))
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
LIVE_CACHE = LIVE / "experiments" / "cache"
EXP = HERE.parent
OUT = EXP / "runs"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import evaluate as Ev  # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.evaluate import build_observers, evaluate  # noqa: E402
from hyperkkl.job import tune_oracle  # noqa: E402
from hyperkkl.observers import _rk4  # noqa: E402
from hyperkkl.train import phase1, phase2  # noqa: E402

import structural as ST  # noqa: E402
import wingb as WG  # noqa: E402

Dm.CACHE = EXP / "cache"
MC = Dm.CACHE / "models"
ST.install(Dm, Ev)
_cached_orig = Dm._cached


def _cached(path, build, **kw):
    path = Path(path)
    try:
        live = LIVE_CACHE / path.relative_to(Dm.CACHE)
        if live.exists():
            return torch.load(live, weights_only=False, map_location="cpu")
    except ValueError:
        pass
    return _cached_orig(path, build, **kw)


Dm._cached = _cached
# system_scales reads/writes its JSON directly under Dm.CACHE / "scales": seed it from the live cache
_ss = Dm.system_scales


def system_scales(cfg, device="cpu"):
    import shutil
    (Dm.CACHE / "scales").mkdir(parents=True, exist_ok=True)
    for p in (LIVE_CACHE / "scales").glob(f"{cfg['system']['name']}_*.json"):
        q = Dm.CACHE / "scales" / p.name
        if not q.exists():
            shutil.copy(p, q)
    return _ss(cfg, device)


Dm.system_scales = system_scales

SYSTEMS = ["quad_tank", "battery", "cartpole", "highway_traffic", "bioreactor", "vdp", "lorenz"]
NODE = {s: ("TEST-C3" if s in ("cartpole", "highway_traffic") else "C3-realistic") for s in SYSTEMS}
BASELINES = ["ekf", "ukf", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
# A4 of experiments/beat_greybox/PREREG.md: S (SOAP) + P (base decoder trainable, lr x 0.1) +
# H (strided z-window context) + R (decoder width 256, budget 280k, 6000 phase-2 steps)
A4_P1 = {"optimizer": "soap", "lr": 1e-2, "dec_width": 256}
A4_P2 = {"optimizer": "soap", "lr": 1e-2, "budget": 280000, "steps": 6000, "base_lr_factor": 0.1, "window": 20}


def stride_rule(cfg, des) -> int:
    a = float(cfg["system"]["alpha"]) * float(cfg["system"].get("alpha_scale", 1.0))
    return max(1, int(round(6.0 / (a * des.dt) / 20.0)))


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


@torch.no_grad()
def repairability(model, gb_model, val, scales, dt, H, seed, device, n_per_ep=64) -> dict:
    """A-priori repairability (PREREG.md): open-loop H-step prediction of the MEASURED output from
    the true state x_k (validation episodes, k after the washout), by the modeller's model and by the
    grey-box model (model + fitted residual). rho = 1 - MSE_greybox / MSE_model (errors in y_std
    units, clipped at 100). 1 = the residual removes all model error visible in the output."""
    rng = np.random.RandomState(seed + 5)
    E, N1 = val.x.shape[0], val.x.shape[1]
    ks = np.stack([rng.randint(val.burn, N1 - 1 - H, size=n_per_ep) for _ in range(E)])
    ep = np.repeat(np.arange(E), n_per_ep)
    k = ks.reshape(-1)
    X = val.x.to(device).double(); U = val.u.to(device).double(); Yc = val.y_clean.to(device).double()
    ys = torch.as_tensor(np.asarray(scales["y_std"], float), dtype=torch.float64, device=device)
    sens = list(model.sensors)
    out = {}
    for nm, m in (("model", model), ("greybox", gb_model)):
        x = X[ep, k]
        for j in range(H):
            uj = U[ep, k + j]
            x = _rk4(m, x, uj, uj, uj, float(dt))
            x = torch.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
        e = ((x[:, sens] - Yc[ep, k + H]) / ys).clamp(-100, 100)
        out[f"mse_{nm}"] = float(e.pow(2).mean())
    out["rho"] = 1.0 - out["mse_greybox"] / max(out["mse_model"], 1e-12)
    out["H"] = int(H)
    return out


def run(system, tier, seed, device, smoke=False, log_to=None):
    src = LIVE / "experiments" / "tree" / NODE[system] / "runs" / system / f"seed{seed}" / "results.json"
    cfg = json.load(open(src))["config"]
    cfg["seed"] = seed
    cfg["methods"] = []
    cfg["baselines"] = list(BASELINES)
    cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
    if tier == "struct":
        cfg["system"]["structural"] = ST.STRUCTURAL[system].NAME_S
    elif tier != "ref":
        raise ValueError(tier)
    if smoke:
        cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40},
                               "phase2": {"steps": 20, "val_every": 10},
                               "data_baselines": {"gb_steps": 30, "gb_val_every": 10}})
    out = OUT / ("smoke" if smoke else "") / system / tier / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        print("exists", out); return
    logf = open(out / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    t_start = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    log(f"structural job {system} tier {tier} seed {seed} device {device} smoke {smoke}")
    scales = Dm.system_scales(cfg, device)
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, seed, device)
    log(f"  data: p1 {len(data['p1'])} train {len(data['train'])} val {len(data['val'])} "
        f"rejected {data['train'].n_rejected}  burn {data['train'].burn}  dt {des.dt:.4f} N {des.N}")
    mh = module_hash(MODEL_MODULES)
    tag = ("smoke_" if smoke else "") + f"{system}_{tier}_s{seed}"
    info = {}

    # --- HyperKKL A4 (primary) and A0 (secondary)
    learned = {}
    for arm, p1o in (("A4", A4_P1), ("A0", {})):
        c1 = deep_merge(cfg, {"phase1": p1o})
        k1 = digest(c1, "system", "data", "phase1", extra=[seed, mh, "structural"])
        t0 = time.time()
        base = Cm.cached(MC / f"{tag}_p1_{k1}.pt", lambda: phase1(c1, data, scales, des, device, seed, log))
        base = {kk: (v.to(device) if isinstance(v, torch.nn.Module) else v) for kk, v in base.items()}
        t_p1 = time.time() - t0
        t0 = time.time()
        if arm == "A4":
            p2 = dict(A4_P2, stride=stride_rule(cfg, des))
            if smoke:
                p2["steps"] = 20
            k2 = digest(c1, "system", "data", "phase1", "phase2", extra=[seed, mh, p2, "structural"])
            r = Cm.cached(MC / f"{tag}_p2A4_{k2}.pt",
                          lambda: BG.phase2_bg("lora", c1, data, scales, des, base, device, seed, p2, log))
        else:
            k2 = digest(c1, "system", "data", "phase1", "phase2", extra=[seed, mh, "lora", "structural"])
            r = Cm.cached(MC / f"{tag}_p2A0_{k2}.pt",
                          lambda: phase2("lora", c1, data, scales, des, base, device, seed, log))
        r = {kk: (v.to(device) if isinstance(v, torch.nn.Module) else v) for kk, v in r.items()}
        name = "hyperkkl_A4" if arm == "A4" else "hyperkkl_A0"
        learned[name] = BG.make_observer(name, des, base, r if arm == "A4" else {**r, "hist": None}, device)
        info[name] = {"phase1_s": t_p1, "phase2_s": time.time() - t0, "diag": base["diag"],
                      "n_params": r.get("n_params"), "trainable_total": r.get("trainable_total"),
                      "val_loss": r.get("val")}
        info[name]["val"] = BG.validation_nrmse(cfg, learned[name], data["val"], scales, device)
        log(f"  {name}: val nrmse {info[name]['val']['val_nrmse']:.4f}")

    # --- filters
    cfg["eval"]["oracle_tuning"], info["oracle_tuning"] = tune_oracle(cfg, des, scales, data["val"], device, log)
    fits = BD.fit_all(cfg, data, scales, des, device, seed, log)
    info["greybox"] = fits["info"].get("greybox_ekf")
    obs = build_observers(cfg, des, scales, None, {}, None, device, fits)
    model = Dm.model_plant(cfg["system"])
    wf = WG.fit_wingb(model, data, scales, des.dt, float(cfg["system"].get("u_scale", 1.0)),
                      float(cfg["system"].get("t_ref", 1.0)), seed, device, BD.options(cfg), log)
    info["gbw"] = wf["info"]
    q = float(cfg["data"].get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
    rr = max(float(cfg["data"]["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(scales["y_std"])
    box = np.asarray(cfg["system"]["ic_box"], float)
    p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
    obs["gbw_ekf"] = WG.WinEKF(wf["model"], des.dt, q, rr, p0, "gbw_ekf")
    obs["gbw_ukf"] = WG.WinUKF(wf["model"], des.dt, q, rr, p0, "gbw_ukf")
    tr = data["train"]
    obs["climatology"] = WG.Climatology(tr.x[:, tr.burn:].reshape(-1, tr.x.shape[-1]).mean(0).numpy())
    obs.update(learned)

    # --- a-priori repairability (validation episodes only)
    H = max(1, int(round(float(cfg["system"].get("t_ref", 1.0)) / des.dt)))
    info["repairability"] = repairability(model, fits["greybox"]["model"], data["val"], scales, des.dt, H, seed, device)
    log(f"  repairability: {info['repairability']}")

    t0 = time.time()
    timing = {}
    res = evaluate(cfg, des, scales, obs, device, log, timing=timing)
    per = {}
    for t, cs in res.items():
        per[t] = {m: {k: v for k, v in d.items() if k in ("nrmse", "nrmse_unmeasured", "nrmse_measured", "diverged")}
                  for m, d in cs["clean"]["methods"].items()}
    payload = {"system": system, "tier": tier, "seed": seed, "structural": cfg["system"].get("structural"),
               "info": info, "results": per, "eval_s": time.time() - t0, "eval_timing": timing,
               "wall_s": time.time() - t_start, "scales": scales, "config": cfg, "module_hash": mh,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = out / "results.json.tmp"
    tmp.write_text(json.dumps(payload, default=_jd))
    tmp.rename(out / "results.json")
    log("  seq means: " + "  ".join(f"{m} {np.mean(v['nrmse']):.4f}" for m, v in per["sequence_ood"].items()))
    log(f"done in {time.time() - t_start:.0f}s; peak mem "
        f"{torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.2f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--tier", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    try:
        run(a.system, a.tier, a.seed, a.device, a.smoke)
    except Exception:
        d = OUT / ("smoke" if a.smoke else "") / a.system / a.tier / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
