"""Stage B1 job: hybrid innovation correction on top of a trained Stage-A arm (PREREG addendum B).

    python run_b.py --node C3-partial --system duffing --seed 0 --base-arm A3
Writes runs/<node>/<system>/seed<k>/B1-<base>.json (+ greybox_ekf_refit.json, the in-copy refit
of the comparator, as a reproduction check of the stored grey-box numbers)."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import run_bg as RB
from hyperkkl import baselines_data as BD
from hyperkkl import beat_b as BB
from hyperkkl.config import deep_merge
from hyperkkl.evaluate import evaluate, initial_guess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--base-arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_num_threads(2)
    j = RB.Job(a.node, a.system, a.seed, a.device)
    name = f"B1-{a.base_arm}"
    if (j.out / f"{name}.json").exists():
        j.log(f"{name} done"); return
    t0 = time.time()
    kkl, info = j.train_arm(a.base_arm)                 # cached models: load only
    cfg = j.cfg
    fits = BD.fit_all(dict(cfg, baselines=["greybox_ekf"]), j.data0, j.scales, j.des0, a.device, a.seed, j.log)
    gm = fits["greybox"]["model"]
    sc, dcfg = cfg["system"], cfg["data"]
    q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(j.scales["x_std"])
    r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(j.scales["y_std"])
    box = np.asarray(sc["ic_box"], float)
    p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
    # Sigma_KKL and c on the VALIDATION episodes only
    des, data, scl = j.design(info["spec"]["design"])
    va = data["val"].to(a.device)
    x0 = initial_guess(cfg, len(va), "center", 0, a.device)
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (va.x.shape[1] - 1))
    s_kkl = BB.kkl_error_std(kkl, va, x0, settle)
    rows = []
    for c in BB.C_GRID:
        hob = BB.HybridKKL(kkl, gm, j.des0.dt, q, r, p0, s_kkl, c)
        v = RB.BG.validation_nrmse(cfg, hob, data["val"], scl, a.device)
        rows.append({"c": c, **v})
        j.log(f"  B1 c={c}: val nrmse {v['val_nrmse']:.4f} div {v['val_diverged']:.2f}")
    best = min(rows, key=lambda x: x["val_nrmse"] + x["val_diverged"])
    hob = BB.HybridKKL(kkl, gm, j.des0.dt, q, r, p0, s_kkl, best["c"])
    obs = {name: hob, "greybox_ekf_refit": BD.make_observer("greybox_ekf", cfg, j.des0, j.scales, q, r, p0, fits)}
    timing = {}
    res = evaluate(cfg, j.des0, j.scales, obs, a.device, j.log, timing=timing)
    for nm in obs:
        per = {t: {"nrmse": cs["clean"]["methods"][nm]["nrmse"], "diverged": cs["clean"]["methods"][nm]["diverged"],
                   "scorecard": cs["clean"]["scorecard"][nm]} for t, cs in res.items()}
        payload = {"node": a.node, "system": a.system, "seed": a.seed, "arm": nm, "base_arm": a.base_arm,
                   "info": {"val": {"val_nrmse": best["val_nrmse"], "val_diverged": best["val_diverged"]} if nm == name else None,
                            "c_grid": rows, "c": best["c"], "sigma_kkl": s_kkl.tolist(), "base_info": info,
                            "greybox_fit": fits["info"]["greybox_ekf"], "wall_s": time.time() - t0},
                   "results": per, "eval_timing": timing.get(nm), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (j.out / f"{nm}.json").write_text(json.dumps(payload, default=RB._jd))
        j.log(f"  saved {nm}: seq {np.mean(per['sequence_ood']['nrmse']):.4f}")


if __name__ == "__main__":
    main()
