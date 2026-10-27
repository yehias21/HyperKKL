"""GRU / Neural CDE observers on the real battery (PREREG.md section 5).

Same split, windows, masked loss (labelled states SOC, dT, Vm; V1, V2 are model placeholders) and
settled SOC RMSE [%] as experiments/real (hyperkkl.realdata_run: prep.pt read-only, expand,
soc_metrics). lr from the 4-value grid chosen on the validation cycle (Cycle_4) settled SOC RMSE.

    CUDA_VISIBLE_DEVICES=5 python real_run.py --kind gru --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import b2  # noqa: F401
from b2 import HERE
from b2 import models as M
from b2 import train as T
from b2.job import LRS, STEPS
from hyperkkl import realdata as R
from hyperkkl.realdata_run import CACHE, design, expand, run_soc, soc_metrics

OUT = HERE / "real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["gru", "ncde"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=140_000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"log_{a.kind}_seed{a.seed}.txt", "a")

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    prep = torch.load(CACHE / "prep.pt", weights_only=False)        # read-only
    sc, des = prep["scales"], design()
    sens = [3, 4]
    nx, nu, ny = 5, 1, 2
    xm = np.asarray(sc["x_mean"]); xs = np.asarray(sc["x_std"])
    norm = lambda: M._Norm(float(prep["scfg"]["u_scale"]), xm[sens], sc["y_std"], xm, xs)
    w = torch.zeros(nx, device=dev); w[list(R.LABELLED)] = 1.0
    tr = prep["sets"]["train"].to(dev)
    va = prep["sets"]["val"].to(dev)
    Yv, Uv, Sv, X0v, _, _ = expand(prep["sets"]["val"])
    steps = 20 if a.smoke else STEPS
    lrs = LRS[1:2] if a.smoke else LRS
    cands, best = [], None
    for lr in lrs:
        torch.manual_seed(a.seed)
        if a.kind == "gru":
            H = M.solve_gru(nu, ny, nx, a.budget)
            net = M.GRUNet(nu, ny, nx, H, 2, norm()); arch = {"hidden": H, "layers": 2}
        else:
            W = M.solve_ncde(nu, ny, nx, 32, a.budget)
            net = M.NCDENet(nu, ny, nx, 32, W, norm(), des.dt / 1.0); arch = {"dz": 32, "width": W}
        info = T.train_seq(net, tr, va, steps, lr, dev, a.seed, log, val_every=10 if a.smoke else 250,
                           state_w=w)
        ob = M.SeqObserver(a.kind, net)
        sh = run_soc(ob, Yv, Uv, X0v, dev)
        v = float(np.mean(soc_metrics(sh, Sv.numpy())["rmse_settled"]))
        log(f"  {a.kind} lr {lr:g}: val loss {info['val_loss']:.4g}  val settled SOC RMSE {v:.3f} %")
        cands.append({"lr": lr, "val_rmse_settled": v, "val_loss": info["val_loss"], "wall_s": info["wall_s"]})
        if best is None or v <= min(c["val_rmse_settled"] for c in cands):
            best = net
    i = min(range(len(cands)), key=lambda j: cands[j]["val_rmse_settled"])
    ob = M.SeqObserver(a.kind, best)
    res = {}
    for name, es in prep["sets"].items():
        if name == "train":
            continue
        Y, U, S, X0, idx, sgn = expand(es)
        sh = run_soc(ob, Y, U, X0, dev)
        m = soc_metrics(sh, S.numpy())
        res[name] = {"rmse_settled": m["rmse_settled"].tolist(), "rmse_all": m["rmse_all"].tolist(),
                     "start": [es.meta[j]["start"] for j in idx]}
        log(f"  {name}: settled SOC RMSE {np.mean(m['rmse_settled']):.3f} %")
    out = {"kind": a.kind, "seed": a.seed, "lr": cands[i]["lr"], "candidates": cands, "arch": arch,
           "n_params": sum(p.numel() for p in best.parameters()), "results": res, "smoke": a.smoke}
    (OUT / f"{a.kind}_seed{a.seed}{'_smoke' if a.smoke else ''}.json").write_text(json.dumps(out))
    torch.save(best.state_dict(), OUT / f"{a.kind}_seed{a.seed}{'_smoke' if a.smoke else ''}.pt")
    log("done")


if __name__ == "__main__":
    main()
