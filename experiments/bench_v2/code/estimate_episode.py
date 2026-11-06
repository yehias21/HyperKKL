"""Illustration only: re-train the HyperKKL default (hk_base = A3 + mixed, no C5) for one seed with the lr that the
pre-registered selection chose for that seed (results_default.json), and save its estimate on one public test episode.
Phase-2 weights of the study were not stored; this re-run is not used for any reported number.

    CUDA_VISIBLE_DEVICES=2 python estimate_episode.py driven cartpole_b realistic --seed 0 --episode 2
Writes ../illustration/<family>__<plant>__seed<k>__ep<e>.npz (t, u, y, x, xhat, sensors, settle, lr, nrmse_unmeasured).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import env
from v2run import Cell, get_p1, p2_opts, run_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family"); ap.add_argument("plant"); ap.add_argument("variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode", type=int, default=2)
    a = ap.parse_args()
    import jf
    torch.set_num_threads(2)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
    res = json.loads((env.RUNS.parent / "results_default.json").read_text())
    lr = res["cells"][f"{a.family}/{a.plant}/{a.variant}"]["summary"]["hk_base"]["lr"][str(a.seed)]
    cell = Cell(a.family, a.plant, a.variant, dev)
    base = get_p1(cell, a.seed, log)
    cfg = dict(cell.cfg); cfg["seed"] = a.seed
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rb = jf.phase2_jf(cfg, cell.data, cell.sc, cell.des, base, dev, a.seed, p2_opts(cell, lr), log, {})
    ob = jf.make_observer("hk_base", cell.des, base, rb, dev)
    split = "test_op_id" if a.family == "driven" else "test_id"
    es, Y, U, X0 = cell.condition(cell.L(split), split, "clean")
    e = slice(a.episode, a.episode + 1)
    with torch.no_grad():
        xh = run_safe(ob, Y[e].to(dev), U[e].to(dev), X0[e].to(dev), 1, log, "illustration")
    sc = cell.score(es.select([a.episode]), xh)
    out = env.RUNS.parent / "illustration"
    out.mkdir(exist_ok=True)
    f = out / f"{a.family}__{a.plant}__seed{a.seed}__ep{a.episode}.npz"
    np.savez(f, t=np.arange(es.x.shape[1]) * cell.des.dt, u=U[e][0].numpy(), y=Y[e][0].numpy(),
             x=es.x[a.episode].numpy(), xhat=xh[0].cpu().numpy(), sensors=np.array(cell.sensors),
             settle=cell.settle, lr=lr, nrmse_unmeasured=np.mean(sc["nrmse_unmeasured"]))
    log(f"wrote {f}: lr {lr}, episode NRMSE (unmeasured) {np.mean(sc['nrmse_unmeasured']):.4f}")


if __name__ == "__main__":
    main()
