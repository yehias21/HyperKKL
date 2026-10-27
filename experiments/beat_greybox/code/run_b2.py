"""Stage B2 job (PREREG addendum B): FINAL's recipe retrained in phase 2 with the grey-box one-step
prediction of the frozen FINAL observer's previous estimate as one more hypernetwork-context channel.

    python run_b2.py --node C3-partial --system duffing --seed 0 --base-arm A4+D3
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import run_bg as RB
from hyperkkl import baselines_data as BD
from hyperkkl import beat_b as BB
from hyperkkl import cache as Cm
from hyperkkl.config import deep_merge, digest
from hyperkkl.evaluate import _run_chunks, evaluate, initial_guess
from hyperkkl.observers import LatentStep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--base-arm", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = a.device
    j = RB.Job(a.node, a.system, a.seed, dev)
    name = f"B2-{a.base_arm}"
    if (j.out / f"{name}.json").exists():
        j.log(f"{name} done"); return
    t0 = time.time()
    ob1, info = j.train_arm(a.base_arm)                       # frozen stage-1 (cached)
    spec = info["spec"]
    des, data, sc = j.design(spec["design"])
    cfg = j.cfg
    fits = BD.fit_all(dict(cfg, baselines=["greybox_ekf"]), j.data0, j.scales, j.des0, dev, a.seed, j.log)
    gm = fits["greybox"]["model"]
    sig = torch.as_tensor(j.scales["x_std"], dtype=torch.float32, device=dev)

    def ext_of(es):
        es = es.to(dev)
        x0 = initial_guess(cfg, len(es), "center", 0, dev)
        with torch.no_grad():
            xh = _run_chunks(ob1, es.y, es.u, x0, 1024)
        return BB.ext_sequence(gm, xh, es.u, j.des0.dt, sig).cpu()

    t1 = time.time()
    extra = {"train": ext_of(data["train"]), "val": ext_of(data["val"])}
    j.log(f"  B2 stage-1 features on train/val ({time.time() - t1:.0f}s)")
    cfg1 = deep_merge(cfg, {"phase1": spec["p1"]})
    base, _ = j.base(spec["design"], spec["p1"])
    k2 = digest(cfg1, "system", "data", "phase1", "phase2",
                extra=[a.seed, j.mh, a.node, spec["design"], spec["p2"], "B2", fits["info"]["greybox_ekf"]["val_loss"]])
    r = Cm.cached(RB.MC / f"{j.tag()}_B2_{k2}.pt",
                  lambda: BB.phase2_ext("lora", cfg1, data, sc, des, base, dev, a.seed, spec["p2"], j.log, extra=extra))
    r = {kk: (v.to(dev) if isinstance(v, torch.nn.Module) else v) for kk, v in r.items()}
    ob = BB.ExtCtxObserver(name, LatentStep(des, dev), r["cond"], r["ctx"], r["hist"], r["gx"], ob1, gm,
                           j.des0.dt, sig, z0_map=base["T0"])
    v = RB.BG.validation_nrmse(cfg, ob, data["val"], sc, dev)
    j.log(f"  B2 val nrmse {v['val_nrmse']:.4f}")
    timing = {}
    res = evaluate(cfg, j.des0, j.scales, {name: ob}, dev, j.log, timing=timing)
    per = {t: {"nrmse": cs["clean"]["methods"][name]["nrmse"], "diverged": cs["clean"]["methods"][name]["diverged"],
               "scorecard": cs["clean"]["scorecard"][name]} for t, cs in res.items()}
    payload = {"node": a.node, "system": a.system, "seed": a.seed, "arm": name, "base_arm": a.base_arm,
               "info": {"val": v, "base_info": info, "gx_params": r["gx_params"], "phase2_wall_s": r["wall_s"],
                        "greybox_fit": fits["info"]["greybox_ekf"], "wall_s": time.time() - t0},
               "results": per, "eval_timing": timing.get(name), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (j.out / f"{name}.json").write_text(json.dumps(payload, default=RB._jd))
    j.log(f"  saved {name}: seq {np.mean(per['sequence_ood']['nrmse']):.4f}")


if __name__ == "__main__":
    main()
