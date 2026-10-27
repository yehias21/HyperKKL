"""Exp 1+2: same evaluation episodes, sensor noise 0 / 1 / 3 / 10 % (z recomputed online by
every observer from the new y). Cached learned models; EKF and grey-box EKF (fitted in memory,
as reeval does) keep their nominal 3% R.  usage: exp12.py NODE SYSTEM SEED DEVICE OUTDIR"""
import copy, json, sys, time
from pathlib import Path
import numpy as np, torch
import common as K
from hyperkkl import baselines_data as BD
from hyperkkl.evaluate import build_observers, evaluate

node, system, seed, device, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
torch.set_num_threads(2)
t0 = time.time()
payload, cfg, scales, des, data, base, trained = K.load_run(node, system, seed, device, ["latent", "lora", "hyper"])
conds = {"clean": {}, "n0": {"sensor": {"noise": 0.0}}, "n3": {"sensor": {"noise": 0.03}}}
if node == "C1-where":
    conds.update({"n1": {"sensor": {"noise": 0.01}}, "n10": {"sensor": {"noise": 0.10}}})
cfg["conditions"] = {**cfg.get("conditions", {}), **conds}
cfg["eval"]["conditions"] = list(conds)
cfg["eval"]["save_traj"] = False
cfg["baselines"] = ["ekf", "greybox_ekf", "greybox_ukf"]
fits = BD.fit_all(cfg, data, scales, des, device, seed, log=lambda *a: None)
obs = build_observers(cfg, des, scales, base, trained, None, device, fits)
res = evaluate(cfg, des, scales, obs, device, log=print)
out = {"node": node, "system": system, "seed": seed, "wall_s": time.time() - t0, "res": {}}
stored = payload["results"]
for tier, cs in res.items():
    for c, blk in cs.items():
        for m, v in blk["methods"].items():
            out["res"].setdefault(tier, {}).setdefault(c, {})[m] = {"nrmse": v["nrmse"]}
        out["res"][tier]["_stored_clean"] = {m: {"nrmse": stored[tier]["clean"]["methods"][m]["nrmse"]}
                                            for m in stored[tier]["clean"]["methods"] if m in obs}
outdir.mkdir(parents=True, exist_ok=True)
(outdir / f"{node}_{system}_s{seed}.json").write_text(json.dumps(out))
print("done", node, system, seed, f"{time.time() - t0:.0f}s")
