"""Re-fit the grey-box baselines in the copy where the tree has not merged them (PREREG section 6):
same code path as job.run_job (tune_oracle on validation, baselines_data.fit_all, evaluate).

    python run_gbrefit.py --node TEST-C3 --system network --seed 0
Writes runs/<node>/<system>/seed<k>/{greybox_ekf,greybox_ukf,greybox_ekf_oracle}.json."""
import argparse
import json
import time

import numpy as np
import torch

import run_bg as RB
from hyperkkl import baselines_data as BD
from hyperkkl.config import deep_merge
from hyperkkl.evaluate import BlockOracle, evaluate, oracle_maker
from hyperkkl.job import tune_oracle

NAMES = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]

ap = argparse.ArgumentParser()
ap.add_argument("--node"); ap.add_argument("--system"); ap.add_argument("--seed", type=int); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()
torch.set_num_threads(2)
j = RB.Job(a.node, a.system, a.seed, a.device)
cfg = deep_merge(j.cfg, {"baselines": NAMES})
if not cfg["eval"].get("oracle_tuning"):
    cfg["eval"]["oracle_tuning"], _ = tune_oracle(cfg, j.des0, j.scales, j.data0["val"], a.device, j.log)
fits = BD.fit_all(cfg, j.data0, j.scales, j.des0, a.device, a.seed, j.log)
sc, dcfg = cfg["system"], cfg["data"]
q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(j.scales["x_std"])
r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(j.scales["y_std"])
box = np.asarray(sc["ic_box"], float)
p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
obs = {n: BD.make_observer(n, cfg, j.des0, j.scales, q, r, p0, fits, oracle_maker, BlockOracle) for n in NAMES}
res = evaluate(cfg, j.des0, j.scales, obs, a.device, j.log)
for n in NAMES:
    per = {t: {"nrmse": cs["clean"]["methods"][n]["nrmse"], "diverged": cs["clean"]["methods"][n]["diverged"]} for t, cs in res.items()}
    (j.out / f"{n}.json").write_text(json.dumps({"node": a.node, "system": a.system, "seed": a.seed, "arm": n, "refit_in_copy": True,
                                                 "info": {"fit": fits["info"][n], "oracle_tuning": cfg["eval"]["oracle_tuning"]},
                                                 "results": per, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=RB._jd))
    j.log(f"  saved {n}: seq {np.mean(per['sequence_ood']['nrmse']):.4f}")
