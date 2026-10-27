"""Per-system information budgets (bits) and implied NRMSE lower bounds for the 12 plants.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/information/budgets.py

Inputs: the cached evaluation episodes (id, sequence_ood) and training episodes (seed 0) of the
finished runs (read-only; C1-where for dev plants, TEST-C1 for test plants), their stored scales,
and the plant with full knowledge. Output: experiments/information/budgets.json.
Windows: {1, 2, 3, 5, 10} / alpha seconds (the latent filter forgets at rate alpha) and the full
past since t = 0 (the bound for an observer with unbounded memory).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import infolib as L  # noqa: E402

torch.set_num_threads(8)
WIN_MULT = [1, 2, 3, 5, 10]
OUT = HERE / "budgets.json"


def one(system: str) -> dict:
    pay = L.run_payload(system)
    cfg, scales = pay["config"], pay["scales"]
    scfg = cfg["system"]
    plant = L.build_plant(scfg)
    sensors = list(plant.sensors)
    alpha = float(scfg["alpha"])
    windows = [m / alpha for m in WIN_MULT]
    f_slow = L.slow_mode_rate(plant, scales["x_mean"])
    res = {"system": system, "node": L.RUN_FULL[system], "nx": plant.nx, "ny": len(sensors), "sensors": sensors,
           "alpha": alpha, "dt": scales["dt"], "T": scfg["T"], "t_ref": scfg.get("t_ref", 1.0),
           "slow_mode_rate": f_slow, "slow_mode_tau_s": 1.0 / max(f_slow, 1e-12),
           "x_std": scales["x_std"], "y_std": scales["y_std"], "windows_s": windows, "dists": {}}
    es_id = L.load_cached_eval(cfg, "id")
    settle = int(0.5 * (es_id.x.shape[1] - 1))
    Sigma_id = L.climatology(es_id.x[:, settle:].double())     # the NRMSE's climatology prior, fixed per system
    for name, loader in (("eval_id", lambda: es_id), ("eval_seq", lambda: L.load_cached_eval(cfg, "sequence_ood")),
                         ("train", lambda: L.load_cached_train(cfg, 0)["train"])):
        try:
            es = loader()
        except FileNotFoundError as e:
            res["dists"][name] = {"error": str(e)}
            continue
        t0 = time.time()
        b = L.budget_along(plant, es.x, es.u, cfg, scales, sensors, windows, n_ep=min(64, es.x.shape[0]), n_t=12,
                           Sigma=Sigma_id)
        b["input"] = L.input_descriptors(es.u[:64], scales["dt"], f_slow / (2 * np.pi))
        b["families"] = sorted(set(es.families))
        b["wall_s"] = time.time() - t0
        res["dists"][name] = b
        f, w3 = b["bayes"]["full"], b["bayes"]["windows"]["%g" % (3 / alpha)]
        print(f"  {system:16s} {name:8s} bits full {f['bits']:6.2f} (3/a: {w3['bits']:6.2f})  "
              f"lb total {f['nrmse_lb']:.4f} unm {f['unmeasured_lb']:.4f} | 3/a total {w3['nrmse_lb']:.4f} "
              f"unm {w3['unmeasured_lb']:.4f}  ({b['wall_s']:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    systems = sys.argv[1:] or L.ALL
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for s in systems:
        print(s, flush=True)
        try:
            out[s] = one(s)
        except Exception as e:                                   # keep going, report at the end
            import traceback
            traceback.print_exc()
            out[s] = {"system": s, "error": repr(e)}
        OUT.write_text(json.dumps(out, indent=1))
    print("written", OUT)
