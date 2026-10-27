"""POST HOC (2026-10-26) budget fix: information budgets under the TEST-C3 noise model and true plant.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/information/budgets_c3.py [plants]

For each test plant: the TEST-C3 run's config (seed 0), its cached evaluation episodes (generated on
the TRUE plant build_plant(scfg, perturb=0.3) with colored-t sensor noise), Jacobians of the TRUE
plant, and three sensor models (decomposition):
  white_gauss : the old assumption (white Gaussian 0.03) but on the true-plant episodes
  colored_gauss : AR(1) rho = 0.9 Gaussian (whitened)
  c3          : AR(1) rho = 0.9 Student-t nu = 3 (whitened, t Fisher factor) = the corrected budget
Distributions: eval_id and eval_seq (the sequence_ood tier). Output budgets_c3.json.
Dev plants: their comparison (C3-regimes, clean) uses white Gaussian 0.03, mismatch 0, i.e. the
setting of budgets.json (C1-where) -- not recomputed (see BUDGET_FIX.md).
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
import infolib_c3 as C  # noqa: E402

torch.set_num_threads(8)
WIN_MULT = [1, 3, 10]
OUT = HERE / "budgets_c3.json"
NODE = "TEST-C3"


def one(system, node=NODE):
    pay = L.run_payload(system, node)
    cfg, scales = pay["config"], pay["scales"]
    scfg = cfg["system"]
    true_cfg = {k: v for k, v in scfg.items() if k != "knowledge"}
    plant = L.build_plant(true_cfg, perturb=float(scfg.get("mismatch", 0.0)))   # data generator
    sensors = list(plant.sensors)
    alpha = float(scfg["alpha"])
    windows = [m / alpha for m in WIN_MULT]
    res = {"system": system, "node": node, "mismatch": scfg.get("mismatch", 0.0), "sensor": cfg["data"]["sensor"],
           "alpha": alpha, "sensors": sensors, "nx": plant.nx, "windows_s": windows, "dists": {}}
    es_id = L.load_cached_eval(cfg, "id")
    settle = int(0.5 * (es_id.x.shape[1] - 1))
    Sigma = L.climatology(es_id.x[:, settle:].double())
    variants = {"white_gauss": {"noise": 0.03},
                "colored_gauss": {"noise": 0.03, "kind": "colored", "rho": 0.9},
                "c3": None}
    for dname, tier in (("eval_id", "id"), ("eval_seq", "sequence_ood")):
        es = es_id if tier == "id" else L.load_cached_eval(cfg, tier)
        res["dists"][dname] = {}
        for vname, sen in variants.items():
            t0 = time.time()
            b = C.budget_c3(plant, es.x, es.u, cfg, scales, sensors, windows, Sigma, sensor_override=sen)
            res["dists"][dname][vname] = b
            f, w3 = b["bayes"]["full"], b["bayes"]["windows"]["%g" % (3 / alpha)]
            print(f"  {system:16s} {dname:8s} {vname:13s} full unm {f['unmeasured_lb']:.4f} tot {f['nrmse_lb']:.4f} | "
                  f"3/a unm {w3['unmeasured_lb']:.4f} tot {w3['nrmse_lb']:.4f} bits {w3['bits']:.1f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return res


def selftest():
    """rho = 0, J_t = 1 must reproduce infolib's white-Gaussian Bayesian PCRB."""
    s = "fhn"
    pay = L.run_payload(s, "TEST-C1"); cfg, scales = pay["config"], pay["scales"]
    plant = L.build_plant(cfg["system"]); es = L.load_cached_eval(cfg, "id")
    x, u = es.x[:16].double(), es.u[:16].double()
    F, H = L.jacobians(plant, x, u, scales["dt"])
    Q, R = L.noise_matrices(cfg, scales)
    Sigma = L.climatology(x[:, 200:])
    k = torch.tensor([300, 400])
    a = L.pcrb_bayes_full(F, H, Q, R, Sigma, k)
    rho, Jv, Je, Jt = C.sensor_fisher(cfg, scales)
    b = C.bayes_full(F, H, Q, rho, Jv, Je, Sigma, k)
    err = float((a - b).abs().max() / a.abs().max())
    print("selftest rel err", err); assert err < 1e-8
    a = L.pcrb_bayes_window(F, H, Q, R, Sigma, k, 50); b = C.bayes_window(F, H, Q, rho, Jv, Je, Sigma, k, 50)
    err = float((a - b).abs().max() / a.abs().max()); print("selftest window rel err", err); assert err < 1e-8


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "selftest":
        selftest(); sys.exit()
    node = NODE
    if args and args[0].startswith("--node="):
        node = args.pop(0).split("=", 1)[1]
    out_path = OUT if node == NODE else HERE / f"budgets_c3_{node}.json"
    systems = args or L.TEST
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for s in systems:
        print(s, flush=True)
        try:
            out[s] = one(s, node)
        except Exception as e:
            import traceback; traceback.print_exc()
            out[s] = {"system": s, "error": repr(e)}
        out_path.write_text(json.dumps(out, indent=1))
    print("written", out_path)
