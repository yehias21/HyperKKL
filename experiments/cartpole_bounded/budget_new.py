"""Corrected information budget (Part A method) for the new plants under the TEST-C3 setting. CPU.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/cartpole_bounded/budget_new.py cartpole_b
Episodes: hyperkkl.data.eval_data on CPU (same generator as the runs; CPU noise draws, so not the runs'
exact episodes) with TEST-C3's overrides (knowledge partial, mismatch 0.3, colored_t 0.03 / 0.9 / 3),
written to a scratch cache. Bound: infolib_c3 (whitened AR(1), Student-t Fisher factor), true plant.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "information"))
import infolib as L  # noqa: E402
import infolib_c3 as C  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.config import load  # noqa: E402

torch.set_num_threads(8)
HERE = Path(__file__).resolve().parent
Dm.CACHE = Path(os.environ.get("INCL_CACHE", "/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/"
                               "7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/incl_cache"))
C3 = {"system.knowledge": "partial", "system.mismatch": 0.3,
      "data.sensor": {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}}


def main(name):
    cfg = load(name, C3)
    scales = Dm.system_scales(cfg, "cpu")
    scfg = cfg["system"]
    plant = L.build_plant({k: v for k, v in scfg.items() if k != "knowledge"}, perturb=0.3)
    sensors = list(plant.sensors); a = float(scfg["alpha"])
    es_id = Dm.eval_data(cfg, "id", device="cpu")
    settle = int(0.5 * (es_id.x.shape[1] - 1))
    Sig = L.climatology(es_id.x[:, settle:].double())
    out = {"plant": name, "alpha": a, "x_std": scales["x_std"], "dists": {}}
    for d, tier in (("eval_id", "id"), ("eval_seq", "sequence_ood")):
        es = es_id if tier == "id" else Dm.eval_data(cfg, tier, device="cpu")
        out["dists"][d] = {}
        for v, sen in (("white_gauss", {"noise": 0.03}), ("c3", None)):
            b = C.budget_c3(plant, es.x, es.u, cfg, scales, sensors, [1 / a, 3 / a, 10 / a], Sig, sensor_override=sen)
            out["dists"][d][v] = b
            w = b["bayes"]["windows"][f"{3 / a:g}"]; f = b["bayes"]["full"]
            print(f"{name} {d} {v}: A(3/a) unm {w['unmeasured_lb']:.4f} state {[round(c, 4) for c in w['crb_state']]} | "
                  f"full unm {f['unmeasured_lb']:.4f} state {[round(c, 4) for c in f['crb_state']]}", flush=True)
    (HERE / f"budget_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
