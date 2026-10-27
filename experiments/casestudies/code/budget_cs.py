"""Information budgets A (unmeasured Bayesian PCRB over the 3/alpha window) per case x knowledge level, BEFORE any
observer run (PREREG.md). CPU.

    CUDA_VISIBLE_DEVICES="" python budget_cs.py A [B]

Episodes: hyperkkl.data.eval_data (GPU-generated = the runs' cached episodes; recursion on CPU) with the cell's own config (true plant build_plant(scfg,
perturb=mismatch); the cell's sensor model incl. per-output noise, multi-rate hold and, at the realistic
level, AR(1) Student-t noise). The modeller's knowledge does not enter the bound (it bounds ANY
estimator); it enters only through the comparator's achieved error. Output: budgets/<case>.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as C  # noqa: E402

import torch  # noqa: E402

import infolib_cs as I  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402

torch.set_num_threads(8)
OUT = C.HERE / "budgets"


DEV = "cuda" if torch.cuda.is_available() else "cpu"     # episodes: the runs' own (GPU-generated, cached)


def one(case, level):
    cfg = C.case_cfg(case, level, 0, DEV)
    scales = Dm.system_scales(cfg, DEV)
    scfg = cfg["system"]
    plant = I.L.build_plant({k: v for k, v in scfg.items() if k != "knowledge"}, perturb=float(scfg.get("mismatch", 0.0)))
    sensors = list(plant.sensors)
    a = float(scfg["alpha"])
    es_id = Dm.eval_data(cfg, "id", device=DEV)
    settle = int(0.5 * (es_id.x.shape[1] - 1))
    Sig = I.L.climatology(es_id.x[:, settle:].double())
    res = {"case": case, "level": level, "alpha": a, "x_std": scales["x_std"], "sensors": sensors, "dists": {}}
    for d, tier in (("eval_id", "id"), ("eval_seq", "sequence_ood"), ("eval_shape", "shape_ood")):
        t0 = time.time()
        es = es_id if tier == "id" else Dm.eval_data(cfg, tier, device=DEV)
        b = I.budget(plant, es.x, es.u, cfg, scales, sensors, [1 / a, 3 / a], Sig)
        res["dists"][d] = b
        w = b["bayes"]["windows"][f"{3 / a:g}"]; f = b["bayes"]["full"]
        print(f"{case} {level:9s} {d:10s} A(3/a) unm {w['unmeasured_lb']:.4f} state {[round(c, 4) for c in w['crb_state']]}"
              f" | full unm {f['unmeasured_lb']:.4f} bits3/a {w['bits']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("infolib_cs selftest", I.selftest(), flush=True)
    for case in sys.argv[1:]:
        out = {}
        for level in C.LEVELS:
            out[level] = one(case, level)
            (OUT / f"{case}.json").write_text(json.dumps(out, indent=1, default=C.jdefault))
