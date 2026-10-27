"""Pre-observer checks per n_x (CPU only, before any observer run):

  (I) the pre-registered INCLUSION rule (experiments/cartpole_bounded/INCLUSION.md), applied
      unchanged with the functions of cartpole_bounded/inclusion.py: (1a) full-past Bayesian
      PCRB of every unmeasured state < 1 (sigma-normalised), (1b) local observability Gramian
      over 3/alpha full rank at >= 90 % of (episode, time) points, (2) bounded (5T rollouts).
      Setting: the exact model, white sensor noise (the C1 setting). Extra (not part of the rule):
      boundedness of the 30 %-perturbed real plant used by the off30 / realistic levels.
  (B) the information budget A per dataset for the bits rule (experiments/information):
      Bayesian PCRB of the unmeasured states for the window 3/alpha and the full past, under the
      real plant and sensor of each dataset (infolib_c3.budget_c3; white noise = rho 0, J_t 1).

    python code/precheck.py NX   -> inclusion/ring{NX}.json
"""
import json
import math
import sys
import time

import common as C  # noqa: F401  (must be first)
import numpy as np
import torch

sys.path.insert(2, str(C.LIVE / "experiments" / "cartpole_bounded"))
sys.path.insert(3, str(C.LIVE / "experiments" / "information"))
import inclusion as INC  # noqa: E402  (functions only; its main does not run)
import infolib as L  # noqa: E402
import infolib_c3 as L3  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

import scaleplants as SP  # noqa: E402


def main(nx):
    t0 = time.time()
    cfg = SP.make_cfg(nx, "exact", 0)
    scales = Dm.system_scales(cfg)
    es = Dm.eval_data(cfg, "id")
    plant = build_plant(cfg["system"])
    sensors = list(plant.sensors)
    unm = [j for j in range(plant.nx) if j not in sensors]
    alpha = float(cfg["system"]["alpha"])
    settle = int(0.5 * (es.x.shape[1] - 1))
    Sig = L.climatology(es.x[:, settle:].double())
    b = L.budget_along(plant, es.x, es.u, cfg, scales, sensors, [3 / alpha], n_ep=64, Sigma=Sig)
    crb = b["bayes"]["full"]["crb_state"]
    crb_u = [crb[j] for j in unm]
    o1a = all(math.isfinite(c) and c < 1 for c in crb_u)
    g = INC.gramian(plant, es, cfg, scales)
    o1b = g["frac_full_rank"] >= 0.9
    weak = max(crb_u) >= 0.1 or g["ratio_median"] < 1e-6
    bd = INC.bounded(plant, cfg, scales)
    bd30 = INC.bounded(build_plant(cfg["system"], perturb=0.3), cfg, scales)
    res = {"nx": nx, "system": cfg["system"], "design_check": SP.design_check(cfg["system"]),
           "unmeasured": unm, "crb_full_state": crb, "crb_unmeasured_max": max(crb_u),
           "obs_1a": o1a, "gramian": g, "obs_1b": o1b, "observable": o1a and o1b, "weak": weak,
           "bounded": bd["pass"], "bounded_detail": bd, "bounded_perturbed30": bd30["pass"],
           "bounded_perturbed30_detail": bd30, "scales": scales}
    print(nx, "observable", res["observable"], "weak", weak, "bounded", bd["pass"], "bounded30", bd30["pass"],
          f"{time.time() - t0:.0f}s", flush=True)
    # (B) budgets per dataset
    res["budget"] = {}
    for ds, lev in SP.DATASET_LEVEL.items():
        c2 = SP.make_cfg(nx, lev, 0)
        e2 = Dm.eval_data(c2, "sequence_ood")
        pl = build_plant(c2["system"], perturb=float(c2["system"]["mismatch"]))
        S2 = L.climatology(e2.x[:, settle:].double())
        bb = L3.budget_c3(pl, e2.x, e2.u, c2, scales, sensors, [3 / alpha], S2, n_ep=64)
        w = bb["bayes"]["windows"][f"{3 / alpha:g}"]
        res["budget"][ds] = {"A_window_unmeasured_lb": w["unmeasured_lb"],
                             "full_unmeasured_lb": bb["bayes"]["full"]["unmeasured_lb"],
                             "bits_window": w.get("bits"), "bits_full": bb["bayes"]["full"].get("bits")}
        print(nx, ds, res["budget"][ds], flush=True)
    res["wall_s"] = time.time() - t0
    (C.SC / "inclusion").mkdir(exist_ok=True)
    (C.SC / "inclusion" / f"ring{nx}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(int(sys.argv[1]))
