"""Information budget on the real battery (Panasonic 18650PF, experiments/real): the posterior CRB of the
SOC along the identified 2-RC ECM driven by the MEASURED current of the held-out 25 degC cycles, in the
same 30-min windows (prior SOC 20% off, settled half scored) as the real-data evaluation.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python experiments/information/real_battery.py

Two noise settings, because the real sensor noise is not identified: (i) the tuned EKF's own (Q, R) from
experiments/real/results.json (R_vm = 0.1 V absorbs model error: 'what the filter believes'), and (ii) a
sensor-level setting (R_vm = 3 mV, R_dT = 0.1 K, q = 0.01 x_std/sqrt(min), q_soc = 1e-4: the smallest grid
values): the information the measurements would carry if the model were right. Output: real_battery.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "information"))
import infolib as L  # noqa: E402
from hyperkkl import realdata as R  # noqa: E402
from hyperkkl.realdata_run import INIT_ERR, N_STEPS, R_DT, STRIDE_EVAL  # noqa: E402

torch.set_num_threads(8)
res = json.loads((R.REAL / "results.json").read_text())
params = res["identification"]["params"]
plant = R.Pan18650PF(**params)
Q_AH, OFF = res["Q_Ah"], res["thermocouple_offset_K"]
scales = res["scales"]
xs = np.asarray(scales["x_std"])
dt_m = 1.0 / R.TIME_UNIT                      # 1 s in model time units (minutes)
tuned = res["filter_tuning"]["ekf"]["choice"]
settings = {
    "tuned_ekf": {"q": np.r_[tuned["q_soc"], tuned["c"] * xs[1:]], "r": np.array([R_DT, tuned["r_vm"]])},
    "sensor_level": {"q": np.r_[1e-4, 0.01 * xs[1:]], "r": np.array([0.1, 0.003])},
}
p0 = np.array([INIT_ERR, xs[1], xs[2], 0.5, 0.01])
files = R.drive_cycle_files(25.0)
out = {"settings": {k: {"q": v["q"].tolist(), "r": v["r"].tolist()} for k, v in settings.items()},
       "p0": p0.tolist(), "window_s": N_STEPS, "stride_s": STRIDE_EVAL, "cycles": {}}
for cyc in ("US06", "HWFTa", "HWFTb", "NN", "Cycle_4"):
    rec = R.load_recording(files[cyc], Q=Q_AH, offset=OFF)
    xr = R.replay(plant, rec)                              # (n, 5) model replay along the measured current
    starts = R.window_starts(len(rec), N_STEPS, STRIDE_EVAL)
    X = torch.as_tensor(np.stack([xr[s:s + N_STEPS + 1] for s in starts]), dtype=torch.float64)
    U = torch.as_tensor(np.stack([rec.u[s:s + N_STEPS + 1] for s in starts]), dtype=torch.float64)[..., None]
    F, H = L.jacobians(plant, X, U, dt_m)
    k_eval = torch.arange(N_STEPS // 2, N_STEPS + 1, 60)
    P0 = torch.diag(torch.as_tensor(p0 ** 2, dtype=torch.float64))
    cres = {"n_windows": len(starts), "soc_rmse_bound_pct": {}}
    for name, st in settings.items():
        Q = torch.diag(torch.as_tensor(st["q"] ** 2 * dt_m, dtype=torch.float64))
        Rm = torch.diag(torch.as_tensor(st["r"] ** 2, dtype=torch.float64))
        P = L.pcrb_full(F, H, Q, Rm, P0, k_eval)             # local recursion per window (the EKF's own object)
        Pb = L.pcrb_bayes_full(F, H, Q, Rm, P0, k_eval)      # Bayesian, averaged over the windows
        cres["soc_rmse_bound_pct"][name] = {
            "local_pooled": 100 * float(P[..., 0, 0].mean().sqrt()),
            "bayes": 100 * float(Pb[..., 0, 0].mean().sqrt()),
            "v1_v2_bound_sigma": [float(Pb[..., j, j].mean().sqrt() / xs[j]) for j in (1, 2)],
        }
    out["cycles"][cyc] = cres
    print(cyc, cres, flush=True)
# achieved settled SOC RMSE (%) from the real-data results for comparison
ach = {}
for cyc in ("US06", "HWFTa", "HWFTb", "NN"):
    t = res["tables"].get("test25", {}).get(cyc) or {}
    ach[cyc] = {m: v["rmse_settled"]["mean"] for m, v in t.items() if "rmse_settled" in v}
val = res["tables"].get("val", {}).get("Cycle_4", {})
ach["Cycle_4"] = {m: v["rmse_settled"]["mean"] for m, v in val.items() if "rmse_settled" in v}
out["achieved_soc_rmse_pct"] = ach
(Path(__file__).resolve().parent / "real_battery.json").write_text(json.dumps(out, indent=1))
print(json.dumps(ach, indent=0)[:2000])
