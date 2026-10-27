"""Shared set-up of the deployment case studies (experiments/casestudies; CASE_A.md, CASE_B.md, PREREG.md).

Import this FIRST: it puts the package copy code/pkg (live package + SOAP/beat patch + casestudies
patch) on sys.path, redirects the data cache to experiments/casestudies/cache (the live cache is never
read or written) and switches every filter to skip stale (held) samples.

    cfg = case_cfg("A", "realistic", seed)        # full run config of one cell
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # experiments/casestudies
CODE = HERE / "code"
PKG = CODE / "pkg"
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
for p in (str(CODE), str(PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import hyperkkl  # noqa: E402

assert Path(hyperkkl.__file__).resolve().parent == (PKG / "hyperkkl").resolve(), hyperkkl.__file__
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import observers as Ob  # noqa: E402
from hyperkkl import systems_casestudy  # noqa: F401,E402
from hyperkkl.config import deep_merge, load  # noqa: E402

CACHE = HERE / "cache"
Dm.CACHE = CACHE
assert Dm.CACHE.resolve() != (LIVE / "experiments" / "cache").resolve()
Ob.HOLD_MISSING_DEFAULT = True          # every filter skips a sample identical to the previous one (stale)

RUNS, MODELS, LOGS = HERE / "runs", HERE / "models", HERE / "logs"

# ------------------------------------------------------------------------------------------ cases
CASES = {
    "A": {
        "system": "quad_planar",
        # physical sensor noise std per measured output (px GPS [m], pz baro [m], gyro [rad/s], accel [m/s^2])
        "sigma": [0.25, 0.10, 0.03, 0.15],
        "hold": [10, 4, 1, 1],                       # GPS 10 Hz, baro 25 Hz, IMU 100 Hz (dt = 0.01 s)
        "data": {"n_p1": 30, "n_train": 60, "n_val": 12, "noise_proc": 0.02},
        "labels": None,                              # motion capture: dense ground truth
        "robust": {
            "poor_init": {"init": "poor", "tiers": ["id"]},
            "outliers": {"sensor": {"outlier_rate": 0.01, "outlier_scale": 8.0}, "tiers": ["id"]},
            "gps_loss": {"sensor": {"burst": {"start_frac": 0.5, "len_tref": 5.0, "channels": [0]}},
                         "tiers": ["id"]},
        },
    },
    "B": {
        "system": "cstr_exo",
        "sigma": [0.10, 0.10],                       # thermocouples / RTDs [K]
        "hold": None,
        "data": {"n_p1": 6, "n_train": 27, "n_val": 6, "noise_proc": 0.005},
        # lab assays of C_A every 45 +- 15 min, assay error 0.05 x_std (~0.012 mol/L)
        "labels": {"states": [0], "every": 45.0, "jitter": 15.0, "noise": 0.05},
        "robust": {
            "poor_init": {"init": "poor", "tiers": ["id"]},
            "outliers": {"sensor": {"outlier_rate": 0.01, "outlier_scale": 8.0}, "tiers": ["id"]},
            "sensor_loss": {"sensor": {"burst": {"start_frac": 0.5, "len_tref": 1.5, "channels": [0]}},
                            "tiers": ["id"]},
        },
    },
}

LEVELS = ["exact", "mm10", "mm30", "partial", "realistic"]
LEVEL_SYS = {
    "exact": {"knowledge": "full", "mismatch": 0.0},
    "mm10": {"knowledge": "full", "mismatch": 0.1},
    "mm30": {"knowledge": "full", "mismatch": 0.3},
    "partial": {"knowledge": "partial", "mismatch": 0.0},
    "realistic": {"knowledge": "partial", "mismatch": 0.3},
}
COLORED = {"kind": "colored_t", "rho": 0.9, "nu": 3.0}      # realistic level: AR(1) Student-t (as TEST-C3)
SEEDS = [0, 1, 2, 3, 4]


def base_cfg(case: str, level: str) -> dict:
    """Config without the noise multipliers (they need the system scales)."""
    c = CASES[case]
    cfg = load(c["system"])
    cfg["system"].update(LEVEL_SYS[level])
    cfg["data"] = deep_merge(cfg["data"], c["data"])
    sen = {"noise": 1.0}
    if c["hold"]:
        sen["hold"] = list(c["hold"])
    if level == "realistic":
        sen.update(COLORED)
    cfg["data"]["sensor"] = sen
    if c["labels"]:
        cfg["data"]["labels"] = copy.deepcopy(c["labels"])
    cfg["conditions"] = deep_merge(cfg["conditions"], copy.deepcopy(c["robust"]))
    cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"] + list(c["robust"]), "save_traj": False,
                                           "memory_probe": False, "n_episodes": 64})
    cfg["baselines"] = []
    cfg["casestudy"] = {"case": case, "level": level}
    return cfg


def case_cfg(case: str, level: str, seed: int = 0, device: str = "cpu") -> dict:
    cfg = base_cfg(case, level)
    sc = Dm.system_scales(cfg, device)
    ys = np.asarray(sc["y_std"], float)
    cfg["data"]["sensor"]["noise_mult"] = [float(v) for v in np.asarray(CASES[case]["sigma"]) / ys]
    cfg["seed"] = int(seed)
    return cfg


def run_dir(case, level, seed) -> Path:
    d = RUNS / case / level / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir(case, level, seed) -> Path:
    d = MODELS / case / level / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def status(msg: str):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    with open(HERE / "STATUS.md", "a") as f:
        f.write(f"{datetime.now(ZoneInfo('Asia/Dubai')):%Y-%m-%d %H:%M} Dubai  {msg}\n")


def jdefault(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


def save_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, default=jdefault))
    tmp.rename(path)
