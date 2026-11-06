"""Freeze the deployment-track episodes (experiments/casestudies: Case A quad_planar drone, Case B cstr_exo
with sparse lab labels) as Parquet in data_extra/, from the research cache (read-only; never rebuilds:
a cache miss is an error). Same schema as the core export; in the cstr_exo train / validation / phase1
splits the C_A column of x is NaN except at the (noisy) lab-assay samples, i.e. exactly the labels the
research runs trained on. Test tiers keep the dense truth (scoring).

Data variants (as the core plants): white = level 'exact' (= 'partial' data, checked), mismatch30 = 'mm30',
realistic = 'realistic' (30 % mismatch, colored Student-t sensor noise). Training seed 0.

    CUDA_VISIBLE_DEVICES="" python tools/export_casestudies.py
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CHECKOUT = Path(__file__).resolve().parents[1]
CS = CHECKOUT.parent / "experiments" / "casestudies"
sys.path.insert(0, str(CS / "code"))
import numpy as np            # noqa: E402
import pyarrow as pa          # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch                  # noqa: E402
import cs_common as CC        # noqa: E402  (puts the casestudies hyperkkl package first on sys.path)
from hyperkkl import cache as Cm          # noqa: E402
from hyperkkl import data as Dm           # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

def _no_build(path, build, **kw):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss (refusing to regenerate): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


Dm._cached = _no_build
Cm.cached = _no_build
DEV = "cuda"          # cache key of the research runs (device type only; nothing runs on a GPU)
OUT = CHECKOUT / "data_extra"
STATE_NAMES = {"quad_planar": ["px (m)", "pz (m)", "vx (m/s)", "vz (m/s)", "theta (rad)", "q (rad/s)", "a_m (m/s^2)"],
               "cstr_exo": ["C_A (mol/L)", "T (K)", "T_j (K)"]}
VARIANTS = {"white": "exact", "mismatch30": "mm30", "realistic": "realistic"}
SPLITS = {"train": "train", "validation": "val", "phase1": "p1"}


sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_casestudies_io import sha256, write  # noqa: E402


def export(case):
    plant = CC.CASES[case]["system"]
    meta = {"plant": plant, "track": "deployment", "case": case, "settings": {}, "files": {}}
    ref_y = None
    for var, level in VARIANTS.items():
        cfg = CC.case_cfg(case, level, 0, DEV)
        scfg = cfg["system"]
        scales = Dm.system_scales(cfg, DEV)
        tr = Dm.training_data(cfg, 0, DEV)
        info = {"source": f"experiments/casestudies cache, level {level}, training seed 0",
                "mismatch": float(scfg.get("mismatch", 0.0)), "sensor": cfg["data"]["sensor"],
                "labels": cfg["data"].get("labels"), "n": {}}
        for s, key in SPLITS.items():
            p = OUT / "data" / plant / var / f"{s}.parquet"
            info["n"][s] = write(tr[key], p)
            meta["files"][str(p.relative_to(OUT))] = {"sha256": sha256(p), "episodes": info["n"][s]}
        for t in ("id", "shape_ood", "sequence_ood"):
            es = Dm.eval_data(cfg, t, device=DEV)
            p = OUT / "data" / plant / var / f"{t}.parquet"
            info["n"][t] = write(es, p)
            meta["files"][str(p.relative_to(OUT))] = {"sha256": sha256(p), "episodes": info["n"][t]}
            if var == "white" and t == "id":
                ref_y = es.y.clone()
        meta["settings"][var] = info
        print(case, var, info["n"], flush=True)
        if var == "white":
            cp = CC.case_cfg(case, "partial", 0, DEV)
            ep = Dm.eval_data(cp, "id", device=DEV)
            assert torch.equal(ep.y, ref_y), "partial data differ from exact data"
            base = build_plant({k: v for k, v in scfg.items() if k not in ("knowledge", "mismatch")})
            raw = base.base if hasattr(base, "base") else base
            sens = list(scfg["sensors"])
            meta.update({
                "role": "deployment", "n_x": raw.nx, "n_y": len(sens), "n_u": 1, "state_names": STATE_NAMES[plant],
                "sensors": sens, "sensor_names": [STATE_NAMES[plant][j] for j in sens],
                "unmeasured": [j for j in range(raw.nx) if j not in sens], "group": None,
                "unmodelled_term": raw.UNMODELLED, "nominal_params": raw.params(),
                "not_perturbed_params": list(raw.NOT_PERTURBED), "mismatch30_params": raw.perturbed(0.3).params(),
                "x_std": scales["x_std"], "y_std": scales["y_std"], "x_mean": scales["x_mean"],
                "operating_domain": scales["domain"], "dt": scales["dt"], "N": scales["N"], "T": float(scfg["T"]),
                "t_ref": float(scfg.get("t_ref", 1.0)), "alpha": float(scfg["alpha"]), "nz": int(scfg["nz"]),
                "settle_frac": 0.5, "settle_index": int(0.5 * scales["N"]),
                "input_box": {"v_range": [-1.0, 1.0], "u_scale": float(scfg.get("u_scale", 1.0)),
                              "u_range": [-float(scfg.get("u_scale", 1.0)), float(scfg.get("u_scale", 1.0))],
                              "u_bounds": scfg.get("u_bounds") or None},
                "actuator": {"kind": (scfg.get("actuator") or {}).get("kind", "affine")},
                "ic_box": scfg["ic_box"], "process_noise_rel": float(cfg["data"].get("noise_proc", 0.0)),
                "sensor_noise_std_physical": CC.CASES[case]["sigma"], "sensor_hold_samples": CC.CASES[case]["hold"],
                "robust_conditions": CC.CASES[case]["robust"],
                "system_config": {k: v for k, v in scfg.items() if k not in ("knowledge", "mismatch")},
                "requires": "the casestudies plant classes (hyperkkl_bench.plants.systems_casestudy)",
            })
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata" / f"{plant}.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    for c in sys.argv[1:] or ["A", "B"]:
        export(c)
