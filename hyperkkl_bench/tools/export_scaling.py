"""Freeze the scaling-track episodes (experiments/scaling, FHN ring, n_x in 4..64) as Parquet in
data_extra/ from the research cache_rev3 (read-only; a cache miss is an error, nothing is regenerated).

Benchmark plant names fhnring_n{n_x}. Data variants: white = level 'exact' (= 'partial' data),
mismatch30 = 'off30', realistic = 'realistic'. Train / validation / phase1 of training seed 0 with the
pre-registered data rule (n_train = 600 n_x / 8, i.e. 300 .. 4800 episodes); test tiers of split 'test'.

    python tools/export_scaling.py [nx ...]          (one process per n_x is fine)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
CHECKOUT = Path(__file__).resolve().parents[1]
SC = CHECKOUT.parent / "experiments" / "scaling"
sys.path.insert(0, str(SC / "code"))
import common as CM           # noqa: E402,F401  (path order, cache redirection, plants)
import numpy as np            # noqa: E402
import torch                  # noqa: E402
import scaleplants as SP      # noqa: E402
from hyperkkl import cache as Cm          # noqa: E402
from hyperkkl import data as Dm           # noqa: E402
from hyperkkl.systems import build_plant  # noqa: E402

sys.path.insert(0, str(CHECKOUT / "tools"))
from export_casestudies_io import sha256, write  # noqa: E402


def _no_build(path, build, **kw):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cache miss (refusing to regenerate): {path.name}")
    return torch.load(path, weights_only=False, map_location="cpu")


Dm._cached = _no_build
Cm.cached = _no_build
OUT = CHECKOUT / "data_extra"
VARIANTS = {"white": "exact", "mismatch30": "off30", "realistic": "realistic"}
SPLITS = {"train": "train", "validation": "val", "phase1": "p1"}


def export(nx):
    name = f"fhnring_n{nx}"
    meta = {"plant": name, "track": "scaling", "settings": {}, "files": {}}
    for var, level in VARIANTS.items():
        cfg = SP.make_cfg(nx, level, 0)
        scfg = cfg["system"]
        scales = Dm.system_scales(cfg)
        info = {"source": f"experiments/scaling/cache_rev3, level {level}, training seed 0",
                "mismatch": float(scfg.get("mismatch", 0.0)), "sensor": cfg["data"]["sensor"], "n": {}}
        tr = CM.training_data(cfg, 0)
        for s, key in SPLITS.items():
            p = OUT / "data" / name / var / f"{s}.parquet"
            info["n"][s] = write(tr[key], p)
            meta["files"][str(p.relative_to(OUT))] = {"sha256": sha256(p), "episodes": info["n"][s]}
        del tr
        for t in ("id", "shape_ood", "sequence_ood"):
            es = Dm.eval_data(cfg, t)
            p = OUT / "data" / name / var / f"{t}.parquet"
            info["n"][t] = write(es, p)
            meta["files"][str(p.relative_to(OUT))] = {"sha256": sha256(p), "episodes": info["n"][t]}
        meta["settings"][var] = info
        print(name, var, info["n"], flush=True)
        if var == "white":
            sc = {k: v for k, v in scfg.items() if k not in ("knowledge", "mismatch", "bank", "name", "role")}
            sc["name"] = name
            base = build_plant({k: v for k, v in scfg.items() if k not in ("knowledge", "mismatch")})
            raw = base.base if hasattr(base, "base") else base
            sens = list(scfg["sensors"])
            names = [f"{c}{i}" for i in range(nx // 2) for c in ("v", "w")]
            meta.update({
                "role": "scaling", "n_x": raw.nx, "n_y": len(sens), "n_u": 1, "state_names": names,
                "sensors": sens, "sensor_names": [names[j] for j in sens],
                "unmeasured": [j for j in range(raw.nx) if j not in sens], "group": None,
                "unmodelled_term": raw.UNMODELLED, "nominal_params": raw.params(),
                "not_perturbed_params": list(raw.NOT_PERTURBED), "mismatch30_params": raw.perturbed(0.3).params(),
                "x_std": scales["x_std"], "y_std": scales["y_std"], "x_mean": scales["x_mean"],
                "operating_domain": scales["domain"], "dt": scales["dt"], "N": scales["N"], "T": float(scfg["T"]),
                "t_ref": float(scfg.get("t_ref", 1.0)), "alpha": float(scfg["alpha"]), "nz": int(scfg["nz"]),
                "settle_frac": 0.5, "settle_index": int(0.5 * scales["N"]),
                "input_box": {"v_range": [-1.0, 1.0], "u_scale": float(scfg.get("u_scale", 1.0)),
                              "u_range": [-float(scfg.get("u_scale", 1.0)), float(scfg.get("u_scale", 1.0))],
                              "u_bounds": None},
                "actuator": {"kind": "saturating", "u_bar": float(base.u_bar), "k_backemf": float(base.k),
                             "j_state": int(base.j), "x_ref": float(base.x_ref)} if hasattr(base, "u_bar") else {"kind": "affine"},
                "ic_box": scfg["ic_box"], "process_noise_rel": float(cfg["data"].get("noise_proc", 0.0)),
                "system_config": sc, "data_rule": cfg["scaling"]["sizes"],
            })
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    (OUT / "metadata" / f"{name}.json").write_text(json.dumps(meta, indent=1, default=float))


if __name__ == "__main__":
    for nx in (sys.argv[1:] or SP.NX_GRID):
        export(int(nx))
