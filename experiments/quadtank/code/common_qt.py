"""Read-only loader for the quad_tank PE study (scratch copy of the package; live cache read-only)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
LIVE_CACHE = LIVE / "experiments" / "cache"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402
from hyperkkl.systems import build_plant  # noqa: E402

assert Dm.CACHE.resolve() != LIVE_CACHE.resolve(), "refusing to run against the live cache"


def _readonly(path, build, **kw):
    rel = Path(path).relative_to(Dm.CACHE)
    live = LIVE_CACHE / rel
    if live.exists():
        return torch.load(live, weights_only=False, map_location="cpu")
    raise FileNotFoundError(f"not in the live cache (no rebuild by design): {rel}")


Dm._cached = _readonly
_orig_scales = Dm.system_scales


def _scales_ro(cfg, device="cpu"):
    # system_scales returns the cached JSON if present; refuse to build (it would write)
    import hashlib  # noqa: F401
    from hyperkkl.config import digest
    from hyperkkl.cache import DATA_MODULES, module_hash
    scfg, dcfg = cfg["system"], cfg["data"]
    key = digest(cfg, "system.plant", "system.params", "system.sensors", "system.ic_box",
                 "system.T", "system.nz", "system.alpha", "system.actuator",
                 "system.u_scale", "system.u_bounds", "system.dt", "system.alpha_scale",
                 "system.t_ref", "data.envelope_safety", "data.eps_label",
                 extra=[module_hash(DATA_MODULES), torch.device(device).type])
    p = LIVE_CACHE / "scales" / f"{scfg['name']}_{key}.json"
    if not p.exists():
        raise FileNotFoundError(f"scales not in live cache: {p.name}")
    return json.loads(p.read_text())


Dm.system_scales = _scales_ro


def run_cfg(node, system, seed=0):
    src = LIVE / "experiments" / "tree" / node / "runs" / system / f"seed{seed}" / "results.json"
    cfg = json.load(open(src))["config"]
    cfg["seed"] = seed
    return cfg


def c3_results(node, system, seed):
    src = LIVE / "experiments" / "tree" / node / "runs" / system / f"seed{seed}" / "results.json"
    return json.load(open(src))["results"]


DEV = "cuda"   # the cache key holds the device TYPE the data were built on; tensors load on CPU


# Every cached artefact of the C3 runs was built on CUDA, and the cache key holds the device TYPE.
# Key every read with "cuda" whatever device the caller computes on (tensors load on the CPU;
# callers move them), so the grey-box filters can run on the CPU against the same episodes.
_orig_train, _orig_eval, _ro_scales = Dm.training_data, Dm.eval_data, Dm.system_scales
Dm.training_data = lambda cfg, seed, device="cpu": _orig_train(cfg, seed, DEV)
Dm.eval_data = lambda cfg, tier, plant_shift=0.0, device="cpu", **kw: _orig_eval(cfg, tier, plant_shift, DEV, **kw)
Dm.system_scales = lambda cfg, device="cpu": _ro_scales(cfg, DEV)
