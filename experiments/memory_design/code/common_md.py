"""Shared setup for the memory-design study: scratch package copy (code/pkg), live cache READ-ONLY."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
LIVE_CACHE = LIVE / "experiments" / "cache"
CAMPAIGN = Path("/tmp/claude-1010/-mnt-raid0-gpu4-demo-data-papers-TMLR/7db42dae-1ccc-4e01-b21a-1b973b014914/scratchpad/beat_greybox")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402

assert Dm.CACHE.resolve() != LIVE_CACHE.resolve(), "refusing to run against the live cache"


def _readonly(path, build, **kw):
    rel = Path(path).relative_to(Dm.CACHE)
    live = LIVE_CACHE / rel
    if live.exists():
        return torch.load(live, weights_only=False, map_location="cpu")
    raise FileNotFoundError(f"data not in the live cache (no rebuild by design): {rel}")


Dm._cached = _readonly


def run_cfg(node, system, seed=0):
    src = LIVE / "experiments" / "tree" / node / "runs" / system / f"seed{seed}" / "results.json"
    cfg = json.load(open(src))["config"]
    cfg["seed"] = seed
    return cfg


# Every cached artefact of the C3 runs was built on CUDA and the cache key holds the device TYPE:
# key every read with "cuda" whatever device the caller computes on (tensors load on the CPU).
DEV = "cuda"
_orig_train, _orig_eval, _orig_scales = Dm.training_data, Dm.eval_data, Dm.system_scales
Dm.training_data = lambda cfg, seed, device="cpu": _orig_train(cfg, seed, DEV)
Dm.eval_data = lambda cfg, tier, plant_shift=0.0, device="cpu", **kw: _orig_eval(cfg, tier, plant_shift, DEV, **kw)
Dm.system_scales = lambda cfg, device="cpu": _orig_scales(cfg, DEV)
