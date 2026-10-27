"""baselines2: learned, recipe-independent baselines (GRU, Neural CDE, Neural ODE / UDE + EKF/UKF,
capacity-matched NLOX, re-fitted grey-box filters) on the C3 cells.

Isolation (PREREG.md section 0): the live package ``hyperkkl`` is imported unchanged. Its data cache
is redirected: a file that exists in the live cache is READ from there (seeds 0-2: exactly the episodes
of the stored runs); a missing one (seeds 3-4 training data) is built by the unchanged code into
experiments/baselines2/cache. Nothing is written to the live cache.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
HERE = LIVE / "experiments" / "baselines2"
LIVE_CACHE = LIVE / "experiments" / "cache"
LOCAL_CACHE = HERE / "cache"
if str(LIVE) not in sys.path:
    sys.path.insert(0, str(LIVE))

import torch  # noqa: E402

from hyperkkl import cache as _Cm  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402

_orig_cached = _Cm.cached


def _redirected_cached(path, build, **kw):
    path = Path(path)
    rel = path.relative_to(LOCAL_CACHE)
    live = LIVE_CACHE / rel
    if live.exists():
        return torch.load(live, weights_only=False, map_location="cpu")
    return _orig_cached(path, build, **kw)


def _install():
    (LOCAL_CACHE / "scales").mkdir(parents=True, exist_ok=True)
    (LOCAL_CACHE / "data").mkdir(parents=True, exist_ok=True)
    for p in (LIVE_CACHE / "scales").glob("*.json"):          # small; system_scales reads CACHE/scales
        q = LOCAL_CACHE / "scales" / p.name
        if not q.exists():
            try:
                tmp = q.with_suffix(f".tmp{os.getpid()}")
                shutil.copyfile(p, tmp)
                os.replace(tmp, q)
            except OSError:
                pass
    Dm.CACHE = LOCAL_CACHE
    Dm._cached = _redirected_cached


_install()
assert Dm.CACHE.resolve() != LIVE_CACHE.resolve()
