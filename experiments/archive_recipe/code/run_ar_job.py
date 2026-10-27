"""Run one archive-recipe job from the SCRATCH copy of the package.

    python run_ar_job.py --node AR-partial --system vdp --seed 0 --device cuda:0

Isolation: ROOT of this copy is the scratch directory, so runs, model caches and code
snapshots go under ./experiments. Data / evaluation episodes are read READ-ONLY from the
live cache when an identical file exists there (same keys: the data modules are unchanged
in this copy), so the episodes are exactly those of the C3 node runs; anything missing is
built into the scratch cache, never into the live one.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/cache")

import torch  # noqa: E402

from hyperkkl import cache as C  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402

assert Dm.CACHE.resolve() != LIVE.resolve(), "refusing to run against the live cache"
_orig = C.cached


def _layered(path, build, **kw):
    path = Path(path)
    try:
        rel = path.relative_to(Dm.CACHE)
    except ValueError:
        return _orig(path, build, **kw)
    live = LIVE / rel
    if not path.exists() and live.exists():
        print(f"[ar] data from live cache (read-only): {rel}", flush=True)
        return torch.load(live, weights_only=False, map_location="cpu")
    print(f"[ar] data built/loaded in scratch cache: {rel}", flush=True)
    return _orig(path, build, **kw)


Dm._cached = _layered
(Dm.CACHE / "scales").mkdir(parents=True, exist_ok=True)
for f in (LIVE / "scales").glob("*.json"):              # tiny; copied, never written back
    t = Dm.CACHE / "scales" / f.name
    if not t.exists():
        shutil.copy(f, t)

from hyperkkl import job  # noqa: E402

if __name__ == "__main__":
    import atexit
    if torch.cuda.is_available():
        atexit.register(lambda: print(f"[ar] peak CUDA memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
                                      f"reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True))
    job.main()
