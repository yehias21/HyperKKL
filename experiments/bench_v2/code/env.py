"""bench_v2 environment: import this BEFORE hyperkkl / hyperkkl_bench.

Puts the frozen copies first on sys.path, so a run never depends on the live hyperkkl/ (read only) or on
hyperkkl_bench/ (owned by other agents while this study runs):

    code/pkg             research package copy (l96_default pkg = eventmix pkg + systems_chaos, + systems_casestudy)
    code/hkb_snapshot    copy of hyperkkl_bench/src taken at the start of the study (scorer, filters, real tracks)

All writes go below experiments/bench_v2. The live cache is never touched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
HERE = CODE.parent
LIVE = HERE.parent.parent
EXPORT = LIVE / "experiments" / "hf_benchmark_export" / "hf_dataset"
BENCH = LIVE / "hyperkkl_bench"
DATA = HERE / "data"
RUNS = HERE / "runs"

for p in (str(CODE / "hkb_snapshot"), str(CODE / "pkg"), str(CODE)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
os.environ.setdefault("HYPERKKL_BENCH_DATA", str(EXPORT))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")


def check_imports():
    import hyperkkl
    import hyperkkl_bench
    assert Path(hyperkkl.__file__).resolve().parent == (CODE / "pkg" / "hyperkkl").resolve(), hyperkkl.__file__
    assert Path(hyperkkl_bench.__file__).resolve().parent == (CODE / "hkb_snapshot" / "hyperkkl_bench").resolve()
    return True
