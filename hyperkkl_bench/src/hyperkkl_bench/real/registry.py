"""Registry and folders of the real-data tracks.

The raw files are never shipped with the package; ``fetch`` downloads them from the original
source into ``<real root>/<dataset>/raw`` and ``prepare`` writes the frozen episodes into
``<real root>/<dataset>/processed``. The real root is ``HYPERKKL_BENCH_REAL`` or, by default,
``<package checkout>/data_real`` (gitignored).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from ..paths import CHECKOUT, PKG_DIR

RESOURCES = PKG_DIR / "resources"

# name -> python module (in this package) that implements prepare / load / score / context / card
DATASETS = {
    "battery_pf": "battery_pf",
    "battery_lg": "battery_lg",
    "emps": "emps",
    "silverbox": "silverbox",
    "neurobem": "neurobem",
    "calce": "calce",
}


def real_root(root=None) -> Path:
    return Path(root or os.environ.get("HYPERKKL_BENCH_REAL") or CHECKOUT / "data_real")


def raw_dir(dataset: str, root=None) -> Path:
    return real_root(root) / dataset / "raw"


def processed_dir(dataset: str, root=None) -> Path:
    return real_root(root) / dataset / "processed"


def check_name(dataset: str) -> None:
    if dataset not in DATASETS:
        raise KeyError(f"unknown real-data track {dataset!r}; available: {sorted(DATASETS)}")


@lru_cache(maxsize=None)
def spec(dataset: str) -> dict:
    """resources/real_<dataset>.json: source, licence, citation, files (+ SHA-256), frozen constants."""
    check_name(dataset)
    return json.loads((RESOURCES / f"real_{dataset}.json").read_text())


def module(dataset: str):
    import importlib
    check_name(dataset)
    return importlib.import_module(f"hyperkkl_bench.real.{DATASETS[dataset]}")


def list_real() -> list:
    return list(DATASETS)
