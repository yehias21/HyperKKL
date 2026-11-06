"""Where the data live.

Resolution order for the data root (the folder holding ``data/``, ``metadata/``):
  1. the ``root=`` argument of a function,
  2. the environment variable ``HYPERKKL_BENCH_DATA``,
  3. the local export of this repository (experiments/hf_benchmark_export/hf_dataset),
  4. a local snapshot of the Hugging Face dataset made by ``fetch_remote()`` (~/.cache).

The extra splits of this package (``val_ood`` and the label-withheld ``hidden`` test) live in a
second root, ``HYPERKKL_BENCH_EXTRA`` (default: ``<package checkout>/data_extra``). Hidden labels
live in a THIRD root (``HYPERKKL_BENCH_LABELS``) that a leaderboard host keeps private; the
package never reads it except in ``hidden.score_hidden``.
"""

from __future__ import annotations

import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
CHECKOUT = PKG_DIR.parents[1]                      # .../hyperkkl_bench (the repo folder)
REPO = CHECKOUT.parent                             # .../hyperkkl (research repo), may not exist
LOCAL_EXPORT = REPO / "experiments" / "hf_benchmark_export" / "hf_dataset"
HF_REPO_ID = "Yahias21/HyperKKL-benchmark"         # PRIVATE dataset; read access needs a token
HF_CACHE = Path(os.environ.get("HYPERKKL_BENCH_CACHE", Path.home() / ".cache" / "hyperkkl_bench"))


class DataNotFound(FileNotFoundError):
    pass


def data_root(root=None) -> Path:
    env = os.environ.get("HYPERKKL_BENCH_DATA")
    if root is None and env:                       # an explicit setting is never silently overridden
        if (Path(env) / "metadata").is_dir():
            return Path(env)
        raise DataNotFound(f"HYPERKKL_BENCH_DATA={env} has no metadata/ folder")
    cands = [root, env, LOCAL_EXPORT, HF_CACHE / "hf_dataset"]
    for c in cands:
        if c and (Path(c) / "metadata").is_dir():
            return Path(c)
    raise DataNotFound(
        "HyperKKL benchmark data not found. Set HYPERKKL_BENCH_DATA to the folder that holds "
        "data/ and metadata/ (a copy of the dataset), or call hyperkkl_bench.paths.fetch_remote() "
        f"(needs read access to the private Hugging Face dataset {HF_REPO_ID}). Tried: "
        + ", ".join(str(c) for c in cands if c))


def extra_root(root=None) -> Path:
    return Path(root or os.environ.get("HYPERKKL_BENCH_EXTRA") or CHECKOUT / "data_extra")


def labels_root(root=None) -> Path:
    """Hidden-test labels (kept by the leaderboard host; NOT part of any public release)."""
    return Path(root or os.environ.get("HYPERKKL_BENCH_LABELS") or CHECKOUT / "hidden_labels")


def fetch_remote(plants=None, variants=None, token=None, local_dir=None) -> Path:
    """Download (a part of) the dataset from the Hugging Face Hub. READ ONLY: this package
    never uploads. ``plants`` / ``variants`` (white, mismatch30, realistic) restrict the
    download (the full dataset is 5.2 GB)."""
    from huggingface_hub import snapshot_download   # optional dependency ([hub] extra)
    pats = ["metadata/*", "README.md"]
    for p in plants or ["*"]:
        for v in variants or ["*"]:
            pats.append(f"data/{p}/{v}/*")
    out = Path(local_dir or HF_CACHE / "hf_dataset")
    snapshot_download(HF_REPO_ID, repo_type="dataset", allow_patterns=pats, token=token,
                      local_dir=str(out))
    return out
