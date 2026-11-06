import os
import pytest
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from hyperkkl_bench.paths import REPO, data_root, DataNotFound


def _has_data():
    try:
        data_root()
        return True
    except DataNotFound:
        return False


needs_data = pytest.mark.skipif(not _has_data(), reason="benchmark data not found (HYPERKKL_BENCH_DATA)")
needs_repo = pytest.mark.skipif(not (REPO / "experiments" / "tree").is_dir(), reason="research repository not present")
