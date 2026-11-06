"""End-to-end pipeline on the tiny CI fixture (tests/fixtures/mini, built by tools/make_mini_dataset.py).
Runs without the real dataset; checks plumbing only, not numbers."""
from pathlib import Path
import numpy as np
import hyperkkl_bench as hkb

MINI = Path(__file__).resolve().parent / "fixtures" / "mini"


def test_mini_end_to_end(tmp_path):
    hkb.data.metadata.cache_clear() if hasattr(hkb.data.metadata, "cache_clear") else None
    hkb.data._metadata_raw.cache_clear()
    try:
        es = hkb.load("vdp", "realistic", "test_id", root=str(MINI))
        assert es.u.shape == (6, 401)
        for est in (hkb.baselines.EKF(gate=True), hkb.baselines.GRUObserver(hidden=4, steps=3, window=32, batch=2),
                    hkb.baselines.PriorMean()):
            rep = hkb.evaluate(est, "vdp", "realistic", root=str(MINI), conditions=("clean", "outliers", "poor_init"),
                               verbose=False)
            assert np.isfinite(rep.value("test_sequence_ood"))
            rep.save(tmp_path / f"{est.name}.json")
            assert hkb.Report.load(tmp_path / f"{est.name}.json").value("test_id") == rep.value("test_id")
    finally:
        hkb.data._metadata_raw.cache_clear()
