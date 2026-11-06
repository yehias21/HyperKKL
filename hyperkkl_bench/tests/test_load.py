import json
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench.data import HIDDEN_SPLITS
from conftest import needs_data

PLANTS = hkb.list_plants()


@needs_data
@pytest.mark.parametrize("plant", PLANTS)
@pytest.mark.parametrize("knowledge", list(hkb.KNOWLEDGE))
def test_load_test_and_val_ood(plant, knowledge):
    meta = hkb.metadata(plant)
    for split in ("test_id", "val_ood"):
        es = hkb.load(plant, knowledge, split)
        E, N1 = es.u.shape
        assert N1 == meta["N"] + 1
        assert es.y.shape == (E, N1, meta["n_y"]) and es.x.shape == (E, N1, meta["n_x"])
        assert abs(es.dt - meta["dt"]) < 1e-12
        assert np.isfinite(es.x).all() and np.isfinite(es.y).all() and np.isfinite(es.u).all()
        assert E >= 20
    ep = es[0]
    assert ep.u.shape == (N1,) and ep.meta["family"]


@needs_data
@pytest.mark.parametrize("plant", PLANTS)
def test_hidden_public_files_have_no_labels(plant):
    import pyarrow.parquet as pq
    from hyperkkl_bench.data import _file_for
    for s in HIDDEN_SPLITS:
        cols = pq.read_schema(_file_for(plant, "realistic", s)).names
        assert "x" not in cols and "x0" not in cols, cols
        es = hkb.load(plant, "realistic", s)
        assert es.x is None


@needs_data
def test_train_subset_nested():
    a = hkb.load("vdp", "exact", "train", n=50, seed=3)
    b = hkb.load("vdp", "exact", "train", n=100, seed=3)
    assert len(a) == 50
    # deterministic
    a2 = hkb.load("vdp", "exact", "train", n=50, seed=3)
    assert np.array_equal(a.episode_ids, a2.episode_ids)


@needs_data
def test_knowledge_levels_share_data_as_documented():
    e = hkb.load("vdp", "exact", "test_id")
    p = hkb.load("vdp", "partial", "test_id")
    m = hkb.load("vdp", "mismatch", "test_id")
    r = hkb.load("vdp", "realistic", "test_id")
    assert np.array_equal(e.y, p.y) and np.array_equal(e.x, p.x)
    assert np.array_equal(m.x, r.x) and not np.array_equal(m.y, r.y)


def test_unknown_names():
    with pytest.raises(KeyError):
        hkb.metadata("nope")
    with pytest.raises(ValueError):
        hkb.load("vdp", "oracle", "train")
