import numpy as np
import pytest
import hyperkkl_bench as hkb
from conftest import needs_data


@needs_data
@pytest.mark.parametrize("plant", ["quad_planar", "cstr_exo"])
@pytest.mark.parametrize("knowledge", list(hkb.KNOWLEDGE))
def test_deployment_loads(plant, knowledge):
    meta = hkb.metadata(plant)
    te = hkb.load(plant, knowledge, "test_sequence_ood")
    assert te.x.shape == (64, meta["N"] + 1, meta["n_x"]) and np.isfinite(te.x).all()
    tr = hkb.load(plant, knowledge, "train")
    if plant == "cstr_exo":          # sparse lab labels: C_A known only at assay samples
        frac = np.isfinite(tr.x[..., 0]).mean()
        assert 0.0 < frac < 0.05 and np.isfinite(tr.x[..., 1:]).all()
    else:
        assert np.isfinite(tr.x).all()


@needs_data
def test_deployment_conditions_are_plant_specific():
    from hyperkkl_bench import conditions as C
    meta = hkb.metadata("quad_planar")
    assert C.spec_for("outliers", meta)["outlier_rate"] == 0.01
    es = hkb.load("quad_planar", "exact", "test_id")
    Y, *_ , info = C.apply("burst_outage", es, meta)
    assert info["burst"]["channels"] == [0]                    # GPS loss only
    assert "deployment" in [hkb.registry.PLANTS[p]["role"] for p in hkb.list_plants(role="deployment")]
    assert "quad_planar" not in hkb.list_plants()             # core list unchanged


@needs_data
@pytest.mark.parametrize("nx", [4, 8, 16, 32, 64])
def test_scaling_plants_load(nx):
    p = f"fhnring_n{nx}"
    meta = hkb.metadata(p)
    assert meta["n_x"] == nx and meta["n_y"] == max(1, nx // 4)
    for k in ("exact", "realistic"):
        es = hkb.load(p, k, "test_sequence_ood")
        assert es.x.shape[-1] == nx and np.isfinite(es.x).all()
    ctx = hkb.build_context(p, "partial")
    import torch
    x = torch.zeros(2, nx, dtype=torch.float64)
    assert ctx["model"].f(x, torch.zeros(2, 1, dtype=torch.float64)).shape == (2, nx)
