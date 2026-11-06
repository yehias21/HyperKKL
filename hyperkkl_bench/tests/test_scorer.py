"""The scorer reproduces the stored per-episode NRMSE of the research runs to 1e-6, using the
estimates the runs saved (experiments/tree/<node>/runs/<plant>/seed0/traj.npz: first 4 episodes of
the OOD tiers) and the exported true states."""
import json
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench.metrics import episode_metrics
from hyperkkl_bench.paths import REPO
from conftest import needs_data, needs_repo

CELLS = [("vdp", "C1-where", "exact"), ("battery", "C1-where", "exact"), ("quad_tank", "C3-realistic", "realistic"),
         ("lorenz", "C3-mismatch30", "mismatch"), ("bioreactor", "C3-partial", "partial"),
         ("fhn", "TEST-C3", "realistic"), ("cartpole_b", "TEST-C1-B", "exact"), ("two_link_arm", "TEST-C1", "exact")]


@needs_data
@needs_repo
@pytest.mark.parametrize("plant,node,knowledge", CELLS)
def test_scorer_matches_stored(plant, node, knowledge):
    run = REPO / "experiments" / "tree" / node / "runs" / plant / "seed0"
    tj = np.load(run / "traj.npz")
    res = json.loads((run / "results.json").read_text())["results"]
    meta = hkb.metadata(plant)
    n_checked = 0
    for tier in ("shape_ood", "sequence_ood"):
        es = hkb.load(plant, knowledge, f"test_{tier}")
        x = es.x[:4]
        assert np.array_equal(tj[f"{tier}/clean/true"], x), "exported x differs from the run's x"
        for key in tj.files:
            t, c, m = key.split("/")
            if t != tier or m == "true":
                continue
            em = episode_metrics(x, tj[key], meta["x_std"], meta["settle_index"], meta["sensors"])
            st = res[tier]["clean"]["methods"][m]
            for f in ("nrmse", "nrmse_unmeasured"):
                assert np.allclose(em[f], st[f][:4], rtol=0, atol=1e-6), (m, f, em[f], st[f][:4])
            n_checked += 1
    assert n_checked >= 2


def test_metric_definitions():
    x = np.zeros((2, 11, 3))
    xh = x.copy()
    xh[:, 5:, 1] = 2.0                      # error 2 on state 2 in the settled window
    m = episode_metrics(x, xh, [1.0, 2.0, 1.0], 5, sensors=[0])
    assert np.allclose(m["nrmse_x2"], 1.0)
    assert np.allclose(m["nrmse_unmeasured"], 0.5)      # mean of states 2 (1.0) and 3 (0.0)
    assert np.allclose(m["nrmse"], np.sqrt(1 / 3))
    xh[0, 0, 0] = np.nan
    m = episode_metrics(x, xh, [1.0, 2.0, 1.0], 5, sensors=[0])
    assert m["diverged"][0] and not m["diverged"][1]


def test_calibration_ideal_gaussian():
    from hyperkkl_bench.metrics import calibration
    rng = np.random.RandomState(0)
    x = np.zeros((4, 2001, 2))
    P = np.tile(np.diag([0.25, 4.0]), (4, 2001, 1, 1))
    xh = rng.randn(4, 2001, 2) * np.array([0.5, 2.0])
    c = calibration(x, xh, P, 0)
    assert abs(c["coverage68"] - 0.683) < 0.02 and abs(c["coverage95"] - 0.95) < 0.01
    assert abs(c["anees"] - 1.0) < 0.05
