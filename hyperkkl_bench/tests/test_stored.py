"""Stored research checkpoints, run through the package adapter on the exported episodes, reproduce
the stored per-episode NRMSE (research runs were on GPU float32; CPU float32 differs slightly: <= 1e-5 relative for the
UDE+EKF, <= 4e-3 relative for the GRU, whose 2400-step recurrence accumulates kernel rounding)."""
import json
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench.paths import REPO
from conftest import needs_data, needs_repo


@pytest.mark.slow
@needs_data
@needs_repo
@pytest.mark.parametrize("model,file,tol", [("ude5k1_ekf", "ude5k1", 1e-4), ("gru140", "gru140", 1e-2)])
def test_stored_reproduces(model, file, tol):
    from hyperkkl_bench.baselines.stored import StoredModel
    rep = hkb.evaluate(StoredModel(model, 0), "vdp", "partial", splits=["test_id"], conditions=("clean",),
                       n_episodes=4, cost=False, verbose=False)
    st = np.array(json.loads((REPO / "experiments/baselines2/runs/C3-partial/vdp/seed0" / f"{file}.json").read_text())
                  ["results"]["id"][model]["nrmse"][:4])
    mine = np.array(rep.results["test_id"]["clean"]["per_episode"]["nrmse"])
    assert np.max(np.abs(mine - st) / st) < tol, (mine, st)


@pytest.mark.slow
@needs_data
@needs_repo
def test_stored_hyperkkl_c5_reproduces():
    """The paper's method (HyperKKL C5, jitter_fix checkpoint) through the package, CPU, 4 episodes."""
    from hyperkkl_bench.baselines.stored import StoredHyperKKL
    rep = hkb.evaluate(StoredHyperKKL("C5_j3_s0.3_k0.5", 0), "cartpole_b", "realistic", splits=["test_id"],
                       conditions=("clean",), n_episodes=4, verbose=False)
    st = np.array(json.loads((REPO / "experiments/jitter_fix/runs/cartpole/C5_j3_s0.3_k0.5/seed0.json").read_text())
                  ["metrics"]["id/clean"]["raw"]["nrmse"][:4])
    mine = np.array(rep.results["test_id"]["clean"]["per_episode"]["nrmse"])
    assert np.max(np.abs(mine - st) / st) < 1e-3
    assert "latency_note" in rep.cost
