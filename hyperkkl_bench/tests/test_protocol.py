"""Estimator protocol: step-by-step and run_batch agree; evaluate() produces a full report."""
import numpy as np
import pytest
import hyperkkl_bench as hkb
from hyperkkl_bench.estimator import run_episodes
from conftest import needs_data


class _Wrap:
    """Hides run_batch so run_episodes uses the per-step loop."""
    def __init__(self, e):
        self.e, self.name, self.context = e, e.name, e.context

    def reset(self, x0, p0=None):
        return self.e.reset(x0, p0)

    def step(self, u, y):
        return self.e.step(u, y)


@needs_data
def test_ekf_step_equals_batch():
    ctx = hkb.build_context("duffing", "partial")
    es = hkb.load("duffing", "partial", "test_id").subset([0, 1])
    est = hkb.baselines.EKF(gate=True)
    est.fit(None, None, ctx)
    N = 300
    U, Y = es.u[:, :N].astype(float), es.y[:, :N].astype(float)
    X0 = np.tile(ctx["prior_mean"], (2, 1))
    a, Pa = run_episodes(est, U, Y, X0)
    b, Pb = run_episodes(_Wrap(est), U, Y, X0)
    assert np.allclose(a, b, atol=1e-10) and np.allclose(Pa, Pb, atol=1e-10)


@needs_data
def test_gru_step_equals_batch_and_report():
    est = hkb.baselines.GRUObserver(hidden=8, steps=30, window=64, warmup=8, batch=8)
    rep = hkb.evaluate(est, "vdp", "exact", splits=["test_id", "test_sequence_ood"],
                       conditions=("clean", "outliers", "burst_outage", "poor_init"),
                       train_n=20, n_episodes=4, verbose=False)
    es = hkb.load("vdp", "exact", "test_id").subset([0])
    X0 = np.tile(est.context["prior_mean"], (1, 1))
    a, _ = run_episodes(est, es.u.astype(float), es.y.astype(float), X0)
    b, _ = run_episodes(_Wrap(est), es.u.astype(float), es.y.astype(float), X0)
    assert np.allclose(a, b, atol=1e-5)
    d = rep.to_dict()
    assert set(d["results"]["test_id"]) == {"clean", "outliers", "burst_outage", "poor_init"}
    assert d["cost"]["n_params"] > 0 and d["cost"]["latency_us_median"] > 0 and d["cost"]["flops_per_step"] > 0
    assert "input_ood_gap" in d["meta"] and "test_sequence_ood" in d["meta"]["input_ood_gap"]
    assert "| test_id |" in rep.table()


@needs_data
def test_context_hides_real_parameters():
    ctx = hkb.build_context("vdp", "realistic")
    assert "mismatch30_params" not in ctx["meta"]
    assert "mismatch30_params" not in hkb.metadata("vdp")
    assert "mismatch30_params" in hkb.metadata("vdp", include_real_plant=True)
    assert ctx["model_knowledge"] == "partial"
    # modeller's model is nominal: differs from the real (perturbed) plant
    import torch
    from hyperkkl_bench.plants import build_plant
    real = build_plant(hkb.metadata("vdp")["system_config"], perturb=0.3)
    x = torch.randn(5, 2, dtype=torch.float64); u = torch.zeros(5, 1, dtype=torch.float64)
    assert not torch.allclose(ctx["model"].f(x, u), real.f(x, u))


@needs_data
def test_conditions_deterministic_and_effective():
    from hyperkkl_bench import conditions as C
    es = hkb.load("vdp", "exact", "test_id")
    meta = hkb.metadata("vdp")
    y1, *_ = C.apply("outliers", es, meta)
    y2, *_ = C.apply("outliers", es, meta)
    assert np.array_equal(y1, y2)
    frac = np.mean(np.abs(y1 - es.y) > 5 * np.asarray(meta["y_std"]))
    assert 0.01 < frac < 0.03
    yb, _, _, info = C.apply("burst_outage", es, meta)
    k0, L = info["burst"]["k0"], info["burst"]["len"]
    assert L > 0 and np.all(yb[:, k0:k0 + L + 1] == yb[:, k0:k0 + 1])
    _, x0c, _, _ = C.apply("clean", es, meta)
    _, x0p, _, _ = C.apply("poor_init", es, meta)
    assert not np.allclose(x0c, x0p)


@pytest.mark.slow
@needs_data
def test_tuned_filter_selects_on_val_only():
    est = hkb.baselines.TunedKF("ekf", gate=True, q_grid=(1, 100), r_grid=(1,), n_val=4)
    est.fit(None, hkb.load("vdp", "mismatch", "val"), hkb.build_context("vdp", "mismatch"))
    assert set(est.fit_info["val_scores"]) == {"q1_r1", "q100_r1"}
    assert est.fit_info["chosen"]["q_inflate"] in (1.0, 100.0)


def test_track_score_against_stored_reference():
    from hyperkkl_bench import tracks
    ref = tracks.reference_reports("ekf_gated", "core")
    if len(ref) < 4:
        pytest.skip("reference runs not present")
    sc = tracks.track_score(ref, ref)
    assert abs(sc["geo_mean_ratio"] - 1.0) < 1e-12 and sc["wins"] == 0


@pytest.mark.slow
@needs_data
def test_ude_filter_trains_through_rk4_and_handles_sparse_labels():
    est = hkb.baselines.UDEFilter(steps=20, K=4, batch=32, budget=500, lrs=(1e-3,), n_val_deployed=2)
    rep = hkb.evaluate(est, "cstr_exo", "partial", splits=["test_id"], conditions=("clean",), n_episodes=2,
                       cost=False, verbose=False)
    assert np.isfinite(est.fit_info["val_rollout_mse_norm"])
    assert rep.value("test_id") is not None and est.n_params() > 0


@pytest.mark.slow
@needs_data
def test_reference_filter_behaviour_p12():
    """PITFALLS P12, as documented: the stale-sample rule handles the burst outage; the published
    gate is outlier-robust on battery; the optional safeguards fix the lorenz lock-out."""
    r = hkb.evaluate(hkb.baselines.EKF(gate=True), "battery", "exact", splits=["test_sequence_ood"],
                     conditions=("outliers", "burst_outage"), n_episodes=16, cost=False, verbose=False)
    assert r.value(condition="outliers") < 0.02 and r.value(condition="burst_outage") < 0.02
    safe = hkb.baselines.EKF(gate=True, gate_warmup_tref=3.0, lockout_reset=True, lockout_huber=True)
    r = hkb.evaluate(safe, "lorenz", "exact", splits=["test_sequence_ood"], conditions=("clean",),
                     n_episodes=16, cost=False, verbose=False)
    assert r.value() < 0.02

