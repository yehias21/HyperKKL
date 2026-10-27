"""Deployment scorecard, robust / oracle filter wiring, incomplete knowledge, colored_t
sensor noise, burst outage, long episodes and the C2 memory probe.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_scorecard.py -q
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from hyperkkl import config as Cf
from hyperkkl import data as Dm
from hyperkkl import design as D
from hyperkkl import evaluate as Ev
from hyperkkl import metrics as M
from hyperkkl import sched as Sch
from hyperkkl import tree as Tr
from hyperkkl.__main__ import main as cli
from hyperkkl.nets import BaseMap, SFMEncoder
from hyperkkl.observers import EKF, UKF, HuberEKF, HuberUKF, Observer, oracle_noise, run
from hyperkkl.systems import KnownPart, build_plant


# ===========================================================================
# colored_t sensor noise
# ===========================================================================

def _ct(rho=0.9, nu=5.0, E=128, N1=4001, seed=3, ny=2):
    yc = torch.zeros(E, N1, ny, dtype=torch.float64)
    ys = [0.5, 2.0][:ny]
    y = Dm.measure(yc, {"noise": 0.1, "kind": "colored_t", "rho": rho, "nu": nu}, ys, seed)
    return (y / torch.tensor(ys, dtype=torch.float64) / 0.1).numpy()


def _lag1(e):
    return float((e[:, 1:] * e[:, :-1]).mean() / (e * e).mean())


def test_colored_t_stationary_variance_and_autocorrelation():
    e = _ct(rho=0.9, nu=5.0)
    assert np.allclose(e.var((0, 1)), 1.0, rtol=0.04), e.var((0, 1))
    assert abs(_lag1(e) - 0.9) < 0.01
    assert abs(e[:, 0].var() - 1.0) < 0.15                    # stationary from the first sample
    e2 = _ct(rho=0.5, nu=5.0)
    assert abs(_lag1(e2) - 0.5) < 0.01 and np.allclose(e2.var(), 1.0, rtol=0.04)


def test_colored_t_heavy_tails():
    for nu in (3.0, 5.0):
        e = _ct(rho=0.9, nu=nu).ravel()
        kurt = float(((e - e.mean()) ** 4).mean() / e.var() ** 2 - 3.0)
        assert kurt > 0.2, (nu, kurt)
    # white Student-t driving noise (rho = 0) keeps its own tails: theory 6 / (nu - 4) = 6 at nu = 5
    e = _ct(rho=0.0, nu=5.0).ravel()
    assert float(((e - e.mean()) ** 4).mean() / e.var() ** 2 - 3.0) > 2.0
    # a Gaussian AR(1) has none
    yc = torch.zeros(128, 4001, 1, dtype=torch.float64)
    g = Dm.measure(yc, {"noise": 0.1, "kind": "colored", "rho": 0.9}, [1.0], 3).numpy().ravel() / 0.1
    assert abs(float(((g - g.mean()) ** 4).mean() / g.var() ** 2 - 3.0)) < 0.1


def test_colored_t_reproducible_and_validated():
    a, b, c = _ct(seed=3, E=8, N1=300), _ct(seed=3, E=8, N1=300), _ct(seed=4, E=8, N1=300)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    yc = torch.zeros(2, 50, 1, dtype=torch.float64)
    for bad in ({"nu": 2.0}, {"nu": 1.5}, {"rho": 1.0}, {"rho": -0.1}):
        with pytest.raises(ValueError):
            Dm.measure(yc, {"noise": 0.1, "kind": "colored_t", **bad}, [1.0], 0)
    with pytest.raises(ValueError):
        Dm.measure(yc, {"noise": 0.1, "kind": "pink"}, [1.0], 0)
    # float32 input keeps its dtype (the generator's draws depend on dtype, as for every kind)
    y32 = Dm.measure(yc.float(), {"noise": 0.1, "kind": "colored_t"}, [1.0], 0)
    assert y32.dtype == torch.float32 and torch.isfinite(y32).all()


def test_oracle_noise_colored_t():
    SC = {"y_std": [2.0], "x_std": [1.0, 1.0]}
    ct = {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}
    _, r = oracle_noise(ct, SC)
    assert np.allclose(r, 0.03 * 2.0)
    _, r = oracle_noise(ct, SC, colored="longrun")
    assert np.allclose(r, 0.03 * 2.0 * np.sqrt(1.9 / 0.1))
    _, r = oracle_noise({"noise": 0.03, "kind": "colored_t"}, SC, colored="longrun")   # default rho 0.9
    assert np.allclose(r, 0.03 * 2.0 * np.sqrt(1.9 / 0.1))


def test_base_conditions_present():
    cfg = Cf.load("vdp")
    c = cfg["conditions"]
    assert c["colored_t"] == {"sensor": {"kind": "colored_t", "rho": 0.9, "nu": 3.0}}
    assert c["burst_outage"] == {"sensor": {"burst": {"start_frac": 0.6, "len_tref": 2.0}}}
    assert c["long"]["tiers"] == ["id"] and c["long"]["horizon"] >= 3 and c["long"]["n_episodes"] == 16
    assert cfg["eval"]["memory_probe"] is False and cfg["system"]["knowledge"] == "full"
    assert "long" not in cfg["eval"]["conditions"] and "burst_outage" not in cfg["eval"]["conditions"]


# ===========================================================================
# burst outage
# ===========================================================================

def test_burst_is_a_deterministic_sample_and_hold():
    yc = torch.randn(4, 201, 2, dtype=torch.float64)
    sens = {"noise": 0.05, "burst": {"start_frac": 0.6, "len_tref": 2.0}}
    y = Dm.measure(yc, sens, [1.0, 1.0], 9, dt=0.1, t_ref=1.5)
    y0 = Dm.measure(yc, {"noise": 0.05}, [1.0, 1.0], 9, dt=0.1, t_ref=1.5)
    k0, L = Dm.burst_window(sens["burst"], 201, 0.1, 1.5)
    assert (k0, L) == (120, 30)
    assert torch.equal(y[:, k0 + 1:k0 + L + 1], y0[:, k0:k0 + 1].expand(-1, L, -1))
    mask = torch.ones(201, dtype=torch.bool)
    mask[k0 + 1:k0 + L + 1] = False
    assert torch.equal(y[:, mask], y0[:, mask])
    assert torch.equal(y, Dm.measure(yc, sens, [1.0, 1.0], 9, dt=0.1, t_ref=1.5))
    assert Dm.burst_window({"start_frac": 0.95, "len_tref": 5}, 101, 0.1) == (95, 5)   # clipped
    with pytest.raises(ValueError):
        Dm.measure(yc, sens, [1.0, 1.0], 9)                    # needs dt


def test_moving_rms():
    rng = np.random.RandomState(0)
    ne = rng.rand(3, 50)
    m = M.moving_rms(ne, 5)
    for k in (0, 1, 2, 25, 48, 49):
        lo, hi = max(0, k - 2), min(50, k + 3)
        assert np.allclose(m[:, k], np.sqrt((ne[:, lo:hi] ** 2).mean(1)))
    assert np.allclose(M.moving_rms(ne, 1), ne)


def test_forget_time():
    E, N1, dt = 4, 401, 0.05
    ref = torch.randn(E, N1, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    t = torch.arange(N1, dtype=torch.float64) * dt
    tau = torch.tensor([0.2, 0.5, 1.0, 2.0], dtype=torch.float64)
    xh = ref + 5.0 * torch.exp(-t[None] / tau[:, None])[..., None]           # prior forgotten at rate 1/tau
    f = M.forget_time(xh, ref, [1.0, 1.0], dt, t_ref=0.5)
    # the gap 5 e^{-t/tau} falls below 0.05 at tau ln(100) (+ < half a smoothing window)
    want = (tau * np.log(100.0)).numpy()
    assert abs(f["t_forget_median_s"] - np.median(want)) < 0.3 and f["forget_censored"] == 0.0
    assert M.forget_time(ref, ref, [1.0, 1.0], dt)["t_forget_median_s"] == 0.0
    assert M.forget_time(ref + 1.0, ref, [1.0, 1.0], dt)["forget_censored"] == 1.0


def test_recovery_time_definition():
    E, N1, dt = 3, 401, 0.1
    ne = np.full((E, N1), 0.1)
    k0, L = 200, 20
    k1 = k0 + L + 1
    ne[0, k0 + 1:k1] = 1.0                                      # error during the outage
    ne[0, k1:k1 + 30] = 1.0                                     # stays high 30 samples, then back
    ne[1, k1:] = 1.0                                            # never recovers
    r = M.recovery_time(ne, k0, L, W=5)
    assert r["steps"].tolist() == [30, -1, 0]
    assert np.allclose(r["pre"], 0.1) and np.isclose(r["during"][0], 10.0) and np.isclose(r["during"][2], 1.0)


# ===========================================================================
# scorecard metrics
# ===========================================================================

def test_episode_metrics_measured_unmeasured():
    x = torch.zeros(2, 21, 3)
    xh = torch.zeros(2, 21, 3)
    xh[..., 0], xh[..., 1], xh[..., 2] = 0.1, 0.2, 0.4
    m = M.episode_metrics(x, xh, [1.0, 1.0, 2.0], 10, sensors=[1])
    assert np.allclose(m["nrmse_measured"], 0.2) and np.allclose(m["nrmse_unmeasured"], (0.1 + 0.2) / 2)
    assert "nrmse_unmeasured" not in M.episode_metrics(x, xh, [1.0, 1.0, 2.0], 10)
    m = M.episode_metrics(x, xh, [1.0, 1.0, 2.0], 10, sensors=[0, 1, 2])
    assert np.isnan(m["nrmse_unmeasured"]).all()


def test_auroc_matches_pairwise_definition_with_ties():
    rng = np.random.RandomState(0)
    s = rng.randint(0, 5, 300).astype(float)                    # many ties
    l = rng.rand(300) < 0.3 + 0.1 * s
    pos, neg = s[l], s[~l]
    ref = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert np.isclose(M.auroc(s, l), ref)
    assert M.auroc([1, 2, 3], [0, 1, 1]) == 1.0 and M.auroc([3, 2, 1], [0, 1, 1]) == 0.0
    assert np.isnan(M.auroc([1, 2], [1, 1]))


def _scorecard_case(E=40, N1=601, dt=0.05):
    """x = 0, xh = a decaying transient + a small floor on the unmeasured state 1; the
    measured state 0 is exact, so the residual alarm sees exactly the error of state 0."""
    rng = np.random.RandomState(0)
    t = np.arange(N1) * dt
    tau = rng.uniform(0.5, 3.0, E)
    xh = np.zeros((E, N1, 2))
    xh[:, :, 0] = np.exp(-t[None] / tau[:, None]) * 3.0
    xh[:, :, 1] = 0.02 * rng.randn(E, N1)
    x = np.zeros_like(xh)
    return torch.tensor(x), torch.tensor(xh), tau, t


def test_block_scorecard_convergence_tails_alarm():
    x, xh, tau, t = _scorecard_case()
    E, N1, _ = x.shape
    dt, settle = 0.05, 300
    y = torch.zeros(E, N1, 1, dtype=torch.float64)          # residual y - xhat_0 = -error of state 0
    sc = M.block_scorecard(x, xh, y, [1.0, 1.0], [1.0], [0], settle, dt, t_ref=2.0)
    em = M.episode_metrics(x, xh, [1.0, 1.0], settle, [0])
    assert np.isclose(sc["nrmse_p95"], np.percentile(em["nrmse"], 95), rtol=1e-6)
    assert np.isclose(sc["nrmse_p99"], np.percentile(em["nrmse"], 99), rtol=1e-6)
    assert sc["div_rate"] == 0.0 and len(sc["nrmse_state"]) == 2
    ne = M.moving_rms(np.sqrt((xh.numpy() ** 2).mean(-1)), int(round(2.0 / dt)))
    thr = max(2 * np.median(ne[:, settle:]), M.CONV_BAND)
    assert np.isclose(sc["conv_thr"], thr)
    want = np.array([np.where(ne[e] > thr)[0].max() + 1 for e in range(E)]) * dt
    # slow transients converge later (threshold crossing of a decaying exponential)
    assert np.corrcoef(want, tau)[0, 1] > 0.9
    assert np.isclose(sc["t_conv_median_s"], np.median(want)) and np.isclose(sc["t_conv_median_tref"], np.median(want) / 2.0)
    assert np.isclose(sc["t_conv_p90_s"], np.percentile(want, 90)) and sc["conv_censored"] == 0.0
    assert sc["alarm_auroc"] > 0.9               # residual = error of the measured state (EMA lag costs a little)
    # a residual that carries no information: AUROC ~ 0.5
    yr = torch.tensor(np.random.RandomState(1).randn(E, N1, 1)) + xh[..., :1]
    sc2 = M.block_scorecard(x, xh, yr + 0 * y, [1.0, 1.0], [1.0], [0], settle, dt, t_ref=2.0)
    yr2 = torch.tensor(np.random.RandomState(1).randn(E, N1, 1)) * 1e3 + xh[..., :1]
    assert abs(M.block_scorecard(x, xh, yr2, [1.0, 1.0], [1.0], [0], settle, dt, t_ref=2.0)["alarm_auroc"] - 0.5) < 0.1
    assert sc2["alarm_n"] <= 20000


def test_block_scorecard_censoring_divergence_drift():
    E, N1, dt = 10, 501, 0.1
    x = torch.zeros(E, N1, 1)
    xh = torch.full((E, N1, 1), 0.01)
    xh[0, -1] = 5.0                                              # above threshold at the end
    xh[1, 200] = 1e9                                             # diverged
    xh[2:, :] += torch.linspace(0, 1, N1)[None, :, None] * 0.1   # slow drift
    sc = M.block_scorecard(x, xh, xh.clone(), [1.0], [1.0], [0], 250, dt, drift_start=50)
    assert np.isclose(sc["conv_censored"], 0.1) and np.isclose(sc["div_rate"], 0.1)
    assert sc["t_conv_p90_s"] <= (N1 - 1) * dt + 1e-9
    assert sc["drift_ratio"] > 1.5 and sc["drift_ratio_median"] > 1.5
    flat = M.block_scorecard(x, torch.full((E, N1, 1), 0.01), torch.zeros(E, N1, 1), [1.0], [1.0], [0], 250, dt,
                             drift_start=50)
    assert np.isclose(flat["drift_ratio"], 1.0) and "rec_median_s" not in flat


def test_block_scorecard_recovery():
    E, N1, dt = 6, 801, 0.05
    k0, L = 480, 40
    k1 = k0 + L + 1
    x = torch.zeros(E, N1, 1, dtype=torch.float64)
    xh = torch.full((E, N1, 1), 0.05, dtype=torch.float64)
    t = torch.arange(N1 - k1, dtype=torch.float64) * dt
    xh[:, k1:, 0] = 0.05 + 2.0 * torch.exp(-t / 0.5)             # decays back after the outage
    sc = M.block_scorecard(x, xh, torch.zeros(E, N1, 1), [1.0], [1.0], [0], 400, dt, burst=(k0, L))
    W = int(round(0.5 / dt))
    r = M.recovery_time(np.abs(xh[..., 0].numpy()), k0, L, W)
    assert np.isclose(sc["rec_median_s"], np.median(r["steps"]) * dt) and sc["rec_not_recovered"] == 0.0
    # analytic: RMS over the next W samples of 0.05 + 2 e^{-t/0.5} <= 0.075 -> t ~ 0.5 ln(80) ~ 2.2 s
    assert 1.5 < sc["rec_median_s"] < 2.6 and sc["burst_k0"] == k0 and sc["burst_len"] == L


def test_fit_decay_exact():
    L = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    d = 3.0 * np.exp(-0.7 * L * 0.02) + 2.0 * np.exp(-5.0 * L * 0.02)
    f = M.fit_decay(L, d, 0.02)
    assert abs(f["rate"] - 0.7) / 0.7 < 0.02
    d2 = np.where(L < 100, np.exp(-0.7 * L * 0.02), 0.0)
    f2 = M.fit_decay(L, d2, 0.02)
    assert np.isclose(f2["L_floor_s"], 128 * 0.02) and abs(f2["rate"] - 0.7) < 0.01
    assert np.isnan(M.fit_decay(L, np.zeros_like(d), 0.02)["rate"])


# ===========================================================================
# C2 memory probe
# ===========================================================================

@pytest.mark.parametrize("poles", [[-0.5, -2.0], [-0.5 + 1j, -0.5 - 1j, -2 + 3j, -2 - 3j]])
def test_memory_probe_recovers_slowest_pole(poles):
    torch.manual_seed(0)
    dt = 0.05
    enc = SFMEncoder(1, 4, 16, dt, np.array(poles), n_blocks=2).double()
    assert abs(enc.alpha() - 0.5) < 1e-6
    dec = lambda z, c: torch.cat([c[:, :2] + 0.3 * torch.tanh(c[:, 2:4]), z[:, :1]], 1)
    g = torch.Generator().manual_seed(1)
    u = torch.randn(16, 1201, 1, generator=g, dtype=torch.float64)
    z = torch.randn(16, 1201, 3, generator=g, dtype=torch.float64)
    t_idx = np.linspace(600, 1200, 8).round().astype(int)
    Ls = Ev.probe_lengths(512)
    cur = Ev.probe_curves(enc, dec, u, z, t_idx, Ls, [1.0, 1.0, 1.0])
    for v in ("other", "zero"):
        d = np.asarray(cur[v])
        assert d[-1] < 1e-3 * d[0]                                  # decays with L
        rate = M.fit_decay(Ls, d, dt)["rate"]
        assert abs(rate - 0.5) / 0.5 < 0.10, (v, rate)
    # with the whole history kept nothing changes, exactly
    full = Ev.probe_curves(enc, dec, u, z, t_idx, [int(t_idx.max()) + 1], [1.0, 1.0, 1.0])
    assert full["other"][0] < 1e-12 and full["zero"][0] < 1e-12        # FFT round-off only


def test_memory_probe_on_trained_dict(monkeypatch):
    """memory_probe(): eligibility per method, float64 copies, results keys."""
    from hyperkkl.conditioning import ConditionedDecoder
    from hyperkkl.nets import CurrentEncoder, WindowEncoder
    cfg = Cf.load("vdp", {"eval.tiers": ["id"]})
    A, B = D.butterworth_AB(3, 1.0, 1)
    des = Dm.Design(A, B, 0.05, 400, 1.0, D.latent_step(A, B, 0.05))
    n = 6
    g = torch.Generator().manual_seed(0)
    es = Dm.EpisodeSet(torch.zeros(n, 401, 2), torch.randn(n, 401, 1, generator=g), torch.zeros(n, 401, 1),
                       torch.zeros(n, 401, 1), torch.zeros(n, 401, 3), torch.randn(n, 401, 3, generator=g),
                       ["f"] * n, burn=0, dt=0.05)
    monkeypatch.setattr(Dm, "eval_data", lambda c, tier, **k: es)
    Ti = BaseMap(3, 2, 16, 2, "tanh", (torch.zeros(3), torch.ones(3)), (torch.zeros(2), torch.ones(2)))
    torch.manual_seed(0)
    sfm = SFMEncoder(1, 4, 16, 0.05, np.array([-0.8, -3.0]), n_blocks=2)
    dec = ConditionedDecoder(Ti, "lora", 4, 8, 2)
    dw = ConditionedDecoder(Ti, "lora", 4, 8, 2)
    for v in list(dec.V) + list(dw.V):
        torch.nn.init.normal_(v.weight, std=0.3)
    trained = {"lora": {"cond": dec, "ctx": sfm}, "autonomous": {"cond": ConditionedDecoder(Ti, "none", 0, 0), "ctx": None},
               "latent": {"cond": object(), "ctx": sfm},
               "lora@window": {"cond": dw, "ctx": WindowEncoder(1, 4, 8, 20)},
               "lora@current": {"cond": dec, "ctx": CurrentEncoder(1, 4, 8)}}
    out = Ev.memory_probe(cfg, des, {"x_std": [1.0, 1.0]}, trained, "cpu", log=lambda *a: None)
    assert set(out) == {"lora", "latent", "lora@window", "lora@current"}
    assert "skipped" in out["latent"] and "skipped" in out["lora@current"]
    r = out["lora"]
    assert r["context"] == "sfm" and r["n_episodes"] == 6 and len(r["t_idx"]) == 8
    assert np.isclose(r["alpha_sfm"], 0.8, rtol=1e-5) and r["alpha_design"] == 1.0
    assert abs(r["rate_zero"] - 0.8) / 0.8 < 0.15 and max(r["L"]) <= min(r["t_idx"])
    assert next(sfm.parameters()).dtype == torch.float32          # the trained module is untouched
    w = out["lora@window"]
    assert w["context"] == "window" and w["d_other"][0] > 1e-6
    # a window of 20 samples forgets everything older exactly: d = 0 from L = 20 on
    assert np.isclose(w["fit_other"]["L_floor_s"], min(L for L in w["L"] if L >= 20) * 0.05)
    json.dumps(out)


# ===========================================================================
# robust / oracle filters in build_observers and evaluate
# ===========================================================================

def _maps(nz):
    T0 = BaseMap(2, nz, 16, 2, "tanh", (torch.zeros(2), torch.ones(2)), (torch.zeros(nz), torch.ones(nz)))
    Ti = BaseMap(nz, 2, 16, 2, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(2), torch.ones(2)))
    return {"T0": T0, "T0inv": Ti}


SC = {"x_std": [1.1, 0.9], "y_std": [1.3], "domain": [[-4.0, 4.0], [-5.0, 5.0]]}
ALL_B = ["ekf", "ukf", "pf", "kkl_mb", "ekf_huber", "ukf_huber", "ekf_oracle", "ukf_oracle", "ekf_oracle_huber"]


def test_build_observers_robust_and_oracle():
    cfg = Cf.load("vdp", {"baselines": ALL_B, "eval.pf_particles": 16})
    des = Dm.make_design(cfg["system"])
    obs = Ev.build_observers(cfg, des, SC, _maps(des.A.shape[0]), {}, None, "cpu")
    assert set(obs) == set(ALL_B)
    assert type(obs["ekf_huber"]) is HuberEKF and type(obs["ukf_huber"]) is HuberUKF
    for b in ("ekf_huber", "ukf_huber"):
        f = obs[b]
        assert f.robust == "huber" and not f.gate and not f.hold_missing
        assert np.allclose(f.r2, (0.03 * 1.3) ** 2) and np.allclose(f.q2, (0.005 * np.array(SC["x_std"])) ** 2 * des.dt)
    for b in Ev.ORACLE:
        assert isinstance(obs[b], Ev.BlockOracle)
    eo, uo, ho = (obs[b] for b in Ev.ORACLE)
    assert type(eo.default) is EKF and type(uo.default) is UKF and type(ho.default) is HuberEKF
    assert ho.default.gate and ho.default.robust == "huber"
    assert np.allclose(eo.default.r2, (0.03 * 1.3) ** 2) and np.allclose(eo.default.q2, obs["ekf"].q2)   # clean = nominal
    nominal = {"noise": 0.03}
    out = eo.for_block(dict(nominal, outlier_rate=0.02, outlier_scale=12.0), 0.0)
    assert np.allclose(out.r2, (0.03 ** 2 + 0.02 * 144) * 1.3 ** 2) and out.name == "ekf_oracle"
    hb = ho.for_block(dict(nominal, outlier_rate=0.02, outlier_scale=12.0), 0.0)
    assert np.allclose(hb.r2, (0.03 * 1.3) ** 2)                              # nominal sigma for Huber
    assert eo.for_block(dict(nominal, noise=0.1), 0.0).r2[0] == pytest.approx((0.1 * 1.3) ** 2)
    assert eo.for_block(dict(nominal, dropout=0.1), 0.0).hold_missing
    assert not eo.for_block(dict(nominal, dropout=0.1, quant=0.25), 0.0).hold_missing
    assert eo.for_block(dict(nominal, burst={"start_frac": 0.6, "len_tref": 2}), 0.0).hold_missing
    assert not eo.for_block(nominal, 0.0).hold_missing
    dr = eo.for_block(nominal, 0.3)
    assert dr.q2[0] == pytest.approx(eo.default.q2[0]) and dr.q2[1] > 10 * eo.default.q2[1]   # mu: state 2 only
    assert dr.r2 == pytest.approx(eo.default.r2)
    assert eo.for_block(nominal, 0.3) is dr                                  # cached per (sensor, shift)
    assert obs["pf"].m.params() == build_plant(cfg["system"]).params()


def _synth(tier, n, N1, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(N1, dtype=torch.float32) * 0.05
    x = torch.stack([torch.sin(t[None] * (1 + 0.1 * torch.arange(n)[:, None])),
                     torch.cos(t[None] * (1 + 0.1 * torch.arange(n)[:, None]))], -1)
    y = x[..., :1].clone()
    u = 0.3 * torch.randn(n, N1, 1, generator=g)
    z = torch.zeros(n, N1, 2)
    return Dm.EpisodeSet(x, u, y, y.clone(), z, z.clone(), [f"f{tier}"] * n, burn=0, dt=0.05)


def test_evaluate_oracle_per_block_long_and_burst(monkeypatch):
    NE = 60
    A, B = D.butterworth_AB(2, 1.0, 1)
    des = Dm.Design(A, B, 0.05, NE, 1.0, D.latent_step(A, B, 0.05))
    conds = ["clean", "noise10", "outliers", "burst_outage", "plant_drift10", "poor_init", "long"]
    cfg = Cf.load("vdp", {"eval.tiers": ["id", "shape_ood"], "eval.conditions": conds, "eval.chunk": 3,
                          "baselines": ["ekf", "ekf_oracle", "ekf_oracle_huber"], "eval.save_traj": False,
                          "conditions.long": {"horizon": 3.0, "n_episodes": 4, "tiers": ["id"]}})
    calls = []

    def fake_eval_data(cfg_, tier, plant_shift=0.0, device="cpu", horizon=1.0, n_episodes=None):
        calls.append((tier, plant_shift, horizon, n_episodes))
        n = n_episodes or {"id": 5, "shape_ood": 4}[tier]
        return _synth(tier, n, int(NE * horizon) + 1, {"id": 1, "shape_ood": 2}[tier] + int(10 * plant_shift))

    monkeypatch.setattr(Dm, "eval_data", fake_eval_data)
    obs = Ev.build_observers(cfg, des, SC, None, {}, None, "cpu")
    timing = {}
    out = Ev.evaluate(cfg, des, SC, obs, "cpu", log=lambda *a: None, timing=timing)
    assert ("id", 0.0, 3.0, 4) in calls and not any(c[0] == "shape_ood" and c[2] != 1.0 for c in calls)
    assert "long" in out["id"] and "long" not in out["shape_ood"]
    assert len(out["id"]["long"]["families"]) == 4
    assert {r["N1"] for r in timing["ekf_oracle"]} == {NE + 1, 3 * NE + 1}
    # oracle results equal the per-block filter run alone (chunks clipped at block ends)
    seed = Dm.split_seed("vdp", "id", "dev", 0)
    for cname, spec in (("outliers", {"outlier_rate": 0.02, "outlier_scale": 12.0}), ("noise10", {"noise": 0.10}),
                        ("burst_outage", {"burst": {"start_frac": 0.6, "len_tref": 2.0}})):
        sens = Cf.deep_merge(cfg["data"]["sensor"], spec)
        es = fake_eval_data(cfg, "id").with_sensor(des, sens, SC["y_std"], seed + 1, t_ref=1.0)
        x0 = Ev.initial_guess(cfg, len(es), "center", seed + 3, "cpu")
        for name in ("ekf_oracle", "ekf_oracle_huber"):
            f = obs[name].for_block(sens, 0.0)
            xh = torch.cat([run(f, es.y[i:i + 3], es.u[i:i + 3], x0[i:i + 3]) for i in range(0, len(es), 3)])
            m = M.episode_metrics(es.x, xh, SC["x_std"], int(0.5 * NE))
            assert np.allclose(out["id"][cname]["methods"][name]["nrmse"], m["nrmse"], rtol=1e-5), (cname, name)
    # every block has a scorecard for every method; burst / long windows only where they belong
    for tier, cs in out.items():
        for cname, blk in cs.items():
            assert set(blk["scorecard"]) == set(obs)
            for name, sc in blk["scorecard"].items():
                assert ("rec_median_s" in sc) == (cname == "burst_outage")
                assert ("drift_ratio" in sc) == (cname == "long")
                assert ("t_forget_median_s" in sc) == (cname == "poor_init")
                assert "alarm_auroc" in sc and "t_conv_median_s" in sc and "nrmse_p99" in sc
                assert len(blk["methods"][name]["nrmse_unmeasured"]) == len(blk["families"])
    json.dumps(out)


# ===========================================================================
# incomplete knowledge
# ===========================================================================

@pytest.mark.parametrize("system", ["vdp", "bioreactor", "lorenz"])
def test_partial_knowledge_reaches_every_model_based_observer(system):
    cfg = Cf.load(system, {"system.knowledge": "partial", "baselines": ALL_B, "eval.pf_particles": 8})
    des = Dm.make_design(cfg["system"])
    nx = cfg["system"]["ic_box"].__len__()
    sc = {"x_std": [1.0] * nx, "y_std": [1.0] * len(cfg["system"]["sensors"]), "domain": cfg["system"]["ic_box"]}
    base = {"T0": BaseMap(nx, des.A.shape[0], 8, 1, "tanh", (torch.zeros(nx), torch.ones(nx)),
                          (torch.zeros(des.A.shape[0]), torch.ones(des.A.shape[0]))),
            "T0inv": BaseMap(des.A.shape[0], nx, 8, 1, "tanh", (torch.zeros(des.A.shape[0]), torch.ones(des.A.shape[0])),
                             (torch.zeros(nx), torch.ones(nx)))}
    obs = Ev.build_observers(cfg, des, sc, base, {}, None, "cpu")
    known = build_plant(cfg["system"], knowledge="partial")
    full = build_plant(cfg["system"])
    box = np.asarray(cfg["system"]["ic_box"], float)
    g = torch.Generator().manual_seed(0)
    x = torch.as_tensor(box[:, 0]) + torch.rand(64, nx, generator=g, dtype=torch.float64) * torch.as_tensor(box[:, 1] - box[:, 0])
    u = 0.5 * torch.randn(64, 1, generator=g, dtype=torch.float64)
    assert not torch.allclose(known.f(x, u), full.f(x, u))
    for b, ob in obs.items():
        m = ob.model if b == "kkl_mb" else ob.m
        assert torch.equal(m.f(x, u), known.f(x, u)), b
        inner = m.base if hasattr(m, "u_bar") else m
        assert isinstance(inner, KnownPart), b
    # the oracle Q includes the (large) model error between f_known and the real plant
    assert (obs["ekf_oracle"].default.q2 > obs["ekf"].q2 * 10).any()


@pytest.fixture()
def tiny_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(Dm, "CACHE", tmp_path)
    sc = {"domain": [[-3.5, 3.5], [-3.5, 3.5]], "x_std": [1.0, 1.0], "y_std": [1.0], "t_burn": 0.5,
          "x_mean": [0.0, 0.0]}
    monkeypatch.setattr(Dm, "system_scales", lambda cfg, device="cpu": sc)
    return tmp_path


def _tiny(knowledge):
    return Cf.load("vdp", {"system.knowledge": knowledge, "system.T": 1.0, "data.n_p1": 4, "data.n_train": 6,
                           "data.n_val": 3, "data.n_coll": 64, "eval.n_episodes": 3})


def test_partial_knowledge_data_and_collocation(tiny_cache):
    full, part = _tiny("full"), _tiny("partial")
    a, b = Dm.training_data(full, 0), Dm.training_data(part, 0)
    for s in ("p1", "train", "val"):
        for k in ("x", "u", "y", "z", "z_obs"):
            assert torch.equal(getattr(a[s], k), getattr(b[s], k)), (s, k)      # the REAL plant generates data
    assert torch.equal(a["coll"]["x"], b["coll"]["x"]) and torch.equal(a["coll"]["y"], b["coll"]["y"])
    xc = a["coll"]["x"].double()
    zero = torch.zeros(len(xc), 1, dtype=torch.float64)
    known = build_plant(part["system"], knowledge="partial")
    assert torch.allclose(b["coll"]["f0"].double(), known.f(xc, zero).float().double())
    assert torch.allclose(a["coll"]["f0"].double(), build_plant(full["system"]).f(xc, zero).float().double())
    assert not torch.allclose(a["coll"]["f0"], b["coll"]["f0"])
    assert len(list((tiny_cache / "data").glob("vdp_train_*.pt"))) == 2           # caches split
    # evaluation episodes do not depend on knowledge: one cache entry
    e1, e2 = Dm.eval_data(full, "id"), Dm.eval_data(part, "id")
    assert torch.equal(e1.x, e2.x) and len(list((tiny_cache / "data").glob("vdp_eval_id_*.pt"))) == 1
    el = Dm.eval_data(full, "id", horizon=3.0, n_episodes=2)
    assert el.x.shape[0] == 2 and el.x.shape[1] - 1 == 3 * (e1.x.shape[1] - 1)
    assert Dm.model_plant(part["system"]).f(xc, zero).allclose(known.f(xc, zero))


def test_knowledge_in_model_cache_keys():
    from hyperkkl.config import digest
    f, p = _tiny("full"), _tiny("partial")
    assert digest(f, "system", "data", "phase1") != digest(p, "system", "data", "phase1")
    with pytest.raises(ValueError):
        build_plant(Cf.load("vdp")["system"], knowledge="half")


# ===========================================================================
# tree: deployment scorecard in reports + CLI
# ===========================================================================

@pytest.fixture()
def exp(tmp_path, monkeypatch):
    E = tmp_path / "experiments"
    monkeypatch.setattr(Tr, "EXP", E)
    monkeypatch.setattr(Tr, "TREE", E / "tree")
    monkeypatch.setattr(Tr, "QUEUE", E / "queue")
    monkeypatch.setattr(Sch, "PENDING", E / "queue" / "pending")
    monkeypatch.setattr(Sch, "RUNNING", E / "queue" / "running")
    monkeypatch.setattr(Sch, "FAILED", E / "queue" / "failed")
    monkeypatch.setattr(Sch, "REPORT_ASYNC", False)
    return E


def _sc(scale):
    return {"alarm_auroc": 0.8, "t_conv_median_s": 3.0 * scale, "conv_censored": 0.0, "t_forget_median_s": scale,
            "forget_censored": 0.0,
            "rec_median_s": 1.5 * scale, "rec_not_recovered": 0.25, "drift_ratio": 1.1 * scale,
            "nrmse_p99": 0.5}


def _payload(node, seed, v2=True):
    fams = ["a", "a", "b", "b"]
    meths = {"lora": 1.0, "ekf": 2.0}
    blocks = {}
    conds = {"id": ["clean", "poor_init", "long"], "shape_ood": ["clean", "outliers"],
             "sequence_ood": ["clean", "outliers", "burst_outage", "poor_init"]}
    for t, cs in conds.items():
        blocks[t] = {}
        for c in cs:
            k = 3.0 if c == "outliers" else 1.0
            ms = {m: {"nrmse": [0.1 * s * k * (i + 1) for i in range(4)], "diverged": [False, False, False, c == "outliers" and m == "ekf"]}
                  for m, s in meths.items()}
            if v2:
                for m, s in meths.items():
                    ms[m]["nrmse_unmeasured"] = [0.2 * s] * 4
            blocks[t][c] = {"families": fams, "methods": ms}
            if v2:
                blocks[t][c]["scorecard"] = {m: _sc(s) for m, s in meths.items()}
    cfg = Cf.load("vdp")
    return {"node": node, "system": "vdp", "seed": seed, "code": "c0", "wall_s": 10.0,
            "phase1": {"diag": {"eps_dec": 0.01}, "wall_s": 1.0}, "train": {"lora": {"n_params": 5}},
            "cost": {"lora": {"us_cpu_b1": 50.0}, "ekf": {"us_cpu_b1": 300.0}}, "results": blocks,
            "config": cfg, "scales": {}}


def _write(node, seed, payload):
    d = Tr.run_dir(node, "vdp", seed)
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(payload))


def test_scorecard_table_values(exp):
    Tr.new_node("S", None, "s", systems=["vdp"], seeds=[0, 1])
    for k in (0, 1):
        _write("S", k, _payload("S", k))
    rows = Tr.scorecard_table(Tr._load_runs("S")["vdp"])
    r = rows["ekf"]
    assert np.isclose(r["id"][0], 0.2 * 2.5) and np.isclose(r["sequence_ood"][0], 0.5)
    assert r["worst_sensor"][1] == "outliers" and np.isclose(r["worst_sensor"][0], 1.5)
    assert np.isclose(r["p99"][0], np.percentile([0.2, 0.4, 0.6, 0.8], 99))
    assert np.isclose(r["div"][0], 0.25) and r["div"][1].endswith("/outliers")
    assert np.isclose(r["forget"][0], 2.0) and np.isclose(r["conv"][0], 6.0) and np.isclose(r["recovery"][0], 3.0) and r["recovery"][1] == "25% cens."
    assert np.isclose(r["drift"][0], 2.2) and np.isclose(r["auroc"][0], 0.8) and r["us"][0] == 300.0
    assert np.isclose(rows["lora"]["unmeasured"][0], 0.2)


def test_report_has_scorecard_and_survives_v1_payloads(exp):
    Tr.new_node("S", None, "s", systems=["vdp"], seeds=[0])
    _write("S", 0, _payload("S", 0))
    txt = Tr.report("S").read_text()
    assert "## Deployment scorecard" in txt and "recovery (burst)" in txt and "drift (long)" in txt
    Tr.new_node("V1", None, "v1", systems=["vdp"], seeds=[0])
    p = _payload("V1", 0, v2=False)
    for t in p["results"]:                                       # v1: no burst / long / poor_init blocks
        for c in [c for c in p["results"][t] if c not in ("clean", "outliers")]:
            del p["results"][t][c]
    _write("V1", 0, p)
    txt = Tr.report("V1").read_text()
    sec = txt[txt.index("## Deployment scorecard"):]
    head = [l for l in sec.splitlines() if l.startswith("| method")][0]
    assert "recovery" not in head and "drift" not in head and "forget" not in head and "AUROC" not in head and "unmeasured" not in head
    assert "worst sensor fault" in head and "p99 (seq)" in head


def test_cli_scorecard_writes_md_and_tex(exp, capsys):
    Tr.new_node("S", None, "s", systems=["vdp"], seeds=[0])
    Tr.new_node("T", None, "t", systems=["vdp"], seeds=[0])
    _write("S", 0, _payload("S", 0))
    _write("T", 0, _payload("T", 0, v2=False))
    cli(["scorecard", "S", "T"])
    md, tex = exp / "scorecard" / "scorecard_S_T.md", exp / "scorecard" / "scorecard_S_T.tex"
    assert md.exists() and tex.exists()
    m = md.read_text()
    assert "| S:lora |" in m and "| T:ekf |" in m and "## vdp" in m
    t = tex.read_text()
    assert "\\begin{tabular}" in t and "S:lora" in t and "\\_" in t and "\\toprule" in t
    assert "scorecard_S_T.md" in capsys.readouterr().out


# ===========================================================================
# oracle tuning on validation episodes (job.tune_oracle)
# ===========================================================================

def _val_set(cfg, n=6, N=80, seed=0):
    """Real vdp episodes (true plant, the job's sensor) on a short grid."""
    sc = cfg["system"]
    full = Dm.make_design(sc)
    des = Dm.Design(full.A, full.B, full.dt, N, full.alpha, full.step)
    scales = {"domain": [[-4.0, 4.0], [-5.0, 5.0]], "x_std": [1.0, 1.5], "y_std": [1.0], "t_burn": 0.0}
    es = Dm.generate(build_plant(sc), des, scales, sc, dict(cfg["data"], k_cross=1), ["sinusoid", "prbs"], n,
                     "id", seed, "cpu")
    return des, scales, es


def test_tune_oracle_picks_lowest_validation_error():
    from hyperkkl import job as J
    cfg = Cf.load("vdp", {"system.knowledge": "partial", "baselines": ["ekf_oracle"],
                          "eval.oracle_grid": {"q_inflate": [0.0, 1.0, 10.0], "r_inflate": [1.0, 2.0]}})
    des, scales, es = _val_set(cfg)
    choice, rec = J.tune_oracle(cfg, des, scales, es, "cpu", log=lambda *a: None)
    sc = rec["scores"]
    assert len(sc) == 6 and rec["n_val"] == 6
    key = lambda s: s["val_nrmse"] + s["val_diverged"]
    best = min(sc, key=key)
    assert choice == {"q_inflate": best["q_inflate"], "r_inflate": best["r_inflate"]} == rec["choice"]
    assert len({round(s["val_nrmse"], 6) for s in sc}) > 1                    # the grid matters
    # every score is reproduced by a direct run of the configured filter
    s0 = sc[0]
    c = Cf.deep_merge(cfg, {"eval": {"oracle_tuning": {"q_inflate": s0["q_inflate"], "r_inflate": s0["r_inflate"]}}})
    f = Ev.build_observers(c, des, scales, None, {}, None, "cpu")["ekf_oracle"]
    xh = run(f, es.y, es.u, Ev.initial_guess(cfg, len(es), "center", 0, "cpu"))
    assert np.isclose(np.mean(M.episode_metrics(es.x, xh, scales["x_std"], int(0.5 * 80))["nrmse"]), s0["val_nrmse"], rtol=1e-5)
    # an exact model (full knowledge, no mismatch): only r_inflate is searched
    cf = Cf.load("vdp", {"baselines": ["ekf_oracle"], "eval.oracle_grid": {"q_inflate": [0.0, 1.0, 10.0], "r_inflate": [1.0, 2.0]}})
    _, rf = J.tune_oracle(cf, des, scales, es, "cpu", log=lambda *a: None)
    assert [s["q_inflate"] for s in rf["scores"]] == [1.0, 1.0]
    assert Cf.load("vdp")["eval"]["oracle_grid"] == {"q_inflate": [0.0, 0.3, 1.0, 3.0, 10.0], "r_inflate": [1.0, 2.0]}


def test_oracle_tuning_scales_model_error_and_R():
    cfg = Cf.load("vdp", {"system.knowledge": "partial", "baselines": list(Ev.ORACLE)})
    des = Dm.make_design(cfg["system"])
    o1 = Ev.build_observers(cfg, des, SC, None, {}, None, "cpu")
    o3 = Ev.build_observers(Cf.deep_merge(cfg, {"eval": {"oracle_tuning": {"q_inflate": 3.0, "r_inflate": 2.0}}}),
                            des, SC, None, {}, None, "cpu")
    o0 = Ev.build_observers(Cf.deep_merge(cfg, {"eval": {"oracle_tuning": {"q_inflate": 0.0, "r_inflate": 1.0}}}),
                            des, SC, None, {}, None, "cpu")
    proc = (0.005 * np.array(SC["x_std"])) ** 2 * des.dt
    for b in Ev.ORACLE:
        q1, q3, q0 = o1[b].default.q2, o3[b].default.q2, o0[b].default.q2
        assert np.allclose(q0, proc)                                          # model-error term off
        assert np.allclose(q3 - proc, 3.0 * (q1 - proc)) and (q1 > proc).any()
        assert np.allclose(o3[b].default.r2, 2.0 * o1[b].default.r2)
        # per block, the condition's sensor still sets R (times r_inflate)
        r_out = o3[b].for_block({"noise": 0.1}, 0.0).r2
        assert np.allclose(r_out, 2.0 * (0.1 * 1.3) ** 2)


def test_run_job_records_oracle_tuning(exp, monkeypatch):
    from hyperkkl import job as J

    class _Set:
        n_rejected = 0

        def __len__(self):
            return 3

    Tr.new_node("O", None, "o", methods=["autonomous"], baselines=["ekf", "ekf_oracle"], systems=["vdp"], seeds=[0])
    seen = {}
    monkeypatch.setattr(J, "snapshot_code", lambda ch: "x")
    monkeypatch.setattr(J.Dm, "system_scales", lambda cfg, device: {
        "design_check": {"ok": True}, "t_burn": 1.0, "memory_s": 2.0, "x_std": [1, 1], "y_std": [1]})
    monkeypatch.setattr(J.Dm, "training_data", lambda cfg, seed, device: {"p1": _Set(), "train": _Set(), "val": _Set()})
    monkeypatch.setattr(J, "_cached_model", lambda path, build: build())
    monkeypatch.setattr(J, "phase1", lambda *a: {"diag": {"eps_dec": 0.01}, "wall_s": 1.0})
    monkeypatch.setattr(J, "phase2", lambda *a: {"n_params": 0, "cond": None, "ctx": None})
    monkeypatch.setattr(J, "tune_oracle", lambda cfg, des, sc, val, dev, log: (
        {"q_inflate": 3.0, "r_inflate": 1.0}, {"choice": {"q_inflate": 3.0, "r_inflate": 1.0}, "scores": [], "n_val": 3}))
    monkeypatch.setattr(J, "build_observers", lambda cfg, *a: seen.setdefault("tun", cfg["eval"].get("oracle_tuning")) and {})
    monkeypatch.setattr(J, "state_coverage", lambda *a: {})
    monkeypatch.setattr(J, "evaluate", lambda *a, **k: {})
    monkeypatch.setattr(J, "_costs", lambda *a: {})
    pay = J.run_job("O", "vdp", 0, "cpu")
    on_disk = json.loads((Tr.run_dir("O", "vdp", 0) / "results.json").read_text())
    assert on_disk["oracle_tuning"]["choice"] == {"q_inflate": 3.0, "r_inflate": 1.0} == pay["oracle_tuning"]["choice"]
    assert on_disk["config"]["eval"]["oracle_tuning"] == {"q_inflate": 3.0, "r_inflate": 1.0}
    assert seen["tun"] == {"q_inflate": 3.0, "r_inflate": 1.0}               # the evaluated filters use it
