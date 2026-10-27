"""Robust (Huber / gated) measurement updates and condition-aware oracle noise
(hyperkkl/observers.py: _Filter._robust_update, HuberEKF, HuberUKF, oracle_noise,
model_error_rate).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_robust_filters.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import chi2

from hyperkkl import design as D
from hyperkkl.observers import (EKF, UKF, HuberEKF, HuberUKF, _Filter, model_error_rate,
                                oracle_noise, run)
from hyperkkl.systems import Plant, build_plant

T = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)


class Lin3(Plant):
    """x' = F x + G u, y = (x1, x3)."""
    nx = 3
    PARAMS = {}
    DEFAULT_SENSORS = (0, 2)
    F = np.array([[0.0, 1.0, 0.0], [-2.0, -0.3, 0.5], [0.2, 0.0, -0.7]])
    G = np.array([0.0, 1.0, 0.3])

    def f(self, x, u):
        F = torch.as_tensor(self.F, dtype=x.dtype, device=x.device)
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        return x @ F.T + u[:, :1] * G


class Lin3Full(Lin3):
    """Every state measured: the error is dominated by the measured coordinates, so the
    effect of the measurement update (and of outliers) is not masked by an unmeasured
    state's error."""
    DEFAULT_SENSORS = (0, 1, 2)


PL, DT, N = Lin3(), 0.05, 400
Q = np.array([0.1, 0.2, 0.1])
R = np.array([0.05, 0.08])
P0 = np.array([1.0, 1.5, 0.8])
PLF, RF = Lin3Full(), np.array([0.05, 0.08, 0.05])


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    """3x3 problems: intra-op threading only adds contention (10x slower on a busy host)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def simulate(E, seed, outlier_rate=0.0, outlier_scale=12.0, N=N, pl=None, R=R):
    """E exact linear-Gaussian episodes (ZOH input); outliers +-scale*sigma at the given
    rate, independently per sample and component, as data.measure applies them."""
    pl = PL if pl is None else pl
    rng = np.random.RandomState(seed)
    st = D.latent_step(PL.F, PL.G[:, None], DT)
    t = np.arange(N + 1) * DT
    ph = rng.uniform(0, 2 * np.pi, E)
    u = (np.sin(1.3 * t[None] + ph[:, None]) + 0.5 * np.cos(0.4 * t[None]))[..., None]
    x = np.zeros((E, N + 1, 3))
    x[:, 0] = P0 * rng.randn(E, 3)
    for k in range(N):
        x[:, k + 1] = x[:, k] @ st["Ad"].T + u[:, k] @ (st["W0"] + st["W1"]).T \
            + Q * np.sqrt(DT) * rng.randn(E, 3)
    ny = len(pl.sensors)
    y = x[:, :, list(pl.sensors)] + R * rng.randn(E, N + 1, ny)
    m = rng.rand(E, N + 1, ny) < outlier_rate
    y = np.where(m, y + outlier_scale * R * rng.choice([-1.0, 1.0], size=y.shape), y)
    return T(x), T(y), T(u), m


def rmse(xh, x, burn=50):
    return float(((xh - x)[:, burn:] ** 2).mean().sqrt())


def go(ob, x, y, u, dt=torch.float64):
    return run(ob, y.to(dt), u.to(dt), torch.zeros(len(y), 3, dtype=dt)).double()


@pytest.fixture(scope="module")
def clean():
    return simulate(64, 0)


@pytest.fixture(scope="module")
def dirty():
    return simulate(64, 0, outlier_rate=0.02)


# ---------------------------------------------------------------------------
# 1. The default filters are bit-identical to the pre-robust code
# ---------------------------------------------------------------------------

def _old_update(self, y):
    """Verbatim copy of _Filter._update before the robust options were added."""
    H, P = self.H, self.P
    S = H @ P @ H.T + self.R
    K = P @ H.T @ torch.linalg.inv(S)
    self.x = self.x + (K @ (y - self.x @ H.T)[:, :, None])[:, :, 0]
    I = torch.eye(P.shape[1], dtype=P.dtype, device=P.device)
    A = I - K @ H
    self.P = A @ P @ A.transpose(1, 2) + K @ self.R @ K.transpose(1, 2)
    self.x = torch.nan_to_num(self.x, nan=1e6, posinf=1e6, neginf=-1e6)


@pytest.mark.parametrize("cls", [EKF, UKF])
@pytest.mark.parametrize("dt", [torch.float32, torch.float64])
def test_default_filters_bit_identical(cls, dt, dirty):
    x, y, u, _ = dirty
    y = y.clone()
    y[3, 100] = float("nan")                                  # NaN path identical too
    Old = type("Old", (cls,), {"_update": _old_update})
    new, old = cls(PL, DT, Q, R, P0), Old(PL, DT, Q, R, P0)
    assert new._plain
    a, b = go(new, x, y, u, dt), go(old, x, y, u, dt)
    assert torch.equal(a, b)
    assert torch.equal(new.P, old.P)


# ---------------------------------------------------------------------------
# 2. Statistical behaviour on a linear-Gaussian toy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HuberEKF, HuberUKF])
def test_huber_with_infinite_threshold_is_the_kalman_filter(cls, clean):
    x, y, u, _ = clean
    base = EKF if cls is HuberEKF else UKF
    a = go(cls(PL, DT, Q, R, P0, huber_k=float("inf")), x, y, u)
    b = go(base(PL, DT, Q, R, P0), x, y, u)
    assert (a - b).abs().max() < 1e-10


@pytest.mark.parametrize("cls,base", [(HuberEKF, EKF), (HuberUKF, UKF)])
def test_huber_efficiency_without_outliers(cls, base, clean):
    """Gaussian data: the Huber filter (k = 1.345, ~95% asymptotic efficiency) costs at
    most 5% RMSE relative to the (optimal) Kalman filter."""
    x, y, u, _ = clean
    e_h, e_k = rmse(go(cls(PL, DT, Q, R, P0), x, y, u), x), rmse(go(base(PL, DT, Q, R, P0), x, y, u), x)
    assert e_h < 1.05 * e_k, (e_h, e_k)


@pytest.fixture(scope="module")
def full():
    return simulate(64, 1, pl=PLF, R=RF), simulate(64, 1, outlier_rate=0.02, pl=PLF, R=RF)


@pytest.mark.parametrize("cls,base", [(HuberEKF, EKF), (HuberUKF, UKF)])
def test_huber_beats_plain_filter_with_outliers(cls, base, full, dirty):
    """2% outliers at 12 sigma per component (the `outliers` stress condition)."""
    (xc, yc, uc, _), (x, y, u, _) = full
    e_h = rmse(go(cls(PLF, DT, Q, RF, P0), x, y, u), x)
    e_k = rmse(go(base(PLF, DT, Q, RF, P0), x, y, u), x)
    e_clean = rmse(go(base(PLF, DT, Q, RF, P0), xc, yc, uc), xc)
    assert e_h < 0.7 * e_k, (e_h, e_k)
    assert e_h < 1.15 * e_clean, (e_h, e_clean)                 # close to outlier-free KF
    # the gate alone (plain KF update + chi-square rejection) also helps
    e_g = rmse(go(base(PLF, DT, Q, RF, P0, gate=True), x, y, u), x)
    assert e_g < 0.8 * e_k, (e_g, e_k)
    # partially observed plant: smaller effect (unmeasured x2 dominates), same direction
    x, y, u, _ = dirty
    e_h = rmse(go(cls(PL, DT, Q, R, P0), x, y, u), x)
    e_k = rmse(go(base(PL, DT, Q, R, P0), x, y, u), x)
    assert e_h < 0.92 * e_k, (e_h, e_k)


def test_oracle_mixture_R_is_not_enough(dirty):
    """Re-tuning R to the outlier-mixture variance (the condition-aware oracle) is the
    best a Gaussian KF can do, and the Huber filter still beats it."""
    x, y, u, _ = dirty
    r_mix = R * np.sqrt(1 + 0.02 * 144)
    e_o = rmse(go(EKF(PL, DT, Q, r_mix, P0), x, y, u), x)
    e_h = rmse(go(HuberEKF(PL, DT, Q, R, P0), x, y, u), x)
    e_k = rmse(go(EKF(PL, DT, Q, R, P0), x, y, u), x)
    assert e_o < e_k and e_h < e_o, (e_h, e_o, e_k)


def test_gate_rejects_injected_outliers(clean):
    x, y, u, _ = clean
    y = y.clone()
    ks = np.arange(60, N, 37)
    y[:, ks, 0] += 50 * R[0]                                   # 50 sigma on sensor 0
    ob = EKF(PL, DT, Q, R, P0, gate=True)
    xh = go(ob, x, y, u)
    n_inj = len(ks) * len(y)
    n_rej = int(ob.n_rejected)
    # every injected outlier rejected; false alarms ~ (1 - 0.999) of the clean updates
    false_budget = chi2.sf(chi2.ppf(0.999, 2), 2) * len(y) * (N + 1)
    assert n_inj <= n_rej <= n_inj + 5 * false_budget + 5, (n_inj, n_rej)
    e_clean = rmse(go(EKF(PL, DT, Q, R, P0), x, clean[1], u), x)
    assert rmse(xh, x) < 1.05 * e_clean
    # per-step: every episode's update is rejected exactly at the injected steps
    ob2 = EKF(PL, DT, Q, R, P0, gate=True)
    ob2.reset(len(y), torch.zeros(len(y), 3, dtype=torch.float64), u[:, 0])
    for k in range(N):
        n0 = int(ob2.n_rejected)
        ob2.advance(y[:, k], u[:, k], y[:, k + 1], u[:, k + 1])
        if k + 1 in ks:
            assert int(ob2.n_rejected) - n0 == len(y), k


# ---------------------------------------------------------------------------
# 3. Batching, missing data, dtypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HuberEKF, HuberUKF])
def test_batched_equals_per_episode(cls, dirty):
    x, y, u, _ = dirty
    x, y, u = x[:6], y[:6], u[:6]
    kw = dict(gate=True)
    a = go(cls(PL, DT, Q, R, P0, **kw), x, y, u)
    b = torch.cat([go(cls(PL, DT, Q, R, P0, **kw), x[i:i + 1], y[i:i + 1], u[i:i + 1])
                   for i in range(len(y))])
    assert (a - b).abs().max() < 1e-12


@pytest.mark.parametrize("cls", [HuberEKF, HuberUKF])
def test_nan_measurements(cls, clean):
    """Default: a NaN measurement poisons the state, exactly as for the plain filters
    (run() reports +-1e6, i.e. a divergence). skip_nan=True treats it as missing."""
    x, y, u, _ = clean
    y = y.clone()
    y[:, 200, 1] = float("nan")
    y[:5, 250] = float("nan")
    xh = go(cls(PL, DT, Q, R, P0), x, y, u)
    xp = go((EKF if cls is HuberEKF else UKF)(PL, DT, Q, R, P0), x, y, u)
    from hyperkkl.metrics import episode_metrics
    for a in (xh, xp):                                         # poisoned at k = 200 ...
        assert (a[:, 200].abs() == 1e6).all()
        assert episode_metrics(x.float(), a.float(), [1.0] * 3, 50)["diverged"].all()
    xs = go(cls(PL, DT, Q, R, P0, skip_nan=True, gate=True), x, y, u)
    assert torch.isfinite(xs).all() and xs.abs().max() < 1e3
    ref = go(cls(PL, DT, Q, R, P0), x, clean[1], u)
    assert rmse(xs, x) < 1.02 * rmse(ref, x)


def test_hold_missing_for_dropout(clean):
    """Dropout in data.measure is a sample-and-hold (no NaN). With hold_missing the
    held samples are skipped; without it the filter treats a stale value as fresh."""
    x, y, u, _ = clean
    y = y.clone()
    rng = np.random.RandomState(5)
    hold = rng.rand(len(y), N + 1) < 0.3
    for k in range(1, N + 1):
        y[:, k] = torch.where(T(hold[:, k, None], torch.bool), y[:, k - 1], y[:, k])
    e_skip = rmse(go(EKF(PL, DT, Q, R, P0, hold_missing=True), x, y, u), x)
    e_plain = rmse(go(EKF(PL, DT, Q, R, P0), x, y, u), x)
    assert e_skip < e_plain, (e_skip, e_plain)


@pytest.mark.parametrize("cls", [HuberEKF, HuberUKF])
def test_float32_matches_float64(cls, dirty):
    x, y, u, _ = dirty
    a = go(cls(PL, DT, Q, R, P0, gate=True), x, y, u, torch.float32)
    b = go(cls(PL, DT, Q, R, P0, gate=True), x, y, u, torch.float64)
    assert torch.isfinite(a).all()
    assert (a - b).abs().max() < 1e-3 * b.abs().max()
    assert abs(rmse(a, x) - rmse(b, x)) < 1e-3 * rmse(b, x)


def test_invalid_robust_option():
    with pytest.raises(ValueError):
        EKF(PL, DT, Q, R, P0, robust="cauchy")


# ---------------------------------------------------------------------------
# 4. Condition-aware oracle noise
# ---------------------------------------------------------------------------

SC = {"x_std": [2.0, 1.0, 0.5], "y_std": [2.0, 0.5]}


def test_oracle_noise_levels():
    base = {"noise": 0.03}
    q, r = oracle_noise(base, SC, noise_proc=0.005)
    assert np.allclose(q, 0.005 * np.array(SC["x_std"]))
    assert np.allclose(r, 0.03 * np.array(SC["y_std"]))       # = evaluate.build_observers
    _, r = oracle_noise({"noise": 0.10}, SC)
    assert np.allclose(r, 0.10 * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, outlier_rate=0.02, outlier_scale=12.0), SC)
    assert np.allclose(r, np.sqrt(0.03 ** 2 + 0.02 * 144) * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, outlier_rate=0.02), SC, outliers="ignore")
    assert np.allclose(r, 0.03 * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, quant=0.25), SC)
    assert np.allclose(r, np.sqrt(0.03 ** 2 + 0.25 ** 2 / 12) * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, kind="colored", rho=0.95), SC)
    assert np.allclose(r, 0.03 * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, kind="colored", rho=0.95), SC, colored="longrun")
    assert np.allclose(r, 0.03 * np.sqrt(39.0) * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, drift=0.3), SC)
    assert np.allclose(r, 0.03 * np.array(SC["y_std"]))
    _, r = oracle_noise(dict(base, drift=0.3), SC, drift="variance")
    assert np.allclose(r, np.sqrt(0.03 ** 2 + 0.24 * 0.09) * np.array(SC["y_std"]))
    _, r = oracle_noise({}, SC)                                # noise floor 1e-3
    assert np.allclose(r, 1e-3 * np.array(SC["y_std"]))
    q, _ = oracle_noise(base, SC, noise_proc=0.005, dt=0.01, model_err=[1.0, 0.0, 2.0],
                        q_inflate=2.0)
    assert np.allclose(q ** 2, (0.01 * np.array(SC["x_std"])) ** 2 + np.array([1, 0, 4]) * 0.01)
    with pytest.raises(ValueError):
        oracle_noise(base, SC, model_err=[1.0, 0.0, 0.0])


def test_oracle_quantisation_variance_is_right():
    """q^2/12 is the empirical variance of data.measure's quantisation error."""
    from hyperkkl.data import measure
    yc = torch.linspace(-3, 3, 200001, dtype=torch.float64)[None, :, None]
    yq = measure(yc, {"quant": 0.25}, [1.0], 0)
    assert abs(float((yq - yc).var()) / (0.25 ** 2 / 12) - 1) < 0.01


def test_model_error_rate():
    nom = build_plant({"plant": "vdp", "sensors": [0]})
    assert np.allclose(model_error_rate(nom, nom, [[-2, 2], [-2, 2]]), 0.0)
    me = model_error_rate(nom, nom.perturbed(0.3), [[-2, 2], [-2, 2]])
    assert me.shape == (2,) and (me >= 0).all() and me.max() > 0

