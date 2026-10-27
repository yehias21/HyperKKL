"""Numerical verification of hyperkkl/observers.py: the observer contract, the exact latent
step, the classical filters against the exact Kalman filter, the particle filter's
resampling, the model-based KKL correction, the context timing of the learned family, and
the baselines' noise statistics on real evaluation data.

    .venv/bin/python -m pytest tests/test_verify_observers.py -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from hyperkkl import design as D
from hyperkkl import signals as S
from hyperkkl.conditioning import ConditionedDecoder, build_conditioner
from hyperkkl.data import Design, integrate, run_latent
from hyperkkl.metrics import episode_metrics
from hyperkkl.nets import BaseMap, ContextEncoder, build_context
from hyperkkl.observers import (EKF, UKF, KKLObserver, LatentStep, ModelBasedKKL,
                                ParticleFilter, run, time_per_step)
from hyperkkl.systems import Plant

torch.set_default_dtype(torch.float32)
T = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)


# ---------------------------------------------------------------------------
# Test plants (h = selection of sensors, as for every benchmark plant)
# ---------------------------------------------------------------------------

class Lin2(Plant):
    """x' = F x + G u, y = x1."""
    nx = 2
    PARAMS = {}
    DEFAULT_SENSORS = (0,)
    F = np.array([[0.0, 1.0], [-2.0, -0.3]])
    G = np.array([0.0, 1.0])

    def f(self, x, u):
        F = torch.as_tensor(self.F, dtype=x.dtype, device=x.device)
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        return x @ F.T + u[:, :1] * G


class Lin3(Lin2):
    """Three states, two sensors (x1, x3)."""
    nx = 3
    DEFAULT_SENSORS = (0, 2)
    F = np.array([[0.0, 1.0, 0.0], [-2.0, -0.3, 0.5], [0.2, 0.0, -0.7]])
    G = np.array([0.0, 1.0, 0.3])


class Static(Plant):
    """x' = 0, y = x (for the resampling test)."""
    nx = 1
    PARAMS = {}

    def f(self, x, u):
        return torch.zeros_like(x)


class _Lin(torch.nn.Module):
    """x -> M x, also usable as a decoder (z, c) -> M z."""

    def __init__(self, M, dtype=torch.float32):
        super().__init__()
        self.register_buffer("M", torch.as_tensor(np.asarray(M), dtype=dtype))

    def forward(self, x, c=None):
        return x @ self.M.T


def _H(pl):
    H = np.zeros((len(pl.sensors), pl.nx))
    for i, s in enumerate(pl.sensors):
        H[i, s] = 1.0
    return H


def _design(dt, N, nz=4, alpha=1.0, ny=1):
    A, B = D.butterworth_AB(nz, alpha, ny)
    return Design(A, B, dt, N, alpha, D.latent_step(A, B, dt))


def _lat64(des):
    lat = LatentStep(des, "cpu")
    for k in ("Ad", "W0", "W1", "Wg"):
        setattr(lat, k, T(des.step[k]))
    lat.A, lat.B = T(des.A), T(des.B)
    return lat


def _simulate_linear(pl, dt, N, q, r, seed, x0_std):
    """Exact discrete linear-Gaussian model with the input on a zero-order hold (as
    data.integrate applies it): x_{k+1} = Phi x_k + (W0 + W1) u_k + q sqrt(dt) w_k,
    y_k = H x_k + r v_k  (W0 + W1 = A^{-1}(Ad - I) G, the exact ZOH input matrix)."""
    rng = np.random.RandomState(seed)
    st = D.latent_step(pl.F, pl.G[:, None], dt)
    t = np.arange(N + 1) * dt
    u = (np.sin(1.3 * t) + 0.5 * np.cos(0.4 * t))[:, None]
    x = np.zeros((N + 1, pl.nx))
    x[0] = x0_std * rng.randn(pl.nx)
    for k in range(N):
        x[k + 1] = st["Ad"] @ x[k] + (st["W0"] + st["W1"]) @ u[k] \
            + q * np.sqrt(dt) * rng.randn(pl.nx)
    y = x @ _H(pl).T + r * rng.randn(N + 1, len(pl.sensors))
    return x, y, u


def _kalman(pl, dt, q, r, p0, x0, y, u):
    """The exact discrete-time Kalman filter. Prior N(x0, diag p0^2) on x_0, y_0 is
    assimilated before the first prediction; xs[0] is the prior (observer contract),
    xs[k] = E[x_k | y_0..y_k] for k >= 1. Standard (I - K H) P covariance update."""
    st = D.latent_step(pl.F, pl.G[:, None], dt)
    Phi, W0, W1 = st["Ad"], st["W0"], st["W1"]
    H, Q, R = _H(pl), np.diag(q ** 2 * dt), np.diag(r ** 2)
    I = np.eye(pl.nx)

    def upd(x, P, yk):
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        return x + K @ (yk - H @ x), (I - K @ H) @ P

    x, P = np.asarray(x0, float).copy(), np.diag(p0 ** 2)
    xs, Ps = [x.copy()], [P.copy()]
    x, P = upd(x, P, y[0])
    for k in range(len(y) - 1):
        x, P = Phi @ x + (W0 + W1) @ u[k], Phi @ P @ Phi.T + Q
        x, P = upd(x, P, y[k + 1])
        xs.append(x.copy()); Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


def _steps(ob, y, u, x0):
    """Run an observer step by step, keeping its mean and covariance after every step."""
    ob.reset(y.shape[0], x0, u[:, 0])
    xs, Ps = [ob.estimate().clone()], [ob.P.clone()]
    for k in range(y.shape[1] - 1):
        ob.advance(y[:, k], u[:, k], y[:, k + 1], u[:, k + 1])
        xs.append(ob.estimate().clone()); Ps.append(ob.P.clone())
    return torch.stack(xs, 1), torch.stack(Ps, 1)


# ---------------------------------------------------------------------------
# 1. Contract: N+1 outputs, xhat[0] = prior, no look-ahead, batch == single
# ---------------------------------------------------------------------------

class _NoCtx(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z, c):
        return self.dec(z)


def _randomise(m, std=0.1):
    for p in m.parameters():
        if p.requires_grad:
            p.data.normal_(0, std)
    return m


def _all_observers(des, pl):
    """One instance of every observer class (and every KKLObserver configuration the
    evaluation builds: autonomous, decoder-conditioned, latent injection, NLOX)."""
    from hyperkkl.evaluate import RawInput
    torch.manual_seed(0)
    nz, nx = des.A.shape[0], pl.nx
    lat = LatentStep(des, "cpu")
    P = D.sylvester_reference(pl.F, _H(pl), des.A, des.B)
    T0, Ti = _Lin(P), _Lin(np.linalg.pinv(P))
    base = BaseMap(nz, nx, 16, 2, "tanh", (torch.zeros(nz), torch.ones(nz)),
                   (torch.zeros(nx), torch.ones(nx)))
    ccfg = {"dim": 4, "hidden": 8, "sfm_blocks": 2, "window": 5}
    poles = np.linalg.eigvals(des.A)
    ctx1 = _randomise(build_context("sfm", ccfg, 1, des.dt, poles, 1.0), 0.3)
    ctx2 = _randomise(build_context("gru", ccfg, 1, des.dt, poles, 1.0), 0.3)
    film = _randomise(build_conditioner("film", base, 4, 8, 4, torch.ones(nz)), 0.3)
    inj = _randomise(build_conditioner("latent", base, 4, 8, 4, torch.ones(nz)), 0.3)
    omega = _randomise(torch.nn.Linear(nz, nz), 0.1)
    q, r, p0 = np.array([0.05, 0.1]), np.array([0.05]), np.array([1.0, 1.0])
    return {
        "kkl_auto": KKLObserver("auto", lat, Ti, z0_map=T0),
        "kkl_film_sfm": KKLObserver("film", lat, film, ctx=ctx1, z0_map=T0),
        "kkl_latent_gru": KKLObserver("latent", lat, ConditionedDecoder(base, "none", 0, 0),
                                      ctx=ctx2, injection=inj, z0_map=T0),
        "nlox": KKLObserver("nlox", lat, _NoCtx(base), ctx=RawInput(),
                            injection=lambda z, c: omega(z) * c, z0_map=None),
        "kkl_mb": ModelBasedKKL(lat, T0, Ti, pl),
        "ekf": EKF(pl, des.dt, q, r, p0),
        "ukf": UKF(pl, des.dt, q, r, p0),
        "pf": ParticleFilter(pl, des.dt, q, r, p0, n_particles=256),
    }


@pytest.fixture(scope="module")
def contract_data():
    dt, N = 0.05, 60
    des = _design(dt, N)
    pl = Lin2()
    rng = np.random.RandomState(0)
    sigs = [S.sample_signal(f, "train", rng, dt, N * dt) for f in ("sinusoid", "prbs", "square")]
    x, _, u = integrate(pl, des, np.array([[0.5, -0.3], [1.0, 0.2], [-0.4, 0.6]]), sigs)
    y = (x[:, :, :1] + 0.02 * torch.randn(x[:, :, :1].shape, dtype=x.dtype,
                                          generator=torch.Generator().manual_seed(1))).float()
    return des, pl, y, u.float(), torch.tensor([[0.2, 0.1], [0.0, 0.0], [-0.3, 0.2]])


@pytest.mark.parametrize("name", ["kkl_auto", "kkl_film_sfm", "kkl_latent_gru", "nlox",
                                  "kkl_mb", "ekf", "ukf", "pf"])
def test_contract(contract_data, name):
    des, pl, y, u, x0 = contract_data
    ob = _all_observers(des, pl)[name]
    B, N1 = y.shape[:2]
    xh = run(ob, y, u, x0)
    assert xh.shape == (B, N1, pl.nx)
    assert torch.isfinite(xh).all()
    # xhat[0] = prior: exactly for the filters, through z0 = T0(x0_hat) for the KKL family
    if name in ("ekf", "ukf", "pf"):
        assert torch.equal(xh[:, 0], x0)
    elif name in ("kkl_auto", "kkl_mb"):                      # exact maps: T0+ T0 = I
        assert torch.allclose(xh[:, 0], x0, atol=1e-5)
    elif name == "nlox":                                      # documented: z0 = 0
        assert torch.allclose(xh[:, 0], ob.dec(torch.zeros(B, des.A.shape[0]), None))
    # no look-ahead: perturbing y and u after k leaves xhat[0..k] bit-identical,
    # and xhat[k+1] does depend on y_{k+1}
    g = torch.Generator().manual_seed(2)
    for k in (0, 7, 33, N1 - 2):
        y2, u2 = y.clone(), u.clone()
        y2[:, k + 1:] += torch.randn(y2[:, k + 1:].shape, generator=g)
        u2[:, k + 1:] += torch.randn(u2[:, k + 1:].shape, generator=g)
        xh2 = run(ob, y2, u2, x0)
        assert torch.equal(xh2[:, :k + 1], xh[:, :k + 1]), (name, k)
        assert not torch.allclose(xh2[:, k + 1], xh[:, k + 1]), (name, k)
    # batch == single episode (deterministic observers; the PF shares one random stream
    # across the batch, so it agrees only in distribution: see test_pf_matches_kalman)
    if name != "pf":
        for i in range(B):
            xi = run(ob, y[i:i + 1], u[i:i + 1], x0[i:i + 1])
            assert torch.allclose(xi[0], xh[i], atol=1e-5, rtol=1e-5), (name, i)
    # repeated runs are identical (reset really resets, including the PF's generator)
    assert torch.equal(run(ob, y, u, x0), xh)


def test_time_per_step_every_observer(contract_data):
    des, pl, *_ = contract_data
    for name, ob in _all_observers(des, pl).items():
        us = time_per_step(ob, 1, 1, pl.nx, "cpu", B=4, n=20)
        assert np.isfinite(us) and us > 0, name


# ---------------------------------------------------------------------------
# 2. LatentStep
# ---------------------------------------------------------------------------

def _augmented_reference(A, B, dt, z, yk, yk1, g):
    """expm of  [z, y, ydot, g]' = [A z + B y + g, ydot, 0, 0]."""
    nz, ny = B.shape
    n = nz + 2 * ny + nz
    M = np.zeros((n, n))
    M[:nz, :nz] = A
    M[:nz, nz:nz + ny] = B
    M[nz:nz + ny, nz + ny:nz + 2 * ny] = np.eye(ny)
    M[:nz, nz + 2 * ny:] = np.eye(nz)
    E = expm(M * dt)
    w = np.concatenate([z, yk, (yk1 - yk) / dt, g], 1)
    return (w @ E.T)[:, :nz]


@pytest.mark.parametrize("ny", [1, 2])
def test_latent_step_exact_for_linear_y_and_constant_injection(ny):
    rng = np.random.RandomState(0)
    des = _design(0.07, 10, nz=5, alpha=1.3, ny=ny)
    z, yk, yk1, g = rng.randn(6, 5), rng.randn(6, ny), rng.randn(6, ny), rng.randn(6, 5)
    ref0 = _augmented_reference(des.A, des.B, des.dt, z, yk, yk1, 0 * g)
    ref = _augmented_reference(des.A, des.B, des.dt, z, yk, yk1, g)
    lat = _lat64(des)                                         # float64: the weights themselves
    assert np.abs(lat(T(z), T(yk), T(yk1)).numpy() - ref0).max() < 1e-12
    assert np.abs(lat(T(z), T(yk), T(yk1), lambda zz: T(g)).numpy() - ref).max() < 1e-12
    lat32 = LatentStep(des, "cpu")                            # the deployed float32 step
    f = lambda a: T(a, torch.float32)
    out = lat32(f(z), f(yk), f(yk1), lambda zz: f(g)).double().numpy()
    assert np.abs(out - ref).max() < 2e-6 * max(1.0, np.abs(ref).max())


def test_latent_step_exponential_midpoint_is_second_order():
    rng = np.random.RandomState(1)
    nz, alpha, Tend = 4, 1.0, 2.0
    A, B = D.butterworth_AB(nz, alpha, 1)
    W = 0.8 * rng.randn(nz, nz)
    phi_np = lambda zz: np.tanh(zz @ W.T) + 0.3 * np.sin(zz)
    phi_t = lambda zz: torch.tanh(zz @ T(W).T) + 0.3 * torch.sin(zz)
    yfun = lambda t: np.sin(2.0 * t) + 0.5
    z0 = rng.randn(nz)
    errs = []
    for dt in (0.1, 0.05, 0.025, 0.0125):
        N = int(round(Tend / dt))
        ts = np.arange(N + 1) * dt
        ys = yfun(ts)
        # reference with y linearly interpolated between samples: only the injection's
        # quadrature error remains
        sol = solve_ivp(lambda t, zz: A @ zz + B[:, 0] * np.interp(t, ts, ys) + phi_np(zz),
                        (0, Tend), z0, method="DOP853", rtol=1e-12, atol=1e-13,
                        max_step=dt / 4)
        lat = _lat64(Design(A, B, dt, N, alpha, D.latent_step(A, B, dt)))
        z = T(z0[None])
        for k in range(N):
            z = lat(z, T([[ys[k]]]), T([[ys[k + 1]]]), phi_t)
        errs.append(np.abs(z[0].numpy() - sol.y[:, -1]).max())
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert (order > 1.85).all() and (order < 2.3).all(), (errs, order)


# ---------------------------------------------------------------------------
# 3. EKF / UKF == exact Kalman filter on a linear-Gaussian plant
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lg():
    pl, dt, N = Lin3(), 0.05, 300
    q, r, p0 = np.array([0.1, 0.2, 0.1]), np.array([0.05, 0.08]), np.array([1.0, 1.5, 0.8])
    x, y, u = _simulate_linear(pl, dt, N, q, r, seed=0, x0_std=p0)
    x0h = np.zeros(pl.nx)
    xk, Pk = _kalman(pl, dt, q, r, p0, x0h, y, u)
    return dict(pl=pl, dt=dt, q=q, r=r, p0=p0, x=x, y=y, u=u, x0h=x0h, xk=xk, Pk=Pk)


@pytest.mark.parametrize("cls", [EKF, UKF])
def test_filter_equals_exact_kalman(lg, cls):
    ob = cls(lg["pl"], lg["dt"], lg["q"], lg["r"], lg["p0"])
    xs, Ps = _steps(ob, T(lg["y"])[None], T(lg["u"])[None], T(lg["x0h"])[None])
    xs, Ps = xs[0].numpy(), Ps[0].numpy()
    # RK4 (EKF mean, UKF sigma points) vs expm: O(dt^5) per step
    assert np.abs(xs - lg["xk"]).max() < 1e-5, np.abs(xs - lg["xk"]).max()
    assert np.abs(Ps[1:] - lg["Pk"][1:]).max() < 1e-6, np.abs(Ps[1:] - lg["Pk"][1:]).max()
    # covariance symmetric and positive definite at every step
    assert np.abs(Ps - Ps.transpose(0, 2, 1)).max() < 1e-12
    assert np.linalg.eigvalsh(Ps).min() > 0
    # the posterior mean beats the prior-only estimate (sanity: the filter is used)
    mse = ((xs - lg["x"]) ** 2)[50:].mean()
    assert mse < 0.05 * (lg["x"][50:] ** 2).mean()


def test_ekf_uses_y0():
    """The first measurement is assimilated: xhat[1] = E[x_1 | y_0, y_1]."""
    pl, dt = Lin3(), 0.05
    q, r, p0 = np.array([0.1, 0.2, 0.1]), np.array([0.05, 0.08]), np.array([1.0, 1.5, 0.8])
    x, y, u = _simulate_linear(pl, dt, 3, q, r, seed=3, x0_std=p0)
    xk, _ = _kalman(pl, dt, q, r, p0, np.zeros(3), y, u)
    for cls in (EKF, UKF):
        xh = run(cls(pl, dt, q, r, p0), T(y)[None], T(u)[None], T(np.zeros((1, 3))))[0].numpy()
        assert np.abs(xh - xk).max() < 1e-6, cls.name


def test_joseph_update():
    """_update is the Joseph form A P A' + K R K' with the optimal gain: equal to the
    textbook (I - K H) P in exact arithmetic, and symmetric PSD in float32 even for an
    ill-conditioned prior covariance."""
    pl = Lin3()
    rng = np.random.RandomState(4)
    q, r, p0 = np.ones(3), np.array([0.05, 0.08]), np.ones(3)
    ob = EKF(pl, 0.05, q, r, p0)
    for dt_, cond in ((torch.float64, 1e2), (torch.float32, 1e7)):
        ob.reset(4, torch.zeros(4, 3, dtype=dt_), torch.zeros(4, 1, dtype=dt_))
        Us = [np.linalg.qr(rng.randn(3, 3))[0] for _ in range(4)]
        P = np.stack([U @ np.diag(np.geomspace(1.0, 1.0 / cond, 3)) @ U.T * 100 for U in Us])
        x, y = rng.randn(4, 3), rng.randn(4, 2)
        ob.P, ob.x = T(P, dt_), T(x, dt_)
        ob._update(T(y, dt_))
        H, R = _H(pl), np.diag(r ** 2)
        for b in range(4):
            K = P[b] @ H.T @ np.linalg.inv(H @ P[b] @ H.T + R)
            Aj = np.eye(3) - K @ H
            Pj = Aj @ P[b] @ Aj.T + K @ R @ K.T
            xr = x[b] + K @ (y[b] - H @ x[b])
            tol = 1e-10 if dt_ == torch.float64 else 1e-3
            assert np.abs(ob.P[b].double().numpy() - Pj).max() < tol * np.abs(Pj).max()
            assert np.abs(ob.x[b].double().numpy() - xr).max() < tol * max(1, np.abs(xr).max())
            if dt_ == torch.float64:
                assert np.abs(Pj - Aj @ P[b]).max() < 1e-10 * np.abs(Pj).max()
        Pn = ob.P.double().numpy()
        sym = 1e-12 if dt_ == torch.float64 else 1e-4           # Joseph form is not symmetrised
        assert np.abs(Pn - Pn.transpose(0, 2, 1)).max() < sym * np.abs(Pn).max()
        assert np.linalg.eigvalsh(0.5 * (Pn + Pn.transpose(0, 2, 1))).min() > -1e-6 * np.abs(Pn).max()


# ---------------------------------------------------------------------------
# 4. Particle filter
# ---------------------------------------------------------------------------

PF_REPORT = {}


def test_pf_matches_kalman(lg):
    """With the TRUE process noise (inflate = 1) and many particles the PF mean is the KF
    mean within Monte-Carlo error; the deployed setting (512 particles, inflate 3) is
    reported as a deviation in posterior standard deviations."""
    y, u, x0 = T(lg["y"])[None], T(lg["u"])[None], T(lg["x0h"])[None]
    sd = np.sqrt(np.diagonal(lg["Pk"], axis1=1, axis2=2))
    for Np, infl, seed in ((20000, 1.0, 0), (20000, 1.0, 1), (512, 3.0, 0), (512, 1.0, 0)):
        pf = ParticleFilter(lg["pl"], lg["dt"], lg["q"], lg["r"], lg["p0"],
                            n_particles=Np, inflate=infl, seed=seed)
        xh = run(pf, y, u, x0)[0].numpy()
        d = np.sqrt((((xh - lg["xk"]) / sd)[20:] ** 2).mean())
        PF_REPORT[(Np, infl, seed)] = (d, ((xh - lg["x"]) ** 2)[50:].mean() /
                                       ((lg["xk"] - lg["x"]) ** 2)[50:].mean())
    assert PF_REPORT[(20000, 1.0, 0)][0] < 0.06, PF_REPORT
    assert PF_REPORT[(20000, 1.0, 1)][0] < 0.06, PF_REPORT
    assert PF_REPORT[(512, 1.0, 0)][0] < 0.3, PF_REPORT
    assert PF_REPORT[(512, 3.0, 0)][0] < 1.0, PF_REPORT       # inflated proposal: biased
    print("\nPF vs KF (rms deviation in posterior stds, MSE ratio PF/KF):", PF_REPORT)


def _static_pf(Np, r):
    pf = ParticleFilter(Static(), 0.1, [0.0], [r], [1.0], n_particles=Np)
    pf.reset(1, torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64))
    grid = torch.linspace(-1, 1, Np, dtype=torch.float64)
    pf.X = grid[None, :, None].clone()
    pf._fresh = False                       # weight with y_{k+1} only (first-step y_0 off)
    return pf, grid


def test_pf_systematic_resampling():
    Np, r, yv = 1000, 0.2, 0.3
    pf, grid = _static_pf(Np, r)
    y = torch.full((1, 1), yv, dtype=torch.float64)
    w = torch.softmax(-0.5 * (yv - grid) ** 2 / r ** 2, 0)
    ess = 1.0 / (w * w).sum()
    assert ess < Np / 2
    pf.advance(y, 0 * y, y, 0 * y)
    # estimate = weighted mean BEFORE resampling
    assert abs(float(pf.x[0, 0]) - float((w * grid).sum())) < 1e-9
    # resampled: weights reset, offspring counts are floor or ceil of Np w_i
    assert torch.equal(pf.logw, torch.zeros_like(pf.logw))
    idx = torch.round((pf.X[0, :, 0] + 1) / 2 * (Np - 1)).long()
    assert (pf.X[0, :, 0] - grid[idx]).abs().max() < 1e-6        # only the 1e-9 jitter
    cnt = torch.bincount(idx, minlength=Np).double()
    assert cnt.sum() == Np
    assert ((cnt >= torch.floor(Np * w) - 1e-9) & (cnt <= torch.ceil(Np * w) + 1e-9)).all()


def test_pf_no_resampling_above_ess_threshold():
    Np, r, yv = 1000, 5.0, 0.3                                    # flat likelihood
    pf, grid = _static_pf(Np, r)
    y = torch.full((1, 1), yv, dtype=torch.float64)
    w = torch.softmax(-0.5 * (yv - grid) ** 2 / r ** 2, 0)
    assert 1.0 / (w * w).sum() > Np / 2
    pf.advance(y, 0 * y, y, 0 * y)
    assert (pf.X[0, :, 0] - grid).abs().max() < 1e-6              # particles kept in place
    assert torch.allclose(torch.softmax(pf.logw[0], 0), w, atol=1e-12)   # weights carried


# ---------------------------------------------------------------------------
# 5. KKL observers with EXACT maps on linear plants
# ---------------------------------------------------------------------------

def _exact_maps(pl, des):
    P = D.sylvester_reference(pl.F, _H(pl), des.A, des.B)
    return P, _Lin(P), _Lin(np.linalg.pinv(P))


def _forced(pl, des, fam, regime="train", seed=5, x0=(1.0, -0.5)):
    rng = np.random.RandomState(seed)
    sig = S.sample_signal(fam, regime, rng, des.dt, des.T)
    x, _, u = integrate(pl, des, np.array([x0]), [sig])
    return x.float(), x[:, :, list(pl.sensors)].float(), u.float()


@pytest.mark.parametrize("fam", ["prbs", "square", "staircase"])
def test_model_based_kkl_exact_maps_converge(fam):
    """kkl_mb = autonomous KKL + J_T0 (f(x,u) - f(x,0)). With exact maps the error tends
    to 0 (second order in dt for inputs piecewise constant on the grid); the same observer
    WITHOUT the correction does not converge, so the injection term is what makes it work."""
    pl = Lin2()
    late = []
    for dt in (0.02, 0.01):
        des = _design(dt, int(round(30.0 / dt)))
        P, T0, Ti = _exact_maps(pl, des)
        x, y, u = _forced(pl, des, fam)
        x0h = torch.zeros(1, 2)
        lat = LatentStep(des, "cpu")
        e = (run(ModelBasedKKL(lat, T0, Ti, pl), y, u, x0h) - x)[0].norm(dim=-1)
        ea = (run(KKLObserver("auto", lat, Ti, z0_map=T0), y, u, x0h) - x)[0].norm(dim=-1)
        k2 = des.N // 2
        xs = float(x[0, k2:].norm(dim=-1).max())
        assert float(e[0]) > 0.5                              # starts far
        assert float(e[k2:].max()) < 1e-3 * xs, (fam, dt, float(e[k2:].max()), xs)
        assert float(ea[k2:].max()) > 0.3 * xs                # no correction: no convergence
        late.append(float(e[k2:].max()))
    assert np.log2(late[0] / late[1]) > 1.6, late


def test_model_based_kkl_is_second_order_on_smooth_inputs():
    """The plant receives its input through a zero-order hold, which kkl_mb holds too, so
    no input delay remains: the steady error is the latent step's O(dt^2). (Before the
    plant input was ZOH, kkl_mb had an O(dt) half-step input delay here.)"""
    pl = Lin2()
    late = []
    for dt in (0.02, 0.01):
        des = _design(dt, int(round(30.0 / dt)))
        P, T0, Ti = _exact_maps(pl, des)
        x, y, u = _forced(pl, des, "sinusoid")
        e = (run(ModelBasedKKL(LatentStep(des, "cpu"), T0, Ti, pl), y, u, torch.zeros(1, 2))
             - x)[0, des.N // 2:].norm(dim=-1)
        late.append(float(e.max()))
    order = np.log2(late[0] / late[1])
    assert 1.7 < order < 2.4, (late, order)


def test_autonomous_kkl_exact_maps_meets_the_contraction_bound():
    """Zero input, exact maps: |xhat - x|(t) <= ||P^+|| e^{-alpha t} |P (x0_hat - x0)| + disc."""
    pl = Lin2()
    des = _design(0.02, 1000, alpha=1.0)
    P, T0, Ti = _exact_maps(pl, des)
    x, y, u = _forced(pl, des, "zero", regime="train", x0=(1.5, -1.0))
    x0h = torch.tensor([[-0.5, 0.8]])
    xh = run(KKLObserver("auto", LatentStep(des, "cpu"), Ti, z0_map=T0), y, u, x0h)
    e = (xh - x)[0].norm(dim=-1).double().numpy()
    bound = np.linalg.norm(np.linalg.pinv(P), 2) * np.exp(-des.alpha * des.t) * \
        np.linalg.norm(P @ (x0h[0].numpy() - x[0, 0].numpy()))
    assert (e <= bound + 2e-4).all(), float((e - bound).max())
    assert e[-1] < 1e-4


# ---------------------------------------------------------------------------
# 6. Context timing of the learned family
# ---------------------------------------------------------------------------

class _CumCtx(ContextEncoder):
    """c_k = sum_{j <= k} u_j."""
    dim = 1

    def seq(self, u):
        return u.cumsum(1)

    def init_state(self, B, device):
        return torch.zeros(B, 1, device=device)

    def step(self, u_k, s):
        s = s + u_k
        return s.clone(), s


class _RetC(torch.nn.Module):
    def forward(self, z, c):
        return c


def test_context_timing_decoder_and_injection(contract_data):
    des, pl, y, u, _ = contract_data
    lat = LatentStep(des, "cpu")
    nz = des.A.shape[0]
    cum = u.cumsum(1)
    x0 = torch.zeros(y.shape[0], nz)
    # estimate at k uses c_k = E(u_0..u_k)
    xh = run(KKLObserver("t", lat, _RetC(), ctx=_CumCtx(), z0_map=lambda x: x), y, u, x0)
    assert torch.allclose(xh, cum, atol=1e-5)
    # the injection over [t_k, t_{k+1}) uses c_k (held), not c_{k+1}
    inj = lambda z, c: c.expand(-1, nz) * torch.ones_like(z)
    zh = run(KKLObserver("t", lat, _Lin(np.eye(nz)), ctx=_CumCtx(), injection=inj,
                         z0_map=lambda x: x), y, u, x0)
    z = x0
    for k in range(y.shape[1] - 1):
        z = z @ lat.Ad.T + y[:, k] @ lat.W0.T + y[:, k + 1] @ lat.W1.T + cum[:, k].expand(-1, nz) @ lat.Wg.T
        assert torch.allclose(zh[:, k + 1], z, atol=1e-5), k


def test_deployment_equals_training_computation(contract_data):
    """run() reproduces what phase 2 trains on: decoder variants T+(z_obs_k; seq(u)_k);
    the latent variant the rollout with inj(., c_k) held over each step."""
    des, pl, y, u, x0 = contract_data
    obs = _all_observers(des, pl)
    lat = LatentStep(des, "cpu")
    ob = obs["kkl_film_sfm"]
    xh = run(ob, y, u, x0)
    zo = run_latent(des, y, z0=ob.z0_map(x0))
    c = ob.ctx.seq(u)
    with torch.no_grad():
        ref = torch.stack([ob.dec(zo[:, k], c[:, k]) for k in range(y.shape[1])], 1)
    assert torch.allclose(xh, ref, atol=1e-5)
    ob = obs["kkl_latent_gru"]
    xh = run(ob, y, u, x0)
    c = ob.ctx.seq(u)
    with torch.no_grad():
        z, zs = ob.z0_map(x0), [ob.z0_map(x0)]
        for k in range(y.shape[1] - 1):
            ck = c[:, k]
            z = lat(z, y[:, k], y[:, k + 1], lambda zz: ob.inj(zz, ck))
            zs.append(z)
        ref = ob.dec(torch.stack(zs, 1).reshape(-1, z.shape[-1]), None).view(xh.shape)
    assert torch.allclose(xh, ref, atol=1e-4)


# ---------------------------------------------------------------------------
# 7. NaN handling
# ---------------------------------------------------------------------------

def test_non_finite_estimates_are_reported_as_diverged():
    """A non-finite filter state must surface as a divergence in the metrics, never as a
    silent restart from the origin."""
    pl, dt = Lin2(), 0.05
    x, y, u = _simulate_linear(pl, dt, 80, np.array([0.05, 0.1]), np.array([0.05]), 0, 1.0)
    y = T(y, torch.float32)[None]
    u = T(u, torch.float32)[None]
    xt = T(x, torch.float32)[None]
    y[:, 40] = float("nan")
    for cls in (EKF, UKF):
        xh = run(cls(pl, dt, [0.05, 0.1], [0.05], [1.0, 1.0]), y, u, torch.zeros(1, 2))
        assert torch.isfinite(xh).all()                    # run() maps non-finite to +-1e6
        assert episode_metrics(xt, xh, [1.0, 1.0], 20)["diverged"].all(), cls.name


# ---------------------------------------------------------------------------
# 8. Real evaluation data: noise statistics of the baselines and brief runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lorenz():
    import hyperkkl.config as C
    import hyperkkl.data as Dm
    from hyperkkl.systems import build_plant
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = C.load("lorenz", [{"eval.n_episodes": 4}])
    scales = Dm.system_scales(cfg, dev)
    des = Dm.make_design(cfg["system"])
    es = Dm.eval_data(cfg, "shape_ood", device=dev)
    return cfg, scales, des, es, build_plant(cfg["system"])


def test_baseline_noise_statistics_are_the_data_statistics(lorenz):
    """R = (sensor.noise * y_std)^2, Q = (noise_proc * x_std)^2 dt, P0 = span^2 / 12 are the
    statistics the evaluation data is generated with."""
    from hyperkkl.data import _sobol
    from hyperkkl.observers import _rk4
    cfg, scales, des, es, model = lorenz
    q = float(cfg["data"]["noise_proc"]) * np.asarray(scales["x_std"])
    r = float(cfg["data"]["sensor"]["noise"]) * np.asarray(scales["y_std"])
    meas = (es.y - es.y_clean).double()
    ratio_r = (meas.std((0, 1)).numpy() / r)
    assert np.all(np.abs(ratio_r - 1) < 0.03), ratio_r
    # process noise: residual of the nominal RK4 step (inputs interpolated as the filters
    # do) on steps where the input is smooth, i.e. away from piecewise-constant switches and
    # the sawtooth's one-step flyback
    x, u = es.x.double(), es.u.double()
    um = 0.5 * (u[:, :-1] + u[:, 1:])
    pred = _rk4(model, x[:, :-1].reshape(-1, 3), u[:, :-1].reshape(-1, 1), um.reshape(-1, 1),
                u[:, 1:].reshape(-1, 1), des.dt).view(x[:, 1:].shape)
    res = x[:, 1:] - pred
    smooth = (u[:, 1:] == u[:, :-1])[..., 0] | torch.tensor(
        [f not in S.PIECEWISE_CONSTANT | {"sawtooth"} for f in es.families])[:, None]
    ratio_q = res[smooth].std(0).numpy() / (q * np.sqrt(des.dt))
    assert np.all(np.abs(ratio_q - 1) < 0.05), ratio_q
    # prior: the evaluation's x0 are Sobol-uniform on the IC box, x0_hat its centre
    box = np.asarray(cfg["system"]["ic_box"], float)
    X0 = _sobol(box, 4096, 7)
    assert np.allclose(X0.mean(0), box.mean(1), atol=0.02 * (box[:, 1] - box[:, 0]))
    assert np.allclose(X0.std(0), (box[:, 1] - box[:, 0]) / np.sqrt(12), rtol=0.02)


def test_filters_on_lorenz_briefly(lorenz):
    """EKF / UKF / PF on the first 600 steps of real shape_ood episodes (float32, as deployed):
    finite, symmetric PSD covariance, and the KF-type filters lock on."""
    cfg, scales, des, es, model = lorenz
    from hyperkkl.evaluate import initial_guess
    q = float(cfg["data"]["noise_proc"]) * np.asarray(scales["x_std"])
    r = float(cfg["data"]["sensor"]["noise"]) * np.asarray(scales["y_std"])
    box = np.asarray(cfg["system"]["ic_box"], float)
    p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
    K = 600
    X, Y, U = es.x[:, :K + 1], es.y[:, :K + 1], es.u[:, :K + 1]
    x0 = initial_guess(cfg, len(es), "center", 0, "cpu")
    sig = np.asarray(scales["x_std"])
    for cls in (EKF, UKF):
        ob = cls(model, des.dt, q, r, p0)
        xs, Ps = _steps(ob, Y, U, x0)
        assert torch.isfinite(xs).all() and torch.isfinite(Ps).all()
        Pn = Ps.double()
        assert (Pn - Pn.transpose(-1, -2)).abs().max() < 1e-4 * Pn.abs().max()
        assert torch.linalg.eigvalsh(0.5 * (Pn + Pn.transpose(-1, -2))).min() > 0
        err = ((xs[:, K // 2:] - X[:, K // 2:]) / torch.as_tensor(sig, dtype=X.dtype)).pow(2).mean().sqrt()
        assert err < 0.05, (cls.name, float(err))
    pf = ParticleFilter(model, des.dt, q, r, p0, n_particles=512)
    xh = run(pf, Y, U, x0)
    assert torch.isfinite(xh).all() and (xh.abs() < 1e5).all()
