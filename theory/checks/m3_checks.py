"""Numerical checks for theory/m3_error_bounds.tex (block M3), CPU only, ~1 min.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python theory/checks/m3_checks.py

Linear plant with exact KKL maps (hyperkkl.design.sylvester_reference), so every term of
Proposition 3 (decoder-side ISS bound) is known exactly:
    xdot = F x + G u (ZOH),  y = C x + v,   z = P x + nu,  nu' = A nu - P G u,
    exact conditioned decoder  T+(z; c) = P^+ (z - c),  context c_k = nu_k.
Each check prints measured vs bound and asserts the inequality.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm, solve_discrete_are

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperkkl.design import butterworth_AB, choose_dt, latent_step, spectral_bounds, sylvester_reference  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(0)
nrm = lambda M: float(np.linalg.norm(M, 2))


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


# ---------------------------------------------------------------------------
# Common linear setting
# ---------------------------------------------------------------------------
F = np.array([[0.0, 1.0], [-1.0, -0.4]])
G = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
NZ, ALPHA = 5, 1.5
A, B = butterworth_AB(NZ, ALPHA, 1)
DT = choose_dt(A)
KAPPA, _ = spectral_bounds(A)
ST = latent_step(A, B, DT)
Ad, W0, W1, Wg = ST["Ad"], ST["W0"], ST["W1"], ST["Wg"]
P = sylvester_reference(F, C, A, B)
Pp = np.linalg.pinv(P)
Fd = expm(F * DT)
Gd = np.linalg.solve(F, Fd - np.eye(2)) @ G
NUd = -Wg @ P @ G                     # nu_{k+1} = Ad nu_k + NUd u_k  (exact under ZOH)


def impulse(m_max):
    """Impulse response v -> zhat error of the exact latent step: H_0 = W1, H_m = Ad^{m-1}(W0 + Ad W1)."""
    H = [W1]
    M = W0 + Ad @ W1
    for _ in range(1, m_max):
        H.append(M)
        M = Ad @ M
    return np.array(H)                 # (m_max, nz, ny)


H = impulse(int(40 / (ALPHA * DT)))
GAM_Y = float(sum(nrm(h) for h in H))                                  # exact l1 gain
GAM_CT = KAPPA * nrm(B) / ALPHA                                        # continuous limit bound
L_Z = nrm(Pp)                                                          # Lipschitz of P^+ (z - c) in z
L_C = nrm(Pp)                                                          # ... and in c
# context l1 gain: c_k = nu_k = sum_m Ad^m NUd u_{k-1-m}
L_E = float(sum(nrm(np.linalg.matrix_power(Ad, m) @ NUd) for m in range(len(H))))


def simulate(u, x0, v=None, z0=None, u_obs=None):
    """Plant + deployed decoder-side observer. Returns x, z_ideal, zhat, xhat, c_true, c_obs."""
    N = len(u) - 1
    x = np.zeros((N + 1, 2)); x[0] = x0
    nu = np.zeros((N + 1, NZ)); nuo = np.zeros((N + 1, NZ))
    uo = u if u_obs is None else u_obs
    for k in range(N):
        x[k + 1] = Fd @ x[k] + Gd[:, 0] * u[k]
        nu[k + 1] = Ad @ nu[k] + NUd[:, 0] * u[k]
        nuo[k + 1] = Ad @ nuo[k] + NUd[:, 0] * uo[k]
    z = x @ P.T + nu                                       # ideal latent T(x_k, t_k; u)
    y = x @ C.T + (0 if v is None else v)
    zh = np.zeros((N + 1, NZ)); zh[0] = z[0] if z0 is None else z0
    for k in range(N):
        zh[k + 1] = Ad @ zh[k] + W0 @ y[k] + W1 @ y[k + 1]
    xh = (zh - nuo) @ Pp.T
    return x, z, zh, xh, nu, nuo


def prbs(N, amp=1.0, dwell=150, seed=1):
    r = np.random.default_rng(seed)
    return np.repeat(amp * r.choice([-1.0, 1.0], size=N // dwell + 2), dwell)[:N + 1]


def m2_bound(x, u):
    """sup ||y''|| inside the ZOH intervals: y'' = C F (F x(t) + G u_k), sampled finely."""
    s = np.linspace(0, DT, 9)
    E = [expm(F * si) for si in s]
    best = 0.0
    for k in range(len(u) - 1):
        xd = F @ x[k] + G[:, 0] * u[k]
        best = max(best, max(abs(float((C @ F @ Ei @ xd)[0])) for Ei in E))
    return best * 1.02


# ---------------------------------------------------------------------------
def wfun(alpha, dt):
    r0 = math.exp(-alpha * dt)
    return (1 - r0 * (1 + alpha * dt)) / (alpha ** 2 * dt), (alpha * dt - 1 + r0) / (alpha ** 2 * dt)


def gamma_of(Am, Bm, dt, horizon=60.0):
    st = latent_step(Am, Bm, dt, n_quad=24)
    kap, al = spectral_bounds(Am)
    n = int(horizon / (al * dt)) + 2
    g, M = nrm(st["W1"]), st["W0"] + st["Ad"] @ st["W1"]
    for _ in range(n):
        g += nrm(M); M = st["Ad"] @ M
    return g, kap, al, st


def check_design_constants():
    header("Lemma D (c): merged kernels -- gamma_y <= kappa|B|/alpha at every dt")
    print(f"  dt = {DT:.5f}, alpha dt = {ALPHA * DT:.4f}, kappa = {KAPPA:.6f}, ||B|| = {nrm(B):.4f}")
    ct = float(np.trapezoid([nrm(expm(A * s) @ B) for s in np.linspace(0, 40 / ALPHA, 20001)],
                            np.linspace(0, 40 / ALPHA, 20001)))
    print(f"  gamma_y exact l1 = {GAM_Y:.4f}  <= kappa|B|/alpha = {GAM_CT:.4f};  int ||e^As B|| = {ct:.4f}")
    print(f"  l_z = l_c = ||P^+|| = {L_Z:.4f},  l_E (u -> nu, l1) = {L_E:.4f}")
    assert GAM_Y <= GAM_CT and abs(GAM_Y - ct) / ct < 0.05
    # per-kernel merged bounds and gamma_y <= kappa|B|/alpha across alpha dt, normal and non-normal A
    T = np.eye(NZ) + 0.6 * np.triu(RNG.standard_normal((NZ, NZ)), 1)
    An = T @ A @ np.linalg.inv(T)
    for name, Am in (("Butterworth (normal)", A), ("non-normal, same spectrum", An)):
        for adt in (0.03, 0.3, 1.0, 3.0):
            dt = adt / ALPHA
            g, kap, al, st = gamma_of(Am, B, dt)
            w0, w1 = wfun(al, dt)
            ok = all(nrm(np.linalg.matrix_power(st["Ad"], m) @ st["W0"]) <= kap * nrm(B) * math.exp(-al * dt) ** m * w0 * (1 + 1e-9)
                     and nrm(np.linalg.matrix_power(st["Ad"], m) @ st["W1"]) <= kap * nrm(B) * math.exp(-al * dt) ** m * w1 * (1 + 1e-9)
                     for m in range(0, 40))
            print(f"  {name:26s} alpha dt={adt:4.2f}: kappa={kap:.3f}  gamma_y={g:.4f} <= kappa|B|/alpha="
                  f"{kap * nrm(B) / al:.4f}   merged kernels hold: {ok}")
            assert g <= kap * nrm(B) / al * (1 + 1e-6) and ok


def check_noise_iss():
    header("Proposition m3-iss (i),(iv): latent deviation, worst case and white noise")
    N = 6000
    u = prbs(N)
    x, z, *_ = simulate(u, np.array([1.0, 0.0]))
    delta = 0.05
    # adversarial bounded noise: v_{k-m} = delta sign<w, H_m>, w the best direction found
    best = 0.0
    for _ in range(50):
        w = RNG.standard_normal(NZ); w /= np.linalg.norm(w)
        for _ in range(30):                                 # alternating maximisation
            s = np.sign(H[:, :, 0] @ w)
            e = delta * (s[:, None] * H[:, :, 0]).sum(0)
            w = e / np.linalg.norm(e)
        best = max(best, np.linalg.norm(e))
    print(f"  worst-case latent error found {best:.5f} <= gamma_y delta = {GAM_Y * delta:.5f} <= kappa|B|delta/alpha = {GAM_CT * delta:.5f}"
          f"  (ratio {best / (GAM_Y * delta):.3f})")
    # full observer with that adversarial pattern repeated, plus state error vs l_z bound
    v = delta * np.sign(RNG.standard_normal((N + 1, 1)))
    _, z, zh, xh, *_ = simulate(u, np.array([1.0, 0.0]), v=v)
    _, _, zh0, _, *_ = simulate(u, np.array([1.0, 0.0]))
    dz = np.linalg.norm(zh - zh0, axis=1).max()
    ex = np.linalg.norm(xh - x, axis=1)[N // 2:].max()
    ex0 = np.linalg.norm((zh0 - simulate(u, np.array([1.0, 0.0]))[4]) @ Pp.T - x, axis=1)[N // 2:].max()
    bnd = ex0 + L_Z * GAM_Y * delta
    print(f"  random-sign noise: max |dzhat| {dz:.5f} <= {GAM_Y * delta:.5f};"
          f"  steady max |xhat - x| {ex:.5f} <= e_clean + l_z gamma_y delta = {bnd:.5f}")
    assert best <= GAM_Y * delta * (1 + 1e-9) and dz <= GAM_Y * delta and ex <= bnd
    # white Gaussian: RMS latent error = sigma sqrt(sum ||H_m||_F^2)
    sig = 0.03
    pred = sig * math.sqrt(float((H ** 2).sum()))
    vv = sig * RNG.standard_normal((40000, 1))
    e = np.zeros(NZ); acc = []
    for k in range(len(vv) - 1):
        e = Ad @ e + W0 @ vv[k] + W1 @ vv[k + 1]
        if k > 2000:
            acc.append(e @ e)
    rms = math.sqrt(np.mean(acc))
    print(f"  white noise sigma={sig}: RMS latent error MC {rms:.5f} vs H2 formula {pred:.5f}"
          f"  (worst-case gamma_y*4sigma = {GAM_Y * 4 * sig:.4f}: {GAM_Y * 4 * sig / pred:.0f}x looser)")
    assert abs(rms - pred) / pred < 0.05


def check_interpolation():
    header("Lemma D (b): interpolation error, explicit constant, order 2 (ZOH, PRBS input)")
    rows = []
    global DT, Ad, W0, W1, Wg, Fd, Gd, NUd
    saved = (DT, Ad, W0, W1, Wg, Fd, Gd, NUd)
    for fac in (4, 2, 1, 0.5):
        DT = saved[0] * fac
        st = latent_step(A, B, DT, n_quad=24)
        Ad, W0, W1, Wg = st["Ad"], st["W0"], st["W1"], st["Wg"]
        Fd = expm(F * DT); Gd = np.linalg.solve(F, Fd - np.eye(2)) @ G; NUd = -Wg @ P @ G
        N = int(round(30 / DT))
        dwell = max(1, int(round(1.5 / DT)))                    # switches on the grid
        u = prbs(N, dwell=dwell)
        x, z, zh, *_ = simulate(u, np.array([1.0, 0.5]))
        err = np.linalg.norm(zh - z, axis=1).max()
        M2 = m2_bound(x, u)
        bnd = KAPPA * nrm(B) * M2 * DT ** 3 / (12 * (1 - math.exp(-ALPHA * DT)))
        rows.append((DT, err, bnd))
        print(f"  dt={DT:.5f}: sup |zhat - z| = {err:.3e}  bound e_disc = {bnd:.3e}  (bound/err {bnd / err:.2f})")
    DT, Ad, W0, W1, Wg, Fd, Gd, NUd = saved
    o = [math.log(rows[i][1] / rows[i + 1][1], 2) for i in range(len(rows) - 1)]
    print(f"  observed orders {np.round(o, 3)}")
    assert all(r[2] >= r[1] for r in rows) and min(o) > 1.9


def check_input_corruption():
    header("Proposition m3-iss (iii): input-side term l_c l_E delta_u (bias, delay, noise)")
    N = 8000
    u = prbs(N, dwell=200)
    x0 = np.array([1.0, 0.0])
    x, z, zh, xh_clean, *_ = simulate(u, x0)
    for name, uo in (("bias 0.1", u + 0.1), ("delay 5", np.concatenate([np.full(5, u[0]), u[:-5]])),
                     ("noise 0.1 (bounded)", u + 0.1 * np.clip(RNG.standard_normal(N + 1), -3, 3))):
        _, _, _, xh, nu, nuo = simulate(u, x0, u_obs=uo)
        du = np.abs(uo - u).max()
        dc = np.linalg.norm(nuo - nu, axis=1).max()
        ex = np.linalg.norm(xh - x, axis=1)[1000:].max()
        e0 = np.linalg.norm(xh_clean - x, axis=1)[1000:].max()
        print(f"  {name:20s}: delta_u {du:.3f}  |dc| {dc:.4f} <= l_E delta_u {L_E * du:.4f};"
              f"  max err {ex:.4f} <= {e0 + L_C * L_E * du:.4f}")
        assert dc <= L_E * du + 1e-12 and ex <= e0 + L_C * L_E * du
    # delay: sup-norm delta_u = jump size J, but the error is a pulse of duration d dt per switch
    print("  (delay: delta_u = jump size 2; error confined to d samples after each switch, see Cor. m3-faults)")


def check_outliers():
    header("Corollary m3-faults: outliers -- merged-kernel envelope and outlier-process mean")
    M, k0, N = 12.0, 50, 2000
    v = np.zeros((N + 1, 1)); v[k0] = M
    e = np.zeros(NZ); traj = []
    for k in range(N):
        e = Ad @ e + W0 @ v[k] + W1 @ v[k + 1]
        traj.append((k + 1, np.linalg.norm(e)))
    ks = np.array([t[0] for t in traj if t[0] >= k0]); es = np.array([t[1] for t in traj if t[0] >= k0])
    w0, w1 = wfun(ALPHA, DT)
    env = KAPPA * nrm(B) * M * np.where(ks == k0, w1, (w1 + math.exp(ALPHA * DT) * w0)
                                        * np.exp(-ALPHA * (ks - k0) * DT))
    print(f"  peak |dzhat| {es.max():.4f} <= kappa|B| M (w1 + e^(a dt) w0) = {env.max():.4f};"
          f"  envelope violated at {int((es > env + 1e-12).sum())} of {len(ks)} samples")
    integ = DT * es.sum()
    print(f"  time integral {integ:.5f} <= kappa|B| M dt/alpha = {KAPPA * nrm(B) * M * DT / ALPHA:.5f}")
    assert np.all(es <= env + 1e-12) and integ <= KAPPA * nrm(B) * M * DT / ALPHA
    # outlier process: p per sample, size M, random sign
    p = 0.02
    NN = 200000
    m = RNG.random(NN) < p
    vv = (m * M * np.where(RNG.random(NN) < 0.5, -1, 1))[:, None]
    e = np.zeros(NZ); a1, a2 = [], []
    for k in range(NN - 1):
        e = Ad @ e + W0 @ vv[k] + W1 @ vv[k + 1]
        if k > 3000:
            n_ = np.linalg.norm(e); a1.append(n_); a2.append(n_ * n_)
    mean, rms = np.mean(a1), math.sqrt(np.mean(a2))
    print(f"  process p={p}, M={M}: E|dzhat| {mean:.4f} <= gamma_y p M = {GAM_Y * p * M:.4f};"
          f"  RMS {rms:.4f} vs sqrt(p) M ||H||_2 = {math.sqrt(p) * M * math.sqrt(float((H ** 2).sum())):.4f}")
    assert mean <= GAM_Y * p * M and abs(rms / (math.sqrt(p) * M * math.sqrt(float((H ** 2).sum()))) - 1) < 0.05


def check_dropout_drift():
    header("Corollary m3-faults: dropout (sample-and-hold) and drift")
    N = 20000
    u = prbs(N, dwell=150, seed=3)
    x0 = np.array([1.0, 0.0])
    x, z, zh0, *_ = simulate(u, x0)
    y = x @ C.T
    Ly = np.abs(np.diff(y[:, 0])).max() / DT                 # sup |y'| (sampled)
    for p in (0.1, 0.3):
        hold = RNG.random(N + 1) < p; hold[0] = False
        yd = y.copy(); run = np.zeros(N + 1)
        for k in range(1, N + 1):
            if hold[k]:
                yd[k] = yd[k - 1]; run[k] = run[k - 1] + 1
        v = yd - y
        assert np.all(np.abs(v[:, 0]) <= Ly * DT * run + 1e-12)
        _, _, zh, *_ = simulate(u, x0, v=v)
        dz = np.linalg.norm(zh - zh0, axis=1)[1000:]
        print(f"  dropout p={p}: mean |dzhat| {dz.mean():.5f} <= gamma_y L_y dt p/(1-p) = "
              f"{GAM_Y * Ly * DT * p / (1 - p):.5f};  max {dz.max():.5f} <= gamma_y L_y dt max_run = "
              f"{GAM_Y * Ly * DT * run.max():.5f}")
        assert dz.max() <= GAM_Y * Ly * DT * run.max() + 1e-12
        assert dz.mean() <= GAM_Y * Ly * DT * p / (1 - p) * 1.05   # expectation bound, MC slack 5%
    # constant bias: exact DC gain -A^{-1} B (discrete filter reproduces it exactly)
    dc = np.linalg.solve(np.eye(NZ) - Ad, W0 + W1)
    print(f"  DC gain: (I-Ad)^-1 (W0+W1) = -A^-1 B up to {np.abs(dc + np.linalg.solve(A, B)).max():.1e};"
          f"  ||A^-1 B|| = {nrm(np.linalg.solve(A, B)):.4f} vs gamma_y = {GAM_Y:.4f}, kappa|B|/alpha = {GAM_CT:.4f}")
    assert np.abs(dc + np.linalg.solve(A, B)).max() < 1e-10
    # slow random-walk drift (data.measure): quasi-static prediction -A^{-1} B b_k
    b = np.cumsum(RNG.standard_normal(N + 1)); b = 0.3 * b / np.abs(b).max()
    _, _, zh, *_ = simulate(u, x0, v=b[:, None])
    dz = zh - zh0
    qs = -(np.linalg.solve(A, B) @ b[None])[:, :].T
    S = np.cumsum(H[::-1], 0)[::-1]                          # tail sums S_m
    corr = float(sum(nrm(S[m]) for m in range(1, len(S)))) * np.abs(np.diff(b)).max()
    dev = np.linalg.norm(dz - qs, axis=1)[1000:].max()
    print(f"  drift sup|b|=0.3: max |dzhat| {np.linalg.norm(dz, axis=1).max():.4f} <= gamma_y sup|b| ="
          f" {GAM_Y * 0.3:.4f};  quasi-static deviation {dev:.4f} <= sum_m>=1 |S_m| max|db| = {corr:.4f}")
    assert np.linalg.norm(dz, axis=1).max() <= GAM_Y * 0.3 and dev <= corr


def check_midpoint():
    header("Lemma D (d): exponential-midpoint step with injection -- step-size / small-gain condition")
    nA = nrm(A)
    st = latent_step(A, B, DT)

    def step_matrix(Lm, dt):
        s = latent_step(A, B, dt)
        return s["Ad"] + s["Wg"] @ Lm @ (np.eye(NZ) + 0.5 * dt * (A + Lm))

    for l in (0.5, 1.0, 1.3):
        Lm = l * np.eye(NZ)                                     # Phi(z) = l z: l_Phi = l
        beta = 1 + 0.5 * DT * (nA + l)
        rho = math.exp(-ALPHA * DT) + KAPPA * (1 - math.exp(-ALPHA * DT)) * l * beta / ALPHA
        Mk = step_matrix(Lm, DT)
        g = max(nrm(np.linalg.matrix_power(Mk, k)) ** (1 / k) for k in (1, 10, 100))
        print(f"  l_Phi={l:.2f}: kappa l beta = {KAPPA * l * beta:.4f} (< alpha={ALPHA}),"
              f" measured per-step gain {g:.6f} <= rho = {rho:.6f}")
        assert KAPPA * l * beta < ALPHA and g <= rho + 1e-12
    # random non-symmetric Lipschitz injections
    worst = 0.0
    for _ in range(200):
        Lm = RNG.standard_normal((NZ, NZ)); Lm *= RNG.uniform(0.1, 1.3) / nrm(Lm)
        l = nrm(Lm); beta = 1 + 0.5 * DT * (nA + l)
        rho = math.exp(-ALPHA * DT) + KAPPA * (1 - math.exp(-ALPHA * DT)) * l * beta / ALPHA
        worst = max(worst, nrm(step_matrix(Lm, DT)) / rho)
    print(f"  200 random linear injections with kappa l beta < alpha: max ||step|| / rho = {worst:.4f} (<= 1)")
    assert worst <= 1 + 1e-12
    # non-normal A (kappa ~ 3): incremental bound ||Delta_k|| <= kappa rho^k ||Delta_0|| with ONE kappa
    T = np.eye(NZ) + 0.6 * np.triu(RNG.standard_normal((NZ, NZ)), 1)
    An = T @ A @ np.linalg.inv(T)
    kap_n, al_n = spectral_bounds(An)
    sN = latent_step(An, B, DT)
    worst = 0.0
    for _ in range(50):
        Lm = RNG.standard_normal((NZ, NZ)); Lm *= RNG.uniform(0.05, 0.4) / nrm(Lm)
        l = nrm(Lm); beta = 1 + 0.5 * DT * (nrm(An) + l)
        rho = math.exp(-al_n * DT) + kap_n * (1 - math.exp(-al_n * DT)) * l * beta / al_n
        if kap_n * l * beta >= al_n:
            continue
        Mk = sN["Ad"] + sN["Wg"] @ Lm @ (np.eye(NZ) + 0.5 * DT * (An + Lm))
        P_ = np.eye(NZ)
        for k in range(1, 400):
            P_ = Mk @ P_
            worst = max(worst, nrm(P_) / (kap_n * rho ** k))
    print(f"  non-normal A (kappa = {kap_n:.2f}): max_k ||step^k|| / (kappa rho^k) = {worst:.4f} (<= 1, single kappa)")
    assert worst <= 1 + 1e-9
    # the step-size condition matters: a continuous-time-stable injection Phi = -l z
    # (A - l I is MORE stable) destabilises the explicit midpoint predictor at large l dt
    for l in (5.0, 50.0, 400.0):
        sr = max(abs(np.linalg.eigvals(step_matrix(-l * np.eye(NZ), DT))))
        print(f"  Phi = -{l:g} z (continuous: stable, abscissa {-ALPHA - l:.1f}); l dt = {l * DT:.2f}:"
              f" discrete spectral radius {sr:.4f}")
    assert max(abs(np.linalg.eigvals(step_matrix(-400.0 * np.eye(NZ), DT)))) > 1
    # continuous threshold kappa l_Phi < alpha is sharp (Phi = l z): A + l I has abscissa l - alpha
    print(f"  threshold sharp: Phi = l z with l = alpha(1+1e-3) gives continuous abscissa "
          f"{max(np.linalg.eigvals(A + ALPHA * 1.001 * np.eye(NZ)).real):+.4f} > 0")
    # cross-check the numpy midpoint step against hyperkkl.observers.LatentStep (float32)
    try:
        import torch
        from hyperkkl.observers import LatentStep

        class _D:
            pass
        d = _D(); d.A, d.B, d.dt, d.step = A, B, DT, st
        ls = LatentStep(d, "cpu")
        Lm = 0.7 * np.eye(NZ)
        z = RNG.standard_normal(NZ); yk, yk1 = 0.3, -0.2
        ref = st["Ad"] @ z + st["W0"][:, 0] * yk + st["W1"][:, 0] * yk1 + \
            st["Wg"] @ (Lm @ (z + 0.5 * DT * (A @ z + B[:, 0] * yk + Lm @ z)))
        out = ls(torch.tensor(z[None], dtype=torch.float32), torch.tensor([[yk]], dtype=torch.float32),
                 torch.tensor([[yk1]], dtype=torch.float32),
                 lambda zz: zz @ torch.tensor(Lm.T, dtype=torch.float32))[0].numpy()
        print(f"  numpy midpoint step == hyperkkl LatentStep (float32): max diff {np.abs(out - ref).max():.1e}")
        assert np.abs(out - ref).max() < 1e-5
    except ImportError:
        print("  (torch not available: LatentStep cross-check skipped)")


def check_kf_comparison():
    header("Remark m3-kf: KF vs KKL on a linear plant (both LTI filters of y)")
    q, sig, p, M = 0.05, 0.03, 0.02, 12.0
    N = 300000
    Qd = q ** 2 * DT * np.eye(2)

    def kf_gain(R):
        Pk = solve_discrete_are(Fd.T, C.T, Qd, np.array([[R]]))
        return Pk @ C.T / float((C @ Pk @ C.T)[0, 0] + R)

    w = RNG.standard_normal((N, 2)) * q * math.sqrt(DT)
    out = RNG.random(N) < p
    v_g = sig * RNG.standard_normal(N)
    v_o = v_g + out * M * np.where(RNG.random(N) < 0.5, -1, 1)
    x = np.zeros((N, 2))
    for k in range(N - 1):
        x[k + 1] = Fd @ x[k] + w[k]                      # u = 0: the KF/KKL difference is the gain
    res = {}
    for vname, v in (("gaussian", v_g), ("outliers", v_o)):
        y = x[:, 0] + v
        for fname, R in (("KF R=sig^2", sig ** 2), ("KF R=sig^2+pM^2", sig ** 2 + p * M * M)):
            K = kf_gain(R)[:, 0]
            xh = np.zeros(2); err = []
            for k in range(1, N):
                xp = Fd @ xh
                xh = xp + K * (y[k] - xp[0])
                if k > 5000:
                    err.append(np.sum((xh - x[k]) ** 2))
            res[(vname, fname)] = np.mean(err)
        zh = P @ x[0]; err = []
        for k in range(N - 1):
            zh = Ad @ zh + W0[:, 0] * y[k] + W1[:, 0] * y[k + 1]
            if k > 5000:
                err.append(np.sum((Pp @ zh - x[k + 1]) ** 2))
        res[(vname, "KKL")] = np.mean(err)
    for vname in ("gaussian", "outliers"):
        print(f"  {vname:9s}: " + "  ".join(f"{f} MSE {res[(vname, f)]:.3e}"
                                            for f in ("KF R=sig^2", "KF R=sig^2+pM^2", "KKL")))
    # linear prediction: MSE(outliers) = MSE(gaussian) + p M^2 ||Gamma||_H2^2, Gamma the v -> error
    # impulse response of each filter (KF: Phi^m K, Phi = (I - K C) Fd;  KKL: P^+ H_m)
    gam = {}
    for fname, R in (("KF R=sig^2", sig ** 2), ("KF R=sig^2+pM^2", sig ** 2 + p * M * M)):
        K = kf_gain(R); Phi = (np.eye(2) - K @ C) @ Fd; Mm = K.copy(); h2 = 0.0
        for _ in range(20000):
            h2 += float((Mm ** 2).sum()); Mm = Phi @ Mm
        gam[fname] = h2
    gam["KKL"] = float(((Pp @ H) ** 2).sum())
    for f in ("KF R=sig^2", "KF R=sig^2+pM^2", "KKL"):
        pred = res[("gaussian", f)] + p * M * M * gam[f]
        print(f"  {f:16s}: ||Gamma||_H2^2 = {gam[f]:.3e};  outlier MSE MC {res[('outliers', f)]:.3e}"
              f" vs prediction {pred:.3e}")
        assert abs(res[("outliers", f)] / pred - 1) < 0.1
    K0 = kf_gain(sig ** 2)[:, 0]
    print(f"  per-outlier peak state jump: KF |K| M = {np.linalg.norm(K0) * M:.3f}  vs  KKL |P^+ W1| M = "
          f"{nrm(Pp @ W1) * M:.4f}")
    print("  -> the ranking under outliers is decided by ||Gamma||_H2 alone; here the KKL design"
          " (||P^+|| large) loses to both KFs: no general 'KKL is robust' claim on linear plants")
    assert res[("gaussian", "KF R=sig^2")] <= res[("gaussian", "KKL")]           # KF optimal (Gaussian)
    assert res[("outliers", "KF R=sig^2+pM^2")] <= res[("outliers", "KKL")]      # LMMSE beats any LTI filter


def check_model_mismatch():
    """Proposition m3-mismatch: model incompleteness. Linear plant, the filter's model misses a term
    (Delta F x): fixed-gain predict-correct observer (steady KF / Luenberger) has a non-decaying bias
    (I - (I-KC)F)^{-1} (I-KC) dbar, Q-inflation trades that bias against the noise gain (a floor
    independent of data), and the KKL observer whose maps solve the TRUE plant's Sylvester equation
    (what phase 2 fits from true-plant labels) has no bias term."""
    header("Proposition m3-mismatch: filter bias under a missing term, Q-inflation floor, learned (true-plant) KKL")
    q, sig = 0.05, 0.03
    Qd = q ** 2 * DT * np.eye(2)
    R = np.array([[sig ** 2]])

    def true_system(eps):
        Ft = F + eps * np.array([[0.0, 0.0], [-1.0, 0.0]])          # missing stiffness term (model: F)
        Ftd = expm(Ft * DT); Gtd = np.linalg.solve(Ft, Ftd - np.eye(2)) @ G
        return Ft, Ftd, Gtd

    def gain(lam):
        Pk = solve_discrete_are(Fd.T, C.T, Qd + lam * np.eye(2), R)
        return Pk @ C.T / float((C @ Pk @ C.T)[0, 0] + R[0, 0])

    def filt(K, y, u, x0):
        """predict-correct with the (wrong) nominal model: xhat_{k+1} = Fd xhat_k + Gd u_k + K (y_{k+1} - C(.))"""
        xh = np.zeros((len(u), 2)); xh[0] = x0
        for k in range(len(u) - 1):
            xp = Fd @ xh[k] + Gd[:, 0] * u[k]
            xh[k + 1] = xp + K[:, 0] * (y[k + 1] - float(C @ xp))
        return xh

    # (1) steady bias for a constant input: dbar = (Ftd - Fd) xbar + (Gtd - Gd) u, formula exact
    eps = 0.3
    Ft, Ftd, Gtd = true_system(eps)
    ub = 1.0
    xbar = np.linalg.solve(np.eye(2) - Ftd, Gtd[:, 0] * ub)
    dbar = (Ftd - Fd) @ xbar + (Gtd - Gd)[:, 0] * ub
    N = 4000
    u = np.full(N, ub)
    x = np.zeros((N, 2)); x[0] = np.array([0.3, -0.2])
    for k in range(N - 1):
        x[k + 1] = Ftd @ x[k] + Gtd[:, 0] * u[k]
    y = x @ C.T
    for lam in (0.0, 1e-3):
        K = gain(lam)
        xh = filt(K, y[:, 0], u, np.zeros(2))
        IKC = np.eye(2) - K @ C
        ebar = np.linalg.solve(np.eye(2) - IKC @ Fd, IKC @ dbar)
        e_meas = (x - xh)[-1]
        print(f"  lambda={lam:g}: steady bias measured {e_meas} vs formula (I-(I-KC)F)^-1 (I-KC) dbar = {ebar}"
              f"  (|diff| {np.linalg.norm(e_meas - ebar):.1e}; |K| = {nrm(K):.3f})")
        assert np.linalg.norm(e_meas - ebar) < 1e-8
    # continuous-time analogue: edot = (F - K C) e + d  ->  ebar = -(F - K C)^{-1} dbar_c
    Kc = np.array([[2.0], [1.0]]); dc = eps * np.array([[0.0, 0.0], [-1.0, 0.0]]) @ np.linalg.solve(-Ft, G[:, 0] * ub)
    e_ct = -np.linalg.solve(F - Kc @ C, dc)
    e_sim = np.zeros(2)
    for _ in range(200000):
        e_sim = e_sim + 1e-3 * ((F - Kc @ C) @ e_sim + dc)
    print(f"  continuous Luenberger: steady bias {e_sim} vs -(F-KC)^-1 Delta F xbar = {e_ct}")
    assert np.linalg.norm(e_sim - e_ct) < 1e-6

    # (2) bias-variance trade-off under Q-inflation, persistent input (PRBS), MSE = bias^2 + tr Sigma
    def kf_mse(K, eps, u, x0, Nb=20000):
        Ft, Ftd, Gtd = true_system(eps)
        x = np.zeros((len(u), 2)); x[0] = x0
        for k in range(len(u) - 1):
            x[k + 1] = Ftd @ x[k] + Gtd[:, 0] * u[k]
        xh = filt(K, (x @ C.T)[:, 0], u, x0)                        # noise-free: the bias path
        bias2 = float(np.mean(np.sum((x - xh)[Nb // 4:] ** 2, axis=1)))
        Phi = (np.eye(2) - K @ C) @ Fd
        IKC = np.eye(2) - K @ C
        W = IKC @ Qd @ IKC.T + K @ R @ K.T                            # noise enters through (I-KC) w and K v
        S = solve_discrete_are_lyap(Phi, W)
        return bias2, float(np.trace(S)), x, xh

    def solve_discrete_are_lyap(Phi, W):
        from scipy.linalg import solve_discrete_lyapunov
        return solve_discrete_lyapunov(Phi, W)

    u = prbs(20000, dwell=300, seed=5)[:20000]
    x0 = np.array([0.5, 0.0])
    lams = [0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    for lam in lams:
        K = gain(lam)
        b2, s2, x, xh = kf_mse(K, eps, u, x0)
        rows.append((lam, nrm(K), b2, s2, b2 + s2))
    print(f"  Q-inflation (eps = {eps}; model error |DeltaF x|/|F x| median = "
          f"{np.median(np.linalg.norm(x @ (eps * np.array([[0, 0], [-1, 0]])).T, axis=1) / np.linalg.norm(x @ F.T, axis=1)):.2f}):")
    print(f"  {'lambda':>8s} {'|K|':>6s} {'bias^2':>10s} {'tr Sigma':>10s} {'MSE':>10s}")
    for r in rows:
        print(f"  {r[0]:8.0e} {r[1]:6.3f} {r[2]:10.3e} {r[3]:10.3e} {r[4]:10.3e}")
    b = np.array([r[2] for r in rows]); s = np.array([r[3] for r in rows]); kn = np.array([r[1] for r in rows])
    assert np.all(np.diff(kn) > 0) and np.all(np.diff(b) < 0) and np.all(np.diff(s) > 0), "bias falls, noise gain rises"
    floor = min(r[4] for r in rows)
    # Monte Carlo at two inflation levels confirms MSE = bias^2 + tr Sigma
    rng = np.random.default_rng(7)
    for lam in (0.0, 1e-2):
        K = gain(lam)
        _, Ftd, Gtd = true_system(eps)
        x = np.zeros((20000, 2)); x[0] = x0
        w = rng.standard_normal((20000, 2)) * q * math.sqrt(DT); v = rng.standard_normal(20000) * sig
        for k in range(20000 - 1):
            x[k + 1] = Ftd @ x[k] + Gtd[:, 0] * u[k] + w[k]
        xh = filt(K, (x @ C.T)[:, 0] + v, u, x0)
        mse = float(np.mean(np.sum((x - xh)[5000:] ** 2, axis=1)))
        pred = [r[4] for r in rows if r[0] == lam][0]
        print(f"  MC lambda={lam:g}: MSE {mse:.3e} vs bias^2 + tr Sigma = {pred:.3e}")
        assert abs(mse / pred - 1) < 0.1

    # (3) learned (true-plant) KKL: maps solve the true Sylvester equation; error is noise-driven only
    def kkl_mse(eps, u, x0, rng):
        Ft, Ftd, Gtd = true_system(eps)
        Pt = sylvester_reference(Ft, C, A, B); Ppt = np.linalg.pinv(Pt)
        NUt = -Wg @ Pt @ G
        n = len(u)
        x = np.zeros((n, 2)); x[0] = x0; nu = np.zeros((n, NZ)); zh = np.zeros((n, NZ)); zh[0] = Pt @ x0
        w = rng.standard_normal((n, 2)) * q * math.sqrt(DT); v = rng.standard_normal(n) * sig
        y = np.zeros(n); y[0] = float(C @ x[0]) + v[0]
        for k in range(n - 1):
            x[k + 1] = Ftd @ x[k] + Gtd[:, 0] * u[k] + w[k]
            y[k + 1] = float(C @ x[k + 1]) + v[k + 1]
            nu[k + 1] = Ad @ nu[k] + NUt[:, 0] * u[k]
            zh[k + 1] = Ad @ zh[k] + W0[:, 0] * y[k] + W1[:, 0] * y[k + 1]
        xh = (zh - nu) @ Ppt.T
        e = (x - xh)[5000:]
        return float(np.mean(np.sum(e ** 2, axis=1))), np.linalg.norm(e.mean(0))
    mse_kkl, mean_kkl = kkl_mse(eps, u, x0, np.random.default_rng(11))
    print(f"  learned (true-plant maps) KKL: MSE {mse_kkl:.3e}, |mean error| {mean_kkl:.1e} (no bias);"
          f" filter floor min_lambda MSE = {floor:.3e}  ->  learned {'wins' if mse_kkl < floor else 'loses'} at eps={eps}")

    # (4) the threshold: sweep the size of the missing term; filters win when it is small
    print(f"  {'eps':>5s} {'rel. model err':>14s} {'filter floor':>12s} {'learned MSE':>12s}  winner")
    outcome = []
    for eps in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        best = min(sum(kf_mse(gain(l), eps, u, x0)[:2]) for l in lams)
        mk, _ = kkl_mse(eps, u, x0, np.random.default_rng(11))
        _, Ftd, Gtd = true_system(eps)
        xs = np.zeros((len(u), 2)); xs[0] = x0
        for k in range(len(u) - 1):
            xs[k + 1] = Ftd @ xs[k] + Gtd[:, 0] * u[k]
        rel = np.median(np.linalg.norm(xs @ (eps * np.array([[0, 0], [-1, 0]])).T, axis=1) / np.linalg.norm(xs @ F.T, axis=1))
        outcome.append(mk < best)
        print(f"  {eps:5.2f} {rel:14.3f} {best:12.3e} {mk:12.3e}  {'learned' if mk < best else 'filter'}")
    assert outcome[0] is False and outcome[-1] is True and outcome == sorted(outcome), "one crossing: filter wins below, learned above"


def check_apriori_vacuity():
    """Prop m3-iss (iii) with a-priori constants: the global term l_z kappa |B| delta_y / alpha evaluated with the
    phase-1 proxy l_dec_q95 (C1-where) is 2-3 orders above the measured NRMSE: a decomposition, not a certificate."""
    header("Prop m3-iss (iii): a-priori constants vs measured NRMSE (C1-where, lora, id/clean)")
    rows = {"battery": (7, 1.5, 2, 816, 0.0483), "bioreactor": (5, 0.5, 1, 43.9, 0.0193), "duffing": (5, 3.0, 1, 299, 0.0469),
            "lorenz": (7, 2.0, 1, 10.1, 0.0270), "quad_tank": (6, 0.1, 2, 13.5, 0.0524), "vdp": (5, 1.5, 1, 23.9, 0.0164)}
    dy = 0.03                                     # nominal sensor noise, y_std units
    ratios = []
    for name, (nz, al, ny, l, meas) in rows.items():
        Am, Bm = butterworth_AB(nz, al, ny)
        kap, a = spectral_bounds(Am)
        g = kap * nrm(Bm) / a
        ap = l * g * dy
        ratios.append(ap / meas)
        print(f"  {name:10s}: kappa|B|/alpha = {g:.3f}; l_dec_q95 = {l:6.1f}; a-priori term l g delta_y = {ap:7.2f}"
              f"  vs measured NRMSE {meas:.4f}  (x{ap / meas:.0f})")
    print(f"  a-priori / measured: {min(ratios):.0f}x to {max(ratios):.0f}x")
    assert min(ratios) > 10


def check_lipschitz_estimator():
    header("Remark m3-cert: random-direction finite differences under-estimate ||J||_2")
    # phase1_diagnostics' l_dec: ||T+(z + h d) - T+(z)|| / h, d uniform on the sphere.
    # For a linear decoder J (nx x nz): E||J d||^2 = ||J||_F^2 / nz <= ||J||_2^2.
    J = Pp
    d = RNG.standard_normal((20000, NZ)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = np.linalg.norm(d @ J.T, axis=1)
    print(f"  ||J||_2 = {nrm(J):.4f};  random-direction q95 {np.quantile(r, 0.95):.4f}, mean {r.mean():.4f}"
          f"  (ratio q95/||J|| = {np.quantile(r, 0.95) / nrm(J):.3f})")
    assert np.quantile(r, 0.95) < nrm(J)


def check_identity_and_quantile():
    header("Proposition m3-iss (ii): path identity for a nonlinear decoder; Definition m3-fit: order statistic")
    W1_ = RNG.standard_normal((32, NZ)) * 0.8; b1 = RNG.standard_normal(32) * 0.3
    W2_ = RNG.standard_normal((32, 32 + 3)) * 0.3; W3_ = RNG.standard_normal((2, 32)) * 0.5

    def dec(z, c):
        h = np.tanh(W1_ @ z + b1)
        h = np.tanh(W2_ @ np.concatenate([h, c]))
        return W3_ @ h

    def jac(fun, a, eps=1e-6):
        f0 = fun(a)
        return np.stack([(fun(a + eps * e) - fun(a - eps * e)) / (2 * eps) for e in np.eye(len(a))], 1)
    xg, wg = np.polynomial.legendre.leggauss(20)
    sq, wq = 0.5 * (xg + 1), 0.5 * wg
    worst = 0.0; gains = []
    for _ in range(50):
        z = RNG.standard_normal(NZ); c = RNG.standard_normal(3)
        dz = 0.5 * RNG.standard_normal(NZ); dc = 0.3 * RNG.standard_normal(3)
        Jz = sum(w * jac(lambda a: dec(a, c + dc), z + s * dz) for s, w in zip(sq, wq))
        Jc = sum(w * jac(lambda a: dec(z, a), c + s * dc) for s, w in zip(sq, wq))
        lhs = dec(z + dz, c + dc) - dec(z, c)
        worst = max(worst, np.abs(lhs - Jz @ dz - Jc @ dc).max())
        on = nrm(jac(lambda a: dec(a, c), z))
        gains.append((np.linalg.norm(Jz @ dz) / np.linalg.norm(dz), on))
    print(f"  identity xhat - xhat_nom = Jz_bar dz + Jc_bar dc: max residual {worst:.1e}")
    g = np.array(gains)
    print(f"  realised gain / on-data ||J_z||_2: median {np.median(g[:, 0] / g[:, 1]):.3f}"
          f" (range {np.min(g[:, 0] / g[:, 1]):.2f}-{np.max(g[:, 0] / g[:, 1]):.2f}): off-data slopes differ")
    assert worst < 1e-8
    # order statistic: P(max of n draws >= (1-beta)-quantile) = 1 - (1-beta)^n
    n, beta, R = 64, 0.05, 20000
    q = -math.log(beta)                                       # (1-beta)-quantile of Exp(1)
    cover = np.mean(RNG.exponential(size=(R, n)).max(1) >= q)
    print(f"  n={n}, beta={beta}: coverage {cover:.4f} vs 1-(1-beta)^n = {1 - (1 - beta) ** n:.4f}")
    assert abs(cover - (1 - (1 - beta) ** n)) < 0.01


def check_fault_ratios():
    header("Corollary m3-faults prediction: latent deviation ratios per stress condition (design-invariant)")
    rows = []
    for name, nz, al in (("vdp", 5, 1.5), ("duffing", 5, 3.0), ("bioreactor", 5, 0.5), ("lorenz", 7, 2.0)):
        Am, Bm = butterworth_AB(nz, al, 1)
        dt = choose_dt(Am)
        st = latent_step(Am, Bm, dt)
        n = int(40 / (al * dt))
        Hs = [st["W1"][:, 0]]; Mm = st["W0"] + st["Ad"] @ st["W1"]
        for _ in range(n - 1):
            Hs.append(Mm[:, 0]); Mm = st["Ad"] @ Mm
        Hs = np.array(Hs)
        h2 = float((Hs ** 2).sum())
        Gm = Hs @ Hs.T
        idx = np.arange(len(Hs))
        colored = math.sqrt(float(((0.95 ** np.abs(idx[:, None] - idx[None])) * Gm).sum()) / h2)
        outl = math.sqrt(0.02) * 12.0 / 0.03
        drift = 0.3 * np.linalg.norm(np.linalg.solve(Am, Bm)) / (0.03 * math.sqrt(h2))
        rows.append((colored, outl, drift))
        print(f"  {name:10s}: dt={dt:.4f}  colored x{colored:.2f}  outliers x{outl:.1f}  drift (quasi-static peak) x{drift:.1f}")
    r = np.array(rows)
    assert np.ptp(r[:, 0]) < 0.1 and np.all(r[:, 2] > 30) and np.all(r[:, 2] < 33)


if __name__ == "__main__":
    check_design_constants()
    check_noise_iss()
    check_interpolation()
    check_input_corruption()
    check_outliers()
    check_dropout_drift()
    check_midpoint()
    check_kf_comparison()
    check_model_mismatch()
    check_apriori_vacuity()
    check_lipschitz_estimator()
    check_identity_and_quantile()
    check_fault_ratios()
    print("\nall M3 checks passed")
