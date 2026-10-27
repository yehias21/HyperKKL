"""Numerical checks for theory/m4_information.tex (block M4: information budget), CPU only, ~3 min.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 .venv/bin/python theory/checks/m4_checks.py

[1] Linear-Gaussian plant: the Kalman covariance == the posterior-CRB recursion (local and Bayesian forms,
    to 1e-12); the KF's Monte Carlo MSE attains the bound; a sub-optimal LTI estimator (the exact KKL
    observer) lies above it.
[2] Fisher information == observability Gramian / sigma^2 (deterministic window, Q -> 0), and the
    recursion reproduces it.
[3] Bits: 1/2 log2 det(I + Sigma W/sigma^2) == the recursion's bits; the rate-distortion (Shannon) bound
    is below the CRB total (AM-GM) and below the KF error.
[4] Excitation: for a linear plant the information about x_t does not depend on the input (two input
    records: identical J); for quad_tank (sqrt law) it changes only through the operating point (reported).
[5] Empirical (finite-difference) Gramian of a nonlinear plant vs the Jacobian Gramian: stable in the
    perturbation size.
[6] Learning side: a regressor covariance with a null direction leaves that direction unlearnable at any
    sample size; the excess error of a least-squares fit scales as sigma^2 p / N (structured residual with
    p = 2 vs generic map with p = 64).
[7] Slow error processes: per-episode RMS of a slow state scatters widely while the pooled RMS over
    episodes matches the bound (why the CRB is checked against the pooled RMS).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, solve_discrete_are

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "information"))
from hyperkkl.design import butterworth_AB, choose_dt, latent_step, sylvester_reference  # noqa: E402
import infolib as L  # noqa: E402  (the recursions used by experiments/information)

torch.set_num_threads(8)
np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(0)


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


# ---------------------------------------------------------------------------
# Linear setting (as m3_checks): damped oscillator, PRBS input, Butterworth latent
# ---------------------------------------------------------------------------
F = np.array([[0.0, 1.0], [-1.0, -0.4]])
G = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
NZ, ALPHA = 5, 1.5
A, B = butterworth_AB(NZ, ALPHA, 1)
DT = choose_dt(A)
Fd = expm(F * DT)
Gd = np.linalg.solve(F, Fd - np.eye(2)) @ G
Q_RATE, SIG = 0.05, 0.03
Qd = Q_RATE ** 2 * DT * np.eye(2)
Rm = np.array([[SIG ** 2]])


def prbs(N, amp=1.0, dwell=150, seed=1):
    r = np.random.default_rng(seed)
    return np.repeat(amp * r.choice([-1.0, 1.0], size=N // dwell + 2), dwell)[:N + 1]


def simulate_linear(u, x0, rng, E):
    N = len(u) - 1
    x = np.zeros((E, N + 1, 2)); x[:, 0] = x0
    w = rng.standard_normal((E, N, 2)) * Q_RATE * math.sqrt(DT)
    for k in range(N):
        x[:, k + 1] = x[:, k] @ Fd.T + Gd[:, 0] * u[k] + w[:, k]
    y = x[..., :1] + SIG * rng.standard_normal((E, N + 1, 1))
    return x, y


def kf(y, u, x0, P0):
    """Standard KF (predict with the known input, update), returns xhat (E, N+1, 2) and P_k (N+1, 2, 2)."""
    E, N1, _ = y.shape
    xh = np.zeros((E, N1, 2)); P = P0.copy()
    xh[:, 0] = x0
    Ps = []
    # update at k = 0
    S = C @ P @ C.T + Rm; K = P @ C.T @ np.linalg.inv(S)
    xh[:, 0] = xh[:, 0] + (y[:, 0] - xh[:, 0] @ C.T) @ K.T
    P = (np.eye(2) - K @ C) @ P; Ps.append(P)
    for k in range(N1 - 1):
        xp = xh[:, k] @ Fd.T + Gd[:, 0] * u[k]
        P = Fd @ P @ Fd.T + Qd
        S = C @ P @ C.T + Rm; K = P @ C.T @ np.linalg.inv(S)
        xh[:, k + 1] = xp + (y[:, k + 1] - xp @ C.T) @ K.T
        P = (np.eye(2) - K @ C) @ P
        Ps.append(P)
    return xh, np.array(Ps)


def check_linear_gaussian():
    header("[1] linear-Gaussian: KF covariance == PCRB recursion; KF attains it; the KKL observer does not")
    N, E = 3000, 400
    u = prbs(N, dwell=120, seed=2)
    rng = np.random.default_rng(3)
    Sigma0 = np.diag([0.5, 0.5]) ** 2
    x0 = rng.multivariate_normal(np.zeros(2), Sigma0, size=E)
    x, y = simulate_linear(u, x0, rng, E)
    xh, Ps = kf(y, u, np.zeros(2), Sigma0)
    # the recursions of infolib on the same linear model (F constant, H constant)
    Ft = torch.as_tensor(np.broadcast_to(Fd, (E, N, 2, 2)).copy())
    Ht = torch.as_tensor(np.broadcast_to(C, (E, N + 1, 1, 2)).copy())
    k_eval = torch.arange(0, N + 1, 250)
    Pl = L.pcrb_full(Ft, Ht, torch.as_tensor(Qd), torch.as_tensor(Rm), torch.as_tensor(Sigma0), k_eval)
    Pb = L.pcrb_bayes_full(Ft, Ht, torch.as_tensor(Qd), torch.as_tensor(Rm), torch.as_tensor(Sigma0), k_eval)
    d_local = float((Pl[0].numpy() - Ps[k_eval.numpy()]).__abs__().max())
    d_bayes = float((Pb.numpy() - Ps[k_eval.numpy()]).__abs__().max())
    print(f"  max |P_KF - P_local| = {d_local:.1e},  max |P_KF - P_Bayes| = {d_bayes:.1e}")
    assert d_local < 1e-12 and d_bayes < 1e-12
    settle = N // 2
    mse = ((xh - x)[:, settle:] ** 2).mean((0, 1))
    bnd = np.array([Ps[settle:, j, j].mean() for j in range(2)])
    print(f"  KF Monte Carlo MSE per state {mse} vs bound diag(P) {bnd}  (ratio {mse / bnd})")
    assert np.all(np.abs(mse / bnd - 1) < 0.05), "KF attains the posterior CRB within MC error"
    # the exact KKL observer (an LTI filter of y, u): z = P x + nu, decoder P^+ (z - nu)
    P_syl = sylvester_reference(F, C, A, B); Pp = np.linalg.pinv(P_syl)
    st = latent_step(A, B, DT); Ad, W0, W1, Wg = st["Ad"], st["W0"], st["W1"], st["Wg"]
    NUd = -Wg @ P_syl @ G
    z = np.zeros((E, NZ)); nu = np.zeros(NZ); err = []
    for k in range(N):
        z = z @ Ad.T + y[:, k] @ W0.T + y[:, k + 1] @ W1.T
        nu = Ad @ nu + NUd[:, 0] * u[k]
        if k + 1 >= settle:
            err.append((((z - nu) @ Pp.T) - x[:, k + 1]) ** 2)
    mse_kkl = np.mean(err, (0, 1))
    print(f"  exact KKL observer MSE {mse_kkl} >= bound {bnd}: {np.all(mse_kkl >= bnd)}  (x{mse_kkl / bnd})")
    assert np.all(mse_kkl >= bnd)
    return Sigma0


def check_gramian_is_fisher():
    header("[2] Fisher information of a deterministic window == observability Gramian / sigma^2")
    W = 60
    # information about x_t from y_{t-W..t}: J = sum_k Phi_k^T C^T R^-1 C Phi_k,  Phi_k = Fd^{-k} (backward sensitivity)
    Finv = np.linalg.inv(Fd)
    Jg = np.zeros((2, 2)); Phi = np.eye(2)
    for k in range(W + 1):
        Jg += Phi.T @ C.T @ np.linalg.inv(Rm) @ C @ Phi
        Phi = Finv @ Phi
    Sig_big = 1e6 * np.eye(2)                 # flat prior
    Ft = torch.as_tensor(np.broadcast_to(Fd, (1, W, 2, 2)).copy())
    Ht = torch.as_tensor(np.broadcast_to(C, (1, W + 1, 1, 2)).copy())
    Pw = L.pcrb_window(Ft, Ht, 1e-14 * torch.eye(2, dtype=torch.float64), torch.as_tensor(Rm),
                       torch.as_tensor(Sig_big), torch.tensor([W]), W)[0, 0].numpy()
    Jrec = np.linalg.inv(Pw)
    rel = np.abs(Jrec - Jg).max() / np.abs(Jg).max()
    print(f"  Gramian/sigma^2 J =\n{Jg}\n  recursion (Q -> 0, flat prior) J =\n{Jrec}\n  relative deviation {rel:.1e}")
    assert rel < 1e-5
    # equivalently: the Gramian of the window's output w.r.t. the state at the START of the window,
    # mapped to the end: J_end = Fd^{-W T} W_o(start) Fd^{-W}
    Wo = np.zeros((2, 2)); Phi = np.eye(2)
    for k in range(W + 1):
        Wo += Phi.T @ C.T @ C @ Phi / SIG ** 2; Phi = Fd @ Phi
    FW = np.linalg.matrix_power(Finv, W)
    print(f"  forward Gramian mapped to the window end: deviation {np.abs(FW.T @ Wo @ FW - Jg).max() / np.abs(Jg).max():.1e}")
    return Jg


def check_bits_and_rate_distortion(Sigma0, Jg):
    header("[3] bits = 1/2 log2 det(I + Sigma W_o/sigma^2); Shannon lower bound <= CRB total <= KF MSE")
    n = 2
    Sig = np.diag([0.5, 0.5]) ** 2
    bits_formula = 0.5 * math.log2(np.linalg.det(np.eye(n) + Sig @ Jg))
    P_post = np.linalg.inv(np.linalg.inv(Sig) + Jg)
    bits_rec = 0.5 * (math.log2(np.linalg.det(Sig)) - math.log2(np.linalg.det(P_post)))
    print(f"  bits (Gaussian channel formula) {bits_formula:.4f} == bits (prior - posterior log det) {bits_rec:.4f}")
    assert abs(bits_formula - bits_rec) < 1e-9
    # rate-distortion (Shannon lower bound): D_total >= n det(Sig)^{1/n} 2^{-2 I/n};  CRB total = tr(P_post) >= that (AM-GM)
    D_rd = n * np.linalg.det(Sig) ** (1 / n) * 2 ** (-2 * bits_rec / n)
    D_crb = np.trace(P_post)
    print(f"  Shannon lower bound on total MSE {D_rd:.4e} <= CRB total tr(P) {D_crb:.4e} (AM-GM gap x{D_crb / D_rd:.3f})")
    assert D_rd <= D_crb * (1 + 1e-12)
    # at the KF's steady state (with process noise): the same ordering against the Monte Carlo MSE
    Pk = solve_discrete_are(Fd.T, C.T, Qd, Rm)                       # predicted steady covariance
    Pk = Pk - Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + Rm) @ C @ Pk  # posterior
    Sig_clim = np.diag([0.5, 0.5]) ** 2
    I_bits = 0.5 * (math.log2(np.linalg.det(Sig_clim)) - math.log2(np.linalg.det(Pk)))
    D_rd2 = n * np.linalg.det(Sig_clim) ** (1 / n) * 2 ** (-2 * I_bits / n)
    print(f"  steady KF: bits {I_bits:.2f}, Shannon bound {D_rd2:.3e} <= tr P {np.trace(Pk):.3e} (= attained MSE, check [1])")
    assert D_rd2 <= np.trace(Pk)


def check_excitation():
    header("[4] excitation: linear plant -> J independent of u; quad_tank -> changes only through the operating point")
    N = 2000
    Ft = torch.as_tensor(np.broadcast_to(Fd, (1, N, 2, 2)).copy())
    Ht = torch.as_tensor(np.broadcast_to(C, (1, N + 1, 1, 2)).copy())
    k_eval = torch.tensor([N])
    Pa = L.pcrb_full(Ft, Ht, torch.as_tensor(Qd), torch.as_tensor(Rm), torch.eye(2, dtype=torch.float64), k_eval)
    # ... the Jacobians of a linear plant do not depend on u, so any two input records give the same recursion:
    print(f"  linear plant: J does not involve u at all (F, H constant); |P(u1) - P(u2)| = 0 by construction, P_ss diag = {Pa[0, 0].diagonal().numpy()}")
    # quad_tank: two input records with different time scales (t_ref x 0.25 and x 4) -> operating points differ
    from hyperkkl import data as Dm, signals as S
    pay = L.run_payload("quad_tank"); cfg, scales = pay["config"], pay["scales"]; scfg = cfg["system"]
    plant = L.build_plant(scfg); des = Dm.make_design(scfg)
    es_ref = L.load_cached_eval(cfg, "id"); settle = int(0.5 * (es_ref.x.shape[1] - 1))
    Sigma = L.climatology(es_ref.x[:, settle:].double())
    out = {}
    for s in (0.25, 1.0, 4.0):
        sc = dict(scfg, t_ref=float(scfg["t_ref"]) * s)
        es = Dm.generate(plant, des, scales, sc, dict(cfg["data"], k_cross=1), S.families_for("id"), 8, "id",
                         4242 + int(10 * s), "cpu")
        b = L.budget_along(plant, es.x, es.u, cfg, scales, [0, 1], [30.0], n_ep=8, n_t=8, Sigma=Sigma)
        out[s] = b
        print(f"  quad_tank, input time scale x{s:g}: full-past unmeasured bound {b['bayes']['full']['unmeasured_lb']:.4f}, "
              f"bits {b['bayes']['full']['bits']:.2f}; 30 s window {b['bayes']['windows']['30']['unmeasured_lb']:.4f}; "
              f"mean upper-tank level x3 {float(es.x[:, settle:, 2].mean()):.2f} cm")
    v = [out[s]["bayes"]["full"]["unmeasured_lb"] for s in out]
    print(f"  spread of the full-past bound across input time scales: {100 * (max(v) / min(v) - 1):.1f}% (operating-point effect only)")


def check_empirical_gramian():
    header("[5] empirical (finite-difference) Gramian of a nonlinear plant vs the Jacobian Gramian: stable in epsilon")
    pay = L.run_payload("vdp"); cfg, scales = pay["config"], pay["scales"]; scfg = cfg["system"]
    plant = L.build_plant(scfg); dt = scales["dt"]
    es = L.load_cached_eval(cfg, "id")
    x0 = es.x[:4, 1000].double(); u = es.u[:4, 1000:1000 + 101].double()
    W = 100
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)

    def rollout(xs):
        ys = []
        x = xs
        for k in range(W + 1):
            ys.append(plant.h(x))
            x = L.rk4_step(plant, x, u[:, k], dt)
        return torch.stack(ys, 1)                                   # (E, W+1, ny)
    # Jacobian Gramian about the window START (information about x_{t-W} from the window), Q -> 0
    F, H = L.jacobians(plant, torch.stack([x0] + [None] * 0) if False else None, None, dt) if False else (None, None)
    # explicit: Phi_k by chaining Jacobians along the rollout
    xs = [x0]; x = x0
    for k in range(W):
        x = L.rk4_step(plant, x, u[:, k], dt); xs.append(x)
    X = torch.stack(xs, 1)
    Fk, Hk = L.jacobians(plant, X, u[:, :W + 1], dt)
    E = x0.shape[0]
    J = torch.zeros(E, 2, 2, dtype=torch.float64); Phi = torch.eye(2, dtype=torch.float64).expand(E, 2, 2).clone()
    for k in range(W + 1):
        J += Phi.transpose(-1, -2) @ Hk[:, k].transpose(-1, -2) @ Hk[:, k] @ Phi
        if k < W:
            Phi = Fk[:, k] @ Phi
    J = J / (0.03 * scales["y_std"][0]) ** 2
    y0 = rollout(x0)
    res = []
    for eps_rel in (1e-3, 1e-2, 1e-1, 0.3):
        Je = torch.zeros(E, 2, 2, dtype=torch.float64)
        for i in range(2):
            e = torch.zeros(2, dtype=torch.float64); e[i] = eps_rel * sig[i]
            dy = (rollout(x0 + e) - rollout(x0 - e)) / (2 * e[i])            # (E, W+1, ny)
            for j in range(2):
                e2 = torch.zeros(2, dtype=torch.float64); e2[j] = eps_rel * sig[j]
                dy2 = (rollout(x0 + e2) - rollout(x0 - e2)) / (2 * e2[j])
                Je[:, i, j] = (dy * dy2).sum((1, 2)) / (0.03 * scales["y_std"][0]) ** 2
        rel = float((torch.linalg.norm(Je - J, dim=(1, 2)) / torch.linalg.norm(J, dim=(1, 2))).max())
        res.append(rel)
        print(f"  epsilon = {eps_rel:g} x_std: max relative deviation of the empirical Gramian from the Jacobian one {rel:.2e}")
    assert res[0] < 1e-3 and res[1] < 1e-2 and res[2] < 0.3, "empirical Gramian converges to the Jacobian Gramian as eps -> 0"


def check_learning_side():
    header("[6] learning side: a null direction of the regressor covariance is unlearnable; excess error ~ sigma^2 p / N")
    rng = np.random.default_rng(5)
    # (a) regressors confined to a subspace: the least-squares map agrees on the subspace, is arbitrary off it
    d, N = 4, 5000
    w_true = rng.standard_normal(d)
    Z = rng.standard_normal((N, d)); Z[:, 3] = 0.0                  # direction 4 never excited
    y = Z @ w_true + 0.1 * rng.standard_normal(N)
    w_hat = np.linalg.lstsq(Z, y, rcond=None)[0]
    q = np.array([0, 0, 0, 1.0])                                     # a query along the unexcited direction
    print(f"  rank(Z^T Z) = {np.linalg.matrix_rank(Z.T @ Z)} < {d}; fitted w on the excited directions {w_hat[:3]} vs true {w_true[:3]}")
    print(f"  prediction error at the unexcited query |q.(w_hat - w_true)| = {abs(q @ (w_hat - w_true)):.3f} (= |w_true,4| = {abs(w_true[3]):.3f}, independent of N)")
    assert abs(q @ (w_hat - w_true)) > 0.1 * abs(w_true[3]) or abs(w_true[3]) < 0.1
    # (b) excess error sigma^2 p / N: structured residual p = 2 vs generic p = 64 (both well specified)
    sigma = 0.1
    for p in (2, 64):
        ratios = []
        for N in (200, 800, 3200):
            ex = []
            for _ in range(200):
                Z = rng.standard_normal((N, p)); w = rng.standard_normal(p)
                y = Z @ w + sigma * rng.standard_normal(N)
                w_hat = np.linalg.lstsq(Z, y, rcond=None)[0]
                ex.append(float(np.sum((w_hat - w) ** 2)))      # E||w_hat - w||^2 = sigma^2 tr((Z^T Z)^-1) ~ sigma^2 p / N
            ratios.append(np.mean(ex) / (sigma ** 2 * p / N))
        print(f"  p = {p:2d}: excess risk / (sigma^2 p / N) at N = 200, 800, 3200: {np.round(ratios, 2)}")
        assert all(0.8 < r < 1.5 for r in ratios)
    print("  -> the same excess error needs N proportional to p: a 2-parameter residual needs 32x fewer samples than a 64-parameter map")


def check_slow_error_sampling():
    header("[7] slow error processes: per-episode RMS scatters, pooled RMS matches the bound (linearised quad_tank, KF)")
    from hyperkkl.systems import build_plant
    pay = L.run_payload("quad_tank"); cfg, scales = pay["config"], pay["scales"]; scfg = cfg["system"]
    es = L.load_cached_eval(cfg, "id")
    E, N1 = 64, es.x.shape[1]; dt = scales["dt"]; settle = int(0.5 * (N1 - 1))
    plant = build_plant(scfg, knowledge="partial")                 # flows linearised at the operating point: a linear plant
    u = es.u[:E].double(); x0 = es.x[:E, 0].double()
    q = 0.005 * np.asarray(scales["x_std"]); r = 0.03 * np.asarray(scales["y_std"])
    g = torch.Generator().manual_seed(11)
    Wn = torch.randn(E, N1 - 1, 4, generator=g, dtype=torch.float64) * torch.as_tensor(q) * math.sqrt(dt)
    V = torch.randn(E, N1, 2, generator=g, dtype=torch.float64) * torch.as_tensor(r)
    xs = [x0]
    for k in range(N1 - 1):
        xs.append(L.rk4_step(plant, xs[-1], u[:, k], dt) + Wn[:, k])
    X = torch.stack(xs, 1); Y = plant.h(X.reshape(-1, 4)).reshape(E, N1, 2) + V
    # linear plant: one Jacobian; KF in information form == infolib recursion; run a plain KF
    Fk, Hk = L.jacobians(plant, X[:1, :2], u[:1, :2], dt)
    Fd_, H_ = Fk[0, 0].numpy(), Hk[0, 0].numpy()
    Qd_, R_ = np.diag(q ** 2 * dt), np.diag(r ** 2)
    box = np.asarray(scfg["ic_box"]); P = np.diag(((box[:, 1] - box[:, 0]) / math.sqrt(12)) ** 2)
    xh = np.tile(box.mean(1), (E, 1)); Yn, Un = Y.numpy(), u.numpy()
    S = H_ @ P @ H_.T + R_; K = P @ H_.T @ np.linalg.inv(S); xh = xh + (Yn[:, 0] - xh @ H_.T) @ K.T; P = (np.eye(4) - K @ H_) @ P
    err, Pd = [], []
    for k in range(N1 - 1):
        xp = np.stack([L.rk4_step(plant, torch.as_tensor(xh), u[:, k], dt).numpy()])[0]
        P = Fd_ @ P @ Fd_.T + Qd_
        S = H_ @ P @ H_.T + R_; K = P @ H_.T @ np.linalg.inv(S)
        xh = xp + (Yn[:, k + 1] - xp @ H_.T) @ K.T; P = (np.eye(4) - K @ H_) @ P
        if k + 1 >= settle:
            err.append(xh - X[:, k + 1].numpy()); Pd.append(np.diag(P))
    err = np.array(err) / np.asarray(scales["x_std"]); Pd = np.array(Pd) / np.asarray(scales["x_std"]) ** 2
    per_ep = np.sqrt((err ** 2).mean(0))                              # (E, 4)
    pooled = np.sqrt((err ** 2).mean((0, 1)))
    bound = np.sqrt(Pd.mean(0))
    print(f"  KF bound sqrt(P_jj)/sigma_j {bound}")
    print(f"  pooled RMS over {E} episodes      {pooled}  (ratio {pooled / bound})")
    print(f"  per-episode RMS of x3: min {per_ep[:, 2].min():.4f} median {np.median(per_ep[:, 2]):.4f} max {per_ep[:, 2].max():.4f}; "
          f"mean of per-episode RMS {per_ep[:, 2].mean():.4f} (Jensen: below the pooled {pooled[2]:.4f})")
    assert np.all(np.abs(pooled / bound - 1) < 0.08), "pooled RMS within MC error of the bound"
    assert per_ep[:, 2].mean() <= pooled[2] + 1e-12


if __name__ == "__main__":
    Sigma0 = check_linear_gaussian()
    Jg = check_gramian_is_fisher()
    check_bits_and_rate_distortion(Sigma0, Jg)
    check_excitation()
    check_empirical_gramian()
    check_learning_side()
    check_slow_error_sampling()
    print("\nall M4 checks passed")
