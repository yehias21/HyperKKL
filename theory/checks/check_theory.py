"""Numerical checks for every result in theory/main.tex (CPU, ~1 minute).

    CUDA_VISIBLE_DEVICES="" ../.venv/bin/python theory/checks/check_theory.py

Each check prints the measured quantity next to the bound or the prediction, and
asserts the inequality, so a failing statement cannot go unnoticed.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True)


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


# ---------------------------------------------------------------------------
# Lemma 1 / Lemma 1': the KKL map forgets the input exactly at rate alpha
# ---------------------------------------------------------------------------

def lemma1():
    """Integrator plant xdot = u, y = x, scalar latent zdot = -alpha z + y.
    Two inputs agree on [t-L, t]; before t-L, u = 0 and v = w. Then
    T_t(x;u) - T_t(x;v) = w e^{-alpha L} int_0^{t-L} e^{-alpha r} r dr  (closed form)."""
    header("Lemma 1 (upper bound) and Lemma 1' (matching lower bound)")
    alpha, t, w, xbar = 1.5, 40.0, 0.2, 1.0
    Ls = np.linspace(0.5, 12.0, 24)
    diffs, upper = [], []
    for L in Ls:
        m = t - L
        exact = w * math.exp(-alpha * L) * (1 - math.exp(-alpha * m) * (1 + alpha * m)) / alpha ** 2
        # brute-force check of the closed form by quadrature of the definition
        s = np.linspace(0, t, 200001)
        Xu = np.full_like(s, xbar)                              # u = 0: X(s) = x
        Xv = xbar - w * np.clip(t - L - s, 0, None)             # v = w before t-L
        quad = np.trapezoid(np.exp(-alpha * (t - s)) * (Xu - Xv), s)
        assert abs(quad - exact) < 1e-6 * max(abs(exact), 1e-12), (quad, exact)
        diffs.append(exact)
        ybar = xbar + w * t                                     # |h| on the reached set
        upper.append(2 * ybar * math.exp(-alpha * L) / alpha)   # kappa = ||B|| = 1
    diffs, upper = np.array(diffs), np.array(upper)
    rate = -np.polyfit(Ls, np.log(diffs), 1)[0]
    lower_c = w * (1 - 2 / math.e) / alpha ** 2                 # valid for t - L >= 1/alpha
    print(f"  fitted decay rate of |T(u) - T(v)| vs L: {rate:.4f}   (alpha = {alpha})")
    print(f"  min over L of  upper bound / measured : {np.min(upper / diffs):.3g}  (>= 1)")
    print(f"  min over L of  measured / (c e^-aL)    : {np.min(diffs / (lower_c * np.exp(-alpha * Ls))):.3f}  (>= 1)")
    assert abs(rate - alpha) < 1e-3 and np.all(upper >= diffs) and np.all(diffs >= lower_c * np.exp(-alpha * Ls))


# ---------------------------------------------------------------------------
# Proposition 1: z alone cannot replace the input history (current input insufficient)
# ---------------------------------------------------------------------------

def history_needed():
    """Linear plant xdot = F x + G u, y = C x, finite-time map z_t = M x_t + H w where w
    collects the input BEFORE t - L (the current input is identical). If
    rank[M H] < nx + rank(H), two different states give the same latent and the same
    current input: no decoder G(z, u(t)) can be right for both."""
    header("Proposition 1: the latent state does not encode the input history")
    F = np.array([[0.0, 1.0], [-1.0, -0.2]]); Gv = np.array([0.0, 1.0]); C = np.array([[1.0, 0.0]])
    from hyperkkl.design import butterworth_AB
    A, B = butterworth_AB(5, 1.0, 1)
    dt, t, L, m = 0.01, 30.0, 2.0, 12                           # m past input pieces
    n = int(round(t / dt))
    ts = np.arange(n) * dt
    # z_t = int_0^t e^{A(t-s)} B C x(s) ds with x(s) = Phi(s,t) x_t - int_s^t Phi(s,r) G u(r) dr
    Pb = expm(-F * dt)
    Gd = np.linalg.solve(F, (np.eye(2) - Pb)) @ Gv              # e^{-F dt} int_0^dt e^{F s} ds G
    Ad, Eh = expm(A * dt), expm(A * dt / 2) @ B

    def latent(xt, u):
        x = np.zeros((n + 1, 2)); x[n] = xt
        for k in range(n - 1, -1, -1):                          # exact backward ZOH step
            x[k] = Pb @ x[k + 1] - Gd * u[k]
        z = np.zeros(5)
        for k in range(n):                                      # midpoint quadrature (linear in x, u)
            z = Ad @ z + dt * (Eh @ (C @ (0.5 * (x[k] + x[k + 1]))))
        return z
    # the recent past [t - 6, t - 0.1): within the latent memory, so its effect on z is O(1);
    # the current input (last 0.1 s) is identical for every member of the family
    pieces = np.array_split(np.nonzero((ts >= t - 6.0) & (ts < t - 0.1))[0], m)
    M = np.stack([latent(e, np.zeros(n)) for e in np.eye(2)], 1)            # z linear in x_t
    H = []
    for idx in pieces:
        u = np.zeros(n); u[idx] = 1.0
        H.append(latent(np.zeros(2), u))
    H = np.stack(H, 1)
    rkH = np.linalg.matrix_rank(H, tol=1e-9 * np.abs(H).max())
    # range(H) = R^nz, so for ANY state difference dx there is a past-input difference dw
    # with M dx + H dw = 0: same latent, same current input, different state. The size of
    # the ambiguity per unit of input difference is rho = max |dx| / min |dw| = 1/sigma_min(H^+ M).
    Hp = np.linalg.pinv(H)
    sv = np.linalg.svd(Hp @ M, compute_uv=False)
    rho = 1.0 / sv[-1]
    dx = np.linalg.svd(Hp @ M)[2][-1]                          # the least-protected direction
    dw = -Hp @ M @ dx
    res = np.linalg.norm(M @ dx + H @ dw) / np.linalg.norm(M @ dx)
    print(f"  nz = 5, nx = 2, recent-past input pieces m = {m}: rank(H) = {rkH} = nz")
    print(f"  collision: |dx| = 1 with |dw|_2 = {np.linalg.norm(dw):.1f}, residual {res:.1e} "
          f"(same latent, same current input)")
    print(f"  ambiguity per unit input difference rho = {rho:.2e}: with admissible |u| <= 1 the "
          f"state is undetermined by (z, u(t)) up to ~{rho:.1e}")
    rk = rkH
    assert rkH == 5 and res < 1e-8 and rho > 0


# ---------------------------------------------------------------------------
# Proposition 2: where the input enters (closed-form bilinear example)
# ---------------------------------------------------------------------------

def where_input_enters():
    """Plant xdot = (u - a) x + b, y = x (stable for u < a, equilibrium b/(a - u) > 0);
    latent zdot = -alpha z + y.
      Stationary route: T0(x) = P x + q, P = 1/(alpha - a), q = -P b / alpha; the EXACT
      injection is Phi*(z, u) = J_T0 (f(x,u) - f(x,0)) = u (z - q), so the latent error obeys
      edot = (u - alpha) e: it diverges for constant u > alpha although the plant is stable
      and the injection is exact.
      Decoder-side route: zhat' = -alpha zhat + y (input-free). The exact time-varying map is
      z = m x + n with m' = -alpha m + 1 - (u - a) m, n' = -alpha n - b m; for the tested values
      (a = 3, alpha = 1, u < 2) the coefficient -(alpha + u - a) is POSITIVE, so m, n -> infinity:
      the exact context is unbounded (this is reported, not used). The bounded windowed context of
      Prop. m1-suff(a) is used instead: T_L(x, w) = P_L(u) x + Q_L(u) with
      P_L = int_{-L}^0 e^{(alpha + u - a) tau} dtau, Q_L = int_{-L}^0 e^{alpha tau} b (e^{(u-a) tau} - 1)/(u - a) dtau,
      decoder xhat = (zhat - Q_L)/P_L, error <= (l/2) eps(L), l = 1/P_L (Lemma m1-window)."""
    header("Proposition 2: latent injection vs decoder-side conditioning (bilinear plant)")
    a, b, alpha, T = 3.0, 1.0, 1.0, 30.0
    P = 1.0 / (alpha - a)
    q = -P * b / alpha
    rows = []
    for u0 in (0.5, 0.9, 1.1, 2.0):
        def rhs(t, s):
            x, zl, zd, m, n = s
            y = x
            return [(u0 - a) * x + b,
                    -alpha * zl + y + u0 * (zl - q),             # exact stationary injection
                    -alpha * zd + y,                             # input-free latent filter
                    -alpha * m + 1.0 - (u0 - a) * m,             # context (input history only)
                    -alpha * n - b * m]
        x0, xh0 = 0.2, 0.6                                       # prior x0_hat != x0
        s0 = [x0, P * xh0 + q, 0.0, 0.0, 0.0]
        sol = solve_ivp(rhs, (0, T), s0, method="LSODA", rtol=1e-10, atol=1e-12, dense_output=True)
        tt = np.linspace(5.0, T, 400)
        x, zl, zd, m, n = sol.sol(tt)
        e_lat = np.abs((zl - q) / P - x)
        e_exact = np.abs((zd - n) / m - x).max()                # exact (unbounded) context: still exact
        # bounded windowed context (constant u on the window), L = 6: P_L, Q_L in closed form / quadrature
        L = 6.0
        c = alpha + u0 - a
        PL = (1 - math.exp(-c * L)) / c if abs(c) > 1e-12 else L
        taus = np.linspace(-L, 0, 200001)
        QL = np.trapezoid(np.exp(alpha * taus) * b * (np.exp((u0 - a) * taus) - 1) / (u0 - a), taus)
        e_win = np.abs((zd - QL) / PL - x)[tt >= L].max()
        ybar = max(abs(x).max(), abs(x0))                       # |h| = |x| on the reached set (kappa = |B| = 1)
        bound = 0.5 * (1 / PL) * 2 * ybar * math.exp(-alpha * L) / alpha
        rate = np.polyfit(tt, np.log(e_lat), 1)[0]              # theory: exactly u - alpha
        rows.append((u0, rate, e_exact, e_win, bound, m[-1]))
        print(f"  u = {u0:3.1f}  (alpha = {alpha}, plant pole {u0 - a:+.1f}, x* = {b / (a - u0):.2f}): "
              f"latent-injection error rate {rate:+.4f} (theory {u0 - alpha:+.1f}), "
              f"|e| at T = {e_lat[-1]:8.1e};  decoder-side: exact context error {e_exact:8.1e} (but m(T) = {m[-1]:.1e},"
              f" {'unbounded' if c < 0 else 'bounded'}); windowed context (L = {L:g}, l = 1/P_L = {1 / PL:.2e}) error"
              f" {e_win:8.1e} <= (l/2) eps(L) = {bound:8.1e}")
    assert all(abs(r[1] - (r[0] - alpha)) < 1e-3 for r in rows), rows
    assert all(r[2] < 1e-6 and r[3] <= r[4] * (1 + 1e-3) for r in rows), rows   # the bound is attained (constant input, ODE tol 1e-10)
    assert any(r[5] > 1e3 for r in rows), "the exact context diverges for u < a - alpha"


# ---------------------------------------------------------------------------
# Lemma D: the deployed latent step is second order (ZOH input, y linear interpolation)
# ---------------------------------------------------------------------------

def lemma_d():
    header("Lemma D: sampled-data latent step error is O(dt^2)")
    from hyperkkl.design import butterworth_AB, latent_step
    A, B = butterworth_AB(5, 1.5, 1)
    T = 20.0
    yfun = lambda t: np.sin(1.7 * t) + 0.3 * np.cos(0.4 * t)
    ydd = 1.7 ** 2 + 0.3 * 0.4 ** 2                              # sup |y''|
    sol = solve_ivp(lambda t, z: A @ z + B[:, 0] * yfun(t), (0, T), np.zeros(5), rtol=1e-12, atol=1e-14)
    zT = sol.y[:, -1]
    errs, bounds = [], []
    kappa, a = 1.0, 1.5
    for dt in (0.04, 0.02, 0.01):
        st = latent_step(A, B, dt)
        z = np.zeros(5)
        n = int(round(T / dt))
        for k in range(n):
            z = st["Ad"] @ z + st["W0"][:, 0] * yfun(k * dt) + st["W1"][:, 0] * yfun((k + 1) * dt)
        errs.append(np.linalg.norm(z - zT))
        # Lemma D(b): e_disc = kappa |B| M2 dt^3 / (12 (1 - e^{-alpha dt}))
        bounds.append(kappa * np.linalg.norm(B, 2) * ydd * dt ** 3 / (12 * (1 - math.exp(-a * dt))))
    order = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
    print(f"  errors {np.array(errs)}  observed order {order[0]:.2f}, {order[1]:.2f}")
    print(f"  bound / error: {np.array(bounds) / np.array(errs)}  (>= 1)")
    assert min(order) > 1.9 and all(b >= e for b, e in zip(bounds, errs))


# ---------------------------------------------------------------------------
# Lemma 2 remark: a GRU has no forgetting guarantee
# ---------------------------------------------------------------------------

def gru_memory():
    header("Lemma 2 remark: GRU forgetting rate is not bounded away from zero")
    for b in (0.0, 4.0, 8.0, 12.0):
        zg = 1 / (1 + math.exp(-b))                              # update-gate value with bias b
        l = 1000
        print(f"  gate bias {b:4.1f}: |dh_k/dh_(k-{l})| = z^l = {zg ** l:.3e}  "
              f"(forgetting rate per step {-math.log(zg):.2e})")
    assert (1 / (1 + math.exp(-12.0))) ** 1000 > 0.99


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    lemma1()
    history_needed()
    where_input_enters()
    lemma_d()
    gru_memory()
    print("\nall theory checks passed")
