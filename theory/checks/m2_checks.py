"""Block M2 checks (fading memory, window rule, sampled-data constants, SFM, GRU).
Complements theory/checks/check_theory.py (integrator-plant Lemma 1/1', Lemma D order, GRU gate bias).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python theory/checks/m2_checks.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import expm, solve_sylvester

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperkkl.design import butterworth_AB, latent_step  # noqa: E402

torch.set_default_dtype(torch.float64)


def header(s):
    print("\n" + s + "\n" + "-" * len(s))


def slow_projector(A):
    """Orthogonal projector on the slowest invariant subspace of a normal A."""
    lam, V = np.linalg.eig(A)
    sl = np.isclose(lam.real, lam.real.max())
    Q, _ = np.linalg.qr(np.real_if_close(np.c_[V[:, sl].real, V[:, sl].imag]))
    Q = Q[:, :int(sl.sum())]
    return Q @ Q.T


# --------------------------------------------------------------------------
# Lemma 1 / 1' on a normal Butterworth A (nz = 5) and a linear oscillator
# --------------------------------------------------------------------------
def lemma1_oscillator():
    header("Lemma 1 / 1' : oscillator plant, Butterworth nz = 5, step difference before t-L")
    w0, a, alpha, xnorm = 1.3, 0.5, 1.0, 1.0
    F = np.array([[0, w0], [-w0, 0.]]); G = np.array([[0.], [1.]]); C = np.array([[1., 0.]])
    A, B = butterworth_AB(5, alpha, 1)
    P = solve_sylvester(-A, F, B @ C)                       # P F = A P + B C
    m = np.linalg.solve(A, P @ G)[:, 0] * a                 # Delta T_0 = A^{-1} P G a
    Pi = slow_projector(A)
    Ls = np.linspace(0, 12, 25)
    d = np.array([np.linalg.norm(expm(A * L) @ m) for L in Ls])
    lo = np.exp(-alpha * Ls) * np.linalg.norm(Pi @ m)
    ybar = xnorm + 2 * abs(a) / w0
    up = 2 * np.linalg.norm(B, 2) * ybar * np.exp(-alpha * Ls) / alpha
    # brute-force the definition at two L: T = int_{-inf}^t e^{A(t-s)} B C X(s) ds
    lam, V = np.linalg.eig(A); Vi = np.linalg.solve(V, B[:, 0])
    ss = np.linspace(-60.0, 0.0, 600001)                     # t = 0; e^{-60 alpha} negligible
    eAB = (V[None] * np.exp(-np.outer(ss, lam))[:, None, :]) @ Vi   # e^{A(t-s)} B
    for L in (1.0, 4.0):
        tau = np.clip(-L - ss, 0, None)                    # time spent under the step (backward)
        dX1 = a * (1 - np.cos(w0 * tau)) / w0              # (X_u - X_v)_1, closed form
        quad = np.real(np.trapezoid(eAB * dX1[:, None], ss, axis=0))
        exact = expm(A * L) @ m
        assert np.linalg.norm(quad - exact) < 1e-6 * np.linalg.norm(exact), (quad, exact)
    rate = -np.polyfit(Ls[4:], np.log(d[4:]), 1)[0]
    print(f"  fitted rate {rate:.4f} vs alpha {alpha};  slow-mode lower bound: min d/lo = {np.min(d / lo):.4f} (>=1),"
          f" max d/lo = {np.max(d / lo):.3f};  Lemma-1 slack min up/d = {np.min(up / d):.1f}")
    assert abs(rate - alpha) < 0.02 and np.all(d >= lo * (1 - 1e-12)) and np.all(up >= d)


def lemma1_vdp():
    header("Lemma 1 / 1' : Van der Pol (hyperkkl.systems, saturating actuator), finite-time map")
    from hyperkkl.config import load
    from hyperkkl.systems import build_plant
    cfg = load("vdp")
    plant = build_plant(cfg["system"])
    alpha = 1.0
    A, B = butterworth_AB(5, alpha, 1)
    At, Bt = torch.tensor(A), torch.tensor(B)
    h = 0.005
    f = lambda x, u: plant.f(x, torch.tensor([[u]]))

    def rk4(x, u, h):
        k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
        return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def joint(x, z, u, h):
        def r(x, z):
            return f(x, u), z @ At.T + plant.h(x) @ Bt.T
        a1, b1 = r(x, z); a2, b2 = r(x + h / 2 * a1, z + h / 2 * b1)
        a3, b3 = r(x + h / 2 * a2, z + h / 2 * b2); a4, b4 = r(x + h * a3, z + h * b3)
        return x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4), z + h / 6 * (b1 + 2 * b2 + 2 * b3 + b4)

    xi = torch.tensor([[0.3, -0.2]])
    Tpast = 15.0
    n = int(Tpast / h)
    pasts = {"u": lambda s: 0.4, "v": lambda s: -0.3 * math.sin(2 * s)}
    zedge, ybar = {}, 0.0
    stp = latent_step(A, B, h)                              # exact for y linear on each step
    for key, uf in pasts.items():
        # the backward solution from xi IS the past trajectory (backward vdp is stable near 0);
        # z(t-L) = T_{t0}(xi, t-L; .) = int_{t0}^{t-L} e^{A(t-L-s)} B h(X(s)) ds, z(t0) = 0
        xs = [xi]
        for k in range(n):
            xs.append(rk4(xs[-1], uf(-k * h), -h))
        ys = np.array([float(plant.h(xx)[0, 0]) for xx in xs])[::-1]   # oldest first
        ybar = max(ybar, np.abs(ys).max())
        z = np.zeros(5)
        for k in range(n):
            z = stp["Ad"] @ z + stp["W0"][:, 0] * ys[k] + stp["W1"][:, 0] * ys[k + 1]
        zedge[key] = torch.tensor(z)[None]
    w = lambda s: 0.5 * math.sin(0.7 * s)
    Ls, d = [], []
    xu, zu, zv = xi.clone(), zedge["u"].clone(), zedge["v"].clone()
    xv = xi.clone()
    for k in range(int(12 / h) + 1):
        if k % int(0.5 / h) == 0:
            Ls.append(k * h); d.append(float((zu - zv).norm()))
            ybar = max(ybar, float(plant.h(xu).abs().max()))
        xu, zu = joint(xu, zu, w(k * h), h)
        xv, zv = joint(xv, zv, w(k * h), h)
    Ls, d = np.array(Ls), np.array(d)
    rate = -np.polyfit(Ls[4:], np.log(d[4:]), 1)[0]
    up = 2 * np.linalg.norm(B, 2) * ybar * np.exp(-alpha * Ls) / alpha
    Pi = slow_projector(A)
    lo = np.exp(-alpha * Ls) * np.linalg.norm(Pi @ (zedge["u"] - zedge["v"]).numpy()[0])
    print(f"  fitted rate {rate:.4f} vs alpha {alpha};  min d/lower = {np.min(d / lo):.4f} (>=1);"
          f"  Lemma-1 slack min up/d = {np.min(up / d):.1f} (ybar = {ybar:.2f})")
    assert abs(rate - alpha) < 0.03 and np.all(d >= lo * (1 - 1e-6)) and np.all(up >= d)


# --------------------------------------------------------------------------
# Theorem m2-window (b), linear plant: the fibre diameter delta(L) itself, computed from the
# reachable set Q of the history term q = -int_{-inf}^{t-L} e^{A(t-L-r)} P G u(r) dr, |u| <= ubar.
# Q is a symmetric convex body (zonoid) with support function h_Q(w) = ubar int_0^inf |w' e^{As} P G| ds;
# e^{-AL} P d in Q - Q = 2Q  <=>  gauge_Q(e^{-AL} P d) <= 2.  The gauge is computed by a linear
# programme on a Riemann discretisation of Q (a zonotope):  gauge_Q(v) = min ||u||_inf / (ubar ds)
# s.t. sum_j e^{A s_j} P G u_j = v.  delta(L) = max_theta 2 / gauge_Q(e^{-AL} P e_theta)  (n_x = 2).
# --------------------------------------------------------------------------
def alpha_k_of(A, nx):
    lam = -np.linalg.eigvals(A).real
    re = np.sort(np.unique(np.round(lam, 9)))
    dims = np.cumsum([np.sum(np.isclose(lam, r)) for r in re])
    return re[np.argmax(dims >= A.shape[0] - nx + 1)], re[0], re[-1]


def fibre_diameter():
    header("Theorem m2-window (b): fibre diameter delta(L) by the support function / gauge of the reachable set")
    from scipy.optimize import linprog
    F = np.array([[0, 1.3], [-1.3, 0.]]); G = np.array([[0.], [1.]]); C = np.array([[1., 0.]])
    ubar = 1.0
    for nz in (3, 5):
        A, B = butterworth_AB(nz, 1.0, 1)
        P = solve_sylvester(-A, F, B @ C)
        ak, a1, amax = alpha_k_of(A, 2)
        # zonotope generators g_j = e^{A s_j} P G ds on s in [0, S]
        S, n = 25.0, 500
        ss = np.linspace(0, S, n + 1); ds = ss[1] - ss[0]
        Gz = np.stack([expm(A * s) @ (P @ G)[:, 0] * ds for s in ss], 1)          # (nz, n+1)
        Rq = ubar * sum(np.linalg.norm(expm(A * s) @ (P @ G)[:, 0]) for s in ss) * ds   # radius bound of Q
        # r_q: radius of the largest ball in Q = min_w h_Q(w) over unit w (sampled directions)
        W = np.random.default_rng(0).standard_normal((4000, nz)); W /= np.linalg.norm(W, axis=1, keepdims=True)
        rq = float(np.min(np.abs(W @ Gz).sum(1)) * ubar)

        def gauge(v):
            # min t s.t. Gz u = v, -t ubar <= u_j <= t ubar   (u_j already carries ds)
            m = Gz.shape[1]
            c = np.r_[np.zeros(m), 1.0]
            Aeq = np.c_[Gz, np.zeros((nz, 1))]
            Aub = np.r_[np.c_[np.eye(m), -ubar * np.ones((m, 1))], np.c_[-np.eye(m), -ubar * np.ones((m, 1))]]
            res = linprog(c, A_ub=Aub, b_ub=np.zeros(2 * m), A_eq=Aeq, b_eq=v, bounds=[(None, None)] * m + [(0, None)],
                          method="highs")
            assert res.success, res.message
            return res.x[-1]

        from scipy.optimize import minimize_scalar
        thetas = np.linspace(0, np.pi, 181)[:-1]
        Ls = np.linspace(2.0, 6.0, 9)
        delta, needle, width = [], [], []
        for L in Ls:
            E = expm(-A * L) @ P
            gth = lambda t: gauge(E @ np.array([np.cos(t), np.sin(t)]))
            r = np.array([2.0 / gth(t) for t in thetas])
            # the maximiser is a needle whose angular width shrinks like e^{-(alpha_max - alpha_k) L}:
            # refine the coarse-grid maximum by a bounded 1-D minimisation of the gauge
            t0 = thetas[np.argmax(r)]
            res = minimize_scalar(gth, bounds=(t0 - np.pi / 180, t0 + np.pi / 180), method="bounded",
                                  options={"xatol": 1e-12})
            delta.append(2.0 / res.fun); needle.append(res.x % np.pi)
            width.append(np.mean(r >= 0.5 * delta[-1]) * np.pi)      # angular width at half height
        delta = np.array(delta)
        rate = -np.polyfit(Ls, np.log(delta), 1)[0]
        # the direction with P d in the slow span S_k (predicted needle) and the rate along it
        lam, V = np.linalg.eig(A)
        slow = np.abs(-lam.real - ak) < 1e-6
        slow |= -lam.real < ak - 1e-6
        Vs = np.c_[V[:, slow].real, V[:, slow].imag]
        Qs, _ = np.linalg.qr(Vs); Qs = Qs[:, :int(np.linalg.matrix_rank(Vs))]
        Pi = np.eye(nz) - Qs @ Qs.T                        # projector on the fast complement
        d_needle = np.linalg.svd(Pi @ P)[2][-1]            # P d_needle in S_k
        th_needle = np.arctan2(d_needle[1], d_needle[0]) % np.pi
        r_perp = [2.0 / gauge(expm(-A * L) @ P @ np.array([-np.sin(th_needle), np.cos(th_needle)])) for L in Ls]
        rate_perp = -np.polyfit(Ls, np.log(r_perp), 1)[0]
        lo = rq * np.exp(-ak * Ls) / np.linalg.norm(P, 2)
        up = 2 * Rq * np.exp(-a1 * Ls) / np.linalg.svd(P, compute_uv=False)[-1]
        print(f"  nz = {nz}: delta(L) decays at rate {rate:.3f} (alpha_k = {ak:.3f}; alpha = {a1:.3f}, alpha_max = {amax:.3f});"
              f" needle direction theta = {np.degrees(needle[-1]):.2f} deg (predicted {np.degrees(th_needle):.2f} deg),"
              f" half-height width {np.degrees(width[0]):.2f} -> {np.degrees(width[-1]):.3f} deg over L = {Ls[0]:.0f}..{Ls[-1]:.0f};"
              f" orthogonal direction: rate {rate_perp:.3f} (~alpha_max)")
        print(f"          lower bound r_q e^(-alpha_k L)/|P| <= delta(L) <= 2 R_q e^(-alpha L)/sigma_min(P):"
              f" min delta/lo = {np.min(delta / lo):.2f}, min up/delta = {np.min(up / delta):.1f}")
        assert abs(rate - ak) < 0.05 * ak and np.all(delta >= lo * (1 - 1e-9)) and np.all(up >= delta)
        assert abs((needle[-1] - th_needle + np.pi / 2) % np.pi - np.pi / 2) < np.radians(2.0)
        if nz == 5:
            assert rate_perp > ak + 0.3
        # arithmetic cross-check: sigma_min(e^{-AL} P) has the same rate
        g = [np.linalg.svd(expm(-A * L) @ P, compute_uv=False)[-1] for L in np.linspace(4, 14, 11)]
        rate_sv = np.polyfit(np.linspace(4, 14, 11), np.log(g), 1)[0]
        print(f"          sigma_min(e^(-AL) P) grows at rate {rate_sv:.3f} (same alpha_k)")
        assert abs(rate_sv - ak) < 1e-2


# --------------------------------------------------------------------------
# Window rule with the ln(l) term (Theorem m2-window (a)): per-system table and comparison with the
# C2-window-a1 sweep (read-only from the node report; lora@window, sequence_ood and id, NRMSE).
# --------------------------------------------------------------------------
C2_WINDOW_A1 = {   # experiments/tree/C2-window-a1/report.md, scoreboards (L = 1..512 samples)
    "sequence_ood": {
        "battery":    [0.3469, 0.3365, 0.3106, 0.2981, 0.2722, 0.2373, 0.1883, 0.1613, 0.1081, 0.0625],
        "bioreactor": [1.4752, 1.3101, 1.1953, 0.9757, 0.4280, 0.0670, 0.0402, 0.0390, 0.0526, 0.0930],
        "duffing":    [0.0962, 0.0743, 0.0680, 0.0588, 0.0463, 0.0408, 0.0388, 0.0376, 0.0390, 0.0397],
        "lorenz":     [0.1041, 0.1005, 0.0979, 0.0952, 0.0910, 0.0837, 0.0688, 0.0354, 0.0231, 0.0215],
        "quad_tank":  [0.2598, 0.2670, 0.2523, 0.2306, 0.2434, 0.1962, 0.1391, 0.1061, 0.0887, 0.0820],
        "vdp":        [0.0453, 0.0361, 0.0336, 0.0291, 0.0229, 0.0179, 0.0164, 0.0158, 0.0168, 0.0176]},
    "id": {
        "battery":    [0.1564, 0.1559, 0.1465, 0.1393, 0.1370, 0.1296, 0.1214, 0.1043, 0.0686, 0.0420],
        "bioreactor": [0.0764, 0.0907, 0.0529, 0.0414, 0.0277, 0.0233, 0.0186, 0.0177, 0.0180, 0.0223],
        "duffing":    [0.0525, 0.0486, 0.0454, 0.0416, 0.0386, 0.0367, 0.0345, 0.0342, 0.0341, 0.0296],
        "lorenz":     [0.0348, 0.0341, 0.0323, 0.0328, 0.0316, 0.0317, 0.0307, 0.0273, 0.0276, 0.0208],
        "quad_tank":  [0.1117, 0.1135, 0.1112, 0.1012, 0.0964, 0.0905, 0.0851, 0.0675, 0.0658, 0.0658],
        "vdp":        [0.0270, 0.0244, 0.0229, 0.0204, 0.0168, 0.0142, 0.0141, 0.0142, 0.0148, 0.0148]},
}
L_GRID = 2 ** np.arange(10)
# phase-1 l_dec_q95 of the C2-window-a1 jobs themselves (paper/generated/RESULTS.md, section 2; the base decoder's
# random-direction quotient, a proxy for l). The refined rule is POST HOC: it was not pre-registered in node.yaml.
L_DEC_Q95 = {"battery": 806, "bioreactor": 44, "duffing": 295, "lorenz": 10, "quad_tank": 14, "vdp": 23}


def window_rule_table():
    header("Window rule L*(eta) = (ln(1/eta) + ln l)/alpha vs the C2-window-a1 sweep (eta = 0.05, l = l_dec_q95; POST HOC)")
    import yaml
    root = Path(__file__).resolve().parents[2]
    eta = 0.05
    print(f"  {'system':10s} {'nz':>3s} {'alpha':>6s} {'a*dt':>7s} {'l':>6s} {'L0 [smp]':>9s} {'L* [smp]':>9s}"
          f"  {'seq: e(L>=L0)/e512 ok?':>22s} {'e(L>=L*)/e512 ok?':>18s}  knee(seq) monotone?")
    viol0, violl, mono_n, worst_l, worst_0 = 0, 0, 0, 0.0, 0.0
    for name in ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"):
        sc = yaml.safe_load(open(root / "configs" / "systems" / f"{name}.yaml"))
        nz, al = int(sc["nz"]), float(sc["alpha"])
        dt = float(sc.get("dt") or 0.1 * math.sin(math.pi / (2 * nz)) / al)
        adt = al * dt
        l = L_DEC_Q95[name]
        L0 = math.log(1 / eta) / adt
        Ls = (math.log(1 / eta) + math.log(l)) / adt
        e = np.array(C2_WINDOW_A1["sequence_ood"][name]); e512 = e[-1]
        # monotone (within 10%) on the dyadic grid: no window is worse than a shorter one by > 10%
        mono = all(e[i + 1] <= 1.10 * e[i] for i in range(len(e) - 1))
        mono_n += mono
        def ok(Lstar):
            idx = L_GRID >= Lstar
            if not idx.any():
                return "grid too short", 0.0
            worst = (e[idx] / e512).max()
            return f"{worst:.2f} {'ok' if worst <= 1.05 else 'VIOLATED'}", worst
        (o0, w0), (ol, wl) = ok(L0), ok(Ls)
        if mono:
            viol0 += "VIOLATED" in o0; violl += "VIOLATED" in ol
            worst_0, worst_l = max(worst_0, w0), max(worst_l, wl)
        print(f"  {name:10s} {nz:3d} {al:6.2f} {adt:7.4f} {l:6.1f} {L0:9.0f} {Ls:9.0f}  {o0:>22s} {ol:>18s}  {mono}")
    print(f"  monotone systems: {mono_n}/6; l-free rule violated (5% criterion) on {viol0} of them (worst ratio {worst_0:.2f});"
          f" rule with ln l: {violl} (worst ratio {worst_l:.2f})")
    # the claim in the text: the l-free rule fails by a large factor where l is large (battery), the rule with ln l
    # is an upper bound on the knee within 10% on every monotone system that the grid can resolve
    assert worst_l <= 1.10 and worst_0 >= 1.5


def sfm_history_swap():
    header("Lemma m2-sfm (c): whole-history swap (the memory probe) and the initial fill-in modes")
    from hyperkkl.nets import SFMEncoder
    A, _ = butterworth_AB(5, 1.5, 1)
    poles = np.linalg.eigvals(A)
    alpha = -poles.real.max()
    dt = 0.1 * math.sin(math.pi / 10) / 1.5
    for floor in (0.0, 0.5):
        torch.manual_seed(1)
        E = SFMEncoder(1, 16, 64, dt, poles, 8, floor=floor).double()
        with torch.no_grad():
            E.w_in.add_(0.5 * torch.randn_like(E.w_in))
        a_xi = E.alpha()
        sig = sorted(float(s) for s in -E.sigma().detach())
        print(f"  floor {floor}: alpha_xi = {a_xi:.4f} = {a_xi / alpha:.3f} alpha;  block rates / alpha:"
              f" {np.round(np.array(sig) / alpha, 3)}")
        if floor == 0.0:
            assert abs(a_xi - alpha / 1.5 ** 5) < 1e-6, "slowest fill-in mode at initialisation is alpha / 1.5^5"
        else:
            assert a_xi >= floor * alpha - 1e-9
        lp = E.psi_lipschitz()
        bd = float(torch.linalg.matrix_norm(E._ad_bd()[1].reshape(-1, 1), 2))
        N, ubar = 1500, 1.0
        u = ubar * torch.sign(torch.randn(N, 1)); u2 = ubar * torch.sign(torch.randn(N, 1))
        worst = 0.0
        for L in (50, 100, 200, 400, 800):
            uswap = u.clone(); uswap[:N - L] = u2[:N - L]          # same last L samples, older history swapped
            dc = float((E.seq(u[None])[0, -1] - E.seq(uswap[None])[0, -1]).norm())
            bound = lp * bd * (2 * ubar) * math.exp(-a_xi * L * dt) / (1 - math.exp(-a_xi * dt))
            worst = max(worst, dc / bound)
            if L in (50, 400):
                print(f"    L = {L:4d} samples: |c - c'| = {dc:.3e} <= l_psi |B_d| 2 ubar e^(-alpha_xi L dt)/(1-e^(-alpha_xi dt)) = {bound:.3e}")
        print(f"    max observed/bound over L = {worst:.3f} (<= 1)")
        assert worst <= 1.0


def history_rank_vdp():
    header("Prop. current input (c), nonlinear local: rank of d T(xi, t-L; u_theta) / d theta, Van der Pol")
    from hyperkkl.config import load
    from hyperkkl.systems import build_plant
    plant = build_plant(load("vdp")["system"])
    A, B = butterworth_AB(5, 1.0, 1)
    h, Tpast, npc = 0.005, 8.0, 12
    n = int(Tpast / h)
    stp = latent_step(A, B, h)
    xi = torch.tensor([[0.3, -0.2]])
    f = lambda x, u: plant.f(x, torch.tensor([[u]]))

    def rk4(x, u, hh):
        k1 = f(x, u); k2 = f(x + hh / 2 * k1, u); k3 = f(x + hh / 2 * k2, u); k4 = f(x + hh * k3, u)
        return x + hh / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def edge(theta):          # T^{t0}(xi, t-L; u_theta): pasts = npc constant pieces, backward from xi
        xs = [xi]
        for k in range(n):
            xs.append(rk4(xs[-1], float(theta[min(k * npc // n, npc - 1)]), -h))
        ys = np.array([float(plant.h(x)[0, 0]) for x in xs])[::-1]
        z = np.zeros(5)
        for k in range(n):
            z = stp["Ad"] @ z + stp["W0"][:, 0] * ys[k] + stp["W1"][:, 0] * ys[k + 1]
        return z
    th0 = 0.2 * np.ones(npc)
    e = 1e-5
    J = np.stack([(edge(th0 + e * ei) - edge(th0 - e * ei)) / (2 * e) for ei in np.eye(npc)], 1)
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  singular values of d z / d theta: {sv}  -> rank {int((sv > 1e-6 * sv[0]).sum())} = nz:"
          f" every x near x(t) is reached with the same latent and the same window")
    assert (sv > 1e-6 * sv[0]).sum() == 5


# --------------------------------------------------------------------------
# Lemma D (discrete constants): ZOH, sums with e^{-alpha dt}
# --------------------------------------------------------------------------
def lemma_d_constants():
    header("Lemma D: discrete constants (ZOH linear oscillator, deployed filter)")
    w0, alpha, a = 1.3, 1.0, 0.5
    F = np.array([[0, w0], [-w0, 0.]]); G = np.array([[0.], [1.]]); C = np.array([[1., 0.]])
    A, B = butterworth_AB(5, alpha, 1)
    P = solve_sylvester(-A, F, B @ C)
    for dt in (0.0309, 0.1):
        st = latent_step(A, B, dt)
        Ad = st["Ad"]
        # (i) exact map at sample times, step of size a before sample k-M: Delta = Ad^M A^{-1} P G a
        Gam = np.linalg.solve(A, Ad - np.eye(5))
        M = 30
        S = sum(np.linalg.matrix_power(Ad, l) @ Gam @ P @ G[:, 0] * a for l in range(M, M + 4000))
        ref = np.linalg.matrix_power(Ad, M) @ np.linalg.solve(A, P @ G[:, 0]) * a
        e1 = np.linalg.norm(-S - ref) / np.linalg.norm(ref)
        # (ii) the deployed filter's map T^d driven by samples of the backward trajectories
        Fd = expm(F * dt); Gd = np.linalg.solve(F, Fd - np.eye(2)) @ G
        K = 6000; x = np.array([1.0, 0.0])
        def ys(uhist):                                      # backward exact ZOH, y_{k-j}, j=0..K
            xs = [x]
            for j in range(K):
                xs.append(np.linalg.solve(Fd, xs[-1] - Gd[:, 0] * uhist(j + 1)))
            return np.array([C @ xx for xx in xs])[:, 0]
        yu = ys(lambda j: a if j > M else 0.0); yv = ys(lambda j: 0.0)
        def Td(y):                                          # z_k = sum_j Ad^j (W0 y_{k-1-j} + W1 y_{k-j})
            z = np.zeros(5)
            for j in range(K - 1, -1, -1):
                z = Ad @ z + st["W0"][:, 0] * y[j + 1] + st["W1"][:, 0] * y[j]
            return z
        dTd = np.linalg.norm(Td(yu) - Td(yv))
        ybar = max(np.abs(yu).max(), np.abs(yv).max())
        h = alpha * dt
        wa = (1 - math.exp(-h) * (1 + h)) / (alpha * h)     # w0 * alpha
        wb = (h - 1 + math.exp(-h)) / (alpha * h)           # w1 * alpha
        cd = (wa + math.exp(-h) * wb) / (alpha * (1 - math.exp(-h)))
        upd = 2 * np.linalg.norm(B, 2) * ybar * math.exp(-alpha * M * dt) * cd
        print(f"  dt = {dt}: exact-ZOH step identity rel.err {e1:.1e};  discrete constant c_d = {cd:.4f}"
              f" vs 1/alpha = {1 / alpha:.4f};  |dT^d| = {dTd:.3e} <= bound {upd:.3e}"
              f" (cont. {np.linalg.norm(ref):.3e})")
        assert e1 < 1e-8 and cd <= 1 / alpha and dTd <= upd


# --------------------------------------------------------------------------
# Lemma 2: SFM, tight case and l_E; GRU latch (no fading memory)
# --------------------------------------------------------------------------
def lemma2_sfm():
    header("Lemma 2: SFMEncoder per-lag bound (tight construction) and l_E")
    from hyperkkl.nets import SFMEncoder
    A, _ = butterworth_AB(5, 1.0, 1)
    poles = np.linalg.eigvals(A)
    dt, N = 0.05, 200
    E = SFMEncoder(1, 4, 8, dt, poles, 4).double()
    b = int(torch.argmax(E.sigma()))
    with torch.no_grad():
        E.omega[b] = 2.0; E.w_in.zero_(); E.w_in[b, 0, 0], E.w_in[b, 1, 0] = 0.6, 0.8
        for W in (l.weight for l in E.psi.layers):
            W.zero_()
        W1, W2, W3 = (l.weight for l in E.psi.layers)
        W1[0, 2 * b], W1[1, 2 * b + 1] = 1.0, 1.0
        W2[0, 0], W2[1, 1], W3[0, 0], W3[1, 1] = 1.0, 1.0, 1.0, 1.0
    J = torch.autograd.functional.jacobian(lambda v: E.seq(v[None])[0, -1], torch.zeros(N, 1))[..., 0]
    lp, a = E.psi_lipschitz(), E.alpha()
    bd = float(torch.linalg.matrix_norm(E._ad_bd()[1].reshape(-1, 1), 2))
    r = [float(J[:, N - 1 - l].norm()) / (lp * bd * math.exp(-a * (l - 1) * dt)) for l in range(1, N)]
    print(f"  observed/bound over l = 1..{N - 1}: [{min(r):.10f}, {max(r):.10f}]  (alpha_xi = {a:.3f})")
    assert max(abs(x - 1) for x in r) < 1e-9
    # random SFM: bound holds; l_E bound holds for constant worst-case perturbation
    torch.manual_seed(0)
    E2 = SFMEncoder(1, 16, 64, dt, poles, 8).double()
    with torch.no_grad():
        E2.w_in.add_(0.5 * torch.randn_like(E2.w_in))
    u = torch.randn(N, 1)
    J2 = torch.autograd.functional.jacobian(lambda v: E2.seq(v[None])[0, -1], u)[..., 0]
    lp2, a2 = E2.psi_lipschitz(), E2.alpha()
    bd2 = float(torch.linalg.matrix_norm(E2._ad_bd()[1].reshape(-1, 1), 2))
    worst = max(float(J2[:, N - 1 - l].norm()) / (lp2 * bd2 * math.exp(-a2 * (l - 1) * dt)) for l in range(1, N))
    lE = lp2 * (1 + bd2 / (1 - math.exp(-a2 * dt)))
    dlt = 1e-4
    fd = float((E2.seq((u + dlt)[None])[0, -1] - E2.seq(u[None])[0, -1]).norm()) / dlt
    print(f"  random SFM: max observed/bound = {worst:.3e} (<= 1);  l_E finite difference {fd:.3f} <= bound {lE:.1f}")
    assert worst <= 1 and fd <= lE


def gru_latch():
    header("Lemma 2 remark: a GRU without fading memory (latch), vs SFM")
    from hyperkkl.nets import GRUEncoder
    E = GRUEncoder(1, 1, 1).double()
    with torch.no_grad():
        for p in E.parameters():
            p.zero_()
        # PyTorch gate order (r, z, n)
        E.rnn.bias_hh_l0[0] = 30.0            # r = 1
        E.rnn.bias_hh_l0[1] = -30.0           # z = 0: h' = n
        E.rnn.weight_hh_l0[2, 0] = 3.0        # n = tanh(3 h + 2 u): bistable
        E.rnn.weight_ih_l0[2, 0] = 2.0
        E.rnn.bias_ih_l0[2] = 0.0
        E.proj.weight[0, 0] = 1.0
    N = 5000
    u = torch.zeros(1, N, 1); u[0, 10, 0] = -3.0     # one pulse at k = 10, then zero
    # reference h(0) sits at the fixed point h=0 (unstable but exact at zero input); perturb it
    # to the + well by starting both paths from the default zero state with a tiny bias:
    with torch.no_grad():
        E.rnn.bias_ih_l0[2] = 1e-3
        c = E.seq(u)[0, :, 0]
    print(f"  |c_k(pulse) - c_k(0)| at k = 100, 1000, {N - 1}: "
          f"{abs(float(c[100])):.3f}, {abs(float(c[1000])):.3f}, {abs(float(c[-1])):.3f}  (no decay)")
    assert abs(float(c[-1])) > 1.0


# --------------------------------------------------------------------------
# C2 predictions: windows in samples vs seconds
# --------------------------------------------------------------------------
def c2_predictions():
    header("C2-window predictions (dev systems; grid L in {1,...,512} samples)")
    import json
    import yaml
    root = Path(__file__).resolve().parents[2]
    cache = root / "experiments" / "cache" / "scales"
    grid = 2 ** np.arange(10)
    print(f"  {'system':10s} {'s':>4s} {'alpha':>6s} {'dt':>8s} {'a*dt':>7s} {'smp/e-fold':>10s}"
          f" {'a*L(512)':>8s} {'L1 bound [s]':>12s} {'L1 bound [smp]':>14s} {'a2/a':>5s}")
    for name in ("vdp", "duffing", "bioreactor", "lorenz", "quad_tank"):
        sc = yaml.safe_load(open(root / "configs" / "systems" / f"{name}.yaml"))
        nz, a1 = int(sc["nz"]), float(sc["alpha"])
        # read-only: the cached scales at alpha_scale = 1 (y_bar, z_scale)
        best = None
        for p in cache.glob(f"{name}_*.json"):
            dd = json.loads(p.read_text())
            if abs(dd["design_check"]["alpha"] - a1) < 1e-9:
                best = dd
        k = np.arange(1, nz + 1)
        re = np.sort(np.unique(np.round(np.sin((2 * k - 1) * np.pi / (2 * nz)), 12)))
        for s in (0.5, 1.0, 2.0):
            al = a1 * s
            dt = float(sc.get("dt") or 0.1 * math.sin(math.pi / (2 * nz)) / al)
            A, B = butterworth_AB(nz, al, len(sc["sensors"]))
            Bn = np.linalg.norm(B, 2)
            # Lemma-1 window at eps = 1e-3 z_scale; z_scale(s) ~ z_scale(1)/s (z ~ -A^{-1}B y)
            zs = best["z_scale"] / s
            L1 = math.log(2 * Bn * best["y_bar"] / (al * 1e-3 * zs)) / al
            print(f"  {name:10s} {s:4.1f} {al:6.3f} {dt:8.5f} {al * dt:7.4f} {1 / (al * dt):10.1f}"
                  f" {al * 511 * dt:8.2f} {L1:12.2f} {L1 / dt:14.0f} {re[1] / re[0]:5.2f}")


if __name__ == "__main__":
    lemma1_oscillator()
    lemma1_vdp()
    fibre_diameter()
    window_rule_table()
    history_rank_vdp()
    lemma_d_constants()
    lemma2_sfm()
    sfm_history_swap()
    gru_latch()
    c2_predictions()
    print("\nall M2 checks passed")
