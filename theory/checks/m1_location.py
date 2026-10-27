"""Block M1 checks (CPU, < 1 min):  CUDA_VISIBLE_DEVICES="" .venv/bin/python theory/checks/m1_location.py

  1. zero-input collapse (Prop. M1.4): E(0)=0 for every encoder, D(z,0)=T0+(z), Phi(z,0)=0.
  2. capacity cap (Remark M1.3): at fixed c, FiLM / LoRA / hyper decoders ARE plain networks of
     the base architecture (explicit weights reproduce the output to fp precision).
  3. latent injection (Prop. M1-where (a)): z-Lipschitz constant of the code's LatentInjection is 0
     at c=0, linear for small |c|, and saturates; two observer copies contract at a rate bounded by
     the sup log-norm mu2(A + d_z Phi) on the visited segments (the small-gain margin is far looser).
  4. rotating oscillator xdot=(w0+u)Jx, y=x1 (affine, |x| conserved for every u): exact stationary
     correction Omega(z)=P J P^+ z (the kkl_mb construction); error rate = max Re eig(A+uM): stable
     inside the small-gain ball, UNSTABLE for reversed rotation (u < -w0) although the plant is
     bounded and observable there; the exact time-varying (decoder-side) map stays exact.
  5. decoder z-Jacobian vs |c| (Prop. M1-where (b)) at RANDOM initialisation (not trained decoders): concat
     obeys its small uniform-in-c bound; film/lora/hyper are also c-uniformly bounded (the tanh trunk
     saturates) but their product-form constant is 10^2-10^6 larger before saturation.
  6. linear plant: T_L(x,w) = P_L x + Q_L w exactly, so an n_z-dim linear context is sufficient.
  7. Koopman reading (Remark m1-koopman): for a linear plant the KKL rows are the resolvent of the Koopman
     generator on the output observable, p_i = b_i C (F - lambda_i I)^{-1}, i.e. T_i = b_i (L_f - lambda_i)^{-1} h.
  8. Prop. M1-where (a), a-priori form: contraction assumed on a ball B(r), r >= |e(0)| + sup|d|/mu, which is
     forward invariant; the error never leaves it (linear injection toy with a bounded disturbance).
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm, solve_sylvester
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperkkl.design import butterworth_AB, spectral_bounds
from hyperkkl.nets import BaseMap, CurrentEncoder, WindowEncoder, GRUEncoder, SFMEncoder, MLP
from hyperkkl.conditioning import ConditionedDecoder, LatentInjection

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
nz, nx, dc = 5, 2, 8


def base():
    b = BaseMap(nz, nx, 32, 3, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx)))
    return b.double()


def randomize(m, s=0.3):
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.normal_(0, s)


def check_collapse():
    print("\n[1] zero-input collapse")
    u0 = torch.zeros(4, 60, 1)
    A, _ = butterworth_AB(nz, 1.5)
    poles = np.linalg.eigvals(A)
    encs = [CurrentEncoder(1, dc, 16), WindowEncoder(1, dc, 16, 10), GRUEncoder(1, dc, 16),
            SFMEncoder(1, dc, 16, 0.05, poles)]
    for e in encs:
        e.double(); randomize(e, 0.5)
        c = e.seq(u0.double())
        print(f"  {type(e).__name__:15s} max|E(0)| = {c.abs().max().item():.1e}")
        assert c.abs().max() < 1e-7
    b = base(); z = torch.randn(64, nz); c0 = torch.zeros(64, dc)
    for k in ("concat", "film", "lora", "hyper"):
        d = ConditionedDecoder(b, k, dc, 16).double(); randomize(d)
        err = (d(z, c0) - b(z)).abs().max().item()
        print(f"  decoder {k:6s} max|D(z,0)-T0+(z)| = {err:.1e}")
        assert err < 1e-12
    inj = LatentInjection(nz, dc, 16, torch.ones(nz)).double(); randomize(inj)
    print(f"  latent injection max|Phi(z,0)| = {inj(z, c0).abs().max().item():.1e}")
    assert inj(z, c0).abs().max() < 1e-12


def explicit_weights(d, c):
    """Weights of the plain base-architecture MLP that d(., c) equals (single context c)."""
    b = d.base
    Ws, bs = [], []
    n = len(b.layers)
    if d.kind == "film":
        gb = d.G(c[None])[0]; off = 0
    else:
        t = d.trunk(c[None])
    for i, layer in enumerate(b.layers):
        W, bb = layer.weight.clone(), layer.bias.clone()
        a, o = layer.in_features, layer.out_features
        if d.kind == "film":
            if i < n - 1:
                h = d.hidden[i]; g, be = gb[off:off + h], gb[off + h:off + 2 * h]; off += 2 * h
                W, bb = (1 + g)[:, None] * W, (1 + g) * bb + be
        elif d.kind == "lora":
            U = d.U[i](t).view(a, d.rank); V = d.V[i](t).view(d.rank, o)
            W = W + (U @ V).T / np.sqrt(d.rank)
        else:
            w = d.H[i](t)[0]
            W = W + w[:a * o].view(o, a); bb = bb + w[a * o:]
        Ws.append(W); bs.append(bb)
    return Ws, bs


def check_cap():
    print("\n[2] capacity cap: weight/feature-space decoders at fixed c are base-architecture MLPs")
    b = base(); z = torch.randn(256, nz)
    for k in ("film", "lora", "hyper"):
        d = ConditionedDecoder(b, k, dc, 16).double(); randomize(d)
        worst = 0.0
        for _ in range(5):
            c = torch.randn(dc)
            Ws, bs = explicit_weights(d, c)
            h = b.inp.norm(z)
            for i, (W, bb) in enumerate(zip(Ws, bs)):
                h = h @ W.T + bb
                if i < len(Ws) - 1:
                    h = torch.tanh(h)
            out = b.out.denorm(h)
            worst = max(worst, (out - d(z, c.expand(256, -1))).abs().max().item())
        print(f"  {k:6s}: max |D(z,c) - MLP_theta(c)(z)| = {worst:.1e}  (same width/depth as T0+)")
        assert worst < 1e-10


def lip_z(inj, c, Z):
    J = torch.func.vmap(torch.func.jacrev(lambda zz, cc: inj(zz[None], cc[None])[0]))(Z, c.expand(len(Z), -1))
    return torch.linalg.matrix_norm(J, 2).max().item()


def check_margin():
    print("\n[3] latent injection: Lipschitz growth with |c| and log-norm contraction rate")
    A, B = butterworth_AB(nz, 1.5)
    kappa, alpha = spectral_bounds(A)
    inj = LatentInjection(nz, dc, 32, torch.ones(nz)).double(); randomize(inj, 0.4)
    Z = 2 * torch.randn(400, nz)
    dirc = torch.randn(dc); dirc /= dirc.norm()
    rs = [0.0, 0.25, 0.5, 1.0, 2.0, 8.0, 32.0]
    ls = [lip_z(inj, r * dirc, Z) for r in rs]
    print("  |c|      : " + "  ".join(f"{r:6.2f}" for r in rs))
    print("  l_Phi(c) : " + "  ".join(f"{l:6.3f}" for l in ls) + "   (0 at c=0, linear for small |c|, saturates)")
    assert ls[0] < 1e-12 and ls[1] < ls[2] < ls[3]
    At = torch.as_tensor(A)
    jac = torch.func.vmap(torch.func.jacrev(lambda zz, cc: inj(zz[None], cc[None])[0]))
    for gain in (0.5, 3.0, 12.0):
        c = 1.0 * dirc
        sc = lambda zz: gain * inj(zz, c.expand(len(zz), -1))
        lphi = gain * lip_z(inj, c, Z)
        z1 = torch.randn(1, nz); z2 = z1 + 1e-3 * torch.randn(1, nz)
        dt, T = 1e-3, 8.0
        e0 = (z1 - z2).norm().item(); mu_sup = -np.inf
        for k in range(int(T / dt)):
            if k % 200 == 0:                       # log-norm of A + d_z Phi on the segment [z2, z1]
                for s_ in (0.0, 0.5, 1.0):
                    J = At + gain * jac(z2 + s_ * (z1 - z2), c[None])[0]
                    mu_sup = max(mu_sup, torch.linalg.eigvalsh(0.5 * (J + J.T)).max().item())
            z1 = z1 + dt * (z1 @ At.T + sc(z1)); z2 = z2 + dt * (z2 @ At.T + sc(z2))
        rate = np.log((z1 - z2).norm().item() / e0) / T
        print(f"  gain {gain:5.1f}: small-gain margin alpha-kappa*l_Phi = {alpha - kappa * lphi:+7.3f}; "
              f"sup log-norm mu2(A+d_zPhi) on visited segments = {mu_sup:+.3f}; measured rate {rate:+.3f}")
        assert rate <= mu_sup + 0.05
    print(f"  decoder-side latent: e' = A e, rate = -alpha = {-alpha:.3f} for every c")


def check_jacobian_growth():
    print("\n[5] decoder z-Jacobian vs |c| at random initialisation: additive (concat) small and uniform in c;"
          " multiplicative (film/lora/hyper) c-uniform too (tanh saturates) but with a far larger constant")
    b = base(); Z = torch.randn(200, nz)
    dirc = torch.randn(dc); dirc /= dirc.norm()
    rs = [0.0, 1.0, 4.0, 16.0, 64.0, 256.0]
    for k in ("concat", "film", "lora", "hyper"):
        d = ConditionedDecoder(b, k, dc, 16).double(); randomize(d, 0.3)
        f = torch.func.vmap(torch.func.jacrev(lambda zz, cc: d(zz[None], cc[None])[0]))
        ls = [torch.linalg.matrix_norm(f(Z, (r * dirc).expand(len(Z), -1)), 2).max().item() for r in rs]
        print(f"  {k:6s}: sup_z ||d_z D(z,c)|| at |c| = {rs}: " + " ".join(f"{l:9.3g}" for l in ls))
        if k == "concat":       # bound uniform in c: l_z(T0+) + 2 sup|d_z R| <= products of spectral norms
            sn = lambda W: float(torch.linalg.matrix_norm(W.detach(), 2))
            base_l = np.prod([sn(l.weight) for l in b.layers])
            Wr = [l.weight for l in d.R.layers]
            r_l = sn(Wr[0][:, :nz]) * np.prod([sn(W) for W in Wr[1:]]) * 1.1 ** (len(Wr) - 1)
            print(f"          uniform-in-c bound l_z(T0+) + 2 l_z(R) = {base_l + 2 * r_l:.3g}")
            assert max(ls) <= base_l + 2 * r_l
            cbound = base_l + 2 * r_l
            print(f"          ratio |c|=256 vs 64: {ls[-1] / ls[-2]:.2f} (saturates below the bound)")
        else:
            print(f"          max over |c| = {max(ls):.3g} vs concat's uniform bound {cbound:.3g}: the product-form "
                  f"constant is {max(ls) / cbound:.0f}x larger; ratio |c|=256 vs 64: {ls[-1] / ls[-2]:.2f} (saturation/collapse, not growth)")
            assert max(ls) > cbound


def check_linear_sufficiency():
    print("\n[6] linear plant: T_L(x,w) = P_L x + Q_L w, so c = Q_L w (dim n_z) is an exact sufficient context")
    F = np.array([[0.0, 1.0], [-1.0, -0.2]]); G = np.array([0.0, 1.0]); C = np.array([[1.0, 0.0]])
    A, B = butterworth_AB(nz, 1.0)
    L, dt = 4.0, 0.01; n = int(L / dt)
    rng = np.random.default_rng(0)
    def TL(x, w):                      # integrate x backwards then (x,z) forwards from t-L, z(t-L)=0
        Pb = expm(-F * dt); Gd = np.linalg.solve(F, np.eye(2) - Pb) @ G
        xs = [x]
        for k in range(n - 1, -1, -1):
            xs.append(Pb @ xs[-1] - Gd * w[k])
        xs = xs[::-1]
        z = np.zeros(nz); Ad, Eh = expm(A * dt), expm(A * dt / 2) @ B
        for k in range(n):
            z = Ad @ z + dt * (Eh @ (C @ (0.5 * (xs[k] + xs[k + 1]))))
        return z
    P = solve_sylvester(-A, F, B @ C)
    PL = P - expm(A * L) @ P @ expm(-F * L)
    PLn = np.stack([TL(e, np.zeros(n)) for e in np.eye(2)], 1)              # numerical P_L
    QL = np.stack([TL(np.zeros(2), e) for e in np.eye(n)[:: n // 40]], 1)   # columns for 40 pulses
    # superposition check with random states / windows built from those pulses
    err = 0
    for _ in range(5):
        x = rng.normal(size=2); a = rng.normal(size=QL.shape[1])
        w = np.zeros(n); w[:: n // 40][:len(a)] = a
        err = max(err, np.linalg.norm(TL(x, w) - (PLn @ x + QL @ a)))
    sv = np.linalg.svd(PLn, compute_uv=False)
    print(f"  max |T_L(x,w) - P_L x - Q_L w| = {err:.1e} (affine in x with w-independent slope); "
          f"sigma_min(P_L) = {sv[-1]:.3f} (ell_T = {1 / sv[-1]:.2f}); |P_L(num) - (P - e^AL P e^-FL)| = "
          f"{np.linalg.norm(PLn - PL, 2):.1e} (quadrature); ||P_L - P|| = {np.linalg.norm(PL - P, 2):.1e}")
    assert err < 1e-6 and sv[-1] > 0


def check_oscillator():
    print("\n[4] rotating oscillator xdot = (w0+u) J x, y = x1 (input-affine, |x| conserved for every u)")
    w0 = 1.0
    J = np.array([[0.0, -1.0], [1.0, 0.0]]); C = np.array([[1.0, 0.0]])
    A, B = butterworth_AB(nz, 1.5)
    kappa, alpha = spectral_bounds(A)
    P = solve_sylvester(-A, w0 * J, B @ C)               # P (w0 J) = A P + B C  (drift KKL map)
    Pp = np.linalg.pinv(P)
    M = P @ J @ Pp                                      # exact correction Omega(z) = M z on range(P)
    usg = alpha / (kappa * np.linalg.norm(M, 2))
    print(f"  alpha = {alpha:.2f}, kappa = {kappa:.3f}, l_Omega = ||P J P^+|| = {np.linalg.norm(M, 2):.3f}: "
          f"small-gain certificate |u| < u_sg = {usg:.3f}")
    for u0 in (0.2, 3.0, -0.5, -0.9, -2.0, -4.0):
        def rhs(t, s):
            x, zl, zd, Pt = s[:2], s[2:7], s[7:12], s[12:].reshape(nz, 2)
            y = C @ x
            return np.concatenate([(w0 + u0) * J @ x,
                                   A @ zl + B @ y + u0 * M @ zl,              # exact stationary injection
                                   A @ zd + B @ y,                            # input-free latent
                                   (A @ Pt + B @ C - (w0 + u0) * Pt @ J).ravel()])  # T(x,t;u) = P(t) x
        x0 = np.array([1.0, 0.0]); xh0 = np.array([0.3, -0.5])
        s0 = np.concatenate([x0, P @ xh0, np.zeros(5), np.zeros(10)])
        sol = solve_ivp(rhs, (0, 16), s0, rtol=1e-10, atol=1e-12, dense_output=True)
        tt = np.linspace(6, 16, 200); S = sol.sol(tt)
        x = S[:2]; el = np.linalg.norm(Pp @ S[2:7] - x, axis=0)
        ed = max(np.linalg.norm(np.linalg.pinv(S[12:, i].reshape(nz, 2)) @ S[7:12, i] - x[:, i]) for i in range(len(tt)))
        rate = np.polyfit(tt, np.log(el), 1)[0]
        pred = np.linalg.eigvals(A + u0 * M).real.max()
        print(f"  u = {u0:+5.1f}: |x| = {np.linalg.norm(x[:, -1]):.3f}, latent-injection error rate {rate:+.3f} "
              f"(max Re eig(A+uM) = {pred:+.3f}); decoder-side (time-varying map) error {ed:.1e}")
        assert abs(rate - pred) < 0.05 and ed < 1e-6
        if abs(u0) < usg:
            assert rate < -(alpha - kappa * np.linalg.norm(M, 2) * abs(u0)) + 1e-6


def check_koopman_resolvent():
    print("\n[7] Koopman resolvent: linear plant, diagonalised A: rows of P are b_i C (F - lambda_i I)^-1")
    F = np.array([[0.0, 1.3], [-1.3, -0.2]]); C = np.array([[1.0, 0.0]])
    A, B = butterworth_AB(nz, 1.5)
    P = solve_sylvester(-A, F, B @ C)                       # P F = A P + B C
    lam, V = np.linalg.eig(A)                               # A = V diag(lam) V^-1
    Pd = np.linalg.solve(V, P)                              # rows of V^-1 P solve lam_i p_i - p_i F = -b_i C
    bd = np.linalg.solve(V, B)[:, 0]
    worst = max(np.linalg.norm(Pd[i] - bd[i] * C[0] @ np.linalg.inv(F - lam[i] * np.eye(2))) for i in range(nz))
    print(f"  max_i |p_i - b_i C (F - lambda_i)^-1| = {worst:.1e}  (T_i = b_i (L_f - lambda_i)^-1 h on linear observables)")
    assert worst < 1e-10
    # Laplace-transform form: T_i(x) = b_i int_0^inf e^{lambda_i tau} h(X(x, -tau)) dtau (backward Koopman semigroup)
    x = np.array([0.7, -0.4]); tt = np.linspace(0, 60, 600001)
    muF, VF = np.linalg.eig(F)
    hX = np.real((C[0] @ VF) @ (np.exp(-np.outer(muF, tt)) * np.linalg.solve(VF, x)[:, None]))   # h(X(x, -tau))
    Ti = np.array([bd[i] * np.trapezoid(np.exp(lam[i] * tt) * hX, tt) for i in range(nz)])
    print(f"  |T(x) - Laplace transform of the backward semigroup| = {np.linalg.norm(Ti - Pd @ x):.1e}")
    assert np.linalg.norm(Ti - Pd @ x) < 1e-6


def check_forward_invariant_ball():
    print("\n[8] Prop. m1-where (a), a-priori form: ball B(r), r = |e(0)| + sup|d|/mu, is forward invariant")
    A, B = butterworth_AB(nz, 1.5)
    rng = np.random.default_rng(3)
    M = rng.standard_normal((nz, nz)) * 0.4                 # linear injection Phi = M z (mu2 depends on M only)
    mu = -np.linalg.eigvalsh(0.5 * ((A + M) + (A + M).T)).max()
    dbar = 0.3
    e = rng.standard_normal(nz); e *= 0.5 / np.linalg.norm(e)
    r = np.linalg.norm(e) + dbar / mu
    dt, worst = 1e-3, 0.0
    for k in range(int(40 / dt)):
        d = dbar * np.sin(np.arange(nz) + 0.7 * k * dt); d *= dbar / np.linalg.norm(d)     # |d| = dbar
        e = e + dt * ((A + M) @ e + d)
        worst = max(worst, np.linalg.norm(e))
    print(f"  mu2(A + d_z Phi) = {-mu:.3f} (mu = {mu:.3f}), |e(0)| = 0.5, sup|d| = {dbar}: r = {r:.3f}, max_t |e(t)| = {worst:.3f}"
          f" <= r; asymptotic bound sup|d|/mu = {dbar / mu:.3f}")
    assert mu > 0 and worst <= r


if __name__ == "__main__":
    check_collapse(); check_cap(); check_margin(); check_oscillator(); check_jacobian_growth(); check_linear_sufficiency()
    check_koopman_resolvent(); check_forward_invariant_ball()
    print("\nall M1 checks passed")
