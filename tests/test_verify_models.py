"""Numerical verification of the models: hyperkkl/nets.py and hyperkkl/conditioning.py (C1).

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest tests/test_verify_models.py -q -s

Every model is compared with an explicit reference computation of what its docstring
says (numpy float64, per sample). With -s the measured numbers are printed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm
from scipy.special import erf

from hyperkkl import design as D
from hyperkkl.conditioning import ALL_KINDS, build_conditioner, match_width
from hyperkkl.nets import ACT, MLP, Affine, BaseMap, SFMEncoder, build_context, n_params

torch.set_num_threads(min(2, torch.get_num_threads()))   # tiny ops: more threads only contend
CUDA = torch.cuda.is_available()
DEVS = ["cpu"] + (["cuda"] if CUDA else [])
GPU = "cuda" if CUDA else "cpu"
CTX_KINDS = ["current", "window", "gru", "sfm"]
COND_KINDS = [k for k in ALL_KINDS if k != "none"]
CCFG = {"dim": 16, "hidden": 64, "window": 50, "sfm_blocks": 8}      # configs/base.yaml
SMALL = {"dim": 4, "hidden": 12, "window": 7, "sfm_blocks": 4}       # for full Jacobians
BUDGET, RANK, DC = 140000, 4, 16


@pytest.fixture(autouse=True)
def _float32_default():
    """Other test modules set the global default dtype at import; pin it here."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(old)


def _np(t):
    return t.detach().cpu().double().numpy()


def _design(nz, alpha=1.0):
    A, _ = D.butterworth_AB(nz, alpha)
    return A, np.linalg.eigvals(A), D.choose_dt(A)


def _ctx(kind, ccfg=CCFG, nz=5, u_ref=1.0, seed=0):
    torch.manual_seed(seed)
    _, poles, dt = _design(nz)
    return build_context(kind, ccfg, 1, dt, poles, u_ref)


def _run_step(E, u):
    st = E.init_state(u.shape[0], u.device)
    cs = []
    for k in range(u.shape[1]):
        c, st = E.step(u[:, k], st)
        cs.append(c)
    return torch.stack(cs, 1)


def _base(nz, nx, seed=0):
    """Base decoder of phase 1 (3 x 128, tanh) with non-trivial normalisation stats."""
    g = torch.Generator().manual_seed(seed)
    s_in = (torch.randn(nz, generator=g), 0.5 + 2 * torch.rand(nz, generator=g))
    s_out = (torch.randn(nx, generator=g), 0.5 + 2 * torch.rand(nx, generator=g))
    torch.manual_seed(seed)
    return BaseMap(nz, nx, 128, 3, "tanh", s_in, s_out)


def _randomise(m, std=0.1, seed=1):
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        if p.requires_grad:
            p.data.copy_(torch.randn(p.shape, generator=g, dtype=p.dtype) * std)


def _cond(kind, nz=5, nx=2, randomise=True):
    base = _base(nz, nx)
    zs = 0.5 + torch.rand(nz, generator=torch.Generator().manual_seed(3))
    w, npar = match_width(kind, base, DC, BUDGET, RANK, zs)
    m = build_conditioner(kind, base, DC, w, RANK, zs)
    if randomise:
        _randomise(m)
    return base, m, w, npar


NP_ACT = {"tanh": np.tanh, "silu": lambda v: v / (1 + np.exp(-v)),
          "gelu": lambda v: 0.5 * v * (1 + erf(v / math.sqrt(2))), "softplus": lambda v: np.logaddexp(0, v)}


def _mlp_ref(m: MLP, x, act):
    h = x
    for i, l in enumerate(m.layers):
        h = _np(l.weight) @ h + (_np(l.bias) if l.bias is not None else 0.0)
        if i < len(m.layers) - 1:
            h = NP_ACT[act](h)
    return h


# ---------------------------------------------------------------------------
# MLP / BaseMap / Affine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("act", list(ACT))
def test_mlp_layers_activations_and_zero_last(act, bias):
    torch.manual_seed(0)
    m = MLP(5, 3, 17, 3, act, bias=bias).double()
    assert [(l.in_features, l.out_features) for l in m.layers] == [(5, 17), (17, 17), (17, 17), (17, 3)]
    assert all((l.bias is not None) == bias for l in m.layers)
    x = torch.randn(9, 5, dtype=torch.float64)
    ref = np.stack([_mlp_ref(m, xi, act) for xi in x.numpy()])
    assert np.abs(m(x).detach().numpy() - ref).max() < 1e-12
    mz = MLP(5, 3, 17, 2, act, bias=bias, zero_last=True)
    assert mz.layers[-1].weight.abs().max() == 0 and (not bias or mz.layers[-1].bias.abs().max() == 0)
    assert mz.layers[0].weight.abs().max() > 0
    if not bias and act != "softplus":            # bias-free + act(0) = 0 -> zero-preserving
        assert m(torch.zeros(4, 5, dtype=torch.float64)).abs().max() == 0


def test_basemap_normalisation_roundtrip_and_forward():
    base = _base(5, 2).double()
    x = torch.randn(200, 5, dtype=torch.float64) * 3 + 1
    y = torch.randn(200, 2, dtype=torch.float64)
    e_in = (base.inp.denorm(base.inp.norm(x)) - x).abs().max().item()
    e_out = (base.out.norm(base.out.denorm(y)) - y).abs().max().item()
    a32 = Affine(torch.randn(5), 0.5 + torch.rand(5))
    x32 = torch.randn(200, 5) * 3
    e32 = ((a32.denorm(a32.norm(x32)) - x32).abs().max() / x32.abs().max()).item()
    print(f"\n  Affine round trip: float64 {e_in:.1e} / {e_out:.1e}, float32 relative {e32:.1e}")
    assert e_in < 1e-13 and e_out < 1e-13 and e32 < 1e-6
    manual = base.mlp((x - base.inp.mean) / base.inp.std) * base.out.std + base.out.mean
    assert torch.equal(base(x), manual)
    assert len(base.layers) == 4 and base.act is torch.tanh            # depth 3 = 3 hidden layers
    assert n_params(base) == 5 * 128 + 128 + 2 * (128 * 128 + 128) + 128 * 2 + 2
    assert len(list(base.inp.parameters())) == 0 and len(list(base.out.parameters())) == 0  # buffers
    assert Affine(torch.zeros(2), torch.tensor([0.0, 2.0])).std[0].item() == pytest.approx(1e-8)


# ---------------------------------------------------------------------------
# Context encoders
# ---------------------------------------------------------------------------

def _jacobian(E, u, path):
    """Jn[k, j] = || d c_k / d u_j || for one episode u (N, 1), float64."""
    E = E.double()

    def f(v):
        v = v[None]
        return (E.seq(v) if path == "seq" else _run_step(E, v))[0]
    J = torch.autograd.functional.jacobian(f, u.double())[..., 0]      # (N, dc, N)
    return J, J.norm(dim=1)


@pytest.mark.parametrize("path", ["seq", "step"])
@pytest.mark.parametrize("kind", CTX_KINDS)
def test_encoder_causality_and_memory_support(kind, path):
    E = _ctx(kind, SMALL)
    N, L = 48, SMALL["window"]
    u = torch.randn(N, 1, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    _, Jn = _jacobian(E, u, path)
    k, j = np.indices((N, N))
    Jn = Jn.numpy()
    future = Jn[j > k].max()
    print(f"\n  {kind}/{path}: max ||dc_k/du_j|| future {future:.1e} (past max {Jn.max():.2e})")
    if kind == "sfm" and path == "seq":
        assert future <= 1e-13 * Jn.max()            # FFT round-off only
    else:
        assert future == 0.0
    if kind == "current":
        assert Jn[j != k].max() == 0.0 and Jn[j == k].min() > 0
    elif kind == "window":
        assert Jn[j <= k - L].max() == 0.0                       # nothing older than L samples
        assert Jn[(j > k - L) & (j <= k)].min() > 0              # every one of the last L counts
    else:                                                        # unbounded memory
        assert Jn[j <= k].min() > 0


@pytest.mark.parametrize("kind", CTX_KINDS)
def test_encoder_causality_by_perturbation_long_float32(kind):
    """N = 8001 samples in float32 on the GPU: perturb u_j, look at c_k for k < j."""
    E = _ctx(kind).to(GPU)
    N1, j = 8001, 5000
    u = torch.randn(2, N1, 1, device=GPU, generator=torch.Generator(GPU).manual_seed(0))
    u2 = u.clone()
    u2[:, j] += 1.0
    with torch.no_grad():
        ds = (E.seq(u2) - E.seq(u)).abs().amax((0, 2)).cpu()
        dt_ = (_run_step(E, u2) - _run_step(E, u)).abs().amax((0, 2)).cpu()
    print(f"\n  {kind}: max |dc_k| for k < j: seq {ds[:j].max():.1e}, step {dt_[:j].max():.1e}; "
          f"at k = j: {ds[j]:.2e}")
    assert dt_[:j].max() == 0.0
    assert ds[:j].max() <= (1e-5 if kind == "sfm" else 0.0)
    assert ds[j] > 1e-3 and dt_[j] > 1e-3
    if kind == "current":
        assert ds[j + 1:].max() == 0 and dt_[j + 1:].max() == 0
    if kind == "window":
        L = CCFG["window"]
        assert ds[j + L:].max() == 0 and dt_[j + L:].max() == 0 and dt_[j + L - 1] > 0


@pytest.mark.parametrize("kind", CTX_KINDS)
def test_seq_equals_step_long_float32(kind):
    E = _ctx(kind).to(GPU)
    u = 2 * torch.randn(4, 8001, 1, device=GPU, generator=torch.Generator(GPU).manual_seed(1))
    with torch.no_grad():
        cs, ct = E.seq(u), _run_step(E, u)
        E64 = _ctx(kind).to(GPU).double()
        E64.load_state_dict({k: v.double() for k, v in E.state_dict().items()})
        ref = E64.seq(u.double())
    d = (cs - ct).abs().max().item()
    es, et = (cs - ref).abs().max().item(), (ct - ref).abs().max().item()
    print(f"\n  {kind}: N=8001 float32 |seq-step| {d:.1e}; vs float64: seq {es:.1e}, step {et:.1e}; "
          f"max|c| {ref.abs().max().item():.2f}")
    assert d < 1e-4 and es < 1e-4 and et < 1e-4


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("kind", CTX_KINDS)
def test_encoder_exact_zero_at_zero_input(kind, dev):
    E = _ctx(kind).to(dev)
    worst_seq, worst_step = 0.0, 0.0
    with torch.no_grad():
        for B in (1, 3, 33, 64):
            u = torch.zeros(B, 300, 1, device=dev)
            worst_seq = max(worst_seq, E.seq(u).abs().max().item())
            worst_step = max(worst_step, _run_step(E, u).abs().max().item())
    print(f"\n  {kind}/{dev}: max |c| at u = 0: seq {worst_seq:.1e}, step {worst_step:.1e}")
    assert worst_step == 0.0
    if kind == "gru" and dev == "cpu":
        # the zero-input response is the extra batch row of ONE GRU call; a CPU GEMM can
        # round that row differently from the others (~1e-8). Exact on CUDA and in step().
        assert worst_seq < 1e-6
    else:
        assert worst_seq == 0.0


def test_encoder_input_scaling():
    for kind in CTX_KINDS:
        E1, E3 = _ctx(kind, u_ref=1.0), _ctx(kind, u_ref=3.0)
        u = torch.randn(2, 100, 1)
        assert torch.allclose(E3.seq(3 * u), E1.seq(u), atol=1e-6), kind


def test_gru_zero_input_response_subtraction():
    E = _ctx("gru").double()
    u = torch.randn(3, 200, 1, dtype=torch.float64)
    with torch.no_grad():
        h, _ = E.rnn(u)
        h0, _ = E.rnn(torch.zeros(1, 200, 1, dtype=torch.float64))
        ref = E.proj(h - h0)
        es, et = (E.seq(u) - ref).abs().max().item(), (_run_step(E, u) - ref).abs().max().item()
        r0 = E.proj(h0).abs().max().item()
        st = E.init_state(3, "cpu")
        for k in range(200):
            _, st = E.step(u[:, k], st)
    print(f"\n  gru: seq vs P(h(u)-h(0)) {es:.1e}, step {et:.1e}; |P h(0)| = {r0:.3f} (subtracted)")
    assert es < 1e-12 and et < 1e-12 and r0 > 1e-2
    assert torch.allclose(st[1][0], h0[:, -1].expand(3, -1), atol=1e-12)


# ---------------------------------------------------------------------------
# SFM encoder
# ---------------------------------------------------------------------------

def _sfm_random(nu=2, nb=6, dt=0.07, seed=0):
    torch.manual_seed(seed)
    _, poles, _ = _design(7)
    E = SFMEncoder(nu, 8, 16, dt, poles, nb).double()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        E.raw_sigma.copy_(torch.randn(nb, generator=g, dtype=torch.float64) * 2)
        E.omega.copy_(torch.randn(nb, generator=g, dtype=torch.float64) * 4)
        E.w_in.copy_(torch.randn(nb, 2, nu, generator=g, dtype=torch.float64))
    return E


def test_sfm_ad_bd_equal_expm_and_exact_zoh():
    worst_a, worst_b, worst_q = 0.0, 0.0, 0.0
    for seed in range(5):
        E = _sfm_random(seed=seed)
        Ad, Bd = E._ad_bd()
        s, w, W = _np(E.sigma()), _np(E.omega), _np(E.w_in)
        xg, wg = np.polynomial.legendre.leggauss(30)
        for b in range(E.nb):
            Ab = np.array([[s[b], w[b]], [-w[b], s[b]]])
            M = np.zeros((2 + E.nu, 2 + E.nu))
            M[:2, :2], M[:2, 2:] = Ab, W[b]
            X = expm(M * E.dt)                                   # [[e^{A dt}, int_0^dt e^{A s} ds B]]
            quad = sum(wj * expm(Ab * sj) for sj, wj in zip(0.5 * E.dt * (xg + 1), 0.5 * E.dt * wg)) @ W[b]
            worst_a = max(worst_a, np.abs(_np(Ad[b]) - X[:2, :2]).max())
            worst_b = max(worst_b, np.abs(_np(Bd[b]) - X[:2, 2:]).max() / np.abs(X[:2, 2:]).max())
            worst_q = max(worst_q, np.abs(quad - X[:2, 2:]).max() / np.abs(X[:2, 2:]).max())
            assert np.linalg.norm(_np(Ad[b]), 2) == pytest.approx(np.exp(s[b] * E.dt), rel=1e-12)  # normal
    # float32 at the initialisation actually used (poles from spec(A), dt from the design)
    worst32 = 0.0
    for nz in (5, 6, 7, 12):
        E = _ctx("sfm", nz=nz)
        Ad, Bd = E._ad_bd()
        E64 = _ctx("sfm", nz=nz).double()
        Ad64, Bd64 = E64._ad_bd()
        worst32 = max(worst32, ((Bd.double() - Bd64).abs().max() / Bd64.abs().max()).item(),
                      (Ad.double() - Ad64).abs().max().item())
    print(f"\n  SFM Ad vs expm {worst_a:.1e}; Bd vs exact ZOH (rel) {worst_b:.1e}; quadrature check "
          f"{worst_q:.1e}; float32 Ad/Bd at init (rel) {worst32:.1e}")
    assert worst_a < 1e-13 and worst_b < 1e-6 and worst_q < 1e-10 and worst32 < 1e-5


def test_sfm_impulse_is_repeated_product():
    E = _sfm_random(seed=1)
    n = 400
    H = _np(E.impulse(n))
    Ad, Bd = (_np(t) for t in E._ad_bd())
    P, worst = Bd.copy(), 0.0
    for l in range(n):
        worst = max(worst, np.abs(H[l] - P.reshape(2 * E.nb, E.nu)).max())
        P = np.einsum("nij,nju->niu", Ad, P)
    # float32, long horizon, as used by filter_seq
    E32 = _ctx("sfm", nz=12)
    E64 = _ctx("sfm", nz=12).double()
    h32, h64 = E32.impulse(8001).double(), E64.impulse(8001)
    rel32 = ((h32 - h64).abs().max() / h64.abs().max()).item()
    print(f"\n  impulse vs Ad^l Bd (float64, n=400) {worst:.1e}; float32 n=8001 (rel) {rel32:.1e}")
    assert worst < 1e-13 and rel32 < 1e-5


def test_sfm_hurwitz_for_arbitrary_raw_parameters():
    _, poles, dt = _design(5)
    E = SFMEncoder(1, 4, 8, dt, poles, 4)
    raw = torch.linspace(-80, 80, 16001)
    s32 = -torch.nn.functional.softplus(raw)
    s64 = -torch.nn.functional.softplus(raw.double())
    assert (s32 < 0).all() and (s64 < 0).all()
    deep = torch.linspace(-120, -80, 4001)
    first_zero = deep[(-torch.nn.functional.softplus(deep)) == 0].max().item()
    raw_floor = float(np.log(np.expm1(1e-4)))                       # init clip: -sigma >= 1e-4
    with torch.no_grad():
        E.raw_sigma.copy_(torch.tensor([raw_floor, -3.0, 0.0, 25.0]))
    assert (E.sigma() < 0).all() and E.alpha() > 0
    Ad, _ = E._ad_bd()
    assert all(torch.linalg.matrix_norm(Ad[b], 2) < 1 for b in range(4))   # float32 contraction
    # discrete-time contraction in float32 needs |sigma| dt above the float32 resolution of 1
    with torch.no_grad():
        grid = torch.linspace(-30, raw_floor, 2001)
        nrm = []
        for r in grid:
            E.raw_sigma.fill_(float(r))
            nrm.append(float(torch.linalg.matrix_norm(E._ad_bd()[0][0], 2)))
    lost = grid[torch.tensor(nrm) >= 1].max().item()
    print(f"\n  sigma < 0 for raw in [-80, 80] (float32 and float64); float32 softplus underflows to "
          f"0 for raw <= {first_zero:.1f}; float32 ||Ad|| = 1 (contraction lost) for raw <= {lost:.1f} "
          f"(|sigma| dt ~ 6e-8, dt = {dt:.4f}); init floor raw = {raw_floor:.2f}")


def _tight_sfm(dt, nb=4, dim=4, hidden=8, u_feed=0.0, w_scale=1.0):
    """psi linear-isometric on the slowest block, Bd only in that block: at u = 0 the
    Lemma-2 chain  ||J psi|| ||Ad^{l-1}|| ||Bd||  is an equality."""
    _, poles, _ = _design(5)
    E = SFMEncoder(1, dim, hidden, dt, poles, nb).double()
    b = int(torch.argmax(E.sigma()))
    with torch.no_grad():
        E.omega[b] = 2.0
        E.w_in.zero_()
        E.w_in[b, 0, 0], E.w_in[b, 1, 0] = 0.6 * w_scale, 0.8 * w_scale
        W1, W2, W3 = (l.weight for l in E.psi.layers)
        for W in (W1, W2, W3):
            W.zero_()
        W1[0, 2 * b], W1[1, 2 * b + 1], W1[2, 2 * nb] = 1.0, 1.0, u_feed
        W2[0, 0], W2[1, 1], W2[2, 2] = 1.0, 1.0, 1.0
        W3[0, 0], W3[1, 1], W3[2, 2] = 1.0, 1.0, 1.0
    return E


def test_sfm_lemma2_fading_memory_bound():
    """||dc_k/du_{k-l}|| <= l_psi ||Bd|| e^{-alpha_xi (l-1) dt}  (l >= 1), as proved in
    THEORY.md and implemented (xi_k depends on u_{<k}); the statement there has e^{-alpha l dt}."""
    N = 160
    # (a) generic parameters, random input: the bound holds
    worst = 0.0
    for seed in range(3):
        E = _ctx("sfm", SMALL, nz=5, seed=seed).double()
        with torch.no_grad():
            E.omega.add_(torch.randn(E.nb, dtype=torch.float64))
            E.w_in.add_(0.5 * torch.randn_like(E.w_in))
        u = torch.randn(N, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
        _, Jn = _jacobian(E, u, "seq")
        lp, bd, a = E.psi_lipschitz(), torch.linalg.matrix_norm(E._ad_bd()[1].reshape(-1, 1), 2).item(), E.alpha()
        for l in range(1, N):
            obs = max(Jn[k, k - l].item() for k in range(l, N))
            worst = max(worst, obs / (lp * bd * math.exp(-a * (l - 1) * E.dt)))
    # (b) tight construction at u = 0: equality for the (l-1) exponent, violation for l
    E = _tight_sfm(dt=0.05)
    _, Jn = _jacobian(E, torch.zeros(N, 1, dtype=torch.float64), "seq")
    lp, a = E.psi_lipschitz(), E.alpha()
    bd = torch.linalg.matrix_norm(E._ad_bd()[1].reshape(-1, 1), 2).item()
    r_impl = [Jn[N - 1, N - 1 - l].item() / (lp * bd * math.exp(-a * (l - 1) * E.dt)) for l in range(1, N)]
    r_stat = [Jn[N - 1, N - 1 - l].item() / (lp * bd * math.exp(-a * l * E.dt)) for l in range(1, N)]
    # (c) Prop. 1's l_E = sum_l ||dc_k/du_{k-l}||, with a direct u_k -> c_k path
    Ef = _tight_sfm(dt=0.05, u_feed=1.0, w_scale=1e-3)
    Jl = torch.autograd.functional.jacobian(lambda v: Ef.seq(v[None])[0, -1],
                                            torch.zeros(4000, 1, dtype=torch.float64))[..., 0]
    lE = Jl.norm(dim=0).sum().item()                            # sum_l ||dc_k / du_{k-l}||
    bdf = torch.linalg.matrix_norm(Ef._ad_bd()[1].reshape(-1, 1), 2).item()
    th = Ef.psi_lipschitz() * torch.linalg.matrix_norm(Ef.w_in.reshape(-1, 1), 2).item() / (1 - math.exp(-Ef.alpha() * Ef.dt))
    fix = Ef.psi_lipschitz() * (1 + bdf / (1 - math.exp(-Ef.alpha() * Ef.dt)))
    print(f"\n  Lemma 2, generic: max observed / bound(l-1) = {worst:.3f}"
          f"\n  tight case: observed / bound(l-1) in [{min(r_impl):.6f}, {max(r_impl):.6f}]; "
          f"observed / bound(l) (THEORY statement) = {max(r_stat):.6f} = e^(alpha dt) = {math.exp(a * E.dt):.6f}"
          f"\n  l_E: measured {lE:.4f}; THEORY l_psi ||B_xi||/(1-e^(-a dt)) = {th:.4f}; "
          f"l_psi (1 + ||Bd||/(1-e^(-a dt))) = {fix:.4f}")
    assert worst <= 1.0
    assert max(abs(r - 1) for r in r_impl) < 1e-10
    assert max(r_stat) > 1.0                      # the statement's exponent l is off by one
    assert lE <= fix * (1 + 1e-9) and lE > th     # THEORY's l_E misses the direct u_k term


@pytest.mark.parametrize("nz", [5, 6, 7, 12])
def test_sfm_poles_initialised_from_spec_A(nz):
    _, poles, dt = _design(nz)
    for nb in (2, 8):
        E = SFMEncoder(1, 16, 64, dt, poles, nb)
        up = poles[poles.imag >= -1e-12]
        up = up[np.argsort(-up.real)]
        s, w = _np(E.sigma()), _np(E.omega)
        m = min(nb, len(up))
        assert np.allclose(s[:m], up.real[:m], rtol=1e-6) and np.allclose(w[:m], np.abs(up.imag[:m]), rtol=1e-6)
        for b in range(m, nb):                                          # slower fill-in modes
            assert s[b] == pytest.approx(up.real.max() / 1.5 ** (b - len(up) + 1), rel=1e-6) and w[b] == 0
        if nb == 8:
            print(f"\n  nz={nz}: {len(up)} blocks from spec(A), {nb - m} fill-ins, alpha_xi = "
                  f"{E.alpha():.3f} (latent alpha 1.0), alpha_xi dt = {E.alpha() * dt:.4f}")


# ---------------------------------------------------------------------------
# Conditioners
# ---------------------------------------------------------------------------

def _reference(m, kind, z, c):
    """Per-sample float64 reference of the docstring formulas."""
    out = []
    for zi, ci in zip(z, c):
        if kind == "latent":
            zs = _np(m.z_std)
            out.append((_mlp_ref(m.M, np.r_[zi / zs, ci], "silu") - _mlp_ref(m.M, np.r_[zi / zs, 0 * ci], "silu")) * zs)
            continue
        b = m.base
        h = (zi - _np(b.inp.mean)) / _np(b.inp.std)
        Ws, bs = [_np(l.weight) for l in b.layers], [_np(l.bias) for l in b.layers]
        n = len(Ws)
        if kind == "concat":
            y = _mlp_ref(b.mlp, h, "tanh") + _mlp_ref(m.R, np.r_[h, ci], "silu") - _mlp_ref(m.R, np.r_[h, 0 * ci], "silu")
        elif kind == "film":
            gb, off = _mlp_ref(m.G, ci, "silu"), 0
            for i in range(n):
                h = Ws[i] @ h + bs[i]
                if i < n - 1:                                   # pre-activation, hidden layers
                    d = len(bs[i])
                    g, be = gb[off:off + d], gb[off + d:off + 2 * d]
                    off += 2 * d
                    h = np.tanh(h * (1 + g) + be)
            assert off == len(gb)
            y = h
        else:
            t = _mlp_ref(m.trunk, ci, "silu")
            for i in range(n):
                o, a = Ws[i].shape
                if kind == "lora":
                    r = m.rank
                    U = (_np(m.U[i].weight) @ t + (_np(m.U[i].bias) if m.U[i].bias is not None else 0)).reshape(a, r)
                    V = (_np(m.V[i].weight) @ t).reshape(r, o)
                    W, bb = Ws[i] + (U @ V).T / math.sqrt(r), bs[i]
                    assert np.linalg.matrix_rank(W - Ws[i]) <= r
                else:
                    wv = _np(m.H[i].weight) @ t
                    W, bb = Ws[i] + wv[:a * o].reshape(o, a), bs[i] + wv[a * o:]
                h = W @ h + bb
                if i < n - 1:
                    h = np.tanh(h)
            y = h
        out.append(y * _np(b.out.std) + _np(b.out.mean))
    return np.stack(out)


@pytest.mark.parametrize("kind", COND_KINDS)
def test_conditioner_equals_reference(kind):
    _, m, w, _ = _cond(kind)
    m = m.double()
    g = torch.Generator().manual_seed(7)
    z, c = torch.randn(6, 5, generator=g, dtype=torch.float64), torch.randn(6, DC, generator=g, dtype=torch.float64)
    with torch.no_grad():
        out = m(z, c).numpy()
    ref = _reference(m, kind, z.numpy(), c.numpy())
    err = np.abs(out - ref).max() / np.abs(ref).max()
    print(f"\n  {kind} (width {w}): |forward - per-sample reference| / |ref| = {err:.1e}, |ref| {np.abs(ref).max():.2f}")
    assert err < 1e-12


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("kind", COND_KINDS)
def test_conditioner_exact_collapse_at_zero_context(kind, dev):
    base, m, _, _ = _cond(kind)
    base, m = base.to(dev), m.to(dev)
    with torch.no_grad():
        for B in (1, 7, 11, 256, 4099):
            z = torch.randn(B, 5, device=dev)
            out = m(z, torch.zeros(B, DC, device=dev))
            ref = torch.zeros_like(out) if kind == "latent" else base(z)
            assert torch.equal(out, ref), (kind, dev, B, (out - ref).abs().max().item())
            assert (m(z, torch.randn(B, DC, device=dev)) - ref).abs().max() > 1e-3


@pytest.mark.parametrize("ctx_kind", CTX_KINDS)
@pytest.mark.parametrize("kind", COND_KINDS)
def test_observer_is_autonomous_at_zero_input(kind, ctx_kind):
    """Deployment path: c_k from ctx.step at u = 0, decoder bitwise equal to T0+."""
    base, m, _, _ = _cond(kind)
    E = _ctx(ctx_kind)
    _randomise(E, 0.3, seed=4)
    z = torch.randn(64, 5)
    st = E.init_state(64, "cpu")
    with torch.no_grad():
        for _ in range(5):
            c, st = E.step(torch.zeros(64, 1), st)
            out = m(z, c)
            assert torch.equal(out, torch.zeros_like(out) if kind == "latent" else base(z))


@pytest.mark.parametrize("ctx_kind", CTX_KINDS)
@pytest.mark.parametrize("kind", COND_KINDS)
def test_conditioned_observer_sees_the_sign_of_the_input(kind, ctx_kind):
    """current/window/sfm are odd in u (bias-free, tanh). A conditioner that is even in c
    would give the same estimate for u and -u: the old lora (U(t) V(t), both linear) did."""
    _, m, _, _ = _cond(kind)
    E = _ctx(ctx_kind)
    u = torch.randn(4, 120, 1, generator=torch.Generator().manual_seed(2))
    z = torch.randn(4 * 120, 5)
    with torch.no_grad():
        cp, cm = E.seq(u).reshape(-1, DC), E.seq(-u).reshape(-1, DC)
        d = (m(z, cp) - m(z, cm)).abs().max().item()
        odd = (cp + cm).abs().max().item()
    print(f"\n  {kind}@{ctx_kind}: max|E(u)+E(-u)| = {odd:.1e}; max|T+(z;E(u)) - T+(z;E(-u))| = {d:.2e}")
    assert d > 1e-3


@pytest.mark.parametrize("kind", COND_KINDS)
def test_conditioner_has_an_even_component_in_c(kind):
    """No parity constraint: T+(z;c) + T+(z;-c) - 2 T+(z;0) != 0. With bias-free tanh
    generators FiLM and hyper were exactly odd in c (this was 0 up to base curvature)."""
    _, m, _, _ = _cond(kind)
    z = torch.randn(256, 5, generator=torch.Generator().manual_seed(3))
    c = torch.randn(256, DC, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        zero = m(z, torch.zeros_like(c))
        even = (m(z, c) + m(z, -c) - 2 * zero).abs().max().item()
        odd = (m(z, c) - m(z, -c)).abs().max().item()
    print(f"\n  {kind}: even part {even:.2e}, odd part {odd:.2e}")
    assert even > 1e-3 * max(odd, 1e-12)



@pytest.mark.parametrize("budget", [35000, 140000, 560000])
def test_weight_space_reach_of_lora_and_hyper(budget):
    """Dimension of the set of per-sample weight updates {DeltaW(c)} over many contexts:
    hyper's DeltaW = H t(c) lives in a width-dimensional subspace (width = 1/4/16 at
    35k/140k/560k); lora's per-sample updates are rank <= r but span many directions."""
    base = _base(5, 2)
    dims = {}
    for kind in ("lora", "hyper"):
        w, _ = match_width(kind, base, DC, budget, RANK, torch.ones(5))
        m = build_conditioner(kind, base, DC, w, RANK, torch.ones(5)).double()
        _randomise(m, 0.1)
        c = torch.randn(400, DC, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            t = m.trunk(c)
            if kind == "hyper":
                dW = torch.cat([h(t) for h in m.H], 1)                     # (n, sum a*b + b)
            else:
                dW = torch.cat([torch.bmm(U(t).view(-1, a, RANK), V(t).view(-1, RANK, b)).flatten(1)
                                for U, V, (a, b) in zip(m.U, m.V, [(l.in_features, l.out_features)
                                                                  for l in base.layers])], 1)
        sv = torch.linalg.svdvals(dW)
        dims[kind] = (w, int((sv > 1e-9 * sv[0]).sum()))
    print(f"\n  budget {budget}: dim span{{DeltaW(c)}} over 400 contexts: lora (width {dims['lora'][0]}) "
          f"{dims['lora'][1]}, hyper (width {dims['hyper'][0]}) {dims['hyper'][1]}")
    assert dims["hyper"][1] == dims["hyper"][0]

def _expected_count(kind, nz, nx, dc, w, r):
    dims = [(nz, 128), (128, 128), (128, 128), (128, nx)]
    if kind == "concat":
        return (nz + dc) * w + w + w * w + w + w * nx + nx
    if kind == "latent":
        return (nz + dc) * w + w + w * w + w + w * nz + nz
    if kind == "film":
        return dc * w + w * 2 * 128 * 3
    if kind == "lora":
        return dc * w + w * w + sum(w * a * r + a * r + w * r * b for a, b in dims)
    hid = max(w, 64)
    return dc * hid + hid * w + sum(w * (a * b + b) for a, b in dims)


def test_match_width_table():
    print("\n  budget  nz nx | " + " | ".join(f"{k:>6}: width  params    err" for k in COND_KINDS))
    worst = {}
    for budget in (35000, 140000, 560000):
        for nz in (5, 6, 7, 12):
            for nx in (2, 3, 4, 5):
                base = _base(nz, nx)
                row = []
                for kind in COND_KINDS:
                    w, n = match_width(kind, base, DC, budget, RANK, torch.ones(nz))
                    m = build_conditioner(kind, base, DC, w, RANK, torch.ones(nz))
                    assert n == n_params(m) == _expected_count(kind, nz, nx, DC, w, RANK)
                    for ww in (w - 1, w + 1):                   # no closer width exists
                        if ww >= 1:
                            assert abs(n - budget) <= abs(_expected_count(kind, nz, nx, DC, ww, RANK) - budget)
                    e = 100.0 * (n - budget) / budget
                    worst[(budget, kind)] = max(worst.get((budget, kind), 0.0), abs(e))
                    row.append(f"{w:>14} {n:>7} {e:+5.2f}%")
                if nz in (5, 12) or budget == BUDGET:
                    print(f"  {budget:>6} {nz:>3} {nx:>2} | " + " | ".join(row))
    print("  worst |err| %: " + ", ".join(f"{b // 1000}k {k} {v:.2f}" for (b, k), v in worst.items()))
    assert all(v < 3.0 for (b, k), v in worst.items() if b == BUDGET)
    assert all(v < 5.0 for v in worst.values())


def test_context_encoder_parameter_counts():
    """The budget counts the conditioner only; the context encoder is identical for every
    variant of a job and trained jointly. Printed for the record."""
    counts = {k: n_params(_ctx(k)) for k in CTX_KINDS}
    print("\n  context encoder trainable params (dim 16, hidden 64, window 50, 8 blocks): " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))
    assert counts["sfm"] == 8 + 8 + 16 + 17 * 64 + 64 * 64 + 64 * 16
    assert counts["current"] == 64 + 64 * 64 + 64 * 16


@pytest.mark.parametrize("kind", COND_KINDS)
def test_base_frozen_and_no_double_counting(kind):
    base, m, _, npar = _cond(kind, randomise=False)
    E = _ctx("sfm")
    before = {k: v.clone() for k, v in base.state_dict().items()}
    if kind == "latent":            # T0+ is not part of the injection; phase 1 freezes it
        assert not {id(q) for q in base.parameters()} & {id(q) for q in m.parameters()}
        base.requires_grad_(False)
    assert all(not p.requires_grad for p in base.parameters())
    params = [q for q in list(m.parameters()) + list(E.parameters()) if q.requires_grad]   # train.phase2
    ids = [id(q) for q in params]
    assert len(ids) == len(set(ids))
    assert not {id(q) for q in base.parameters()} & set(ids)
    named = list(m.named_parameters(remove_duplicate=False))
    assert len(named) == len({id(p) for _, p in named})
    assert npar == sum(q.numel() for q in m.parameters() if q.requires_grad)
    opt = torch.optim.Adam(params, lr=1e-2)
    u = torch.randn(4, 50, 1)
    for _ in range(3):
        c = E.seq(u).reshape(-1, DC)
        loss = m(torch.randn(len(c), 5), c).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k, v in base.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert all(p.grad is None for p in base.parameters())


@pytest.mark.parametrize("ctx_kind", CTX_KINDS)
@pytest.mark.parametrize("kind", COND_KINDS)
def test_gradient_reaches_every_parameter_in_two_adam_steps(kind, ctx_kind):
    torch.manual_seed(0)
    base, m, _, _ = _cond(kind, randomise=False)            # true init (zero output layers)
    E = _ctx(ctx_kind)
    named = [(f"cond.{n}", p) for n, p in m.named_parameters() if p.requires_grad] + \
            [(f"ctx.{n}", p) for n, p in E.named_parameters() if p.requires_grad]
    opt = torch.optim.Adam([p for _, p in named], lr=1e-3)
    u = torch.randn(4, 64, 1)
    z, x = torch.randn(4 * 64, 5), torch.randn(4 * 64, 2 if kind != "latent" else 5)
    moved = []
    for step in range(2):
        before = [p.detach().clone() for _, p in named]
        c = E.seq(u).reshape(-1, DC)
        loss = (m(z, c) - x).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        moved.append({n for (n, p), b in zip(named, before) if not torch.equal(p, b)})
    still = [n for n, _ in named if n not in moved[0] | moved[1]]
    late = sorted({n.rsplit(".", 1)[0] for n, _ in named if n not in moved[0]})
    print(f"\n  {kind}@{ctx_kind}: {len(moved[0])}/{len(named)} tensors move at step 1 "
          f"(late: {late}), {len(moved[0] | moved[1])}/{len(named)} by step 2; never: {still}")
    # the last bias of R / M cancels in R([z,c]) - R([z,0]): zero gradient forever
    dead = {"concat": ["cond.R.layers.2.bias"], "latent": ["cond.M.layers.2.bias"]}.get(kind, [])
    assert still == dead
    for n in dead:
        assert dict(named)[n].grad.abs().max() == 0


@pytest.mark.skipif(not CUDA, reason="needs a GPU")
@pytest.mark.parametrize("kind", COND_KINDS)
def test_memory_and_finiteness_at_batch_8192(kind):
    base, m, w, _ = _cond(kind, nz=12, nx=5)
    base, m = base.cuda(), m.cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated()
    z, c = torch.randn(8192, 12, device="cuda"), torch.randn(8192, DC, device="cuda", requires_grad=True)
    out = m(z, c)
    out.pow(2).mean().backward()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20
    print(f"\n  {kind} (width {w}, nz 12, nx 5): batch 8192 forward+backward peak {peak:.0f} MiB")
    assert torch.isfinite(out).all() and torch.isfinite(c.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad and p.grad is not None)
    assert peak < 8192
