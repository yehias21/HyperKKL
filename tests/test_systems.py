"""Verification of the plants: against the archived implementation, against physics, and
for what the benchmark relies on (actuator design, observability, configs, perturbation).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/test_systems.py -q
"""

from __future__ import annotations

import copy
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.func import jacrev, vmap

from hyperkkl import signals as S
from hyperkkl.config import CONFIG_DIR
from hyperkkl.data import make_design
from hyperkkl.systems import REGISTRY, SaturatingActuator, actuator_design, build_plant

ARCHIVE = Path(__file__).resolve().parents[2] / "archive" / "hyperkkl_code"


@pytest.fixture(autouse=True)
def _float64():
    """float64 for these tests only. A module-level set_default_dtype here used to leak
    into tests/test_core.py (same pytest session; it sets float32 at import)."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _cfg(name):
    return yaml.safe_load((CONFIG_DIR / "systems" / f"{name}.yaml").read_text())


def _rk4(f, x, u, dt):
    k1 = f(x, u); k2 = f(x + dt / 2 * k1, u); k3 = f(x + dt / 2 * k2, u); k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _sim(f, x0, T, dt, u=0.0):
    x = torch.as_tensor(x0, dtype=torch.float64)[None]
    uu = torch.full((1, 1), float(u))
    out = [x]
    for _ in range(int(round(T / dt))):
        x = _rk4(f, x, uu, dt)
        out.append(x)
    return torch.cat(out)


# ---------------------------------------------------------------------------
# 1. Same vector field as the archived (audited) implementation
# ---------------------------------------------------------------------------

def _archive_systems():
    if not ARCHIVE.exists():
        pytest.skip("archive not present")
    sys.modules.setdefault("smt", types.ModuleType("smt"))
    sm = types.ModuleType("smt.sampling_methods"); sm.LHS = object
    sys.modules.setdefault("smt.sampling_methods", sm)
    sys.path.insert(0, str(ARCHIVE))
    try:
        import importlib
        return importlib.import_module("src.systems")
    finally:
        sys.path.pop(0)


ARCHIVE_MAP = {  # ours -> (archive class, archive init args)
    "vdp": ("VdP", {"my": 1.0}),
    "duffing": ("RevDuffMult", {"kappa": 0.5}),
    "fhn": ("FitzHughNagumo", {}),
    "bioreactor": ("Bioreactor", {"mu_max": 1.0}),
    "rossler": ("Rossler", {}),
    "lorenz": ("Lorenz", {}),
    "quad_tank": ("QuadrupleTank", {}),
    "cartpole": ("CartPole", {"obs_idx": (0, 2)}),
    "two_link_arm": ("TwoLinkArm", {"obs_idx": (0, 1)}),
    "highway_traffic": ("HighwayTraffic", {"obs_idx": (0, 2, 4)}),
}


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_vector_field_matches_archive(name):
    S = _archive_systems()
    cls, kw = ARCHIVE_MAP[name]
    old = getattr(S, cls)(add_noise=False, **kw)
    sc = _cfg(name)
    new = REGISTRY[sc["plant"]](sensors=sc["sensors"], **sc["params"])
    box = torch.tensor(sc["ic_box"])
    g = torch.Generator().manual_seed(0)
    x = box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(256, len(box), generator=g)
    u = torch.randn(256, 1, generator=g) * sc["u_scale"]
    f_old = old.function(0.0, u[:, 0], x)
    f_new = new.f(x, u)
    assert torch.allclose(f_new, f_old, rtol=1e-10, atol=1e-10), (f_new - f_old).abs().max()
    assert torch.equal(new.h(x), x[:, list(sc["sensors"])])


# ---------------------------------------------------------------------------
# 2. Physics
# ---------------------------------------------------------------------------

def test_vdp_limit_cycle_amplitude():
    """mu = 1: the relaxation limit cycle has max |x1| ~ 2.009."""
    p = REGISTRY["vdp"](mu=1.0)
    x = _sim(p.f, [0.1, 0.0], 60.0, 0.005)
    amp = x[-4000:, 0].abs().max().item()
    assert abs(amp - 2.009) < 0.01, amp


def test_duffing_conserves_energy_at_zero_input():
    """x'' = -x - x^3 is conservative: H = x2^2/2 + x1^2/2 + x1^4/4."""
    p = REGISTRY["duffing"](kappa=0.5)
    x = _sim(p.f, [0.8, -0.3], 30.0, 0.005)
    H = 0.5 * x[:, 1] ** 2 + 0.5 * x[:, 0] ** 2 + 0.25 * x[:, 0] ** 4
    assert (H - H[0]).abs().max() < 1e-8


def test_duffing_input_gain_is_state_dependent():
    p = REGISTRY["duffing"](kappa=0.5)
    x = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    g = p.input_gain(x)[:, 1]
    assert torch.allclose(g, torch.tensor([1.0, 3.0]))           # 1 + kappa x1^2


def test_two_link_arm_conserves_energy_without_damping():
    """kd = 0, u = 0: E = 0.5 dq' M(q) dq + g1 sin q1 + g2 sin(q1 + q2) is conserved.
    This checks M(q), the Coriolis terms and gravity together."""
    p = REGISTRY["two_link_arm"](kd=0.0)
    x = _sim(p.f, [0.3, -0.5, 0.4, -0.2], 10.0, 0.0005)
    a = p.m1 * p.lc1 ** 2 + p.I1 + p.m2 * (p.l1 ** 2 + p.lc2 ** 2) + p.I2
    b, d = p.m2 * p.l1 * p.lc2, p.m2 * p.lc2 ** 2 + p.I2
    g1, g2 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g, p.m2 * p.lc2 * p.g
    q1, q2, w1, w2 = x.T
    c2 = torch.cos(q2)
    ke = 0.5 * ((a + 2 * b * c2) * w1 ** 2 + 2 * (d + b * c2) * w1 * w2 + d * w2 ** 2)
    E = ke + g1 * torch.sin(q1) + g2 * torch.sin(q1 + q2)
    assert (E - E[0]).abs().max() < 1e-7, (E - E[0]).abs().max()


def test_bioreactor_equilibrium_matches_monod_steady_state():
    """mu(s*) = D, x1* = Y (x2f - s*), for the base (affine) plant at u = 0."""
    p = REGISTRY["bioreactor"](mu_max=1.0)
    s_star = p.Ks * p.D_nom / (p.mu_max - p.D_nom)
    x_star = p.Y * (p.x2f - s_star)
    x = _sim(p.f, [1.0, 1.0], 300.0, 0.01)
    assert torch.allclose(x[-1], torch.tensor([x_star, s_star]), atol=1e-4), (x[-1], x_star, s_star)


def test_quad_tank_equilibrium_matches_closed_form():
    """Johansson (2000): valve split of the non-minimum-phase setting P+ (gam1 + gam2 < 1)
    with the pump gains and voltages of P- (k = 3.33, 3.35; v = 3.0), as in the archive."""
    p = REGISTRY["quad_tank"]()
    c = math.sqrt(2 * p.g)
    h3 = ((1 - p.gam2) * p.k2 * p.v2n / (p.a3 * c)) ** 2
    h4 = ((1 - p.gam1) * p.k1 * p.v1n / (p.a4 * c)) ** 2
    h1 = ((p.a3 * c * math.sqrt(h3) + p.gam1 * p.k1 * p.v1n) / (p.a1 * c)) ** 2
    h2 = ((p.a4 * c * math.sqrt(h4) + p.gam2 * p.k2 * p.v2n) / (p.a2 * c)) ** 2
    x = _sim(p.f, [10.0, 10.0, 2.0, 2.0], 3000.0, 0.5)
    ref = torch.tensor([h1, h2, h3, h4])
    assert torch.allclose(x[-1], ref, rtol=2e-3), (x[-1], ref)
    assert p.gam1 + p.gam2 < 1.0                              # non-minimum-phase regime
    # P+ operating levels in the paper (12.6, 13.0, 4.8, 4.9) are at v = 3.15 V; scale check
    assert 10 < h1 < 14 and 10 < h2 < 14 and 3 < h3 < 6 and 3 < h4 < 6


def test_traffic_conserves_vehicles():
    """L * sum(rho') = main inflow + ramp inflow - outflow of the last cell."""
    p = REGISTRY["highway_traffic"]()
    g = torch.Generator().manual_seed(1)
    x = 0.01 + 0.09 * torch.rand(64, 5, generator=g)
    u = 0.1 * torch.randn(64, 1, generator=g)
    lhs = p.L * p.f(x, u).sum(1)
    q5 = p.v_free * (x[:, 4] - x[:, 4] ** 2 / p.rho_max)
    rhs = p.u_main_base + p.u_ramp_base + u[:, 0] - q5
    assert torch.allclose(lhs, rhs, atol=1e-12)


def test_lorenz_and_rossler_stay_on_their_attractors():
    lz = _sim(REGISTRY["lorenz"]().f, [1.0, 1.0, 20.0], 40.0, 0.002)[5000:]
    assert lz[:, 0].abs().max() < 22 and lz[:, 2].min() > 0 and lz[:, 2].max() < 50
    rs = _sim(REGISTRY["rossler"]().f, [1.0, 1.0, 0.0], 200.0, 0.005)[10000:]
    assert rs[:, 0].abs().max() < 30 and rs[:, 2].min() > -1e-6
    assert rs[:, 2].max() > 10                                # the z-spikes of the funnel attractor


def test_lorenz_is_chaotic():
    """Largest Lyapunov exponent ~0.9 (Benettin, renormalised twin trajectory)."""
    f = REGISTRY["lorenz"]().f
    x = _sim(f, [1.0, 1.0, 20.0], 20.0, 0.002)[-1:]
    d0 = 1e-8
    y = x + torch.tensor([[d0, 0.0, 0.0]])
    s, dt, n = 0.0, 0.002, 0
    for k in range(25000):
        x, y = _rk4(f, x, torch.zeros(1, 1), dt), _rk4(f, y, torch.zeros(1, 1), dt)
        if k % 10 == 9:
            d = (y - x).norm()
            s += math.log(d / d0); n += 1
            y = x + (y - x) * d0 / d
    lam = s / (n * 10 * dt)
    assert 0.75 < lam < 1.05, lam


def test_cartpole_balancing_loop_is_stable_in_the_pole_states():
    """Closed-loop linearisation at the upright: pole modes stable, cart modes free."""
    p = REGISTRY["cartpole"]()
    x0 = torch.zeros(1, 4, requires_grad=True)
    J = torch.autograd.functional.jacobian(lambda x: p.f(x, torch.zeros(1, 1))[0], x0)[:, 0]
    ev = np.linalg.eigvals(J.detach().numpy())
    pole = sorted(ev, key=lambda e: -abs(e))[:2]
    assert all(e.real < 0 for e in pole), ev


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_base_plants_are_affine_and_actuated_plants_are_not(name):
    sc = _cfg(name)
    base = REGISTRY[sc["plant"]](sensors=sc["sensors"], **sc["params"])
    full = build_plant(sc)
    assert isinstance(full, SaturatingActuator)
    box = torch.tensor(sc["ic_box"])
    g = torch.Generator().manual_seed(2)
    x = box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(128, len(box), generator=g)
    u = torch.full((128, 1), 0.7 * sc["u_scale"])
    second = lambda p: p.f(x, 2 * u) - 2 * p.f(x, u) + p.f(x, 0 * u)
    assert second(base).abs().max() < 1e-9 * (1 + base.f(x, u).abs().max())
    assert second(full).abs().max() > 1e-6
    # the actuator does not move the operating point: at x_j = x_ref and u = 0, u_eff = 0
    xr = x.clone(); xr[:, full.j] = full.x_ref
    assert torch.allclose(full.f(xr, 0 * u), base.f(xr, 0 * u), atol=1e-12)


def test_perturbed_plant_changes_every_parameter_deterministically():
    p = REGISTRY["lorenz"]()
    q = p.perturbed(0.1)
    for k in p.PARAMS:
        r = getattr(q, k) / getattr(p, k)
        assert abs(math.log(r)) == pytest.approx(0.1)
    assert q.params() == p.perturbed(0.1).params()


# ---------------------------------------------------------------------------
# 3. The saturating actuator (the plants every experiment actually uses)
# ---------------------------------------------------------------------------

def _box(sc, n, seed):
    box = torch.tensor(sc["ic_box"])
    g = torch.Generator().manual_seed(seed)
    return box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(n, len(box), generator=g)


# name -> (j, k, x_ref, u_bar) worked out by hand from the docstring rule
# k * half-span(x_j) = u_bar / 2, x_ref = IC-box centre, u_bar = actuator.u_bar * u_scale;
# rossler (k) and two_link_arm (j, k) are fixed in their YAML (see test below for why).
ACTUATOR = {
    "vdp": (1, 1 / 3, 0.0, 2.0), "duffing": (1, 1.0, 0.0, 2.0), "fhn": (0, 0.4, 0.0, 2.0),
    "bioreactor": (1, 0.15 / 0.9, 1.1, 0.3), "rossler": (1, 0.02, 0.0, 2.0),
    "lorenz": (1, 1 / 30, 0.0, 2.0), "quad_tank": (3, 1 / 1.75, 2.25, 2.0),
    "cartpole": (3, 0.25, 0.0, 2.0), "two_link_arm": (2, 0.1, 0.0, 2.0),
    "highway_traffic": (2, 0.15 / 0.045, 0.055, 0.3),
}


@pytest.mark.parametrize("name", sorted(ACTUATOR))
def test_actuator_design_values(name):
    p = build_plant(_cfg(name))
    j, k, xr, ub = ACTUATOR[name]
    assert (p.j, p.k, p.x_ref, p.u_bar) == (j, pytest.approx(k, rel=1e-12),
                                           pytest.approx(xr, abs=1e-12), pytest.approx(ub))


def test_automatic_actuator_design_is_overridden_where_it_breaks_the_plant():
    """Without the YAML override the rule gives rossler k = 0.1 = a (x2' = x1 + (a - k) x2:
    the term that makes Rossler chaotic cancels) and puts the two_link_arm back-EMF on dq2
    although the torque acts on joint 1 (non-collocated: it pumps energy)."""
    for name, want in (("rossler", (0.1, 1)), ("two_link_arm", (1.0, 3))):
        sc = _cfg(name)
        base = REGISTRY[sc["plant"]](sensors=sc["sensors"], **sc["params"])
        k, j, _ = actuator_design(base, torch.tensor(sc["ic_box"]), 2.0 * sc["u_scale"])
        assert (k, j) == (pytest.approx(want[0]), want[1])


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_actuated_plant_matches_archive_wrapper(name):
    """Ours uses x_j - x_ref, the archive's NonAffineActuator the absolute x_j, so
    ours(x, u) == archive(x, u + k x_ref) exactly, for the same (u_bar, k, j)."""
    Sa = _archive_systems()
    cls, kw = ARCHIVE_MAP[name]
    sc = _cfg(name)
    full = build_plant(sc)
    old = Sa.NonAffineActuator(getattr(Sa, cls)(add_noise=False, **kw), full.u_bar, full.k, full.j)
    x = _box(sc, 256, 3)
    u = 2.5 * sc["u_scale"] * torch.randn(256, 1, generator=torch.Generator().manual_seed(4))
    f_old = old.function(0.0, u[:, 0] + full.k * full.x_ref, x)
    assert torch.allclose(full.f(x, u), f_old, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_actuator_matches_docstring_formula(name):
    """f = f0(x) + G(x) u_bar tanh((u - k (x_j - x_ref)) / u_bar), G from two other
    input values (the base plants are affine), u_eff = 0 at (x_j = x_ref, u = 0)."""
    sc = _cfg(name)
    full = build_plant(sc)
    base = full.base
    x = _box(sc, 256, 5)
    u = 2.5 * sc["u_scale"] * torch.randn(256, 1, generator=torch.Generator().manual_seed(6))
    f0 = base.f(x, 0 * u)
    G = (base.f(x, 0 * u + 2.0) - base.f(x, 0 * u - 1.0)) / 3.0
    ue = full.u_bar * torch.tanh((u[:, 0] - full.k * (x[:, full.j] - full.x_ref)) / full.u_bar)
    assert torch.allclose(full.f(x, u), f0 + G * ue[:, None], rtol=1e-10, atol=1e-12)
    assert torch.equal(full.h(x), base.h(x))
    xr = x.clone(); xr[:, full.j] = full.x_ref
    assert torch.allclose(full.f(xr, 0 * u), base.f(xr, 0 * u), atol=1e-12)


def _signal_values(sc, regime, families, n, seed):
    des = make_design(sc)
    rng = np.random.RandomState(seed)
    out = []
    for r in range(n):
        s = S.sample_signal(families[r % len(families)], regime, rng, des.dt, des.T,
                            stratum=r // len(families), scale=float(sc["u_scale"]))
        s = S.enforce_bounds(s, des.dt, des.T, sc.get("u_bounds"))
        out.append(s(np.linspace(0.0, des.T, 101)))
    return np.concatenate(out)


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_actuator_saturates_somewhere_not_everywhere(name):
    """IC box x actual input values: |arg / u_bar| > 1 is rare in every tier (input
    protocol v2 keeps |u| <= u_scale < u_bar everywhere; only the back-EMF term k (x_j -
    x_ref) can push the argument past u_bar), yet the tanh is visibly curved (q99 > 0.5)."""
    sc = _cfg(name)
    p = build_plant(sc)
    rng = np.random.RandomState(0)
    box = np.asarray(sc["ic_box"], float)
    X = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.rand(50000, len(box))

    def frac(uv):
        a = (rng.choice(uv, len(X)) - p.k * (X[:, p.j] - p.x_ref)) / p.u_bar
        return float((np.abs(a) > 1).mean()), float(np.quantile(np.abs(a), 0.99))

    f_id, q_id = frac(_signal_values(sc, "train", S.TRAIN_FAMILIES, 200, 1))
    assert f_id < 0.02 and q_id > 0.5, (f_id, q_id)        # mildly nonlinear in training
    for k, tier in enumerate(("shape_ood", "sequence_ood")):
        f_t, q_t = frac(_signal_values(sc, tier, S.families_for(tier), 100, 2 + k))
        assert f_t < 0.02 and q_t > 0.4, (tier, f_t, q_t)   # same box: no new saturation


def test_admissible_inputs_keep_plants_physical():
    """tanh bounds |u_eff| < u_bar for ANY input (even far outside the u_scale box), so:
    dilution rate D = D_nom + u_eff in [0, mu(x2f)) (no negative D, no washout),
    ramp inflow >= 0 and total inflow below the Greenshields capacity, pumps v > 0."""
    b = build_plant(_cfg("bioreactor"))
    p = b.base
    assert p.D_nom - b.u_bar >= 0.0
    assert p.D_nom + b.u_bar < p.mu_max * p.x2f / (p.Ks + p.x2f)
    t = build_plant(_cfg("highway_traffic"))
    q = t.base
    assert q.u_ramp_base - t.u_bar >= 0.0
    assert q.u_main_base + q.u_ramp_base + t.u_bar < q.v_free * q.rho_max / 4
    w = build_plant(_cfg("quad_tank"))
    assert w.base.v1n - w.u_bar > 0 and w.base.v2n - 0.5 * w.u_bar > 0
    # and numerically through f: extreme inputs at the box corners, D recovered from x1'
    sc = _cfg("bioreactor")
    x = _box(sc, 512, 7)
    for uu in (-10 * sc["u_scale"], 10 * sc["u_scale"]):
        u = torch.full((512, 1), uu)
        sp = 0.5 * (x[:, 1] + torch.sqrt(x[:, 1] ** 2 + 1e-6))
        D = p.mu_max * sp / (p.Ks + sp) - b.f(x, u)[:, 0] / x[:, 0]
        assert D.min() >= 0.0 and D.max() <= p.D_nom + b.u_bar + 1e-12


# ---------------------------------------------------------------------------
# 4. Balance laws, pointwise, with inputs and the actuator
# ---------------------------------------------------------------------------

def _ueff(p, x, u):
    return p.u_bar * torch.tanh((u[:, 0] - p.k * (x[:, p.j] - p.x_ref)) / p.u_bar)


def test_bioreactor_substrate_mass_balance():
    """M = x1/Y + x2 obeys M' = D (x2f - M) whatever mu is (the Monod term cancels)."""
    sc = _cfg("bioreactor")
    full = build_plant(sc)
    p = full.base
    x = _box(sc, 512, 8)
    u = 3 * sc["u_scale"] * torch.randn(512, 1, generator=torch.Generator().manual_seed(9))
    f = full.f(x, u)
    D = p.D_nom + _ueff(full, x, u)
    M = x[:, 0] / p.Y + x[:, 1]
    assert torch.allclose(f[:, 0] / p.Y + f[:, 1], D * (p.x2f - M), atol=1e-12)


def test_quad_tank_volume_balance():
    """sum A_i h_i' = k1 v1 + k2 v2 - sqrt(2g)(a1 sqrt h1 + a2 sqrt h2): the cross flows
    3 -> 1 and 4 -> 2 cancel; only the two lower tanks drain out of the system."""
    sc = _cfg("quad_tank")
    full = build_plant(sc)
    p = full.base
    x = _box(sc, 512, 10)
    u = 2 * torch.randn(512, 1, generator=torch.Generator().manual_seed(11))
    f = full.f(x, u)
    ue = _ueff(full, x, u)
    v1, v2 = p.v1n + ue, p.v2n - 0.5 * ue
    lhs = p.A1 * f[:, 0] + p.A2 * f[:, 1] + p.A3 * f[:, 2] + p.A4 * f[:, 3]
    c = math.sqrt(2 * p.g)
    rhs = p.k1 * v1 + p.k2 * v2 - c * (p.a1 * x[:, 0].sqrt() + p.a2 * x[:, 1].sqrt())
    assert torch.allclose(lhs, rhs, rtol=1e-6, atol=1e-6)


def _arm_energy(p, x):
    q1, q2, w1, w2 = x.unbind(-1)
    a = p.m1 * p.lc1 ** 2 + p.I1 + p.m2 * (p.l1 ** 2 + p.lc2 ** 2) + p.I2
    b, d = p.m2 * p.l1 * p.lc2, p.m2 * p.lc2 ** 2 + p.I2
    g1, g2 = (p.m1 * p.lc1 + p.m2 * p.l1) * p.g, p.m2 * p.lc2 * p.g
    c2 = torch.cos(q2)
    return (0.5 * ((a + 2 * b * c2) * w1 ** 2 + 2 * (d + b * c2) * w1 * w2 + d * w2 ** 2)
            + g1 * torch.sin(q1) + g2 * torch.sin(q1 + q2))


def test_two_link_arm_power_balance_and_dissipative_actuator():
    """dE/dt = tau1 dq1 - kd |dq|^2 (input enters as M^-1 e1), and at u = 0 the
    collocated back-EMF only removes energy. The automatic j = 3 failed this."""
    sc = _cfg("two_link_arm")
    full = build_plant(sc)
    p = full.base
    x = (6 * _box(sc, 512, 12)).requires_grad_(True)
    u = 2 * torch.randn(512, 1, generator=torch.Generator().manual_seed(13))
    (gE,) = torch.autograd.grad(_arm_energy(p, x).sum(), x)
    x = x.detach()
    for plant, tau in ((p, u[:, 0]), (full, _ueff(full, x, u))):
        dE = (gE * plant.f(x, u)).sum(1)
        assert torch.allclose(dE, tau * x[:, 2] - p.kd * (x[:, 2] ** 2 + x[:, 3] ** 2), atol=1e-9)
    dE0 = (gE * full.f(x, 0 * u)).sum(1)
    assert dE0.max() <= 1e-12
    bad = SaturatingActuator(p, full.u_bar, 1.0, 3, 0.0)          # the automatic design
    assert (gE * bad.f(x, 0 * u)).sum(1).max() > 1.0


def test_cartpole_energy_and_momentum_balance():
    """Balancing loop, damping and the angle limiter switched off: E' = F v and p' = F
    with p = (mc + mp) v + mp l om cos th (cart-pole Lagrangian, F = f_scale u)."""
    p = REGISTRY["cartpole"](kd_v=0.0, kd_w=0.0)
    p.K_BAL, p.TH_MAX = (0.0, 0.0, 0.0, 0.0), 1e6
    g = torch.Generator().manual_seed(14)
    x = torch.randn(256, 4, generator=g) * torch.tensor([1.0, 2.0, 1.0, 3.0])
    u = torch.randn(256, 1, generator=g)
    f = p.f(x, u)
    v, th, om = x[:, 1], x[:, 2], x[:, 3]
    M, m, L = p.mc + p.mp, p.mp, p.l
    dv, dom = f[:, 1], f[:, 3]
    dE = M * v * dv + m * L * (dv * om * torch.cos(th) + v * dom * torch.cos(th) - v * om * om * torch.sin(th)) \
        + m * L * L * om * dom - m * p.g * L * torch.sin(th) * om
    dP = M * dv + m * L * (dom * torch.cos(th) - om * om * torch.sin(th))
    F = p.f_scale * u[:, 0]
    assert torch.allclose(dE, F * v, atol=1e-8) and torch.allclose(dP, F, atol=1e-9)


def test_duffing_and_vdp_energy_rates_with_input():
    g = torch.Generator().manual_seed(15)
    x, u = 2 * torch.randn(256, 2, generator=g), torch.randn(256, 1, generator=g)
    d = REGISTRY["duffing"](kappa=0.5)
    f = d.f(x, u)
    dH = x[:, 1] * f[:, 1] + (x[:, 0] + x[:, 0] ** 3) * f[:, 0]
    assert torch.allclose(dH, x[:, 1] * (1 + 0.5 * x[:, 0] ** 2) * u[:, 0], atol=1e-12)
    v = REGISTRY["vdp"](mu=1.0)
    f = v.f(x, u)
    dE = (x * f).sum(1)
    assert torch.allclose(dE, (1 - x[:, 0] ** 2) * x[:, 1] ** 2 + x[:, 1] * u[:, 0], atol=1e-12)


def test_lorenz_divergence_is_constant():
    """tr J = -(sigma + 1 + beta) everywhere (sum of the Lyapunov exponents)."""
    p = REGISTRY["lorenz"]()
    x = 20 * torch.randn(16, 3, generator=torch.Generator().manual_seed(16))
    J = vmap(jacrev(lambda xi: p.f(xi[None], torch.zeros(1, 1))[0]))(x)
    tr = J.diagonal(dim1=1, dim2=2).sum(1)
    assert torch.allclose(tr, torch.full_like(tr, -(p.sigma + 1 + p.beta)), atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Closed forms and qualitative dynamics of the ACTUATED plants
# ---------------------------------------------------------------------------

def _period(x, dt):
    v = x.numpy()
    i = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]
    return np.diff((i + v[i] / (v[i] - v[i + 1])) * dt)


def test_vdp_period():
    """mu = 1: T = 6.6632868 (relaxation-oscillation period)."""
    x = _sim(REGISTRY["vdp"](mu=1.0).f, [0.1, 0.0], 50.0, 0.005)
    T = _period(x[-4000:, 0], 0.005)
    assert abs(T[-1] - 6.6632868) < 1e-4, T


def test_fhn_unstable_equilibrium_and_limit_cycle():
    """v* solves v - v^3/3 - (v + a)/b + I = 0; tr J = 1 - v*^2 - eps b > 0 (repelling),
    so both the base and the actuated FHN oscillate (base v in [-1.97, 1.85])."""
    p = REGISTRY["fhn"]()
    r = np.roots([-1 / 3, 0.0, 1 - 1 / p.b, p.I_ext - p.a / p.b])
    vs = float(r[np.abs(r.imag) < 1e-12].real[0])
    assert 1 - vs ** 2 - p.eps * p.b > 0.25
    full = build_plant(_cfg("fhn"))
    for plant, lo, hi in ((p, -1.97, 1.85), (full, -1.55, 1.33)):
        x = _sim(plant.f, [0.0, 0.0], 200.0, 0.02)[-4000:, 0]
        assert abs(float(x.min()) - lo) < 0.02 and abs(float(x.max()) - hi) < 0.02, (x.min(), x.max())


def test_actuated_rossler_is_still_chaotic():
    """The funnel attractor's x3 spikes need a - k > 0; with the automatic k = a the
    actuated plant is a neutral oscillator (lambda_max 0.00 vs 0.07) without spikes."""
    full = build_plant(_cfg("rossler"))
    x = _sim(full.f, [8.0, 8.0, 0.0], 150.0, 0.02)
    assert x[:, 2].max() > 10 and x[:, 0].abs().max() > 15
    neutral = SaturatingActuator(full.base, full.u_bar, 0.1, 1, 0.0)
    y = _sim(neutral.f, [1.0, 1.0, 0.0], 150.0, 0.02)          # (tanh compression lets
    assert y[:, 2].max() < 0.1                                  # large orbits grow again)


def test_actuated_lorenz_is_still_chaotic():
    f = build_plant(_cfg("lorenz")).f
    x = _sim(f, [1.0, 1.0, 20.0], 10.0, 0.002)[-1:]
    d0, dt, s, n = 1e-8, 0.002, 0.0, 0
    y = x + torch.tensor([[d0, 0.0, 0.0]])
    for k in range(10000):
        x, y = _rk4(f, x, torch.zeros(1, 1), dt), _rk4(f, y, torch.zeros(1, 1), dt)
        if k % 10 == 9:
            d = (y - x).norm()
            s += math.log(d / d0); n += 1
            y = x + (y - x) * d0 / d
    assert 0.7 < s / (n * 10 * dt) < 1.1


def test_cartpole_linearisation_closed_form():
    """At the upright: cart block {0, -kd_v} (the cart is not fed back), pole block
    s^2 + c1 s + c0 with c0 = -((mc+mp) g + f_scale k3) / (l mc),
    c1 = kd_w - f_scale (k4 + k_emf) / (l mc) (u_eff ~ -k_emf om); k_emf = 0 for the base."""
    for plant in (REGISTRY["cartpole"](), build_plant(_cfg("cartpole"))):
        p = getattr(plant, "base", plant)
        kemf = plant.k if plant is not p else 0.0
        J = torch.autograd.functional.jacobian(lambda x: plant.f(x, torch.zeros(1, 1))[0],
                                               torch.zeros(1, 4))[:, 0]
        ev = np.sort_complex(np.linalg.eigvals(J.numpy()))
        _, _, k3, k4 = p.K_BAL
        c0 = -((p.mc + p.mp) * p.g + p.f_scale * k3) / (p.l * p.mc)
        c1 = p.kd_w - p.f_scale * (k4 + kemf) / (p.l * p.mc)
        want = np.sort_complex(np.concatenate([[0.0, -p.kd_v], np.roots([1.0, c1, c0])]))
        assert np.allclose(ev, want, atol=1e-9), (ev, want)
        assert np.all(np.roots([1.0, c1, c0]).real < -4)


# ---------------------------------------------------------------------------
# 6. Observability of the configured sensors (empirical Gramian)
# ---------------------------------------------------------------------------

# window ~ the latent filter's memory (data.system_scales memory_s), seconds
OBS_WINDOW = {"vdp": 6.0, "duffing": 3.0, "fhn": 17.5, "bioreactor": 18.0, "quad_tank": 17.0,
              "two_link_arm": 5.0, "highway_traffic": 18.0}
# measured median lam_min / lam_max (zero input): vdp 2.9e-2, duffing 1.4e-1, fhn 2.9e-2,
# bioreactor 5.7e-1, quad_tank 5.5e-4 (h3/h4: weakly observable), two_link_arm 3.6e-3
# (5.9e-6 with the automatic, energy-pumping actuator), highway_traffic 1.8e-2.
# Not tested here: rossler, lorenz, cartpole. Their forward Gramian is dominated by strongly
# contracting (slaved) or expanding directions (ratio 1e-7..1e-5), which says nothing
# about reconstructing the current state.


def _fd_gramian(plant, x0, sx, sy, H, dt, eps=1e-5):
    n, nx = x0.shape
    X = torch.cat([x0] + [x0 + s * eps * sx[i] * torch.eye(nx)[i] for i in range(nx) for s in (1, -1)])
    W = torch.zeros(n, nx, nx)
    u = torch.zeros(X.shape[0], 1)
    for k in range(int(round(H / dt)) + 1):
        Y = (plant.h(X) / sy)[n:].view(nx, 2, n, -1)
        dY = (Y[:, 0] - Y[:, 1]) / (2 * eps)
        W += dt * torch.einsum("ina,jna->nij", dY, dY)
        X = _rk4(plant.f, X, u, dt)
    return W


@pytest.mark.parametrize("name", sorted(OBS_WINDOW))
def test_configured_sensors_are_observable(name):
    sc = _cfg(name)
    p = build_plant(sc)
    box = torch.tensor(sc["ic_box"])
    sx = (box[:, 1] - box[:, 0]) / 2
    sy = sx[list(p.sensors)]
    dt, H = make_design(sc).dt, OBS_WINDOW[name]
    x = _box(sc, 8, 17)
    for _ in range(int(round(2 * H / dt))):                   # start ON trajectories
        x = _rk4(p.f, x, torch.zeros(8, 1), dt)
    ev = torch.linalg.eigvalsh(_fd_gramian(p, x, sx, sy, H, dt))
    ratio = float((ev[:, 0] / ev[:, -1]).median())
    assert ratio > 1e-4, ratio


# ---------------------------------------------------------------------------
# 7. Configs
# ---------------------------------------------------------------------------

ARCHIVE_CFG = {"vdp": "vdp", "duffing": "duffing_mult", "fhn": "fhn", "bioreactor": "bioreactor",
               "rossler": "rossler", "lorenz": "lorenz", "quad_tank": "quad_tank",
               "cartpole": "cartpole", "two_link_arm": "two_link_arm",
               "highway_traffic": "highway_traffic"}


@pytest.mark.parametrize("name", sorted(ARCHIVE_CFG))
def test_config_sane_and_carried_over_from_archive(name):
    sc = _cfg(name)
    p = REGISTRY[sc["plant"]](sensors=sc["sensors"], **sc["params"])
    nx, ny, nz = p.nx, len(sc["sensors"]), int(sc["nz"])
    assert sc["name"] == name and sc["role"] in ("dev", "test")
    assert all(0 <= s < nx for s in sc["sensors"]) and len(set(sc["sensors"])) == ny
    box = np.asarray(sc["ic_box"], float)
    assert box.shape == (nx, 2) and np.all(box[:, 0] < box[:, 1])
    assert nx + 1 <= nz <= ny * (2 * nx + 1)
    # washout ln(1e3)/alpha is at most a third of the horizon
    assert float(sc["T"]) * float(sc["alpha"]) >= 3 * math.log(1e3)
    assert float(sc["u_scale"]) > 0
    if sc["u_bounds"]:
        assert float(sc["u_bounds"][0]) < 0 < float(sc["u_bounds"][1])
    path = ARCHIVE / "configs" / "systems" / f"{ARCHIVE_CFG[name]}.yaml"
    if not path.exists():
        pytest.skip("archive not present")
    old = yaml.safe_load(path.read_text())
    assert np.allclose(np.asarray(old["limits"], float), box)
    # documented re-designs (configs/systems/<name>.yaml header says why); everything
    # else must be carried over from the audited archive configs unchanged
    # input protocol v2: T raised so that T / t_ref >= 20 (rossler, highway_traffic)
    redesigned = {"quad_tank": {"T": 300.0, "alpha": 0.1}, "rossler": {"T": 160.0},
                  "highway_traffic": {"T": 200.0}}
    exp_T = redesigned.get(name, {}).get("T", float(old["time_end"]))
    exp_alpha = redesigned.get(name, {}).get("alpha", float(old["design"]["alpha"]))
    assert float(sc["T"]) == exp_T and int(old["z_size"]) == nz
    assert float(sc["alpha"]) == exp_alpha
    assert float(old.get("u_scale", 1.0)) == float(sc["u_scale"])
    assert list(old.get("u_bounds", [])) == list(sc["u_bounds"])


# ---------------------------------------------------------------------------
# 8. Perturbation, params, build_plant, and the EKF's Jacobian path
# ---------------------------------------------------------------------------

def test_perturbed_signs_alternate_over_sorted_names():
    p = REGISTRY["bioreactor"](mu_max=1.0)
    q = p.perturbed(0.3)
    phys = [k for k in sorted(p.PARAMS) if k not in p.NOT_PERTURBED]
    for i, k in enumerate(phys):                            # signs alternate over physics only
        assert getattr(q, k) == pytest.approx(getattr(p, k) * math.exp(0.3 * (-1) ** i))
    for k in p.NOT_PERTURBED:                               # the operating point stays put
        assert getattr(q, k) == getattr(p, k)
    assert set(p.NOT_PERTURBED) == {"x2f", "D_nom"}
    assert p.params()["D_nom"] == 0.3                       # the original is untouched
    assert q.sensors == p.sensors and REGISTRY["vdp"]().perturbed(0.0).params() == {"mu": 1.0}
    with pytest.raises(ValueError):
        REGISTRY["vdp"](nu=1.0)


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_build_plant_perturb_keeps_the_actuator(name):
    """Drift moves the plant's physical parameters only; the actuator (u_bar, k, j,
    x_ref) is hardware and stays at its nominal design."""
    sc = _cfg(name)
    nom, drift = build_plant(sc), build_plant(sc, perturb=0.1)
    assert (drift.u_bar, drift.k, drift.j, drift.x_ref) == (nom.u_bar, nom.k, nom.j, nom.x_ref)
    assert drift.params() == nom.base.perturbed(0.1).params() == nom.perturbed(0.1).params()
    assert drift.params() != nom.params() and drift.sensors == nom.sensors
    x = _box(sc, 64, 18)
    u = sc["u_scale"] * torch.randn(64, 1, generator=torch.Generator().manual_seed(19))
    assert torch.allclose(drift.f(x, u), nom.perturbed(0.1).f(x, u), atol=0, rtol=0)
    assert not torch.allclose(drift.f(x, u), nom.f(x, u))


@pytest.mark.parametrize("name", sorted(ARCHIVE_MAP))
def test_ekf_jacobian_path(name):
    """vmap(jacrev) over f (observers.EKF) works, is finite at the box corners and
    matches central differences."""
    sc = _cfg(name)
    p = build_plant(sc)
    box = torch.tensor(sc["ic_box"])
    x = torch.cat([_box(sc, 6, 20), box[:, 0][None], box[:, 1][None]])
    u = sc["u_scale"] * torch.linspace(-2, 2, 8)[:, None]
    J = vmap(jacrev(lambda xi, ui: p.f(xi[None], ui[None])[0]))(x, u)
    assert torch.isfinite(J).all()
    h = 1e-6 * (box[:, 1] - box[:, 0])
    for i in range(p.nx):
        e = torch.zeros(p.nx); e[i] = h[i]
        fd = (p.f(x + e, u) - p.f(x - e, u)) / (2 * h[i])
        assert torch.allclose(J[:, :, i], fd, rtol=1e-5, atol=1e-6 * (1 + fd.abs().max()))
