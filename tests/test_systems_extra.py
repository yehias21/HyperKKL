"""Verification of the plants in hyperkkl/systems_extra.py (battery, network): contract,
physics, envelope (nominal and perturbed), observability of the configured sensors, configs.

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_systems_extra.py -q -s

Nothing is written to experiments/cache: system_scales runs against a temporary cache.
"""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from torch.func import jacrev, vmap

import hyperkkl.data as Dm
from hyperkkl import design as D
from hyperkkl import signals as S
from hyperkkl.config import CONFIG_DIR, ROOT, load
from hyperkkl.systems import REGISTRY, SaturatingActuator, build_plant

NAMES = ("battery", "network")


@pytest.fixture(autouse=True)
def _float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _cfg(name):
    return yaml.safe_load((CONFIG_DIR / "systems" / f"{name}.yaml").read_text())


def _box(sc, n, seed):
    box = torch.tensor(sc["ic_box"], dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    return box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(n, len(box), generator=g, dtype=torch.float64)


def _rk4(f, x, u, dt):
    k1 = f(x, u); k2 = f(x + dt / 2 * k1, u); k3 = f(x + dt / 2 * k2, u); k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# 1. Contract: shapes, finiteness, batch consistency, dtypes, EKF Jacobian path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_shapes_finite_batch_consistent_and_dtypes(name):
    sc = _cfg(name)
    for p in (REGISTRY[name](sensors=sc["sensors"]), build_plant(sc)):
        x = _box(sc, 256, 0)
        u = 2.5 * sc["u_scale"] * torch.randn(256, 1, generator=torch.Generator().manual_seed(1))
        f = p.f(x, u)
        assert f.shape == (256, p.nx) and f.dtype == torch.float64 and torch.isfinite(f).all()
        assert p.h(x).shape == (256, len(sc["sensors"]))
        assert torch.equal(p.h(x), x[:, list(sc["sensors"])])      # observers._selection assumes this
        rows = torch.cat([p.f(x[i:i + 1], u[i:i + 1]) for i in range(0, 256, 37)])
        assert torch.allclose(rows, f[::37], rtol=0, atol=1e-12)
        f32 = p.f(x.float(), u.float())
        assert f32.dtype == torch.float32 and torch.allclose(f32.double(), f, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_ekf_jacobian_path(name):
    sc = _cfg(name)
    p = build_plant(sc)
    box = torch.tensor(sc["ic_box"])
    x = torch.cat([_box(sc, 6, 2), box[:, 0][None], box[:, 1][None]])
    u = sc["u_scale"] * torch.linspace(-2.5, 2.5, 8)[:, None]
    J = vmap(jacrev(lambda xi, ui: p.f(xi[None], ui[None])[0]))(x, u)
    assert torch.isfinite(J).all()
    h = 1e-6 * (box[:, 1] - box[:, 0])
    for i in range(p.nx):
        e = torch.zeros(p.nx); e[i] = h[i]
        fd = (p.f(x + e, u) - p.f(x - e, u)) / (2 * h[i])
        assert torch.allclose(J[:, :, i], fd, rtol=1e-5, atol=1e-6 * (1 + fd.abs().max()))


# ---------------------------------------------------------------------------
# 2. Battery physics
# ---------------------------------------------------------------------------

def test_battery_coulomb_counting():
    """dSOC/dt = -I / (3600 Q) per second, i.e. x TIME_UNIT per model unit, where the BMS
    derating is inactive; positive current discharges."""
    p = REGISTRY["battery"]()
    x = _box(_cfg("battery"), 128, 3)                                   # SOC in [0.3, 0.8]
    u = torch.randn(128, 1, generator=torch.Generator().manual_seed(4))
    want = -p.TIME_UNIT * u[:, 0] / (3600.0 * p.Q)
    assert torch.allclose(p.f(x, u)[:, 0], want, rtol=1e-4, atol=1e-12)
    assert (p.f(x, 0 * u + 1.0)[:, 0] < 0).all() and (p.f(x, 0 * u - 1.0)[:, 0] > 0).all()
    assert torch.allclose(p.cell_current(x[:, 0], u[:, 0]), u[:, 0], rtol=1e-4)


def test_battery_ocv_monotone_and_matches_chen_rincon_mora():
    p = REGISTRY["battery"]()
    s = torch.linspace(0.02, 0.98, 2001)
    v = p.ocv(s)
    assert (v.diff() > 0).all()
    ref = -1.031 * torch.exp(-35 * s) + 3.685 + 0.2156 * s - 0.1178 * s ** 2 + 0.3201 * s ** 3
    inner = (s >= 0.05) & (s <= 0.95)
    assert torch.allclose(v[inner], ref[inner], atol=1e-4)            # the soft clamp's corners
    assert (v - ref).abs().max() < 1e-3                                # 1 mV at SOC 0.02
    box = torch.linspace(0.3, 0.8, 501)
    slope = p.ocv(box).diff() / box.diff()
    assert slope.min() > 0.2                                            # SOC observable from V
    assert 3.70 < float(p.ocv(torch.tensor(0.3))) < float(p.ocv(torch.tensor(0.8))) < 4.0


def test_battery_rc_relaxation_time_constants():
    """At rest with T-independent resistances, Vi(t) = Vi(0) exp(-t / (Ri Ci))."""
    p = REGISTRY["battery"](Ea_R=0.0)
    x = torch.tensor([[0.5, 0.03, -0.02, 1.0, 0.1]])
    dt, n = 0.005, 400                                                  # 2 min
    for _ in range(n):
        x = _rk4(p.f, x, torch.zeros(1, 1), dt)
    t = n * dt * p.TIME_UNIT
    assert abs(float(x[0, 1]) - 0.03 * math.exp(-t / (p.R1 * p.C1))) < 1e-7
    assert abs(float(x[0, 2]) + 0.02 * math.exp(-t / (p.R2 * p.C2))) < 1e-7
    assert p.R1 * p.C1 == pytest.approx(32.85, abs=0.01) and p.R2 * p.C2 == pytest.approx(223.0, abs=0.1)


def test_battery_heat_balance_and_voltage_channel():
    """C_th dT' = Joule + RC dissipation - I T dU/dT - hA dT (Bernardi 1985), and the
    voltage channel is at rest exactly at Vm = OCV + dUdT dT - V1 - V2 - R0(T) I - V_REF."""
    p = REGISTRY["battery"]()
    sc = _cfg("battery")
    x = _box(sc, 128, 5)
    u = torch.randn(128, 1, generator=torch.Generator().manual_seed(6))
    soc, v1, v2, dT, vm = x.unbind(1)
    I = p.cell_current(soc, u[:, 0])
    T = p.T_amb + dT
    arr = torch.exp(p.Ea_R * (1 / T - 1 / p.T_amb))
    q = I ** 2 * p.R0 * arr + v1 ** 2 / (p.R1 * arr) + v2 ** 2 / (p.R2 * arr) - I * T * p.dUdT - p.hA * dT
    f = p.f(x, u)
    assert torch.allclose(f[:, 3] * p.C_th / p.TIME_UNIT, q, atol=1e-12)
    vt = p.ocv(soc) + p.dUdT * dT - v1 - v2 - p.R0 * arr * I - p.V_REF
    xs = torch.stack([soc, v1, v2, dT, vt], 1)
    assert p.f(xs, u)[:, 4].abs().max() < 1e-12
    # warmer cell -> lower resistance (Arrhenius), and heating is quadratic in current
    assert float(arr[dT.argmax()]) < 1.0
    z = torch.zeros(128, 1)
    xr = torch.cat([x[:, :1].clamp(0.3, 0.7), 0 * x[:, 1:4], x[:, 4:]], 1)
    g = lambda a: p.f(xr, z + a)[:, 3]
    assert torch.allclose((g(2.0) - 2 * g(1.0) + g(0.0)) * p.C_th / p.TIME_UNIT,
                          torch.full((128,), 2 * p.R0), rtol=1e-3)


def test_battery_derating_keeps_soc_physical():
    """2.5 C held for an hour, either sign (far beyond any input protocol): SOC stays in (0, 1)."""
    p = REGISTRY["battery"]()
    x = torch.tensor([[0.3, 0, 0, 0, 0.0], [0.8, 0, 0, 0, 0.3]])
    u = torch.tensor([[-2.5 * 0.85], [2.5 * 0.85]])
    for _ in range(6000):
        x = _rk4(p.f, x, u, 0.01)
    assert 0.9 < float(x[0, 0]) < 0.99 and 0.0 < float(x[1, 0]) < 0.08, x[:, 0]
    assert torch.isfinite(x).all() and (x[:, 3].abs() < 15).all()


# ---------------------------------------------------------------------------
# 3. Network physics
# ---------------------------------------------------------------------------

def _jac(p, x):
    return vmap(jacrev(lambda xi: p.f(xi[None], torch.zeros(1, 1))[0]))(x)


def test_network_zero_coupling_decouples_into_fhn_nodes():
    sc = _cfg("network")
    p = REGISTRY["network"](kappa=0.0)
    x = _box(sc, 32, 7)
    u = torch.randn(32, 1, generator=torch.Generator().manual_seed(8))
    J = _jac(p, x).view(32, 5, 2, 5, 2)
    for i in range(5):
        for j in range(5):
            if i != j:
                assert J[:, i, :, j, :].abs().max() == 0.0
    f = p.f(x, u).view(32, 5, 2)
    for i in range(5):
        node = REGISTRY["fhn"](eps=p.eps * math.exp(p.spread * (i - 2) / 2))
        ui = u if i == 0 else 0 * u
        assert torch.allclose(f[:, i], node.f(x[:, 2 * i:2 * i + 2], ui), atol=1e-12)


def test_network_coupling_is_symmetric_diffusive_ring():
    sc = _cfg("network")
    p = REGISTRY["network"]()
    x = _box(sc, 32, 9)
    Jv = _jac(p, x)[:, 0::2, 0::2]                                      # d v_i' / d v_j
    off = Jv - torch.diag_embed(Jv.diagonal(dim1=1, dim2=2))
    assert torch.allclose(off, off.transpose(1, 2), atol=1e-12)
    A = torch.zeros(5, 5)
    for i, j in p.EDGES:
        A[i, j] = A[j, i] = 1.0
    assert torch.allclose(off, p.kappa * A.expand(32, 5, 5), atol=1e-12)
    L = p.laplacian()
    assert torch.allclose(L.sum(1), torch.zeros(5)) and torch.equal(L, L.T)
    # diffusive: the coupling only redistributes, sum_i of the coupling term vanishes
    f0 = REGISTRY["network"](kappa=0.0).f(x, torch.zeros(32, 1))
    assert torch.allclose((p.f(x, torch.zeros(32, 1)) - f0)[:, 0::2].sum(1), torch.zeros(32), atol=1e-12)
    # identical states on every node: coupling has no effect
    xs = x[:, :2].repeat(1, 5)
    assert torch.allclose(p.f(xs, 0 * x[:, :1])[:, 0::2], REGISTRY["network"](kappa=0.0).f(xs, 0 * x[:, :1])[:, 0::2])


def test_network_input_enters_node_zero_only():
    p = REGISTRY["network"]()
    x = _box(_cfg("network"), 16, 10)
    g = p.input_gain(x)
    assert torch.allclose(g, torch.eye(10)[0].expand(16, 10))


# ---------------------------------------------------------------------------
# 4. Envelope: forced trajectories (data.integrate) nominal and perturbed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scales(tmp_path_factory):
    old = Dm.CACHE
    Dm.CACHE = tmp_path_factory.mktemp("cache")
    try:
        # on a GPU when present: on a loaded host the CPU probes took > 50 min
        yield {n: Dm.system_scales(load(n), "cuda" if torch.cuda.is_available() else "cpu") for n in NAMES}
    finally:
        Dm.CACHE = old


def _forced(name, plant, regime, fams, n, seed):
    sc = load(name)["system"]
    des = Dm.make_design(sc)
    sigs = Dm._signals(fams, n, regime, np.random.RandomState(seed), des, sc)
    x, _, _ = Dm.integrate(plant, des, Dm._sobol(np.asarray(sc["ic_box"]), n, seed), sigs)
    return x


PHYSICAL = {"battery": [(0.0, 1.0), (-0.2, 0.2), (-0.2, 0.2), (-1.0, 15.0), (-0.8, 0.8)],
            "network": [(-3.0, 3.0), (-2.5, 2.5)] * 5}


@pytest.mark.parametrize("name", NAMES)
def test_design_and_scales(name, scales):
    s = scales[name]
    assert s["design_check"]["ok"] and s["n_probe_diverged"] == 0
    print(f"\n{name}: dt {s['dt']:.4f} N {s['N']} t_burn {s['t_burn']:.2f} memory {s['memory_s']:.2f}")
    sc = load(name)["system"]
    assert s["t_burn"] < float(sc["T"]) / 3


@pytest.mark.parametrize("name", NAMES)
def test_training_inputs_stay_in_envelope_and_physical(name, scales):
    dom = torch.tensor(scales[name]["domain"])
    phys = torch.tensor(PHYSICAL[name])
    sc = load(name)["system"]
    for regime in ("train",) + tuple(t for t in S.TIERS if t != "id"):
        fams = S.families_for(regime)
        x = _forced(name, build_plant(sc), regime, fams, 30, 21).reshape(-1, len(dom))
        assert torch.isfinite(x).all()
        assert (x >= phys[:, 0]).all() and (x <= phys[:, 1]).all(), (regime, x.min(0).values, x.max(0).values)
        if regime == "train":
            assert (x >= dom[:, 0]).all() and (x <= dom[:, 1]).all()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("rel", (0.3, -0.3))
def test_perturbed_plant_stays_inside_reject_envelope(name, rel, scales):
    """generate() redraws episodes leaving ctr +- 1.5 half (data.reject_scale): a drifted
    plant whose trajectories leave it loses whole tiers (quad_tank did, before
    NOT_PERTURBED). Checked on every evaluation tier, for both sign patterns."""
    dom = np.asarray(scales[name]["domain"])
    ctr, half = dom.mean(1), (dom[:, 1] - dom[:, 0]) / 2
    sc = load(name)["system"]
    p = build_plant(sc, perturb=rel)
    for regime in S.TIERS:
        x = _forced(name, p, regime, S.families_for(regime), 30, 22).reshape(-1, len(dom))
        reach = ((x - torch.as_tensor(ctr)).abs() / torch.as_tensor(half)).max(0).values
        print(f"\n{name} perturbed({rel:+}) {regime}: max reach / half-width {np.round(reach.numpy(), 2)}")
        assert torch.isfinite(x).all() and (reach < 1.4).all(), reach


def test_not_perturbed_are_operating_conditions():
    for name, keep in (("battery", ("T_amb", "hA", "tau_f")), ("network", ())):
        p = REGISTRY[name]()
        q = p.perturbed(0.3)
        assert set(p.NOT_PERTURBED) == set(keep)
        for k in p.PARAMS:
            r = abs(math.log(getattr(q, k) / getattr(p, k)))
            assert r == (0.0 if k in keep else pytest.approx(0.3)), k
        assert q.params() == p.perturbed(0.3).params() and q.sensors == p.sensors


# ---------------------------------------------------------------------------
# 5. Observability of the configured sensors
# ---------------------------------------------------------------------------

def _fd_gramian(plant, x0, sx, sy, H, dt, eps=1e-5):
    """Empirical (finite-difference) observability Gramian at zero input, as in
    tests/test_systems.py, states and outputs scaled by the IC-box half-widths."""
    n, nx = x0.shape
    X = torch.cat([x0] + [x0 + s * eps * sx[i] * torch.eye(nx)[i] for i in range(nx) for s in (1, -1)])
    W = torch.zeros(n, nx, nx)
    u = torch.zeros(X.shape[0], 1)
    for _ in range(int(round(H / dt)) + 1):
        Y = (plant.h(X) / sy)[n:].view(nx, 2, n, -1)
        dY = (Y[:, 0] - Y[:, 1]) / (2 * eps)
        W += dt * torch.einsum("ina,jna->nij", dY, dY)
        X = _rk4(plant.f, X, u, dt)
    return W


# window ~ latent memory (system_scales memory_s), integration step
OBS = {"battery": (6.0, 0.01), "network": (20.0, 0.05)}


def _obs_ratio(name, sensors=None, n=8, seed=17):
    sc = _cfg(name)
    if sensors is not None:
        sc = dict(sc, sensors=list(sensors))
    p = build_plant(sc)
    box = torch.tensor(sc["ic_box"])
    sx = (box[:, 1] - box[:, 0]) / 2
    H, dt = OBS[name]
    x = _box(sc, n, seed)
    for _ in range(int(round(2 * H / dt))):                             # start ON trajectories
        x = _rk4(p.f, x, torch.zeros(n, 1), dt)
    ev = torch.linalg.eigvalsh(_fd_gramian(p, x, sx, sx[list(p.sensors)], H, dt))
    return float((ev[:, 0] / ev[:, -1]).median()), x, p


@pytest.mark.parametrize("name", NAMES)
def test_configured_sensors_are_observable(name):
    """battery: the threshold of tests/test_systems.py (lam_min / lam_max > 1e-4). network:
    the forward Gramian is dominated by fast-contracting transverse directions of the
    relaxation oscillators (median ~1e-4, IC-dependent), so it must be definite and every
    weakly observed direction must contract (detectability), see test below."""
    ratio, x, p = _obs_ratio(name)
    print(f"\n{name}: empirical Gramian median lam_min / lam_max = {ratio:.2e}")
    assert ratio > (1e-4 if name == "battery" else 1e-5), ratio
    # linearised (Kalman) observability matrix has full rank at typical states
    J = _jac(p.base if hasattr(p, "base") else p, x)
    C = torch.eye(p.nx)[list(p.sensors)]
    for Ji in J:
        O = torch.cat([C @ torch.linalg.matrix_power(Ji, i) for i in range(p.nx)])
        O = O / O.norm(dim=1, keepdim=True).clamp(min=1e-300)
        assert int(torch.linalg.matrix_rank(O, rtol=1e-10)) == p.nx


def test_network_weakly_observed_directions_contract():
    """Gramian eigen-directions with lam / lam_max < 1e-3 shrink to < 1/2 of their size
    over the window (measured <= 0.23): what the forward Gramian sees badly decays by itself."""
    sc = _cfg("network")
    p = build_plant(sc)
    box = torch.tensor(sc["ic_box"])
    sx = (box[:, 1] - box[:, 0]) / 2
    H, dt = OBS["network"]
    x0 = _box(sc, 8, 3)
    for _ in range(int(round(2 * H / dt))):
        x0 = _rk4(p.f, x0, torch.zeros(8, 1), dt)
    W = _fd_gramian(p, x0, sx, sx[list(p.sensors)], H, dt)
    ev, V = torch.linalg.eigh(W)
    r = ev / ev[:, -1:]
    assert (r[:, 0] > 1e-7).all(), r[:, 0]                              # definite: observable
    eps = 1e-5
    X = torch.cat([x0[:, None] + s * eps * (V * sx[:, None]).transpose(1, 2) for s in (1, -1)])
    X = X.reshape(-1, p.nx)
    for _ in range(int(round(H / dt))):
        X = _rk4(p.f, X, torch.zeros(X.shape[0], 1), dt)
    X = X.view(2, 8, p.nx, p.nx)
    gain = ((X[0] - X[1]) / (2 * eps) / sx).norm(dim=-1)               # |Phi v| per eigvector
    weak = r < 1e-3
    print(f"\nnetwork: {int(weak.sum())} weak directions in 8 states, max |Phi v| "
          f"{float(gain[weak].max()):.2f} (strong: {float(gain[~weak].max()):.1f})")
    assert weak.any() and gain[weak].max() < 0.5


def test_battery_voltage_alone_leaves_temperature_weak():
    """Why the thermistor is a sensor: from Vm alone the Gramian is >100x worse."""
    weak, _, _ = _obs_ratio("battery", sensors=(4,), n=4)
    full, _, _ = _obs_ratio("battery", n=4)
    print(f"\nbattery Gramian ratio: Vm only {weak:.1e}, dT + Vm {full:.1e}")
    assert full > 100 * weak


# ---------------------------------------------------------------------------
# 6. Configs and registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_config_sane(name):
    sc = _cfg(name)
    p = REGISTRY[sc["plant"]](sensors=sc["sensors"], **sc["params"])
    nx, ny, nz = p.nx, len(sc["sensors"]), int(sc["nz"])
    assert sc["name"] == name and sc["role"] in ("dev", "test")
    assert all(0 <= s < nx for s in sc["sensors"]) and len(set(sc["sensors"])) == ny
    box = np.asarray(sc["ic_box"], float)
    assert box.shape == (nx, 2) and np.all(box[:, 0] < box[:, 1])
    assert nx + 1 <= nz <= ny * (2 * nx + 1)
    assert float(sc["T"]) * float(sc["alpha"]) >= 3 * math.log(1e3)
    des = Dm.make_design(sc)
    assert D.validate(des.A, des.B, des.dt, des.alpha)["ok"]
    full = build_plant(sc)
    assert isinstance(full, SaturatingActuator) == (sc["actuator"]["kind"] == "saturating")


def test_battery_is_non_affine_without_the_actuator_wrapper():
    sc = _cfg("battery")
    p = build_plant(sc)
    x = _box(sc, 64, 23)
    u = torch.full((64, 1), 0.85)
    second = p.f(x, 2 * u) - 2 * p.f(x, u) + p.f(x, 0 * u)
    assert second[:, 3].abs().min() > 1e-3                              # Joule heat ~ I^2


def test_registration_is_lazy():
    """systems_extra is loaded on the first lookup of one of its names. hyperkkl.job now
    imports it eagerly (it is in job.RESULT_MODULES / cache.DATA_MODULES so its code is hashed
    and recorded; tests/test_verify_pipeline checks job's module set), so laziness is checked
    on hyperkkl.systems alone."""
    code0 = "import sys, hyperkkl.job; print('hyperkkl.systems_extra' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code0], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip() == "True"
    code = ("import sys, hyperkkl.systems; a = 'hyperkkl.systems_extra' in sys.modules; "
            "from hyperkkl.systems import REGISTRY; REGISTRY['network']; "
            "print(a, 'hyperkkl.systems_extra' in sys.modules, sorted(REGISTRY))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout.split(maxsplit=2)
    assert out[:2] == ["False", "True"] and "'battery'" in out[2] and "'vdp'" in out[2]
    with pytest.raises(KeyError):
        REGISTRY["no_such_plant"]


# ---------------------------------------------------------------------------
# 7. Incomplete knowledge (build_plant(..., knowledge="partial"))
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_partial_knowledge_model(name):
    """f_known is finite on the IC box, keeps the state dimension, and differs from f by a
    moderate amount: median ||(f - f_known) / s|| / ||f / s|| (s = box half-widths) over the
    box and |u| <= u_scale, measured 0.22 (battery) and 0.21 (network)."""
    sc = _cfg(name)
    full, part = build_plant(sc), build_plant(sc, knowledge="partial")
    assert REGISTRY[name].UNMODELLED
    box = torch.tensor(sc["ic_box"])
    sx = (box[:, 1] - box[:, 0]) / 2
    x = _box(sc, 4096, 30)
    u = sc["u_scale"] * (2 * torch.rand(4096, 1, generator=torch.Generator().manual_seed(31)) - 1)
    f, fk = full.f(x, u), part.f(x, u)
    assert fk.shape == f.shape and torch.isfinite(fk).all()
    assert torch.equal(part.h(x), full.h(x))
    rel = float((((f - fk) / sx).norm(dim=1) / (f / sx).norm(dim=1)).median())
    print(f"\n{name}: unmodelled = {REGISTRY[name].UNMODELLED}; median relative model error {rel:.3f}")
    assert 0.03 < rel < 0.30, rel
    J = vmap(jacrev(lambda xi, ui: part.f(xi[None], ui[None])[0]))(x[:8], u[:8])   # EKF path
    assert torch.isfinite(J).all()


def test_partial_knowledge_terms_are_the_documented_ones():
    p = REGISTRY["battery"]()
    x = _box(_cfg("battery"), 64, 32)
    u = torch.randn(64, 1, generator=torch.Generator().manual_seed(33))
    d = p.f(x, u) - p.f_known(x, u)
    assert torch.allclose(d[:, :2], torch.zeros(64, 2), atol=1e-15)       # SOC, V1 identical
    assert torch.allclose(d[:, 2], p.TIME_UNIT * p.cell_current(x[:, 0], u[:, 0]) / p.C2, atol=1e-12)
    no_v2 = torch.cat([x[:, :2], 0 * x[:, 2:3], x[:, 3:]], 1)              # V2 = 0: rows agree
    assert torch.allclose((p.f(no_v2, u) - p.f_known(no_v2, u))[:, 3:], torch.zeros(64, 2), atol=1e-12)
    q = REGISTRY["network"]()
    xn = _box(_cfg("network"), 64, 34)
    dn = (q.f(xn, 0 * u) - q.f_known(xn, 0 * u)).view(64, 5, 2)
    v = xn[:, 0::2]
    want = torch.zeros(64, 5)
    want[:, 2], want[:, 3] = q.kappa * (v[:, 3] - v[:, 2]), q.kappa * (v[:, 2] - v[:, 3])
    assert torch.allclose(dn[:, :, 0], want, atol=1e-12) and dn[:, :, 1].abs().max() == 0


def test_partial_knowledge_with_saturating_actuator():
    """network: the modeller's model under the actuator is SaturatingActuator(KnownPart);
    the actuator design (j, k, x_ref, u_bar) is the nominal one (it is known hardware)."""
    sc = _cfg("network")
    full, part = build_plant(sc), build_plant(sc, knowledge="partial")
    assert isinstance(part, SaturatingActuator) and type(part.base).__name__ == "KnownPart"
    assert (part.j, part.k, part.x_ref, part.u_bar) == (full.j, full.k, full.x_ref, full.u_bar)
    x = _box(sc, 64, 35)
    u = 3 * torch.randn(64, 1, generator=torch.Generator().manual_seed(36))
    ue = full.u_bar * torch.tanh((u[:, 0] - full.k * (x[:, full.j] - full.x_ref)) / full.u_bar)
    want = REGISTRY["network"]().f_known(x, ue[:, None])                  # input enters additively
    assert torch.allclose(part.f(x, u), want, atol=1e-12)
    drift = build_plant(sc, perturb=0.3, knowledge="partial")
    assert torch.isfinite(drift.f(x, u)).all()
