"""Unit tests for the case-study plants and the casestudies patch of the package copy. CPU.

    cd experiments/casestudies/code && CUDA_VISIBLE_DEVICES="" ../../../.venv/bin/python -m pytest -q tests
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cs_common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.metrics import episode_metrics  # noqa: E402
from hyperkkl.systems import REGISTRY, build_plant  # noqa: E402
from hyperkkl.train import masked_sq_mean  # noqa: E402

torch.set_default_dtype(torch.float64)


def test_registry_and_shapes():
    for name, nx, ny in (("quad_planar", 7, 4), ("cstr_exo", 3, 2)):
        p = REGISTRY[name]()
        assert p.nx == nx and p.ny == ny
        x = torch.randn(5, nx) * 0.1
        if name == "cstr_exo":
            x = torch.tensor([[0.9, 334.0, 319.0]]).repeat(5, 1) + 0.1 * torch.randn(5, 3)
        u = torch.rand(5, 1) - 0.5
        for f in (p.f, p.f_known):
            out = f(x, u)
            assert out.shape == (5, nx) and torch.isfinite(out).all()
        assert torch.equal(p.h(x), x[:, list(p.sensors)])


def test_live_module_identical():
    live = (C.LIVE / "hyperkkl" / "systems_casestudy.py").read_bytes()
    copy = (C.PKG / "hyperkkl" / "systems_casestudy.py").read_bytes()
    assert live == copy


def test_quad_hover_equilibrium_and_drag():
    p = REGISTRY["quad_planar"]()
    x0 = torch.zeros(1, 7)
    assert torch.allclose(p.f(x0, torch.zeros(1, 1)), torch.zeros(1, 7), atol=1e-6)
    # the partial model is the full model without drag
    q = REGISTRY["quad_planar"](kd_x=0.0, kd_z=0.0)
    x = torch.randn(16, 7) * torch.tensor([1, .3, 1, .3, .2, .5, .3])
    u = torch.rand(16, 1) * 6 - 3
    assert torch.allclose(q.f(x, u), p.f_known(x, u))
    assert not torch.allclose(p.f(x, u), p.f_known(x, u))
    # accelerometer channel relaxes to the body-x drag -kd_x v_bx
    th, vx, vz = x[:, 4], x[:, 2], x[:, 3]
    am_ss = -p.kd_x * (vx * torch.cos(th) - vz * torch.sin(th))
    xs = x.clone(); xs[:, 6] = am_ss
    assert torch.allclose(p.f(xs, u)[:, 6], torch.zeros(16), atol=1e-9)


def test_quad_mismatch_params():
    p = REGISTRY["quad_planar"]().perturbed(0.3)
    assert p.g == 9.81 and p.tau_imu == 0.02
    assert math.isclose(p.Iyy, 2.5e-3 * math.exp(0.3)) and math.isclose(p.m, 0.5 * math.exp(-0.3))


def _cstr_steady(p, u, side=True):
    T = torch.linspace(280.0, 520.0, 24001)
    k1 = p._k(p.k1_ref, p.E1_R, T)
    k2 = p._k(p.k2_ref, p.E2_R, T) * (1.0 if side else 0.0)
    ca = p.C_Af / (1 + p.tau_res * (k1 + k2))
    a = p.Fc_Vj * (1 + u)
    Tj = (a * p.T_cin + p.gamma * T) / (a + p.gamma)
    res = (p.T_f - T) / p.tau_res + p.J1 * k1 * ca + p.J2 * k2 * ca - p.beta * (T - Tj)
    idx = torch.nonzero(torch.sign(res[1:]) != torch.sign(res[:-1]))[:, 0]
    out = []
    f = p.f if side else p.f_known
    for i in idx.tolist():
        x = torch.stack([ca[i], T[i], Tj[i]])
        J = torch.func.jacrev(lambda xx: f(xx[None], torch.tensor([[u]]))[0])(x)
        out.append((float(T[i]), float(ca[i]), float(torch.linalg.eigvals(J).real.max())))
    return out


def test_cstr_unique_stable_steady_state_over_box():
    for rel in (0.0, 0.1, 0.3):
        p = REGISTRY["cstr_exo"]()
        p = p.perturbed(rel) if rel else p
        for u in (-0.5, 0.0, 0.5):
            for side in (True, False):
                s = _cstr_steady(p, u, side)
                assert len(s) == 1, (rel, u, side, s)
                assert s[0][2] < 0, (rel, u, side, s)
    s = _cstr_steady(REGISTRY["cstr_exo"](), 0.0)
    assert 330 < s[0][0] < 337 and 0.8 < s[0][1] < 1.0


def test_cstr_mismatch_activation_energy_quarter():
    p = REGISTRY["cstr_exo"]()
    q = p.perturbed(0.3)
    for k in p.NOT_PERTURBED:
        assert getattr(q, k) == getattr(p, k)
    for k in p.E_PARAMS:
        r = getattr(q, k) / getattr(p, k)
        assert math.isclose(abs(math.log(r)), 0.075, rel_tol=1e-9)


def test_measure_hold_mult_burst():
    E, N1, ny = 3, 50, 3
    yc = torch.zeros(E, N1, ny)
    y = Dm.measure(yc, {"noise": 1.0, "noise_mult": [1.0, 0.0, 2.0], "hold": [10, 1, 1]}, [1.0, 1.0, 1.0], 0)
    assert torch.equal(y[:, :, 1], yc[:, :, 1])                         # mult 0: no noise
    for k in range(N1):
        assert torch.equal(y[:, k, 0], y[:, (k // 10) * 10, 0])        # 10x hold
    assert y[:, 1:10, 2].std() > 1.0                                    # fresh every step, mult 2
    assert not torch.equal(y[:, 1, 0], y[:, 0, 2])
    yb = Dm.measure(torch.randn(E, N1, ny), {"noise": 0.0, "burst": {"start_frac": 0.5, "len_tref": 0.1,
                                                                    "channels": [0]}}, [1.0] * 3, 0, dt=0.01)
    k0, L = Dm.burst_window({"start_frac": 0.5, "len_tref": 0.1}, N1, 0.01)
    assert (yb[:, k0 + 1:k0 + L + 1, 0] == yb[:, k0:k0 + 1, 0]).all()
    assert not (yb[:, k0 + 1:k0 + L + 1, 1] == yb[:, k0:k0 + 1, 1]).all()


def _es(E=4, N1=200, nx=3):
    x = torch.randn(E, N1, nx)
    z = torch.zeros(E, N1, 2)
    return Dm.EpisodeSet(x.float(), torch.zeros(E, N1, 1), x[:, :, 1:].float(), x[:, :, 1:].float(), z, z,
                         ["step"] * E, 10, 1.0 / 6.0)


def test_sparse_labels_and_interp():
    es = _es(N1=2000)
    sp = Dm.sparse_labels(es, {"states": [0], "every": 45.0, "jitter": 15.0, "noise": 0.0}, [1.0, 1.0, 1.0], 3)
    lab = torch.isfinite(sp.x[:, :, 0])
    assert torch.isfinite(sp.x[:, :, 1:]).all()
    n = lab.sum(1)
    assert (n >= 5).all() and (n <= 11).all()                           # 2000 samples = 333 min / (45 +- 15) min
    assert torch.equal(sp.x[:, :, 0][lab], es.x[:, :, 0][lab])
    it = Dm.interp_labels(sp)
    assert torch.isfinite(it.x).all() and torch.equal(it.x[:, :, 0][lab], es.x[:, :, 0][lab])
    assert Dm.interp_labels(es) is es


def test_masked_metrics_and_loss_reduce_to_dense():
    x = torch.randn(4, 100, 3); xh = x + 0.1 * torch.randn(4, 100, 3)
    m = episode_metrics(x, xh, [1.0, 2.0, 3.0], 50, [1, 2])
    xs = x.clone(); xs[:, 1::2, 0] = float("nan")
    ms = episode_metrics(xs, xh, [1.0, 2.0, 3.0], 50, [1, 2])
    assert np.allclose(ms["nrmse_x2"], m["nrmse_x2"], atol=1e-6) and np.allclose(ms["nrmse_x3"], m["nrmse_x3"], atol=1e-6)
    assert np.all(np.isfinite(ms["nrmse"])) and not ms["diverged"].any()
    sig = torch.tensor([1.0, 2.0, 3.0])
    p, t = torch.randn(64, 3), torch.randn(64, 3)
    assert torch.equal(masked_sq_mean(p, t, sig), ((p - t) / sig).pow(2).mean())
    t2 = t.clone(); t2[::2, 0] = float("nan")
    l = masked_sq_mean(p, t2, sig)
    e = ((p - t) / sig).pow(2)
    ref = (e[1::2, 0].mean() + e[:, 1].mean() + e[:, 2].mean()) / 3
    assert torch.allclose(l, ref)


def test_infolib_cs_selftest():
    import infolib_cs
    assert infolib_cs.selftest()


def test_filters_skip_stale_samples():
    from hyperkkl.observers import EKF
    assert C.Ob.HOLD_MISSING_DEFAULT is True
    p = build_plant({"plant": "cstr_exo", "sensors": [1, 2], "ic_box": [[0.3, 1.2], [327, 352], [314, 334]],
                     "actuator": {"kind": "affine"}})
    f = EKF(p, 1 / 6, [0.01, 0.1, 0.1], [0.1, 0.1], [0.3, 7.0, 6.0])
    assert f.hold_missing and not f._plain
