"""Unit tests of the scaling-study additions (CPU).   .venv/bin/python -m pytest experiments/scaling/code/tests -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

import scaleplants as SP  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import design as D  # noqa: E402
from hyperkkl.systems import REGISTRY, build_plant  # noqa: E402


def test_registered_and_sizes():
    for nx in SP.NX_GRID:
        cls = REGISTRY[f"fhnring{nx // 2}"]
        assert cls.nx == nx and cls.N_NODES == nx // 2


def test_known_is_f_without_synapse():
    for nx in (4, 16):
        p = REGISTRY[f"fhnring{nx // 2}"]()
        q = REGISTRY[f"fhnring{nx // 2}"](g_syn=0.0)
        x = torch.randn(32, nx, dtype=torch.float64); u = torch.randn(32, 1, dtype=torch.float64)
        assert torch.allclose(p.f_known(x, u), q.f(x, u))
        assert not torch.allclose(p.f(x, u), p.f_known(x, u))


def test_reference_equations():
    """Explicit loop implementation of the docstring equations."""
    nx = 8; N = 4
    p = REGISTRY["fhnring4"]()
    x = torch.randn(3, nx, dtype=torch.float64); u = torch.randn(3, 1, dtype=torch.float64)
    out = p.f(x, u)
    sig = lambda v: 1 / (1 + torch.exp(-v))
    for b in range(3):
        for i in range(N):
            v, w = x[b, 2 * i], x[b, 2 * i + 1]
            nb = [(i + 1) % N, (i - 1) % N]
            dv = v - v ** 3 / 3 - w + p.I_ext + sum(p.kappa * (x[b, 2 * j] - v) for j in nb) \
                + sum(p.g_syn * (p.E_syn - v) * sig((x[b, 2 * j] - p.th_syn) / p.s_syn) for j in nb) \
                + (u[b, 0] if i == 0 else 0)
            eps = p.eps * np.exp(p.spread * (2 * ((i * SP.GOLDEN) % 1.0) - 1))
            dw = eps * (v + p.a - p.b * w)
            assert abs(float(out[b, 2 * i] - dv)) < 1e-10 and abs(float(out[b, 2 * i + 1] - dw)) < 1e-10


def test_sensor_rule_neighbours():
    for nx in SP.NX_GRID:
        N = nx // 2
        meas = {s // 2 for s in SP.sensors_for(N)}
        assert len(meas) == max(1, N // 2)
        for i in range(N):
            if i not in meas:
                assert {(i + 1) % N, (i - 1) % N} <= meas


def test_bank_design():
    for nx in SP.NX_GRID:
        sc = SP.system_cfg(nx)
        des = Dm.make_design(sc)
        ny = len(sc["sensors"])
        assert des.A.shape[0] == (2 * nx + 1 if ny == 1 else 5 * ny)
        assert des.dt == SP.DT and des.N == 2000
        assert SP.design_check(sc)["ok"]
        ev = np.linalg.eigvals(des.A)
        assert abs(-ev.real.max() - 0.3) < 1e-9 and abs(ev).max() * des.dt <= 0.1 + 1e-9
        if ny > 1:     # block-diagonal, channel j drives only its own bank
            m = 5
            for j in range(ny):
                assert np.all(des.B[j * m:(j + 1) * m, j] == 1) and des.B[:, j].sum() == m


def test_bank_latent_step_exact():
    sc = SP.system_cfg(16)
    des = Dm.make_design(sc)
    y = torch.randn(2, 50, des.B.shape[1], dtype=torch.float64).cumsum(1) * 0.1
    z = Dm.run_latent(des, y)
    # per-channel bank on its own = same numbers
    m = 5
    Am, Bm = D.butterworth_AB(m, 0.3, 1)
    dm = Dm.Design(Am, Bm, des.dt, des.N, 0.3, D.latent_step(Am, Bm, des.dt))
    for j in range(y.shape[-1]):
        assert torch.allclose(Dm.run_latent(dm, y[..., j:j + 1]), z[..., j * m:(j + 1) * m], atol=1e-12)


def test_capacity_and_data_rule():
    z = {nx: SP.sizes(nx) for nx in SP.NX_GRID}
    for nx in SP.NX_GRID:
        r = nx / 8
        assert z[nx]["budget"] == round(140_000 * r) and z[nx]["n_train"] == round(600 * r)
        assert abs(z[nx]["enc_width"] / (192 * np.sqrt(r)) - 1) < 0.01
    f = SP.sizes(64, data_nx=4)
    assert f["n_train"] == 300 and f["budget"] == SP.sizes(64)["budget"]


def test_levels_and_perturbation():
    c = {lev: SP.make_cfg(8, lev, 0) for lev in SP.LEVELS}
    assert c["partial"]["system"]["knowledge"] == "partial" and c["realistic"]["data"]["sensor"]["kind"] == "colored_t"
    p30 = build_plant(c["off30"]["system"], perturb=0.3).base
    assert p30.I_ext == 0.5 and abs(p30.a / 0.7 - np.exp(-0.3)) < 1e-12        # sorted alternating signs


def test_neighbours_heterogeneous():
    for nx in (16, 32, 64):
        p = REGISTRY[f"fhnring{nx // 2}"]()
        eps = p._consts(torch.float64, "cpu")[2].numpy()
        d = np.abs(np.log(eps) - np.log(np.roll(eps, 1)))
        assert d.min() > 0.2 * p.spread                 # no two ring neighbours (nearly) identical
