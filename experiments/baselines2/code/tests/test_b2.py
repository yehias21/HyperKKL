"""Unit tests for baselines2 (CPU): shapes, causality (no future y / u), sequence == online step,
parameter counts, the dynamics models inside EKF/UKF, the cache redirection."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import b2  # noqa: E402,F401  (installs the cache redirection)
from b2 import models as M  # noqa: E402
from b2 import train as T  # noqa: E402
from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.observers import EKF, UKF, run  # noqa: E402

torch.set_default_dtype(torch.float32)
NU, NY, NX = 1, 1, 3


def _norm():
    return M._Norm(2.0, [0.1], [0.5], [0.0, 0.1, -0.2], [1.0, 0.5, 2.0])


def _nets():
    torch.manual_seed(0)
    return [M.GRUNet(NU, NY, NX, 16, 2, _norm()),
            M.NCDENet(NU, NY, NX, 8, 24, _norm(), 0.05)]


def _data(B=3, L=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, NU, generator=g), torch.randn(B, L, NY, generator=g)


@pytest.mark.parametrize("i", [0, 1])
def test_shapes_and_seq_equals_online(i):
    net = _nets()[i].eval()
    u, y = _data()
    with torch.no_grad():
        xs = net.seq(u, y)
    assert xs.shape == (3, 40, NX)
    ob = M.SeqObserver("t", net)
    x0 = torch.full((3, NX), 7.0)
    xo = run(ob, y, u, x0)
    assert xo.shape == (3, 40, NX)
    assert torch.equal(xo[:, 0], x0)                     # x̂_0 = prior: no measurement read yet
    assert torch.allclose(xo[:, 1:], xs[:, 1:], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("i", [0, 1])
def test_causality_no_future_inputs(i):
    """Changing y_j, u_j for j > k must not change x̂_0..x̂_k (online and sequence mode)."""
    net = _nets()[i].eval()
    u, y = _data()
    k = 17
    u2, y2 = u.clone(), y.clone()
    u2[:, k + 1:] += 5.0
    y2[:, k + 1:] -= 3.0
    x0 = torch.zeros(3, NX)
    a, b = run(M.SeqObserver("t", net), y, u, x0), run(M.SeqObserver("t", net), y2, u2, x0)
    assert torch.allclose(a[:, :k + 1], b[:, :k + 1])
    assert not torch.allclose(a[:, k + 1:], b[:, k + 1:])
    with torch.no_grad():
        sa, sb = net.seq(u, y), net.seq(u2, y2)
    assert torch.allclose(sa[:, :k + 1], sb[:, :k + 1])


def test_param_counts():
    for H in (16, 124, 235):
        net = M.GRUNet(NU, NY, NX, H, 2, _norm())
        assert sum(p.numel() for p in net.parameters()) == M.gru_params(NU, NY, NX, H)
    for W in (24, 314):
        net = M.NCDENet(NU, NY, NX, 32, W, _norm(), 0.1)
        assert sum(p.numel() for p in net.parameters()) == M.ncde_params(NU, NY, NX, 32, W)
    for W in (64, 371):
        net = BD.ResidualMLP(NX, NU, W, [0] * NX, [1] * NX, 1.0, np.zeros(NX), np.ones(NX))
        assert sum(p.numel() for p in net.parameters()) == M.mlp_params(NX, NU, W)
    H = M.solve_gru(NU, NY, NX, 140_000)
    assert abs(M.gru_params(NU, NY, NX, H) - 140_000) / 140_000 < 0.02
    W = M.solve_ncde(NU, NY, NX, 32, 140_000)
    assert abs(M.ncde_params(NU, NY, NX, 32, W) - 140_000) / 140_000 < 0.02
    W = M.solve_mlp(NX, NU, 140_000)
    assert abs(M.mlp_params(NX, NU, W) - 140_000) / 140_000 < 0.02


def test_tbptt_windows_match_full_sequence():
    """The stateful TBPTT forward (window by window, state carried) equals the full-sequence pass."""
    for net in _nets():
        net.eval()
        u, y = _data(L=33)
        with torch.no_grad():
            full = net.norm.out_norm(net.seq(u, y))
            if net.kind == "gru":
                o1, h = net.seq_norm(u[:, :16], y[:, :16])
                o2, _ = net.seq_norm(u[:, 16:], y[:, 16:], h)
                w = torch.cat([o1, o2], 1)
                assert torch.allclose(w, full, atol=1e-5)
            else:
                X = net.X(u, y)
                z0 = net.zeta(X[:, 0])
                o1, z = net.seq_norm_from(z0, X[:, :17])
                Xb = net.X(u[:, 16:], y[:, 16:], torch.full((3,), 16))
                o2, _ = net.seq_norm_from(z, Xb)
                assert torch.allclose(torch.cat([o1, o2], 1), full[:, 1:], atol=1e-5)


def test_dynamics_models_in_filters():
    cfg = {"name": "vdp", "plant": "vdp"}
    from hyperkkl.systems import build_plant
    import json
    from b2 import LIVE
    rc = json.load(open(LIVE / "experiments/tree/C3-partial/runs/vdp/seed0/results.json"))["config"]
    model = Dm.model_plant(rc["system"])
    nx, nu = model.nx, model.nu
    net = BD.ResidualMLP(nx, nu, 16, [0.0] * nx, [1.0] * nx, 1.0, np.zeros(nx), 0.1 * np.ones(nx))
    for hybrid in (False, True):
        gm = M.deployed_dynamics(model, net, hybrid)
        x = torch.randn(5, nx)
        u = torch.randn(5, nu)
        ref = net(x, u) + (model.f(x, u) if hybrid else 0)
        assert torch.allclose(gm.f(x, u), ref, atol=1e-5)
        # rk4_train is the same RK4 as the filter's step model
        from hyperkkl.observers import _rk4
        a = M.rk4_train(model if hybrid else None, net, x, u, 0.01)
        b = _rk4(gm, x, u, u, u, 0.01)
        assert torch.allclose(a, b, atol=1e-5)
        for F in (EKF, UKF):
            ob = F(gm, 0.02, 0.01 * np.ones(nx), 0.05 * np.ones(model.ny), np.ones(nx))
            Y = torch.randn(4, 30, model.ny)
            U = torch.randn(4, 30, nu)
            xh = run(ob, Y, U, torch.zeros(4, nx))
            assert xh.shape == (4, 30, nx) and torch.isfinite(xh).all()
            Y2 = Y.clone(); Y2[:, 20:] += 1.0                       # filter causality
            xh2 = run(ob, Y2, U, torch.zeros(4, nx))
            assert torch.allclose(xh[:, :20], xh2[:, :20])


def test_rollout_loss_k1_is_one_step():
    torch.manual_seed(0)
    net = BD.ResidualMLP(2, 1, 8, [0, 0], [1, 1], 1.0, np.zeros(2), np.ones(2))
    X = torch.randn(4, 50, 2)
    U = torch.randn(4, 50, 1)
    e = torch.tensor([0, 1, 2, 3]); s0 = torch.tensor([0, 5, 10, 40])
    sig = torch.ones(2)
    l1 = T._rollout_loss(None, net, X, U, e, s0, 1, 0.01, sig)
    x1 = M.rk4_train(None, net, X[e, s0], U[e, s0], 0.01)
    assert torch.allclose(l1, (x1 - X[e, s0 + 1]).pow(2).mean())


def test_cache_redirect_never_writes_live():
    assert Dm.CACHE.resolve() == b2.LOCAL_CACHE.resolve()
    assert Dm._cached is b2._redirected_cached
