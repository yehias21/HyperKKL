"""Unit tests for hyperkkl/systems_bounded.py (cartpole_b, two_link_arm_b).

    CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest tests/test_systems_bounded.py -q
"""
import math

import numpy as np
import pytest
import torch

from hyperkkl import data as Dm
from hyperkkl.config import load
from hyperkkl.systems import REGISTRY, CartPole, TwoLinkArm, build_plant

torch.set_default_dtype(torch.float64)
NEW = {"cartpole_b": CartPole, "two_link_arm_b": TwoLinkArm}


def _x(name, n=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    box = torch.tensor(load(name)["system"]["ic_box"], dtype=torch.float64)
    return box[:, 0] + (box[:, 1] - box[:, 0]) * torch.rand(n, box.shape[0], generator=g)


@pytest.mark.parametrize("name", list(NEW))
def test_registry_and_dims(name):
    cls = REGISTRY[name]                                   # lazy import via hyperkkl/__init__
    assert issubclass(cls, NEW[name]) and cls is not NEW[name]
    cfg = load(name)
    p = build_plant(cfg["system"])
    x = _x(name); u = torch.zeros(len(x), 1)
    assert p.f(x, u).shape == (len(x), 4) and p.h(x).shape == (len(x), 2)
    assert tuple(p.sensors) == tuple(NEW[name].DEFAULT_SENSORS)
    assert type(cls().perturbed(0.3)) is cls               # mismatch keeps the bounded class
    assert cls.UNMODELLED == NEW[name].UNMODELLED


def test_existing_classes_untouched():
    assert CartPole.K_BAL == (0.0, 0.0, -9.889688, -2.181384)
    assert "two_link_arm" in REGISTRY and "cartpole" in REGISTRY


@pytest.mark.parametrize("name", list(NEW))
def test_f_known_differs_only_by_declared_terms(name):
    p = REGISTRY[name](); x = _x(name, 512); u = torch.randn(len(x), 1)
    d = p.f(x, u) - p.f_known(x, u)
    if name == "cartpole_b":
        # same K in f and f_known; the large-angle terms vanish at th = om = 0
        ref = CartPole(); ref.K_BAL = p.K_BAL
        assert torch.allclose(p.f(x, u), ref.f(x, u)) and torch.allclose(p.f_known(x, u), ref.f_known(x, u))
        x0 = x.clone(); x0[:, 2:] = 0
        assert (p.f(x0, u) - p.f_known(x0, u)).abs().max() < 1e-12
        x1 = x0.clone(); x1[:, 2] = 1e-3
        assert (p.f(x1, u) - p.f_known(x1, u)).abs().max() < 1e-4     # O(th^2..th^3)
    else:
        # the elbow spring is in both: f - f_known equals the original arm's (Coriolis / centrifugal)
        ref = TwoLinkArm()
        assert torch.allclose(d, ref.f(x, u) - ref.f_known(x, u), atol=1e-12)
        x0 = x.clone(); x0[:, 2:] = 0
        assert (p.f(x0, u) - p.f_known(x0, u)).abs().max() < 1e-12      # no velocity -> no Coriolis
        assert (p.f(x, u) - ref.f(x, u)).abs().max() > 1e-3                # the spring is there
    kin = [0, 2] if name == "cartpole_b" else [0, 1]
    assert d[:, kin].abs().max() < 1e-12                                    # kinematic rows identical


@pytest.mark.parametrize("name,limit", [("cartpole_b", (4.0, None, 1.0, None)), ("two_link_arm_b", (None, math.pi, None, None))])
def test_bounded_long_random_inputs(name, limit):
    """5 x T rollouts under piecewise-constant random inputs at the box edges (|v| <= 1)."""
    cfg = load(name); scfg = cfg["system"]
    p = build_plant(scfg)
    des0 = Dm.make_design(scfg)
    des = Dm.Design(des0.A, des0.B, des0.dt, 5 * des0.N, des0.alpha, des0.step)
    rng = np.random.RandomState(3); n = 24
    dwell = int(round(2 * float(scfg.get("t_ref", 1)) / des.dt))
    levels = rng.choice([-1.0, 1.0], size=(n, des.N // dwell + 2)) * rng.uniform(0.5, 1.0, size=(n, des.N // dwell + 2))
    sigs = [lambda t, L=levels[i]: L[np.minimum((np.asarray(t) / des.dt).round().astype(int) // dwell, len(L) - 1)] for i in range(n)]
    x0 = Dm._sobol(np.asarray(scfg["ic_box"], float), n, 2)
    x, _, _ = Dm.integrate(p, des, x0, sigs, None)
    assert torch.isfinite(x).all()
    for j, lim in enumerate(limit):
        if lim is not None:
            assert x[:, des0.N:, j].abs().max() < lim, (name, j, float(x[..., j].abs().max()))
