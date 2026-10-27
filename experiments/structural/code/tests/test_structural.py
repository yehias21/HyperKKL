"""Unit tests for the structural plants and the windowed grey-box (run with pytest, CPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "pkg")); sys.path.insert(0, str(HERE))
from hyperkkl import data as Dm, signals as S, systems as Sy  # noqa: E402
from hyperkkl.observers import EKF, run  # noqa: E402
import structural as ST  # noqa: E402
import wingb as WG  # noqa: E402

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree")
ST.install(Dm)
SYS = list(ST.STRUCTURAL)


def cfg_of(s):
    node = "TEST-C3" if s in ("cartpole", "highway_traffic") else "C3-realistic"
    return json.load(open(LIVE / node / "runs" / s / "seed0" / "results.json"))["config"]


@pytest.mark.parametrize("s", SYS)
def test_dims_and_model_omits_hidden(s):
    sc = dict(cfg_of(s)["system"]); sc["structural"] = ST.STRUCTURAL[s].NAME_S
    tp = Dm.build_plant(sc, perturb=0.3)
    model = Dm.model_plant(sc)                                   # knowledge partial
    ref = Dm.build_plant({k: v for k, v in sc.items() if k != "structural"}, perturb=0.3)
    assert isinstance(tp, ST.HiddenPlant) and not isinstance(model, ST.HiddenPlant)
    assert tp.nx == model.nx == ref.nx and tp.nh >= 1
    assert tp.sensors == ref.sensors
    nf = tp.nx + tp.nh
    x = torch.randn(7, nf, dtype=torch.float64).abs() + 0.5
    u = torch.zeros(7, 1, dtype=torch.float64)
    assert tp.f_full(x, u).shape == (7, nf) and tp.h_full(x).shape == (7, tp.ny)
    # the modeller's model acts on the visible state only: same shapes as the reference plant
    assert model.f(x[:, :tp.nx], u).shape == (7, tp.nx)
    # the hidden state matters to the true visible dynamics or to the measurement
    x2 = x.clone(); x2[:, tp.nx:] += 0.5 * (x[:, tp.nx:].abs() + 1)
    dv = (tp.f_full(x2, u)[:, :tp.nx] - tp.f_full(x, u)[:, :tp.nx]).abs().max()
    dy = (tp.h_full(x2) - tp.h_full(x)).abs().max()
    assert float(dv) > 1e-6 or float(dy) > 1e-6
    # parameters are perturbed exactly as in the reference plant
    base_ref = ref.base if isinstance(ref, Sy.SaturatingActuator) else ref
    assert tp.inner.params() == base_ref.params()


@pytest.mark.parametrize("s", ["vdp", "bioreactor", "cartpole"])
def test_generate_labels_visible_and_reference_unchanged(s):
    cfg = cfg_of(s)
    sc = Dm.system_scales(cfg, "cpu") if False else json.load(open(next(
        (Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/cache/scales")).glob(f"{s}_*.json"))))
    des = Dm.make_design(cfg["system"])
    des = Dm.Design(des.A, des.B, des.dt, 200, des.alpha, des.step)
    sc = dict(sc, t_burn=0.0)
    scfg = dict(cfg["system"]); scfg["structural"] = ST.STRUCTURAL[s].NAME_S
    es = Dm.generate(Dm.build_plant(scfg, 0.3), des, sc, scfg, dict(cfg["data"], k_cross=1),
                     S.families_for("id"), 4, "id", 11, "cpu")
    assert es.x.shape[-1] == len(cfg["system"]["ic_box"]) and es.x_hidden.shape[-1] == 1
    # reference path is the package's own generate, unchanged
    a = Dm.generate(Dm.build_plant(cfg["system"], 0.3), des, sc, cfg["system"], dict(cfg["data"], k_cross=1),
                    S.families_for("id"), 4, "id", 11, "cpu")
    from hyperkkl.data import EpisodeSet  # noqa: F401
    import importlib
    D0 = importlib.import_module("hyperkkl.data")
    assert not hasattr(a, "x_hidden")
    if s == "vdp":      # measurement is the lagged position, not x1
        assert float((es.y_clean[..., 0] - es.x[..., 0]).abs().max()) > 1e-3


def test_hidden_plant_causal_integration():
    """The structural plant's future input cannot change its past state (ZOH RK4 integration)."""
    cfg = cfg_of("quad_tank")
    scfg = dict(cfg["system"]); scfg["structural"] = "tank5"
    tp = Dm.build_plant(scfg, 0.3)
    des = Dm.make_design(cfg["system"]); des = Dm.Design(des.A, des.B, des.dt, 100, des.alpha, des.step)

    class Sig:
        def __init__(self, t0): self.t0 = t0
        def __call__(self, t): return np.where(t >= self.t0, 1.0, 0.0)
    x0 = np.array([[10.0, 10.0, 3.0, 3.0, 3.0]])
    xa, _, _ = Dm.integrate(ST._FullView(tp), des, x0, [Sig(10 * des.dt)])
    xb, _, _ = Dm.integrate(ST._FullView(tp), des, x0, [Sig(50 * des.dt)])
    assert torch.allclose(xa[:, :11], xb[:, :11]) and not torch.allclose(xa[:, 12:], xb[:, 12:])


def test_window_greybox_causal():
    """xhat_k of the windowed grey-box EKF/UKF does not depend on y or u after k."""
    cfg = cfg_of("vdp")
    model = Dm.model_plant(cfg["system"])
    lags = [1, 2, 4]
    net = WG.WinMLP(2, 1, len(lags) * 3, 8)
    gb = WG.WinGreyBox(model, net, [0, 0], [1, 1], [1.0], [0, 0], [1, 1], lags)
    B, N = 3, 30
    g = torch.Generator().manual_seed(0)
    y = torch.randn(B, N + 1, 1, generator=g); u = torch.randn(B, N + 1, 1, generator=g)
    x0 = torch.zeros(B, 2)
    for cls in (WG.WinEKF, WG.WinUKF):
        ob = cls(gb, 0.02, [0.1, 0.1], [0.1], [1.0, 1.0])
        a = run(ob, y, u, x0)
        y2, u2 = y.clone(), u.clone(); y2[:, 16:] += 5.0; u2[:, 16:] -= 3.0
        b = run(ob, y2, u2, x0)
        assert torch.allclose(a[:, :16], b[:, :16]) and not torch.allclose(a[:, 17:], b[:, 17:])


def test_window_greybox_zero_residual_equals_ekf():
    """With a zero residual network the windowed EKF is the plain EKF on the model."""
    cfg = cfg_of("vdp")
    model = Dm.model_plant(cfg["system"])
    net = WG.WinMLP(2, 1, 2 * 3, 8)
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    gb = WG.WinGreyBox(model, net, [0, 0], [1, 1], [1.0], [0, 0], [1, 1], [1, 2])
    B, N = 2, 20
    g = torch.Generator().manual_seed(1)
    y = torch.randn(B, N + 1, 1, generator=g); u = torch.randn(B, N + 1, 1, generator=g)
    x0 = torch.zeros(B, 2)
    a = run(WG.WinEKF(gb, 0.02, [0.1, 0.1], [0.1], [1.0, 1.0]), y, u, x0)
    b = run(EKF(model, 0.02, [0.1, 0.1], [0.1], [1.0, 1.0]), y, u, x0)
    assert torch.allclose(a, b, atol=1e-5)
