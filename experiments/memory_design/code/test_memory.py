"""Unit tests of the memory-design pieces (PREREG section 5). CPU, float64 where it matters.

    CUDA_VISIBLE_DEVICES="" python -m pytest -q test_memory.py
"""
import math

import numpy as np
import pytest
import torch

from common_md import Dm, run_cfg  # noqa: F401
from hyperkkl import beat as BG
from hyperkkl import memory as MM
from hyperkkl.conditioning import build_conditioner, match_width
from hyperkkl.data import run_latent
from hyperkkl.nets import BaseMap, build_context, n_params
from hyperkkl.observers import LatentStep, run
from hyperkkl.train import _stats

torch.manual_seed(0)
SYSTEMS = ["quad_tank", "battery", "vdp", "bioreactor"]


def parts(system="vdp", arm=None, dec_width=128):
    cfg = run_cfg("C3-partial", system, 0)
    des0 = Dm.make_design(cfg["system"])
    des = des0 if arm is None else MM.m_design(cfg, des0, arm)
    nz, nx = des.A.shape[0], len(cfg["system"]["ic_box"])
    zs = torch.randn(512, nz)
    xs = torch.randn(512, nx)
    T0 = BaseMap(nx, nz, 32, 2, "silu", _stats(xs), _stats(zs))
    Ti = BaseMap(nz, nx, dec_width, 3, "tanh", _stats(zs), _stats(xs))
    ctx = build_context("sfm", cfg["phase2"]["context"], 1, des.dt, np.linalg.eigvals(des.A), 1.0)
    return cfg, des0, des, T0, Ti, ctx, nz, nx


# ----------------------------------------------------------------------------- (i) strided window
def test_window_causal_and_train_equals_deploy():
    cfg, des0, des, T0, Ti, ctx, nz, nx = parts("vdp")
    hist = BG.HistoryFeatures(nz, 20, 3, None, z_std=torch.rand(nz) + 0.5)
    for q in hist.parameters():
        torch.nn.init.normal_(q, std=0.3)
    cond = build_conditioner("lora", Ti, ctx.dim + hist.dim, 16, 4, torch.ones(nz))
    for q in cond.parameters():
        if q.requires_grad:
            torch.nn.init.normal_(q, std=0.05)
    r = {"cond": cond, "ctx": ctx, "hist": hist}
    ob = BG.make_observer("m3", des, {"T0": T0}, r, "cpu")
    B, N1 = 3, 200
    y, u = torch.randn(B, N1, 1), torch.randn(B, N1, 1)
    x0 = torch.randn(B, nx)
    with torch.no_grad():
        xh = run(ob, y, u, x0)
        # training path: whole-sequence gather
        zseq = run_latent(des, y.double(), T0(x0).double()).float()
        c = ctx.seq(u)
        k = torch.arange(N1)[None].expand(B, -1)
        h = hist.at(zseq, None, k)
        pred = cond(zseq.reshape(-1, nz), torch.cat([c, h], -1).reshape(-1, ctx.dim + hist.dim)).reshape(B, N1, nx)
        assert torch.allclose(xh, pred, atol=1e-4, rtol=1e-4)
        # causality: perturb y, u after k0
        k0 = 120
        y2, u2 = y.clone(), u.clone()
        y2[:, k0 + 1:] += 5.0; u2[:, k0 + 1:] -= 3.0
        xh2 = run(ob, y2, u2, x0)
        assert torch.equal(xh[:, :k0 + 1], xh2[:, :k0 + 1])
        assert not torch.allclose(xh[:, k0 + 5:], xh2[:, k0 + 5:])


# ----------------------------------------------------------------------------- (ii) M4 warp
@pytest.mark.parametrize("system", SYSTEMS)
def test_warp_contraction(system):
    cfg, des0, des, T0, Ti, ctx, nz, nx = parts(system, "M2")
    w = MM.WarpLatent(des, ctx.dim, rate=1.0)
    a2 = float(-np.linalg.eigvals(des.A).real.max())            # alpha of the M2 design
    # ||Ad(s)||_2 < 1 and = exp(s * max Re(lambda) dt) on a grid of s
    for s in np.linspace(MM.S_MIN, 1.0, 7):
        Ad = torch.linalg.matrix_exp(torch.as_tensor(s * des.A * des.dt))
        nrm = float(torch.linalg.matrix_norm(Ad, 2))
        assert nrm < 1.0
        assert abs(nrm - math.exp(-s * a2 * des.dt)) < 1e-9
    # two latents with different z0, same y, randomly fast-switching s in [s_min, 1]
    N = 3000
    s = MM.S_MIN + (1 - MM.S_MIN) * torch.rand(1, N)
    y = torch.randn(1, N, des.B.shape[1])
    za, zb = torch.randn(1, nz), torch.randn(1, nz)
    d0 = float((za - zb).norm())
    for k in range(N):
        za, zb = w.step(za, y[:, k], s[:, k]), w.step(zb, y[:, k], s[:, k])
        bound = d0 * math.exp(-MM.S_MIN * a2 * des.dt * (k + 1)) * (1 + 1e-4)
        assert float((za - zb).norm()) <= bound + 1e-6


def test_warp_s_equals_one_matches_matrix_exp():
    cfg, des0, des, T0, Ti, ctx, nz, nx = parts("quad_tank", "M2")
    w = MM.WarpLatent(des, ctx.dim, rate=0.1)
    z = torch.randn(4, nz)
    yk = torch.randn(4, des.B.shape[1])
    for s in (1.0, 0.3, 0.1):
        Ad = torch.linalg.matrix_exp(torch.as_tensor(s * des.A * des.dt))
        Wz = (Ad - torch.eye(nz, dtype=Ad.dtype)) @ torch.linalg.inv(torch.as_tensor(des.A)) @ torch.as_tensor(des.B)
        ref = z.double() @ Ad.T + yk.double() @ Wz.T
        got = w.step(z, yk, torch.full((4,), s))
        assert torch.allclose(got.double(), ref, atol=1e-5, rtol=1e-4)


def test_warp_scan_equals_deployed():
    cfg, des0, des, T0, Ti, ctx, nz, nx = parts("vdp", "M2")
    w = MM.WarpLatent(des, ctx.dim, rate=1.5)
    for q in w.g.parameters():
        torch.nn.init.normal_(q, std=1.0)
    cond = build_conditioner("lora", Ti, ctx.dim, 16, 4, torch.ones(nz))
    for q in cond.parameters():
        if q.requires_grad:
            torch.nn.init.normal_(q, std=0.05)
    r = {"cond": cond, "ctx": ctx, "hist": None, "warp": w}
    ob = MM.make_observer_md("m4", des, {"T0": T0}, r, "cpu")
    B, N1 = 3, 700
    y, u = torch.randn(B, N1, 1), torch.randn(B, N1, 1)
    x0 = torch.randn(B, nx)
    with torch.no_grad():
        xh = run(ob, y, u, x0)
        c = ctx.seq(u)
        zseq, s = w.seq(y, c, T0(x0))
        assert float(s.min()) >= MM.S_MIN - 1e-6 and float(s.max()) <= 1.0 + 1e-6
        assert float(s.std()) > 1e-3                         # s really varies in this test
        pred = cond(zseq.reshape(-1, nz), c.reshape(-1, ctx.dim)).reshape(B, N1, nx)
        assert torch.allclose(xh, pred, atol=2e-4, rtol=2e-3), float((xh - pred).abs().max())
        # causality
        k0 = 300
        y2, u2 = y.clone(), u.clone()
        y2[:, k0 + 1:] += 5.0; u2[:, k0 + 1:] -= 3.0
        xh2 = run(ob, y2, u2, x0)
        assert torch.equal(xh[:, :k0 + 1], xh2[:, :k0 + 1])
    # gradients flow into the pole-scaling net through the scan
    c = ctx.seq(u)
    zseq, s = w.seq(y, c, T0(x0).detach())
    zseq.pow(2).mean().backward()
    assert all(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in w.g.parameters())


def test_lin_scan():
    a = torch.rand(2, 1000, 3) * 0.9 + 0.05
    b = torch.randn(2, 1000, 3)
    h = MM.lin_scan(a, b)
    ref, hp = [], torch.zeros(2, 3)
    for k in range(1000):
        hp = a[:, k] * hp + b[:, k]
        ref.append(hp)
    assert torch.allclose(h, torch.stack(ref, 1), atol=1e-5)


# ----------------------------------------------------------------------------- (iii) relabel == deployed latent
@pytest.mark.parametrize("arm", ["M1", "M2"])
def test_relabel_from_equals_deployed_latent(arm):
    cfg, des0, des, T0, Ti, ctx, nz, nx = parts("quad_tank", arm)
    B, N1 = 2, 400
    y = torch.randn(B, N1, 2)
    es = Dm.EpisodeSet(torch.zeros(B, N1, nx), torch.zeros(B, N1, 1), y, y, torch.zeros(B, N1, nz),
                       torch.zeros(B, N1, nz), ["a"] * B, 0, des.dt)
    z0 = torch.randn(nz)
    rl = MM.relabel_from(es, des, 7, z0)
    lat = LatentStep(des, "cpu")
    z = z0[None].expand(B, -1).clone()
    zs = [z]
    for k in range(N1 - 1):
        z = lat(z, y[:, k], y[:, k + 1])
        zs.append(z)
    assert torch.allclose(rl.z_obs, torch.stack(zs, 1), atol=1e-4, rtol=1e-4)
    assert rl.burn == 7


# ----------------------------------------------------------------------------- (iv) parameter matching
@pytest.mark.parametrize("system", SYSTEMS)
def test_param_match(system):
    totals = {}
    for arm in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        dname = {"M0": None, "M3": None, "M1": "M1", "M2": "M2", "M4": "M2", "M5": "M2"}[arm]
        cfg, des0, des, T0, Ti, ctx, nz, nx = parts(system, dname)
        for q in Ti.parameters():          # as in phase 2: the conditioner counts its own params only
            q.requires_grad_(False)
        budget = int(cfg["phase2"]["budget"])
        npar0 = match_width("lora", Ti, 16, budget, 4, torch.ones(nz))[1]
        window = 20 if arm in ("M3", "M5") else 0
        extra = MM.extra_params(nz, window, arm == "M4")
        dc = 16 + (16 if window else 0)
        if arm == "M0":
            cnt = npar0
        else:
            cnt = match_width("lora", Ti, dc, npar0 - extra, 4, torch.ones(nz))[1]
        # extra params as actually built
        built = 0
        if window:
            built += n_params(BG.HistoryFeatures(nz, window, 2, None))
        if arm == "M4":
            built += n_params(MM.WarpLatent(des, 16, 1.0))
        assert built == extra, (arm, built, extra)
        totals[arm] = cnt + n_params(ctx, trainable_only=False) + n_params(Ti, trainable_only=False) + built
    for arm, t in totals.items():
        print(system, arm, t, totals["M0"], round(t / totals["M0"] - 1, 4)); assert abs(t / totals["M0"] - 1) <= 0.01, (system, totals)
