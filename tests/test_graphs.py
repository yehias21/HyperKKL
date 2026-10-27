"""CUDA-graph training (train._Graphed) must compute exactly what eager training computes.

    CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest tests/test_graphs.py -q
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from hyperkkl import design as D
from hyperkkl.conditioning import build_conditioner
from hyperkkl.data import Design, EpisodeSet
from hyperkkl.nets import BaseMap, build_context
from hyperkkl.observers import LatentStep
from hyperkkl.train import _Graphed, _rollout_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda:0"


def _setup(ctx_kind, seed=0):
    torch.manual_seed(seed)
    nz, nx, E, N1 = 5, 2, 12, 400
    A, Bm = D.butterworth_AB(nz, 1.5, 1)
    dt = D.choose_dt(A)
    des = Design(A, Bm, dt, N1 - 1, 1.5, D.latent_step(A, Bm, dt))
    g = torch.Generator().manual_seed(seed)
    es = EpisodeSet(x=torch.randn(E, N1, nx, generator=g), u=torch.randn(E, N1, 1, generator=g),
                    y_clean=torch.randn(E, N1, 1, generator=g), y=torch.randn(E, N1, 1, generator=g),
                    z=torch.randn(E, N1, nz, generator=g), z_obs=torch.randn(E, N1, nz, generator=g),
                    families=["x"] * E, burn=20, dt=dt).to(DEV)
    T0 = BaseMap(nx, nz, 32, 2, "silu", (torch.zeros(nx), torch.ones(nx)), (torch.zeros(nz), torch.ones(nz))).to(DEV)
    Ti = BaseMap(nz, nx, 32, 2, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx))).to(DEV)
    for p in list(T0.parameters()) + list(Ti.parameters()):
        p.requires_grad_(False)
    ccfg = {"dim": 8, "hidden": 16, "window": 7, "sfm_blocks": 4}
    ctx = build_context(ctx_kind, ccfg, 1, dt, np.linalg.eigvals(A), 1.0).to(DEV)
    inj = build_conditioner("latent", Ti, 8, 64, 4, torch.ones(nz)).to(DEV)
    for p in inj.parameters():                     # break the zero init so gradients are rich
        p.data.normal_(0, 0.05)
    return des, es, T0, Ti, ctx, inj


@pytest.mark.parametrize("ctx_kind", ["sfm", "gru"])
def test_graphed_rollout_equals_eager(ctx_kind):
    des, es, T0, Ti, ctx, inj = _setup(ctx_kind)
    lat = LatentStep(des, DEV)
    sig = torch.ones(2, device=DEV)
    K, Be = 64, 6
    params = [q for q in list(inj.parameters()) + list(ctx.parameters()) if q.requires_grad]
    e_buf = torch.zeros(Be, dtype=torch.long, device=DEV)
    k_buf = torch.zeros(Be, dtype=torch.long, device=DEV)
    fn = lambda: _rollout_loss(inj, ctx, lat, T0, Ti, es, e_buf, k_buf, K, sig)
    snapshot = copy.deepcopy([p.detach().clone() for p in params])

    def train(enabled, n=4):
        for p, s0 in zip(params, snapshot):
            p.data.copy_(s0)
        opt = torch.optim.Adam(params, lr=1e-3)
        gr = _Graphed(fn, params, enabled=enabled)
        rng = torch.Generator(device=DEV).manual_seed(1)
        losses, grads = [], []
        for _ in range(n):
            e_buf.copy_(torch.randint(0, len(es), (Be,), device=DEV, generator=rng))
            k_buf.copy_(torch.randint(20, 400 - K - 1, (Be,), device=DEV, generator=rng))
            loss = gr()
            losses.append(float(loss))
            grads.append(torch.cat([p.grad.reshape(-1) for p in params]).clone())
            opt.step()
        return losses, grads, [p.detach().clone() for p in params]

    le, ge, pe = train(False)
    lg, gg, pg = train(True)
    assert np.allclose(le, lg, rtol=1e-5, atol=1e-7), (le, lg)
    for a, b in zip(ge, gg):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-7), (a - b).abs().max()
    for a, b in zip(pe, pg):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)
