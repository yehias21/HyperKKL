"""Tests for the archive-recipe extension (experiments/archive_recipe/RECIPE.md), CPU only.

    .venv/bin/python -m pytest tests/test_archive_recipe.py -q
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hyperkkl import design as D
from hyperkkl.conditioning import build_conditioner, match_width
from hyperkkl.config import load
from hyperkkl.data import Design
from hyperkkl.job import method_cfg
from hyperkkl.nets import BaseMap, build_context, n_params
from hyperkkl.observers import KKLObserver, LatentStep, run
from hyperkkl.soap import SOAP, make_optimizer
from hyperkkl.train import _decoder_loss, _decoder_loss_extras

torch.set_default_dtype(torch.float32)


def _base(d_in, d_out, width=32, act="tanh"):
    return BaseMap(d_in, d_out, width, 3, act, (torch.zeros(d_in), torch.ones(d_in)),
                   (torch.zeros(d_out), torch.ones(d_out)))


# ------------------------------------------------------------------------- SOAP
def test_soap_equals_adam_on_1d_parameters():
    """ndim != 2: SOAP is Adam with the same betas / eps, step for step."""
    torch.manual_seed(0)
    a = torch.randn(7, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    oa = SOAP([a], lr=1e-2)
    ob = torch.optim.Adam([b], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    t = torch.randn(7)
    for _ in range(30):
        for p, o in ((a, oa), (b, ob)):
            o.zero_grad()
            ((p - t) ** 4).sum().backward()
            o.step()
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()


def test_soap_reduces_to_adam_when_preconditioners_are_diagonal():
    """Adam limit: with diagonal gradients L, R are diagonal, the eigenbases are signed
    permutations, and Adam is equivariant to those, so SOAP == Adam (incl. the basis
    refreshes and moment rotations every precond_freq steps)."""
    torch.manual_seed(0)
    d = torch.tensor([1.0, 3.0, 10.0, 30.0])
    W = torch.diag(torch.tensor([1.0, -2.0, 0.5, 1.5])).requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    oa = SOAP([W], lr=1e-2, precond_freq=3)
    ob = torch.optim.Adam([W2], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    for _ in range(40):
        for p, o in ((W, oa), (W2, ob)):
            o.zero_grad()
            (d * torch.diagonal(p) ** 2).sum().backward()      # gradient = diag(2 d w)
            o.step()
    assert torch.allclose(W, W2, atol=1e-5), (W - W2).abs().max()


def test_soap_moment_rotation_preserves_the_matrix():
    """M_new = P_L M P_R expresses the SAME moment in the new eigenbasis."""
    torch.manual_seed(1)
    QLo, _ = torch.linalg.qr(torch.randn(5, 5)); QLn, _ = torch.linalg.qr(torch.randn(5, 5))
    QRo, _ = torch.linalg.qr(torch.randn(3, 3)); QRn, _ = torch.linalg.qr(torch.randn(3, 3))
    M = torch.randn(5, 3)
    Mn = (QLn.T @ QLo) @ M @ (QRo.T @ QRn)
    assert torch.allclose(QLo @ M @ QRo.T, QLn @ Mn @ QRn.T, atol=1e-5)


def test_soap_converges_on_an_ill_conditioned_quadratic():
    """Tiny convergence check (deterministic least squares, cond(X^T X) ~ 1e2). Not a ranking
    test: on this toy Adam with SOAP's betas (0.95, 0.95) converges faster than SOAP, which is
    why the test design carries an 'adam95' control (STUDY/RESULTS_TEST deviation note)."""
    g = torch.Generator().manual_seed(0)
    m, k, n = 16, 8, 256
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g)); V, _ = torch.linalg.qr(torch.randn(m, m, generator=g))
    X = torch.randn(n, m, generator=g) @ (U * torch.logspace(0, -1.0, m)) @ V.T
    Y = X @ torch.randn(m, k, generator=g)
    W = torch.zeros(m, k, requires_grad=True)
    opt = SOAP([W], lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        ((X @ W - Y) ** 2).mean().backward()
        opt.step()
    ratio = float(((X @ W - Y) ** 2).mean().detach() / (Y ** 2).mean())
    assert ratio < 3e-2, ratio
    assert torch.isfinite(W).all()


# ---------------------------------------------------- conditioner: collapse, budget
def _lora_with_encoder(dc=8, width=None, budget=20000):
    torch.manual_seed(0)
    dec, enc = _base(5, 2), _base(2, 5, act="silu")
    w, n = match_width("lora", dec, dc, budget, 4, torch.ones(5), enc_base=enc)
    m = build_conditioner("lora", dec, dc, w if width is None else width, 4, torch.ones(5), enc_base=enc)
    for p in m.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.1)                  # break zero-init: non-trivial collapse test
    return m, dec, enc, w, n


def test_parameter_budget_counts_encoder_heads():
    dec, enc = _base(5, 2), _base(2, 5, act="silu")
    w0, n0 = match_width("lora", dec, 8, 20000, 4, torch.ones(5))
    w1, n1 = match_width("lora", dec, 8, 20000, 4, torch.ones(5), enc_base=enc)
    assert abs(n0 - 20000) / 20000 < 0.05 and abs(n1 - 20000) / 20000 < 0.05, (n0, n1)
    assert w1 < w0                                    # encoder heads share the same budget
    m = build_conditioner("lora", dec, 8, w1, 4, torch.ones(5), enc_base=enc)
    assert n_params(m) == n1
    assert all(not p.requires_grad for p in enc.parameters())


@pytest.mark.parametrize("budget", [140000])
def test_parameter_budget_at_production_size(budget):
    """At the production shapes (dec 128 x 3, enc 192 x 3, dc 16) the encoder half stays at 140k +- 3%."""
    dec = BaseMap(9, 2, 128, 3, "tanh", (torch.zeros(9), torch.ones(9)), (torch.zeros(2), torch.ones(2)))
    enc = BaseMap(2, 9, 192, 3, "silu", (torch.zeros(2), torch.ones(2)), (torch.zeros(9), torch.ones(9)))
    for eb in (None, enc):
        _, n = match_width("lora", dec, 16, budget, 4, torch.ones(9), enc_base=eb)
        assert abs(n - budget) / budget < 0.03, (eb is not None, n)


def test_zero_context_collapse_is_exact_with_encoder_half_and_ctx_scale():
    m, dec, enc, _, _ = _lora_with_encoder()
    m.ctx_scale.fill_(7.3)                            # the guard multiplies c: still exact at c = 0
    z, x = torch.randn(11, 5), torch.randn(11, 2)
    c0 = torch.zeros(11, 8)
    assert torch.equal(m(z, c0), dec(z))
    assert torch.equal(m.encode(x, c0), enc(x))
    assert float(m.delta_sq([c0], [1.0])) == 0.0
    c = torch.randn(11, 8)
    assert not torch.allclose(m(z, c), dec(z), atol=1e-4)
    assert not torch.allclose(m.encode(x, c), enc(x), atol=1e-4)


@pytest.mark.parametrize("kind", ["gru", "sfm"])
def test_deployed_observer_collapses_to_autonomous_at_zero_input(kind):
    """u == 0: the context is exactly 0 (the v2 GRU subtracts its zero-input response; the
    archive's GRU did not), so lora + any context == the autonomous observer, bit for bit."""
    torch.manual_seed(0)
    A, B = D.butterworth_AB(5, 1.0, 1)
    des = Design(A, B, 0.05, 80, 1.0, D.latent_step(A, B, 0.05))
    lat = LatentStep(des, "cpu")
    E = build_context(kind, {"dim": 8, "hidden": 16, "sfm_blocks": 4}, 1, 0.05, np.linalg.eigvals(A), 1.0)
    m, dec, enc, _, _ = _lora_with_encoder()
    y, u = torch.randn(3, 81, 1), torch.zeros(3, 81, 1)
    x0 = torch.randn(3, 2)
    a = run(KKLObserver("a", lat, m, ctx=E, z0_map=enc), y, u, x0)
    b = run(KKLObserver("b", lat, lambda z, c: dec(z), ctx=None, z0_map=enc), y, u, x0)
    assert torch.equal(a, b), (a - b).abs().max()


def test_causality_estimate_at_k_ignores_future_inputs_and_outputs():
    torch.manual_seed(0)
    A, B = D.butterworth_AB(5, 1.0, 1)
    des = Design(A, B, 0.05, 60, 1.0, D.latent_step(A, B, 0.05))
    lat = LatentStep(des, "cpu")
    E = build_context("gru", {"dim": 8, "hidden": 16}, 1, 0.05, None, 1.0)
    m, dec, enc, _, _ = _lora_with_encoder()
    obs = KKLObserver("o", lat, m, ctx=E, z0_map=enc)
    y, u = torch.randn(2, 61, 1), torch.randn(2, 61, 1)
    x0 = torch.zeros(2, 2)
    a = run(obs, y, u, x0)
    k = 30
    y2, u2 = y.clone(), u.clone()
    y2[:, k + 1:] += 5.0
    u2[:, k + 1:] = -u2[:, k + 1:]
    b = run(obs, y2, u2, x0)
    assert torch.equal(a[:, :k + 1], b[:, :k + 1])
    assert not torch.allclose(a[:, k + 1:], b[:, k + 1:])


def test_delta_sq_matches_explicit_weight_deltas():
    m, dec, enc, _, _ = _lora_with_encoder()
    cs = [torch.randn(6, 8) for _ in range(3)]
    w = [1.0, -2.0, 1.0]
    got = m.delta_sq(cs, w)
    ref = 0.0
    fs = [m.lora_factors(c) for c in cs]
    for li in range(len(fs[0])):
        Dsum = sum(wj * torch.bmm(f[li][0], f[li][1]) / math.sqrt(m.rank) for wj, f in zip(w, fs))
        a, o = Dsum.shape[1:]
        ref = ref + Dsum.pow(2).sum((1, 2)).mean() / (a * o)
    assert torch.allclose(got, ref, rtol=1e-4), (float(got), float(ref))


# ------------------------------------------------------------- loss and config plumbing
class _ES:
    def __init__(self, E=4, N=40, nx=2, nz=5, nu=1):
        g = torch.Generator().manual_seed(0)
        self.x, self.u = torch.randn(E, N, nx, generator=g), torch.randn(E, N, nu, generator=g)
        self.y_clean = self.x[..., :1].clone(); self.y = self.y_clean + 0.01
        self.z, self.z_obs = torch.randn(E, N, nz, generator=g), torch.randn(E, N, nz, generator=g)


def test_extras_off_reproduce_the_deployment_loss_and_terms_add_up():
    A, B = D.butterworth_AB(5, 1.0, 1)
    des = Design(A, B, 0.05, 39, 1.0, D.latent_step(A, B, 0.05))
    lat = LatentStep(des, "cpu")
    E = build_context("gru", {"dim": 8, "hidden": 16}, 1, 0.05, None, 1.0)
    m, *_ = _lora_with_encoder()
    es, sig, zs = _ES(), torch.ones(2), torch.ones(5)
    e, k = torch.tensor([0, 2]), torch.randint(5, 37, (2, 6))
    torch.manual_seed(3)
    l0 = _decoder_loss(m, E, es, e, k, sig)
    l1 = _decoder_loss_extras(m, E, es, e, k, sig, lat, zs, des.dt, False, 0.0, 0.0)
    assert torch.allclose(l0, l1)
    la = _decoder_loss_extras(m, E, es, e, k, sig, lat, zs, des.dt, True, 0.0, 0.0)
    lb = _decoder_loss_extras(m, E, es, e, k, sig, lat, zs, des.dt, False, 1e-3, 1e-4)
    assert float(la) > float(l0) and float(lb) > float(l0)
    la.backward()                                     # gradient reaches the encoder heads
    assert m.Ue[0].weight.grad is not None and m.Ue[0].weight.grad.abs().sum() > 0


def test_method_cfg_overrides_are_per_method_and_default_is_untouched():
    cfg = load("vdp")
    k0, c0 = method_cfg(cfg, "lora")
    assert k0 == "lora" and c0 == cfg
    k, c = method_cfg(cfg, "lora[phase1.optimizer=soap,phase1.lr=0.01,phase2.delta_curv=1e-4]@gru")
    assert k == "lora" and c["phase1"]["optimizer"] == "soap" and c["phase1"]["lr"] == 0.01
    assert c["phase2"]["delta_curv"] == 1e-4 and c["phase2"]["context"]["kind"] == "gru"
    assert "optimizer" not in cfg["phase1"]           # the job config itself is not mutated
    k, c = method_cfg(cfg, "autonomous")
    assert k == "none"
