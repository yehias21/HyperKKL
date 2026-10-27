"""Tests for the beat-the-grey-box pieces (hyperkkl/beat.py, soap.py, train.phase1 optimizer option).

    .venv/bin/python -m pytest tests/test_beat.py -q

CPU only; caches go to a pytest tmp dir. The relabel test reads one live-cache file read-only and
is skipped when it is absent.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

import hyperkkl.data as Dm
from hyperkkl import beat as BG
from hyperkkl import design as D
from hyperkkl import train as T
from hyperkkl.config import load
from hyperkkl.observers import KKLObserver, LatentStep, run
from hyperkkl.soap import SOAP, make_optimizer

DEV = "cpu"
SMALL = {"data.n_p1": 32, "data.n_train": 48, "data.n_val": 16, "data.n_coll": 4096,
         "phase1.steps_enc": 120, "phase1.steps_dec": 120, "phase2.steps": 40, "phase2.val_every": 20,
         "phase2.budget": 20000}
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/cache")


@pytest.fixture(scope="module", autouse=True)
def _tmp_cache(tmp_path_factory):
    old = Dm.CACHE
    Dm.CACHE = tmp_path_factory.mktemp("cache")
    yield
    Dm.CACHE = old


@pytest.fixture(scope="module")
def vdp():
    cfg = load("vdp", [SMALL])
    sc = Dm.system_scales(cfg, DEV)
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, 0, DEV)
    base = T.phase1(cfg, data, sc, des, DEV, 0, log=lambda m: None)
    return {"cfg": cfg, "scales": sc, "des": des, "data": data, "base": base}


# ------------------------------------------------------------------ SOAP (archive tests)
def test_soap_equals_adam95_on_1d_parameters():
    torch.manual_seed(0)
    a = torch.randn(7, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    oa, ob = SOAP([a], lr=1e-2), torch.optim.Adam([b], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    t = torch.randn(7)
    for _ in range(30):
        for p, o in ((a, oa), (b, ob)):
            o.zero_grad(); ((p - t) ** 4).sum().backward(); o.step()
    assert torch.allclose(a, b, atol=1e-6)


def test_soap_reduces_to_adam_when_preconditioners_are_diagonal():
    d = torch.tensor([1.0, 3.0, 10.0, 30.0])
    W = torch.diag(torch.tensor([1.0, -2.0, 0.5, 1.5])).requires_grad_(True)
    W2 = W.detach().clone().requires_grad_(True)
    oa = SOAP([W], lr=1e-2, precond_freq=3)
    ob = torch.optim.Adam([W2], lr=1e-2, betas=(0.95, 0.95), eps=1e-8)
    for _ in range(40):
        for p, o in ((W, oa), (W2, ob)):
            o.zero_grad(); (d * torch.diagonal(p) ** 2).sum().backward(); o.step()
    assert torch.allclose(W, W2, atol=1e-5)


def test_soap_param_groups_keep_their_lr():
    """P1 uses two groups (side lr, base lr x 0.1): a zero-lr group must not move."""
    a = torch.randn(4, 3, requires_grad=True); b = torch.randn(3, 2, requires_grad=True)
    b0 = b.detach().clone()
    opt = SOAP([{"params": [a], "lr": 1e-2}, {"params": [b], "lr": 0.0}])
    for _ in range(5):
        opt.zero_grad(); (a.sum() ** 2 + (b ** 2).sum()).backward(); opt.step()
    assert torch.equal(b.detach(), b0)


def test_make_optimizer_default_is_plain_adam():
    p = [torch.zeros(3, requires_grad=True)]
    o = make_optimizer(None, p, 1e-3)
    assert type(o) is torch.optim.Adam and o.defaults["betas"] == (0.9, 0.999)


# ------------------------------------------------------------------ designs
def test_custom_design_default_equals_make_design(vdp):
    cfg, des = vdp["cfg"], vdp["des"]
    d2 = BG.custom_design(cfg["system"], des.dt, des.N)
    assert np.allclose(d2.A, des.A) and np.allclose(d2.B, des.B) and d2.dt == des.dt and d2.N == des.N
    for k in des.step:
        assert np.allclose(d2.step[k], des.step[k])


@pytest.mark.parametrize("nz,a,s", [(6, 1.0, 1.0), (5, 0.5, 4.0), (8, 2.0, 4.0), (22, 1.0, 1.0)])
def test_custom_design_properties(vdp, nz, a, s):
    cfg, des = vdp["cfg"], vdp["des"]
    d = BG.custom_design(cfg["system"], des.dt, des.N, nz=nz, alpha_scale=a, spread=s)
    ev = np.linalg.eigvals(d.A)
    alpha = float(cfg["system"]["alpha"]) * a
    assert d.A.shape == (nz, nz)
    assert abs(-ev.real.max() - alpha) < 1e-9 * max(1, alpha)          # slowest stays at alpha
    r0 = alpha / np.sin(np.pi / (2 * nz))
    assert abs(np.abs(ev).max() - s * r0) < 1e-6 * s * r0                # fastest group scaled by s
    chk = BG.design_checks(d)
    assert chk["ok_relaxed"], chk


def test_relabel_on_default_design_matches_stored_rk4_labels():
    """data.run_latent on y_clean reproduces the joint-RK4 labels of a real cached set."""
    import json
    src = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/tree/C3-partial/runs/vdp/seed0/results.json")
    if not src.exists():
        pytest.skip("no live tree")
    cfg = json.load(open(src))["config"]
    from hyperkkl.config import digest
    from hyperkkl.cache import DATA_MODULES, module_hash
    key = digest(cfg, "system", "data", extra=[0, module_hash(DATA_MODULES), "cuda"])
    f = LIVE / "data" / f"vdp_train_{key}.pt"
    if not f.exists():
        pytest.skip("live data file absent")
    data = torch.load(f, weights_only=False, map_location="cpu")
    des = Dm.make_design(cfg["system"])
    es = data["train"].select(range(32))
    re = BG.relabel(es, des, es.burn)
    zs = es.z[:, es.burn:].reshape(-1, des.A.shape[0])
    err = (re.z[:, es.burn:].reshape(-1, des.A.shape[0]) - zs).pow(2).mean().sqrt() / zs.std()
    assert float(err) < 1e-3, float(err)
    assert torch.allclose(re.z_obs, es.z_obs, atol=1e-4 * float(zs.abs().max()))


def test_redesign_shapes_and_burn(vdp):
    cfg, des, data, sc = vdp["cfg"], vdp["des"], vdp["data"], vdp["scales"]
    d = BG.custom_design(cfg["system"], des.dt, des.N, alpha_scale=0.5)
    nd, nsc = BG.redesign(data, d, sc, 1e-3)
    assert nd["train"].x is data["train"].x and nd["train"].z.shape[-1] == d.A.shape[0]
    assert nsc["t_burn"] > sc["t_burn"] and nd["p1"].burn == nd["train"].burn == nd["val"].burn
    # z_obs is exactly the deployed filter on y
    assert torch.allclose(nd["val"].z_obs, Dm.run_latent(d, data["val"].y.double()).float())


# ------------------------------------------------------------------ phase 2
def test_phase2_bg_without_options_equals_train_phase2(vdp):
    """No option = the original decoder-side phase 2, bit for bit."""
    a = T.phase2("lora", vdp["cfg"], vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, 0, log=lambda m: None)
    b = BG.phase2_bg("lora", vdp["cfg"], vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, 0, {}, log=lambda m: None)
    assert a["n_params"] == b["n_params"] and a["width"] == b["width"]
    assert a["val"] == b["val"]
    for (ka, va), (kb, vb) in zip(a["cond"].state_dict().items(), b["cond"].state_dict().items()):
        assert ka == kb and torch.equal(va, vb), ka


def test_p1_trains_a_copy_of_the_base_decoder_only(vdp):
    base = vdp["base"]
    snap = {n: copy.deepcopy(base[n].state_dict()) for n in ("T0", "T0inv")}
    r = BG.phase2_bg("lora", vdp["cfg"], vdp["data"], vdp["scales"], vdp["des"], base, DEV, 0,
                     {"base_lr_factor": 0.1, "optimizer": "soap", "lr": 1e-2}, log=lambda m: None)
    for n in ("T0", "T0inv"):                                  # the phase-1 maps are untouched
        assert all(torch.equal(v, snap[n][k]) for k, v in base[n].state_dict().items())
    moved = any(not torch.equal(v, snap["T0inv"][k]) for k, v in r["cond"].base.state_dict().items())
    assert moved and r["cond"].base is not base["T0inv"]
    assert r["base_dec_params_trainable"] == sum(q.numel() for q in base["T0inv"].parameters())


# ------------------------------------------------------------------ history channel
def _hist_model(vdp, window=6, stride=3, slow=5.0, seed=0):
    r = BG.phase2_bg("lora", vdp["cfg"], vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, seed,
                     {"window": window, "stride": stride, "slow_factor": slow}, log=lambda m: None)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():                         # non-trivial weights for the equivalence tests
        for m in (r["cond"], r["hist"]):
            for p in m.parameters():
                if p.dtype.is_floating_point and p.numel() > 1:
                    p.add_(0.05 * torch.randn(p.shape, generator=g))
    return r


def test_history_features_are_causal(vdp):
    r = _hist_model(vdp)
    h = r["hist"]
    es = vdp["data"]["val"]
    z = es.z_obs[:4].clone()
    zs = BG.slow_bank_seq(h.slow, es.y[:4])
    k = torch.tensor([[50, 200, 400]] * 4)
    f0 = h.at(z, zs, k)
    z2, zs2 = z.clone(), zs.clone()
    z2[:, 401:] += 10.0; zs2[:, 401:] += 10.0
    assert torch.equal(h.at(z2, zs2, k)[:, 2], f0[:, 2])
    z3 = z.clone(); z3[:, 0] += 1.0                              # sample 0 is inside the window of k = 0..(K-1)s
    k0 = torch.tensor([[0, 3]] * 4)
    assert not torch.equal(h.at(z3, zs, k0), h.at(z, zs, k0))


def test_hist_observer_deployment_equals_training_computation(vdp):
    """With z0 = 0 the deployed latents ARE z_obs; the ring buffer + online slow bank must give
    the decoder output of the training gather at every sample."""
    r = _hist_model(vdp)
    des = vdp["des"]
    es = vdp["data"]["val"].select(range(5))
    ob = BG.HistKKLObserver("h", LatentStep(des, DEV), r["cond"], r["ctx"], r["hist"], z0_map=None)
    xh = run(ob, es.y, es.u, torch.zeros(5, es.x.shape[-1]))
    N1 = es.x.shape[1]
    k = torch.arange(N1)[None].expand(5, -1)
    with torch.no_grad():
        c = r["ctx"].seq(es.u)
        zs = BG.slow_bank_seq(r["hist"].slow, es.y)
        hf = r["hist"].at(es.z_obs, zs, k)
        off = r["cond"](es.z_obs.reshape(-1, es.z_obs.shape[-1]), torch.cat([c, hf], -1).reshape(5 * N1, -1))
    off = off.view(5, N1, -1)
    assert torch.allclose(xh, off, atol=1e-4, rtol=1e-4), float((xh - off).abs().max())


def test_hist_observer_end_to_end_causality(vdp):
    r = _hist_model(vdp)
    des = vdp["des"]
    es = vdp["data"]["val"].select(range(3))
    ob = BG.HistKKLObserver("h", LatentStep(des, DEV), r["cond"], r["ctx"], r["hist"], z0_map=vdp["base"]["T0"])
    x0 = torch.zeros(3, es.x.shape[-1])
    a = run(ob, es.y, es.u, x0)
    y2, u2 = es.y.clone(), es.u.clone()
    y2[:, 301:] += 5.0; u2[:, 301:] -= 3.0
    b = run(ob, y2, u2, x0)
    assert torch.equal(a[:, :301], b[:, :301])
    assert not torch.allclose(a[:, 302:], b[:, 302:])


def test_noise_gain_zero_without_noise(vdp):
    es = vdp["data"]["val"]
    clean = Dm.EpisodeSet(es.x, es.u, es.y_clean, es.y_clean, es.z, es.z, es.families, es.burn, es.dt)
    assert BG.noise_gain(vdp["base"]["T0inv"], clean, vdp["scales"], DEV) == 0.0
    assert BG.noise_gain(vdp["base"]["T0inv"], es, vdp["scales"], DEV) > 0.0
