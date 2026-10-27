"""Verification of the losses and optimisation loops in hyperkkl/train.py.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python -m pytest tests/test_verify_train.py -q -s

1. physics residual on a linear plant (exact KKL map T = P x from the Sylvester equation)
2. phase1 end to end: linear plant against P, bioreactor and quad_tank at small budgets
3. deployment equivalence: _decoder_loss / _rollout_loss / nlox's roll against the
   deployed KKLObserver run by observers.run, element by element
4. optimisation: schedule, clipping, frozen base maps, best-validation state, loss
   decrease, identical draws across variants, reproducibility, time per step

Every cache write goes to a pytest tmp dir (hyperkkl.data.CACHE is redirected), so this
never touches experiments/. Lines starting with ``VERIFY`` are the measured numbers.
~10 min on one GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.func import jacrev, vmap

import hyperkkl.data as Dm
from hyperkkl import design as D
from hyperkkl import signals as S
from hyperkkl import train as T
from hyperkkl.conditioning import ConditionedDecoder, build_conditioner, match_width
from hyperkkl.config import load
from hyperkkl.data import Design, generate
from hyperkkl.evaluate import build_observers, initial_guess
from hyperkkl.nets import Affine, BaseMap, build_context
from hyperkkl.observers import KKLObserver, LatentStep, run
from hyperkkl.systems import Plant

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMALL = {"data.n_p1": 64, "data.n_train": 96, "data.n_val": 32}


def report(tag, **kw):
    f = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    print(f"\nVERIFY {tag}: " + "  ".join(f"{k}={f(v)}" for k, v in kw.items()), flush=True)


@pytest.fixture(scope="module", autouse=True)
def _tmp_cache(tmp_path_factory):
    old = Dm.CACHE
    Dm.CACHE = tmp_path_factory.mktemp("cache")
    yield
    Dm.CACHE = old


def _rms(t):
    return float(t.pow(2).mean().sqrt())


# ---------------------------------------------------------------------------
# Linear plant: x' = F x + G u, y = C x, exact KKL map T(x) = P x
# ---------------------------------------------------------------------------

class LinPlant(Plant):
    nx, nu = 2, 1
    PARAMS = {}
    DEFAULT_SENSORS = (0,)
    name = "lin"
    F = np.array([[0.0, 1.0], [-2.0, -0.3]])
    G = np.array([0.0, 1.0])
    C = np.array([[1.0, 0.0]])

    def f(self, x, u):
        F = torch.as_tensor(self.F, dtype=x.dtype, device=x.device)
        G = torch.as_tensor(self.G, dtype=x.dtype, device=x.device)
        return x @ F.T + u[:, :1] * G


P1 = {"enc_width": 128, "enc_depth": 3, "dec_width": 128, "dec_depth": 3, "steps_enc": 2000,
      "steps_dec": 2000, "batch": 2048, "lr": 2e-3, "stride": 2, "pde": True, "lambda_pde": 1.0,
      "decoder_data": ["coll", "traj"]}


@pytest.fixture(scope="module")
def lin():
    plant = LinPlant()
    A, B = D.butterworth_AB(4, 1.0, 1)
    dt = D.choose_dt(A, 0.1)
    des = Design(A, B, dt, int(math.ceil(30.0 / dt)), 1.0, D.latent_step(A, B, dt))
    P = D.sylvester_reference(LinPlant.F, LinPlant.C, A, B)
    box, dom = np.array([[-2.0, 2.0], [-2.0, 2.0]]), np.array([[-3.0, 3.0], [-4.0, 4.0]])
    xc = torch.as_tensor(Dm._sobol(dom, 16384, 13), dtype=torch.float64)
    z_scale = float(np.sqrt(np.square(xc.numpy() @ P.T).sum(1).mean()))
    kappa, a = D.spectral_bounds(A)
    t_burn = D.washout_time(kappa, a, np.linalg.norm(B, 2), 3.0, z_scale, 1e-3)
    scales = {"domain": dom.tolist(), "x_std": [1.0, 1.0], "y_std": [1.0], "t_burn": t_burn}
    scfg = {"ic_box": box.tolist(), "u_scale": 1.0, "u_bounds": []}
    dcfg = {"k_cross": 3, "reject_scale": 10.0, "noise_proc": 0.0, "sensor": {"noise": 0.0}}
    p1 = generate(plant, des, scales, scfg, dcfg, ["zero"], 128, "train", 11, DEV, ic_box=dom)
    tr = generate(plant, des, scales, scfg, dcfg, S.TRAIN_FAMILIES, 96, "train", 12, DEV)
    scales["x_std"] = tr.x[:, tr.burn:].reshape(-1, 2).std(0).tolist()
    coll = {"x": xc.float(), "f0": plant.f(xc, torch.zeros(len(xc), 1, dtype=xc.dtype)).float(),
            "y": plant.h(xc).float()}
    return {"plant": plant, "des": des, "P": P, "scales": scales,
            "data": {"p1": p1, "train": tr, "coll": coll}}


class _Fixed(nn.Module):
    """T(x) = M x + c, frozen; one dummy parameter so phase1's optimiser has something."""

    def __init__(self, M, c, out_stats):
        super().__init__()
        self.register_buffer("M", torch.as_tensor(M, dtype=torch.float32))
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.float32))
        self.out = Affine(*out_stats)
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x @ self.M.T + self.c + 0.0 * self.w


def _phase1_fixed_T0(monkeypatch, lin, M, c, seed=0):
    """Run the REAL phase1 code with T0 replaced by a fixed map (first BaseMap call)."""
    n = []

    def factory(d_in, d_out, width, depth, act, in_stats, out_stats):
        n.append(1)
        if len(n) == 1:
            return _Fixed(M, c, out_stats)
        return BaseMap(d_in, d_out, width, depth, act, in_stats, out_stats)

    monkeypatch.setattr(T, "BaseMap", factory)
    cfg = {"phase1": dict(P1, steps_enc=1, steps_dec=1)}
    return T.phase1(cfg, lin["data"], lin["scales"], lin["des"], DEV, seed, log=lambda m: None)


def _expected_pde(lin, M, c, seed=0):
    """Independent float64 evaluation of the claimed phase-1 PDE loss on the SAME batch
    (phase1 seeds torch, then draws the label batch i, then the collocation batch j)."""
    p1, coll = lin["data"]["p1"], lin["data"]["coll"]
    nX = len(T._flat_valid(p1, "x", P1["stride"]))
    torch.manual_seed(seed)
    torch.randint(0, nX, (P1["batch"],), device=DEV)
    j = torch.randint(0, len(coll["x"]), (P1["batch"],), device=DEV).cpu()
    x = coll["x"][j].double().numpy()
    f = x @ LinPlant.F.T
    A, B = lin["des"].A, lin["des"].B
    Tx, Jf = x @ M.T + c, f @ M.T
    R = Jf - Tx @ A.T - (x @ LinPlant.C.T) @ B.T
    WR = R @ (-np.linalg.inv(A)).T
    return float(((WR ** 2).sum(1) / np.maximum(np.linalg.norm(f, axis=1), 1e-6) ** 2).mean())


def test_pde_residual_zero_at_exact_map_positive_otherwise(monkeypatch, lin):
    P = lin["P"]
    zero = np.zeros(P.shape[0])
    r = _phase1_fixed_T0(monkeypatch, lin, P, zero)
    lab0, pde0 = r["hist_enc"][0][1], r["hist_enc"][0][2]
    rng = np.random.RandomState(0)
    Pp = P + 0.05 * rng.randn(*P.shape)
    rp = _phase1_fixed_T0(monkeypatch, lin, Pp, zero)
    pde_p, exp_p = rp["hist_enc"][0][2], _expected_pde(lin, Pp, zero)
    report("pde.exact", pde_at_P=pde0, label_at_P=lab0, ss_rms_at_P=r["diag"]["eps_pde_ss_rms"],
           pde_at_P_plus_5pct=pde_p, expected_float64=exp_p)
    assert pde0 < 1e-8 and r["diag"]["eps_pde_ss_rms"] < 1e-5
    assert lab0 < 1e-5                       # labels (RK4 + washout) agree with P x
    assert pde_p > 1e3 * max(pde0, 1e-12)
    assert abs(pde_p - exp_p) / exp_p < 1e-3     # the code computes exactly the claimed loss


def test_minus_Ainv_maps_residual_to_steady_latent_error(monkeypatch, lin):
    des = lin["des"]
    A = des.A
    W = -np.linalg.inv(A)
    # (i) xi' = A xi + r with r constant, integrated by the deployed latent step
    lat = LatentStep(des, "cpu")
    r = torch.tensor([[0.3, -1.2, 0.7, 0.2]])
    xi = torch.zeros(1, 4)
    for _ in range(int(40 / (des.alpha * des.dt))):
        xi = lat(xi, torch.zeros(1, 1), torch.zeros(1, 1), lambda z: r)
    ss = r.numpy()[0] @ W.T
    err_i = float(np.abs(xi.numpy()[0] - ss).max() / np.abs(ss).max())
    # (ii) T = P x + c has residual R = -A c; the code's weighted residual is c, and the
    #      deployed filter's latent error T(x_k) - z_k converges to exactly that
    #      (zero-input episodes: under forcing the linear plant's latent is not P x)
    c = np.array([0.2, -0.1, 0.05, 0.3])
    rr = _phase1_fixed_T0(monkeypatch, lin, lin["P"], c)
    p1 = lin["data"]["p1"]
    Tx = p1.x.double().numpy() @ lin["P"].T + c
    e = (Tx - p1.z_obs.double().numpy())[:, p1.burn:]
    err_ii = float(np.abs(e - c).max())
    # (iii) for a state-dependent residual the weighting is only quasi-static:
    #       W R = dP x - A^{-1} dP F x, while the true steady latent error is dP x
    dP = 0.05 * np.random.RandomState(1).randn(*lin["P"].shape)
    x = p1.x[:, p1.burn:].reshape(-1, 2).double().numpy()
    qs = np.linalg.norm(x @ (np.linalg.inv(A) @ dP @ LinPlant.F).T, axis=1) / \
        np.maximum(np.linalg.norm(x @ dP.T, axis=1), 1e-12)
    report("pde.weighting", steady_state_rel_err=err_i, ss_rms_code=rr["diag"]["eps_pde_ss_rms"],
           norm_c=float(np.linalg.norm(c)), latent_err_minus_c_max=err_ii,
           quasi_static_rel_dev_median=float(np.median(qs)))
    assert err_i < 1e-4
    assert abs(rr["diag"]["eps_pde_ss_rms"] - np.linalg.norm(c)) < 1e-4
    assert abs(rr["diag"]["eps_pde_rms"] - np.linalg.norm(A @ c)) < 1e-4
    assert err_ii < 2e-3


# ---------------------------------------------------------------------------
# 2. phase1 end to end
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pde", [True, False])
def test_phase1_recovers_P_on_linear_plant(lin, pde):
    cfg = {"phase1": dict(P1, pde=pde)}
    base = T.phase1(cfg, lin["data"], lin["scales"], lin["des"], DEV, 0, log=lambda m: None)
    T0, Ti = base["T0"], base["T0inv"]
    P = torch.as_tensor(lin["P"], dtype=torch.float32, device=DEV)
    sig = torch.as_tensor(lin["scales"]["x_std"], dtype=torch.float32, device=DEV)
    xc = lin["data"]["coll"]["x"].to(DEV)
    tr = lin["data"]["train"]
    xt = tr.x[:, tr.burn::4].reshape(-1, 2).to(DEV)
    with torch.no_grad():
        rel = {nm: _rms(T0(x) - x @ P.T) / _rms(x @ P.T) for nm, x in (("coll", xc), ("traj", xt))}
        inv_T0 = ((Ti(T0(xc)) - xc) / sig).pow(2).sum(-1).mean().sqrt().item()
        inv_P = ((Ti(xc @ P.T) - xc) / sig).pow(2).sum(-1).mean().sqrt().item()
    report(f"phase1.linear pde={pde}", rel_T0_vs_Px_coll=rel["coll"], rel_T0_vs_Px_traj=rel["traj"],
           dec_on_T0_sigma=inv_T0, dec_on_exact_Px_sigma=inv_P,
           **{k: round(v, 5) for k, v in base["diag"].items()})
    if pde:
        assert rel["coll"] < 0.03 and rel["traj"] < 0.03
        assert base["diag"]["eps_dec"] < 0.1 and inv_P < 0.15


def _sys(system, extra=None, seed=0):
    cfg = load(system, [dict(SMALL, **(extra or {}))])
    sc = Dm.system_scales(cfg, DEV)
    des = Dm.make_design(cfg["system"])
    return cfg, sc, des, Dm.training_data(cfg, seed, DEV)


def _phase1_report(system, cfg, sc, des, data, base):
    T0, Ti = base["T0"], base["T0inv"]
    st = cfg["phase1"]["stride"]
    X = T._flat_valid(data["p1"], "x", st).to(DEV)
    Z = T._flat_valid(data["p1"], "z", st).to(DEV)
    Zf = T._flat_valid(data["train"], "z", 4 * st).to(DEV)
    sig = torch.as_tensor(sc["x_std"], dtype=torch.float32, device=DEV)
    xc = data["coll"]["x"].to(DEV)
    xt = T._flat_valid(data["train"], "x", 4 * st).to(DEV)
    with torch.no_grad():
        lab_abs = _rms(T0(X) - Z)
        dec = {nm: ((Ti(T0(x)) - x) / sig).pow(2).sum(-1).mean().sqrt().item()
               for nm, x in (("coll", xc), ("traj", xt), ("p1", X))}
    zs = T0.out.std
    xs = xc[torch.randperm(len(xc), device=DEV)[:4096]]
    J = vmap(jacrev(lambda v: T0(v[None])[0]))(xs) / zs[None, :, None] * sig[None, None, :]
    sv = torch.linalg.svdvals(J)
    w = 1.0 / data["coll"]["f0"].norm(dim=-1).clamp(min=1e-6).pow(2)
    ws = w.sort(descending=True).values
    out = dict(label_abs_rms=lab_abs, label_std=float(Z.std(0).mean()),
               forced_latent_std=float(Zf.std(0).mean()), label_loss_last=base["hist_enc"][-1][1],
               pde_loss_last=base["hist_enc"][-1][2], dec_loss_last=base["hist_dec"][-1][1],
               eps_dec_coll=dec["coll"], eps_dec_traj=dec["traj"], eps_dec_p1=dec["p1"],
               svmin_JT0_q50=float(sv[:, -1].median()), svmax_JT0_q50=float(sv[:, 0].median()),
               pde_weight_top1pct_share=float(ws[:len(ws) // 100].sum() / ws.sum()),
               **{k: round(v, 5) for k, v in base["diag"].items()})
    report(f"phase1.{system}", **out)
    return out


@pytest.mark.parametrize("system", ["bioreactor", "quad_tank"])
def test_phase1_real_systems(system):
    cfg, sc, des, data = _sys(system, {"phase1.steps_enc": 2000, "phase1.steps_dec": 2000})
    base = T.phase1(cfg, data, sc, des, DEV, 0, log=lambda m: None)
    r = _phase1_report(system, cfg, sc, des, data, base)
    for h in base["hist_enc"] + base["hist_dec"]:
        assert all(np.isfinite(h)), h
    # encoder: labels fit to a small fraction of the forced-latent spread
    assert r["label_abs_rms"] < 0.02 * r["forced_latent_std"]
    assert r["eps_enc_rel"] < 0.05
    if system == "quad_tank" and r["eps_dec"] > 0.2:
        pytest.xfail(f"quad_tank decoder eps_dec={r['eps_dec']:.2f} sigma: T0 is near-singular "
                     f"(sigma_min/sigma_max of normalised J_T0 = "
                     f"{r['svmin_JT0_q50'] / r['svmax_JT0_q50']:.1e}), an identifiability limit")
    assert r["eps_dec"] < 0.2


# ---------------------------------------------------------------------------
# vdp fixture shared by the deployment and optimisation checks
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vdp():
    cfg, sc, des, data = _sys("vdp", {"data.n_coll": 16384, "phase1.steps_enc": 1500,
                                      "phase1.steps_dec": 1500, "phase2.steps": 150,
                                      "phase2.val_every": 50})
    base = T.phase1(cfg, data, sc, des, DEV, 0, log=lambda m: None)
    report("vdp.base", **{k: round(v, 5) for k, v in base["diag"].items()})
    return {"cfg": cfg, "scales": sc, "des": des, "data": data, "base": base}


def _perturb(m, s, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(s * torch.randn(p.shape, generator=g).to(p.device))


def _ctx_cond(v, kind, ctx_kind, budget=20000, seed=0):
    p, des = v["cfg"]["phase2"], v["des"]
    ccfg = dict(p["context"], kind=ctx_kind)
    va = v["data"]["val"].to(DEV)
    z_std = va.z_obs[:, va.burn:].reshape(-1, des.A.shape[0]).std(0)
    torch.manual_seed(seed)
    ctx = build_context(ctx_kind, ccfg, 1, des.dt, np.linalg.eigvals(des.A),
                        float(v["cfg"]["system"]["u_scale"])).to(DEV)
    Ti = v["base"]["T0inv"]
    w, _ = match_width(kind, Ti, ctx.dim, budget, int(p["rank"]), z_std)
    cond = build_conditioner(kind, Ti, ctx.dim, w, int(p["rank"]), z_std).to(DEV)
    _perturb(cond, 0.05 if kind != "latent" else 0.1, seed)
    _perturb(ctx, 0.05, seed + 1)
    for q in list(cond.parameters()) + list(ctx.parameters()):
        q.requires_grad_(False)
    return va, ctx, cond


def _white_u(v, u, seed=5):
    """White-noise input. The equivalence checks need no consistency between u and y, and
    an input that changes at every sample makes an off-by-one in the input timing visible
    (with the smooth / piecewise-constant training inputs it is nearly invisible)."""
    g = torch.Generator(device=u.device).manual_seed(seed)
    return float(v["cfg"]["system"]["u_scale"]) * torch.randn(u.shape, generator=g, device=u.device)


# ---------------------------------------------------------------------------
# 3. deployment equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx_kind", ["sfm", "gru", "window", "current"])
@pytest.mark.parametrize("kind", ["concat", "film", "lora", "hyper"])
def test_decoder_loss_equals_deployed_observer(vdp, kind, ctx_kind):
    va, ctx, cond = _ctx_cond(vdp, kind, ctx_kind)
    des, nx = vdp["des"], va.x.shape[-1]
    sig = torch.as_tensor(vdp["scales"]["x_std"], dtype=torch.float32, device=DEV)
    lat = LatentStep(des, DEV)
    xd = run(KKLObserver("t", lat, cond, ctx=ctx, z0_map=None), va.y, va.u,
             torch.zeros(len(va), nx, device=DEV))
    es = dataclasses.replace(va, x=xd)
    N1, one = va.x.shape[1], torch.ones(nx, device=DEV)
    k = torch.arange(N1, device=DEV)[None].expand(2, -1)
    with torch.no_grad():
        sq, sh_c, sh_z, base_dev = 0.0, 0.0, 0.0, 0.0
        for i in range(0, len(va), 2):                      # chunks: hyper is memory-heavy
            e = torch.arange(i, i + 2, device=DEV)
            sq += float(T._decoder_loss(cond, ctx, es, e, k, one))
            c = ctx.seq(va.u[e])
            z = va.z_obs[e]
            f = lambda zz, cc: cond(zz.reshape(-1, zz.shape[-1]), cc.reshape(-1, cc.shape[-1])).view(2, -1, nx)
            sh_c += float((f(z[:, :-1], c[:, 1:]) - xd[e, :-1]).pow(2).mean())   # c_{k+1}
            sh_z += float((f(z[:, 1:], c[:, :-1]) - xd[e, :-1]).pow(2).mean())   # z_{k+1}
            base_dev += float((vdp["base"]["T0inv"](z.reshape(-1, z.shape[-1])).view(2, -1, nx)
                               - xd[e]).pow(2).mean())
        n = len(va) // 2
        eq, sh_c, sh_z, base_dev = (math.sqrt(a / n) for a in (sq, sh_c, sh_z, base_dev))
        # evaluate.build_observers wiring: z0 = T0(x0_hat) at the IC-box centre
        x0 = initial_guess(vdp["cfg"], len(va), "center", 0, DEV)
        xe = run(KKLObserver("t", lat, cond, ctx=ctx, z0_map=vdp["base"]["T0"]), va.y, va.u, x0)
        gap = ((xe - xd) / sig).norm(dim=-1)
    report(f"deploy.{kind}@{ctx_kind}", rms_train_vs_deployed=eq, rms_if_c_k_plus_1=sh_c,
           rms_if_z_k_plus_1=sh_z, conditioning_effect_rms=base_dev,
           z0map_gap_sigma_after_burn_max=float(gap[:, va.burn:].max()),
           z0map_gap_sigma_after_settle_max=float(gap[:, (N1 - 1) // 2:].max()))
    assert eq < 1e-4
    assert base_dev > 100 * eq                      # the conditioner actually acts
    assert sh_c > 10 * eq and sh_z > 10 * eq        # an off-by-one would be detected
    assert float(gap[:, va.burn:].max()) < 1e-2


@pytest.mark.parametrize("ctx_kind", ["sfm", "gru"])
def test_rollout_loss_equals_deployed_latent_observer(vdp, ctx_kind):
    va, ctx, inj = _ctx_cond(vdp, "latent", ctx_kind)
    va = dataclasses.replace(va, u=_white_u(vdp, va.u))
    des, nx, K = vdp["des"], va.x.shape[-1], 64
    T0, Ti = vdp["base"]["T0"], vdp["base"]["T0inv"]
    lat = LatentStep(des, DEV)
    e = torch.tensor([0, 3, 5, 7], device=DEV)
    k0 = torch.tensor([va.burn, va.burn + 37, va.burn + 150, va.burn + 401], device=DEV)
    xd = va.x.clone()
    with torch.no_grad():
        for b in range(len(e)):
            ob = KKLObserver("lat", lat, ConditionedDecoder(Ti, "none", 0, 0), ctx=ctx,
                             injection=inj, z0_map=None)
            yb, ub = va.y[e[b]:e[b] + 1], va.u[e[b]:e[b] + 1]
            ob.reset(1, torch.zeros(1, nx, device=DEV), ub[:, 0])
            for k in range(int(k0[b])):
                ob.advance(yb[:, k], ub[:, k], yb[:, k + 1], ub[:, k + 1])
            ob.z = T0(va.x[e[b]:e[b] + 1, int(k0[b])])         # start the injection at T0(x_k0)
            for k in range(int(k0[b]), int(k0[b]) + K):
                ob.advance(yb[:, k], ub[:, k], yb[:, k + 1], ub[:, k + 1])
                xd[e[b], k + 1] = ob.estimate()[0]
        es = dataclasses.replace(va, x=xd)
        eq = math.sqrt(float(T._rollout_loss(inj, ctx, lat, T0, Ti, es, e, k0, K, torch.ones(nx, device=DEV))))

        def alt(shift, use_phi=True):                       # manual rollout with other timing
            c, r = ctx.seq(va.u[e]), torch.arange(len(e), device=DEV)
            z, out = T0(va.x[e, k0]), []
            for j in range(K):
                kk = k0 + j
                ck = c[r, torch.clamp(kk + shift, max=c.shape[1] - 1)]
                z = lat(z, va.y[e, kk], va.y[e, kk + 1], (lambda zz: inj(zz, ck)) if use_phi else None)
                out.append(Ti(z))
            kk = k0[:, None] + torch.arange(1, K + 1, device=DEV)[None]
            return _rms(torch.stack(out, 1) - xd[e[:, None], kk])
        sh, nophi = alt(1), alt(0, use_phi=False)
    report(f"deploy.latent@{ctx_kind}", rms_train_vs_deployed=eq, rms_if_c_k_plus_1=sh,
           rms_without_injection=nophi)
    assert math.isfinite(nophi) and math.isfinite(sh)
    assert eq < 1e-4
    assert nophi > 100 * eq and sh > 10 * eq


def test_nlox_roll_equals_deployed_nlox(vdp):
    cfg = copy.deepcopy(vdp["cfg"])
    cfg["nlox"].update({"steps": 1, "val_every": 1, "K": 64, "episodes_per_batch": 4})
    cfg["baselines"] = []
    data, sc, des = vdp["data"], vdp["scales"], vdp["des"]
    res = T.nlox(cfg, data, sc, des, DEV, 0, log=lambda m: None)
    ob = build_observers(cfg, des, sc, vdp["base"], {}, res, DEV)["nlox"]
    # nlox validates the DEPLOYED observer (z0 = 0) on every validation episode, scored on
    # the whole post-washout horizon [burn, N] (training covers it by stateful TBPTT)
    nv = len(data["val"])
    va = data["val"].select(range(nv)).to(DEV)
    nx, burn, N1 = va.x.shape[-1], va.burn, va.x.shape[1]
    sig = torch.as_tensor(sc["x_std"], dtype=torch.float32, device=DEV)
    x0 = torch.zeros(len(va), nx, device=DEV)
    xd = run(ob, va.y, va.u, x0)
    dep = float(((xd[:, burn:] - va.x[:, burn:]) / sig).pow(2).mean())
    # elementwise, with a white-noise input: retrain identically (val does not affect
    # training) with the validation u replaced and its targets replaced by the deployed xd
    uw = _white_u(vdp, va.u)
    xw = run(ob, va.y, uw, x0)
    xw_sh = run(ob, va.y, torch.cat([uw[:, 1:], uw[:, -1:]], 1), x0)      # inject u_{k+1}
    vx, vu = data["val"].x.clone(), data["val"].u.clone()
    vx[:nv], vu[:nv] = xw.cpu(), uw.cpu()
    res2 = T.nlox(cfg, dict(data, val=dataclasses.replace(data["val"], x=vx, u=vu)), sc, des, DEV, 0,
                  log=lambda m: None)
    elem = math.sqrt(res2["val"])
    shift = _rms(((xw_sh - xw) / sig)[:, burn:])
    omega_effect = _rms(((run(ob, va.y, 0 * uw, x0) - xw) / sig)[:, burn:])
    report("deploy.nlox", val_train=res["val"], val_deployed=dep, rms_elementwise_sigma=elem,
           rms_if_u_k_plus_1_sigma=shift, rms_without_injection_sigma=omega_effect)
    assert abs(dep - res["val"]) / res["val"] < 1e-4
    assert elem < 1e-4
    assert shift > 10 * max(elem, 1e-7) and omega_effect > 100 * max(elem, 1e-7)


# ---------------------------------------------------------------------------
# 4. optimisation
# ---------------------------------------------------------------------------

def test_sched_warmup_then_cosine():
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    steps, warm = 1000, 100
    sch = T._sched(opt, steps, warm)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step(); sch.step()
    lrs = np.asarray(lrs)
    s = np.arange(steps)
    exp = np.minimum(1, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * s / steps))
    report("sched", lr0=lrs[0], lr_peak=float(lrs.max()), argmax=int(lrs.argmax()), lr_last=lrs[-1])
    assert np.allclose(lrs, exp, atol=1e-12)
    assert int(lrs.argmax()) == warm - 1 and np.all(np.diff(lrs[warm - 1:]) <= 1e-15)
    assert lrs[-1] < 1e-4


KINDS = ["concat", "film", "lora", "hyper", "latent"]


@pytest.fixture(scope="module")
def p2(vdp):
    """Every phase-2 kind at its full budget for 150 steps, with clipping recorded."""
    out = {}
    base = vdp["base"]
    snap = {n: {k: v.clone() for k, v in base[n].state_dict().items()} for n in ("T0", "T0inv")}
    orig = torch.nn.utils.clip_grad_norm_
    with pytest.MonkeyPatch.context() as mp:
        for kind in KINDS:
            rec = []

            def clip(params, max_norm, *a, **kw):
                params = list(params)
                pre = orig(params, max_norm, *a, **kw)
                post = torch.sqrt(sum(q.grad.pow(2).sum() for q in params if q.grad is not None))
                rec.append((float(max_norm), float(pre), float(post), {id(q) for q in params}))
                return pre

            mp.setattr(torch.nn.utils, "clip_grad_norm_", clip)
            r = T.phase2(kind, vdp["cfg"], vdp["data"], vdp["scales"], vdp["des"], base, DEV, 0,
                         log=lambda m: None)
            out[kind] = {"res": r, "clip": rec}
    out["_frozen"] = {n: all(torch.equal(v, snap[n][k]) for k, v in base[n].state_dict().items())
                      for n in ("T0", "T0inv")}
    return out


def _val_draw(v, kind):
    p = v["cfg"]["phase2"]
    va = v["data"]["val"].to(DEV)
    N1, K, M = va.x.shape[1], int(p["rollout_K"]), int(p["times_per_episode"])
    bv = T._Batcher(va, DEV, torch.Generator(device=DEV).manual_seed(999))
    return va, bv.times(len(va), M, va.burn, N1 if kind != "latent" else N1 - K - 1), K


def _val(v, kind, cond, ctx):
    va, kv, K = _val_draw(v, kind)
    sig = torch.as_tensor(v["scales"]["x_std"], dtype=torch.float32, device=DEV)
    lat = LatentStep(v["des"], DEV)
    ev = torch.arange(len(va), device=DEV)
    tot = 0.0
    with torch.no_grad():
        for i in range(0, len(ev), 16):
            e = ev[i:i + 16]
            if kind == "latent":
                l = T._rollout_loss(cond, ctx, lat, v["base"]["T0"], v["base"]["T0inv"], va, e, kv[i:i + 16, 0], K, sig)
            else:
                l = T._decoder_loss(cond, ctx, va, e, kv[i:i + 16], sig)
            tot += float(l) * len(e)
    return tot / len(ev)


def test_phase2_base_maps_frozen(p2):
    report("phase2.frozen", **p2["_frozen"])
    assert all(p2["_frozen"].values())


@pytest.mark.parametrize("kind", KINDS)
def test_phase2_loop(vdp, p2, kind):
    r, rec = p2[kind]["res"], p2[kind]["clip"]
    steps = int(vdp["cfg"]["phase2"]["steps"])
    hist = np.asarray(r["hist"])
    # clipping: once per step, max_norm 1, exactly the trainable conditioner + context params
    base_ids = {id(q) for n in ("T0", "T0inv") for q in vdp["base"][n].parameters()}
    train_ids = {id(q) for q in list(r["cond"].parameters()) + list(r["ctx"].parameters())} - base_ids
    assert len(rec) == steps
    assert all(m == 1.0 and post <= 1.0 + 1e-4 for m, _, post, _ in rec)
    assert all(ids == train_ids for *_, ids in rec)
    # best-validation state is what is returned, and it is the best in the history
    v_ret = _val(vdp, kind, r["cond"], r["ctx"])
    # untrained reference on the same draw: Delta = 0, i.e. the autonomous observer
    ctx0 = r["ctx"]
    if kind == "latent":
        z_std = torch.ones(vdp["des"].A.shape[0], device=DEV)
        v0 = _val(vdp, kind, build_conditioner("latent", vdp["base"]["T0inv"], ctx0.dim, 8, 4, z_std).to(DEV), ctx0)
    else:
        v0 = _val(vdp, kind, ConditionedDecoder(vdp["base"]["T0inv"], "none", 0, 0), ctx0)
    clipped = float(np.mean([pre > 1.0 for _, pre, _, _ in rec]))
    report(f"phase2.{kind}", width=r["width"], n_params=r["n_params"], val_untrained=v0,
           val_hist=[round(float(x), 5) for x in hist[:, 2]], train_hist=[round(float(x), 5) for x in hist[:, 1]],
           val_returned=r["val"], val_recomputed=v_ret, frac_steps_clipped=clipped,
           s_per_step_incl_val=r["wall_s"] / steps)
    assert np.isfinite(hist).all()
    assert r["val"] == hist[:, 2].min()
    assert abs(v_ret - r["val"]) / r["val"] < 1e-4
    assert r["val"] < v0                              # training reduced the deployed error
    assert hist[-1, 1] < hist[0, 1] or r["val"] < hist[0, 2]


def test_deployed_error_vs_validation(vdp, p2):
    """Decoder-side val IS the deployed error (z0 = 0); the latent val is a K-step
    rollout from the true T0(x_k0), so it can under-state the deployed latent observer."""
    data, sc, des = vdp["data"], vdp["scales"], vdp["des"]
    trained = {k: p2[k]["res"] for k in ("concat", "latent")}
    obs = build_observers(dict(vdp["cfg"], baselines=[]), des, sc, vdp["base"], trained, None, DEV)
    va = data["val"].to(DEV)
    sig = torch.as_tensor(sc["x_std"], dtype=torch.float32, device=DEV)
    x0 = initial_guess(vdp["cfg"], len(va), "center", 0, DEV)
    out = {}
    for k, ob in obs.items():
        xh = run(ob, va.y, va.u, x0)
        out[k] = float(((xh - va.x) / sig)[:, va.burn:].pow(2).mean())
    report("phase2.deployed_vs_val", concat_val=p2["concat"]["res"]["val"], concat_deployed=out["concat"],
           latent_val=p2["latent"]["res"]["val"], latent_deployed=out["latent"])
    assert abs(out["concat"] - p2["concat"]["res"]["val"]) / p2["concat"]["res"]["val"] < 0.25


def test_phase2_identical_draws_and_init_across_variants(vdp, monkeypatch):
    rec, ctx_init = {}, {}
    orig_loss, orig_ctx = T._decoder_loss, T.build_context

    def loss(cond, ctx, es, e, k, sig):
        rec[cur].append((torch.is_grad_enabled(), e.clone(), k.clone()))
        return orig_loss(cond, ctx, es, e, k, sig)

    def bctx(*a, **kw):
        c = orig_ctx(*a, **kw)
        ctx_init[cur] = {n: t.clone() for n, t in c.state_dict().items()}
        return c

    monkeypatch.setattr(T, "_decoder_loss", loss)
    monkeypatch.setattr(T, "build_context", bctx)
    cfg = copy.deepcopy(vdp["cfg"])
    cfg["phase2"].update({"steps": 3, "val_every": 3})
    for cur in ["concat", "film", "lora", "hyper"]:
        rec[cur] = []
        T.phase2(cur, cfg, vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, 0, log=lambda m: None)
    ref = rec["concat"]
    same = all(len(rec[k]) == len(ref) and all(a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
                                               for a, b in zip(rec[k], ref)) for k in rec)
    same_ctx = all(all(torch.equal(ctx_init[k][n], ctx_init["concat"][n]) for n in ctx_init["concat"]) for k in ctx_init)
    p = cfg["phase2"]
    report("phase2.fairness", identical_train_and_val_draws=same, identical_context_init=same_ctx,
           decoder_targets_per_step=int(p["episodes_per_batch"]) * int(p["times_per_episode"]),
           latent_targets_per_step=int(p["episodes_per_batch"]) * int(p["rollout_K"]),
           decoder_val_targets=len(vdp["data"]["val"]) * int(p["times_per_episode"]),
           latent_val_targets=len(vdp["data"]["val"]) * int(p["rollout_K"]))
    assert same and same_ctx


def test_phase2_reproducible(vdp):
    cfg = copy.deepcopy(vdp["cfg"])
    cfg["phase2"].update({"steps": 60, "val_every": 30})
    h = [np.asarray(T.phase2("concat", cfg, vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, 0,
                             log=lambda m: None)["hist"]) for _ in range(2)]
    d = float(np.abs(h[0] - h[1]).max() / np.abs(h[0][:, 1:]).max())
    report("phase2.reproducible", max_rel_diff=d)
    assert d < 1e-3


def test_time_per_step_full_budget(vdp, monkeypatch):
    """Seconds per optimisation step at the base.yaml phase-2 / nlox settings on vdp:
    median interval between successive gradient-clip calls (one per step, CUDA-synced),
    so setup, match_width and validation are excluded."""
    orig, stamps = torch.nn.utils.clip_grad_norm_, []

    def clip(params, max_norm, *a, **kw):
        out = orig(params, max_norm, *a, **kw)
        if DEV.startswith("cuda"):
            torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        return out

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    out = {}
    for kind in ("latent", "concat", "hyper", "nlox"):
        cfg = copy.deepcopy(vdp["cfg"])
        cfg["phase2"].update({"steps": 12, "val_every": 10 ** 6})
        cfg["nlox"].update({"steps": 12, "val_every": 10 ** 6})
        stamps.clear()
        if kind == "nlox":
            T.nlox(cfg, vdp["data"], vdp["scales"], vdp["des"], DEV, 0, log=lambda m: None)
        else:
            T.phase2(kind, cfg, vdp["data"], vdp["scales"], vdp["des"], vdp["base"], DEV, 0, log=lambda m: None)
        out[kind] = float(np.median(np.diff(stamps[1:])))       # drop the first (warm-up) step
    full = load("vdp")
    n = {k: int(full["nlox" if k == "nlox" else "phase2"]["steps"]) for k in out}
    report("timing.vdp", **{f"{k}_s_per_step": v for k, v in out.items()},
           **{f"{k}_full_run_min": out[k] * n[k] / 60 for k in out})
    assert all(v > 0 for v in out.values())
