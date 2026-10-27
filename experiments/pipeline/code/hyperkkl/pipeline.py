r"""Pipeline study (experiments/pipeline/PREREG.md): is the two-phase, frozen-base pipeline optimal?

Arms (all at the matched 140k conditioner budget, equal compute in loss-batch gradient evaluations):
  P0  phase 1, then phase 2 with the base frozen (train.phase1 / train.phase2, unchanged)
  P1  phase 1, then phase 2 with the base decoder unfrozen at lr x 0.1
  P2  one-shot joint training from scratch: encoder (labels + PDE), base decoder (recon on fresh
      encoded states) and conditioner (phase-2 decoder loss on z_obs), one optimiser, S/3 steps
  P3  phase 1, then an UNCONDITIONED fine-tune: base decoder at lr x 0.1 + an additive residual
      R(z) at the conditioner budget, phase-2 loss on forced data, no input
  P4  grey-box KKL: r_theta(x, u) fitted exactly as baselines_data.fit_greybox, phase 1 with
      f_known + r_theta(x, 0) in the PDE term, then the usual frozen phase 2

This module lives in a scratch copy only. It does not touch MODEL_MODULES, so the cache keys of
P0 (and of the shared phase 1 used by P1 and P3) are the live keys.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

from .conditioning import ConditionedDecoder, build_conditioner, match_width
from .monitor import TrainMonitor
from .nets import MLP, BaseMap, build_context, n_params
from .train import _Batcher, _decoder_loss, _flat_valid, _sched, _stats, phase1_diagnostics

ARMS = ("P0", "P1", "P2", "P3", "P4")
BASE_LR_FACTOR = 0.1


# ---------------------------------------------------------------------------
# Compute accounting: one unit = one loss-batch gradient evaluation
# ---------------------------------------------------------------------------

def compute_plan(arm: str, cfg: dict) -> dict:
    """Optimiser steps per stage and loss-batch gradient evaluations ('units') per arm."""
    e, d, p2 = (int(cfg["phase1"]["steps_enc"]), int(cfg["phase1"]["steps_dec"]),
                int(cfg["phase2"]["steps"]))
    total = e + d + p2
    if arm in ("P0", "P1", "P3", "P4"):
        plan = {"enc_steps": e, "dec_steps": d, "p2_steps": p2, "joint_steps": 0,
                "opt_steps": total, "units": total}
        if arm == "P4":
            plan["residual_fit_steps_max"] = 3000      # shared with the greybox baseline; disclosed
        return plan
    if arm == "P2":
        if total % 3:
            raise ValueError("P2 needs total steps divisible by 3")
        j = total // 3
        return {"enc_steps": 0, "dec_steps": 0, "p2_steps": 0, "joint_steps": j,
                "opt_steps": j, "units": 3 * j, "per_loss_evals": {"enc": j, "dec": j, "p2": j}}
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# P3 module: unconditioned additive residual on a trainable base decoder
# ---------------------------------------------------------------------------

class UncondDecoder(nn.Module):
    """x = T+(z) + R(z) (normalised space); c is ignored. R zero-initialised at its output."""

    def __init__(self, base: BaseMap, width: int):
        super().__init__()
        self.base = base
        nz, nx = base.layers[0].in_features, base.layers[-1].out_features
        self.R = MLP(nz, nx, width, 2, "silu", zero_last=True)
        self.kind, self.width = "uncond", width

    def forward(self, z, c=None):
        h = self.base.inp.norm(z)
        return self.base.out.denorm(self.base.mlp(h) + self.R(h))


def match_uncond_width(base: BaseMap, budget: int) -> tuple:
    nz, nx = base.layers[0].in_features, base.layers[-1].out_features
    count = lambda w: (nz + 1) * w + (w + 1) * w + (w + 1) * nx      # MLP(nz, nx, w, depth 2)
    w = min(range(1, 4096), key=lambda w: abs(count(w) - budget))
    assert count(w) == n_params(MLP(nz, nx, w, 2, "silu"))
    return w, count(w)


class NullContext(nn.Module):
    """Context of one constant-zero channel (P3 has no input pathway); no parameters."""
    dim = 1

    def seq(self, u):
        return torch.zeros(*u.shape[:-1], 1, device=u.device, dtype=u.dtype)

    def init_state(self, B, device):
        return None

    def step(self, u_k, state):
        return torch.zeros(u_k.shape[0], 1, device=u_k.device, dtype=u_k.dtype), None


# ---------------------------------------------------------------------------
# Decoder-side phase 2 with a trainable base (P1) or no conditioning (P3)
# ---------------------------------------------------------------------------

def phase2_variant(arm, kind, cfg, data, scales, des, base, device, seed, log=print) -> dict:
    """train.phase2 with the base decoder trainable (P1: conditioned, P3: unconditioned)."""
    assert arm in ("P1", "P3")
    torch.manual_seed(seed)
    t0 = time.time()
    p = cfg["phase2"]
    ccfg = p["context"]
    tr, va = data["train"].to(device), data["val"].to(device)
    Ti = copy.deepcopy(base["T0inv"]).to(device)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    nz = des.A.shape[0]
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    u_ref = float(cfg["system"].get("u_scale", 1.0))
    budget = int(p["budget"])
    if arm == "P1":
        ctx = build_context(ccfg["kind"], ccfg, 1, des.dt, np.linalg.eigvals(des.A), u_ref).to(device)
        width, npar = match_width(kind, base["T0inv"], ctx.dim, budget, int(p["rank"]), z_std)
        cond = build_conditioner(kind, Ti, ctx.dim, width, int(p["rank"]), z_std).to(device)
        side = [q for q in list(cond.parameters()) if q.requires_grad] + list(ctx.parameters())
    else:
        ctx = NullContext()
        width, npar = match_uncond_width(Ti, budget)
        cond = UncondDecoder(Ti, width).to(device)
        side = list(cond.R.parameters())
    for q in Ti.parameters():
        q.requires_grad_(True)
    base_params = list(Ti.parameters())
    lr = float(p["lr"])
    opt = torch.optim.Adam([{"params": side, "lr": lr}, {"params": base_params, "lr": lr * BASE_LR_FACTOR}])
    params = side + base_params
    n_base = sum(q.numel() for q in base_params)
    log(f"    [{arm}-{kind}] width {width}  conditioner params {npar}  base decoder params {n_base} "
        f"(lr x{BASE_LR_FACTOR})  context params {n_params(ctx)}")
    steps = int(p["steps"])
    sch = _sched(opt, steps, int(p.get("warmup", 100)))
    g = torch.Generator(device=device).manual_seed(seed)
    bt, bv = _Batcher(tr, device, g), _Batcher(va, device, torch.Generator(device=device).manual_seed(999))
    N1, burn = tr.x.shape[1], tr.burn
    Be, M = int(p["episodes_per_batch"]), int(p["times_per_episode"])
    ev = torch.arange(len(va), device=device)
    kv = bv.times(len(va), M, burn, N1)

    def val_loss():
        with torch.no_grad():
            return float(sum(_decoder_loss(cond, ctx, va, ev[i:i + 16], kv[i:i + 16], sig) * len(ev[i:i + 16])
                             for i in range(0, len(ev), 16)) / len(ev))

    best, best_state, hist = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    every = int(p.get("val_every", 250))
    for s in range(steps):
        e = bt.episodes(Be)
        loss = _decoder_loss(cond, ctx, tr, e, bt.times(Be, M, burn, N1), sig)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if (s + 1) % every == 0 or s == steps - 1:
            v = val_loss()
            hist.append([s + 1, loss.detach().item(), v])
            log(f"    [{arm}-{kind}] step {s + 1:5d}  train {hist[-1][1]:.3e}  val {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:
                best = v
                best_state = (copy.deepcopy(cond.state_dict()), copy.deepcopy(ctx.state_dict()))
    cond.load_state_dict(best_state[0])
    ctx.load_state_dict(best_state[1])
    for q in params:
        q.requires_grad_(False)
    return {"cond": cond, "ctx": ctx, "n_params": npar, "width": width, "base_dec_params": n_base,
            "ctx_params": n_params(ctx, trainable_only=False),
            "trainable_total": npar + n_base + n_params(ctx, trainable_only=False),
            "wall_s": time.time() - t0, "hist": hist, "val": best, "opt": mon.windows, "steps": steps}


# ---------------------------------------------------------------------------
# P2: one-shot joint training from scratch
# ---------------------------------------------------------------------------

def joint(kind, cfg, data, scales, des, device, seed, log=print, steps: int = None) -> dict:
    """Encoder, base decoder and conditioner trained together from scratch with the same losses
    as the two-phase pipeline, summed: label + ramped PDE (encoder), reconstruction of fresh
    states encoded by the CURRENT encoder, detached (base decoder), phase-2 decoder loss on z_obs
    (conditioner and base decoder). One Adam, lr = phase1.lr for all, cosine schedule."""
    torch.manual_seed(seed)
    t0 = time.time()
    p, q2 = cfg["phase1"], cfg["phase2"]
    steps = int(steps if steps is not None else compute_plan("P2", cfg)["joint_steps"])
    p1, coll = data["p1"], data["coll"]
    X = _flat_valid(p1, "x", p["stride"]).to(device)
    Z = _flat_valid(p1, "z", p["stride"]).to(device)
    xc, fc, yc = (coll[k].to(device) for k in ("x", "f0", "y"))
    nx, nz = X.shape[1], Z.shape[1]
    A = torch.as_tensor(des.A, dtype=torch.float32, device=device)
    B = torch.as_tensor(des.B, dtype=torch.float32, device=device)
    W = torch.as_tensor(-np.linalg.inv(des.A), dtype=torch.float32, device=device)
    Zf = _flat_valid(data["train"], "z", p["stride"] * 4).to(device)
    T0 = BaseMap(nx, nz, p["enc_width"], p["enc_depth"], "silu",
                 _stats(torch.cat([X, xc])), _stats(torch.cat([Z, Zf]))).to(device)
    sources = []
    if "coll" in p["decoder_data"]:
        sources.append(xc)
    if "traj" in p["decoder_data"]:
        sources.append(torch.cat([X, _flat_valid(data["train"], "x", p["stride"] * 4).to(device)]))
    Xd = torch.cat(sources)
    # decoder input statistics: the encoder is untrained, so use the label latents (zero-input
    # and forced) that the encoder is trained towards (phase 1 used stats of T0(sources))
    Ti = BaseMap(nz, nx, p["dec_width"], p["dec_depth"], "tanh", _stats(torch.cat([Z, Zf])), _stats(Xd)).to(device)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    tr, va = data["train"].to(device), data["val"].to(device)
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    ccfg = q2["context"]
    ctx = build_context(ccfg["kind"], ccfg, 1, des.dt, np.linalg.eigvals(des.A),
                        float(cfg["system"].get("u_scale", 1.0))).to(device)
    width, npar = match_width(kind, Ti, ctx.dim, int(q2["budget"]), int(q2["rank"]), z_std)
    cond = build_conditioner(kind, Ti, ctx.dim, width, int(q2["rank"]), z_std).to(device)
    for q in Ti.parameters():
        q.requires_grad_(True)                       # ConditionedDecoder froze it
    params = list(T0.parameters()) + list(Ti.parameters()) + \
        [q for n, q in cond.named_parameters() if not n.startswith("base.")] + list(ctx.parameters())
    n_enc, n_dec = n_params(T0), n_params(Ti)
    log(f"    [P2-{kind}] joint {steps} steps  conditioner {npar}  encoder {n_enc}  base decoder {n_dec}  "
        f"context {n_params(ctx)}")
    opt = torch.optim.Adam(params, lr=float(p["lr"]))
    sch = _sched(opt, steps, 100)
    zs, bs = T0.out.std, int(p["batch"])
    per = bs // len(sources)
    g = torch.Generator(device=device).manual_seed(seed)
    bt, bv = _Batcher(tr, device, g), _Batcher(va, device, torch.Generator(device=device).manual_seed(999))
    N1, burn = tr.x.shape[1], tr.burn
    Be, M = int(q2["episodes_per_batch"]), int(q2["times_per_episode"])
    ev = torch.arange(len(va), device=device)
    kv = bv.times(len(va), M, burn, N1)

    def val_loss():
        with torch.no_grad():
            return float(sum(_decoder_loss(cond, ctx, va, ev[i:i + 16], kv[i:i + 16], sig) * len(ev[i:i + 16])
                             for i in range(0, len(ev), 16)) / len(ev))

    best, best_state, hist = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    every = int(q2.get("val_every", 250))
    for s in range(steps):
        i = torch.randint(0, len(X), (bs,), device=device)
        l_lab = ((T0(X[i]) - Z[i]) / zs).pow(2).mean()
        l_pde = torch.zeros((), device=device)
        if p["pde"]:
            j = torch.randint(0, len(xc), (bs,), device=device)
            zj, Jf = jvp(T0, (xc[j],), (fc[j],))
            R = (Jf - zj @ A.T - yc[j] @ B.T) @ W.T
            l_pde = (R / fc[j].norm(dim=-1, keepdim=True).clamp(min=1e-6)).pow(2).sum(-1).mean()
        idx = [torch.randint(0, len(A_), (per,), device=device) for A_ in sources]
        xb = torch.cat([A_[k] for A_, k in zip(sources, idx)])
        l_rec = ((Ti(T0(xb).detach()) - xb) / sig).pow(2).mean()
        e = bt.episodes(Be)
        l_p2 = _decoder_loss(cond, ctx, tr, e, bt.times(Be, M, burn, N1), sig)
        loss = l_lab + float(p["lambda_pde"]) * min(1.0, s / (0.25 * steps)) * l_pde + l_rec + l_p2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if (s + 1) % every == 0 or s == steps - 1:
            v = val_loss()
            hist.append([s + 1, l_lab.item(), l_pde.item(), l_rec.item(), l_p2.item(), v])
            log(f"    [P2-{kind}] step {s + 1:5d}  lab {hist[-1][1]:.2e} pde {hist[-1][2]:.2e} rec {hist[-1][3]:.2e} "
                f"p2 {hist[-1][4]:.2e}  val {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:
                best = v
                best_state = [copy.deepcopy(m.state_dict()) for m in (T0, Ti, cond, ctx)]
    for m, sd in zip((T0, Ti, cond, ctx), best_state):
        m.load_state_dict(sd)
    for q in params:
        q.requires_grad_(False)
    diag = phase1_diagnostics(T0, Ti, X, Z, xc, fc, yc, A, B, W, sig)
    return {"T0": T0, "T0inv": Ti, "cond": cond, "ctx": ctx, "n_params": npar, "width": width,
            "enc_params": n_enc, "base_dec_params": n_dec, "ctx_params": n_params(ctx, trainable_only=False),
            "trainable_total": npar + n_enc + n_dec + n_params(ctx, trainable_only=False),
            "wall_s": time.time() - t0, "hist": hist, "val": best, "opt": mon.windows, "diag": diag,
            "steps": steps}


# ---------------------------------------------------------------------------
# P4: grey-box PDE term
# ---------------------------------------------------------------------------

@torch.no_grad()
def greybox_collocation(data: dict, gb_model) -> dict:
    """data with coll['f0'] = f_known(x, 0) + r_theta(x, 0) (the grey-box model at zero input)."""
    coll = dict(data["coll"])
    xt = coll["x"].double()
    coll["f0"] = gb_model.f(xt, torch.zeros(len(xt), gb_model.nu, dtype=xt.dtype)).float()
    coll["f0_known"] = data["coll"]["f0"]
    out = dict(data)
    out["coll"] = coll
    return out


# ---------------------------------------------------------------------------
# Zero-input collapse diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def collapse(dec, ref_dec, es, sig, n: int = 8192, dc: int = None) -> dict:
    """On zero-input episodes (z_obs after washout): the decoded state at c = 0 vs the phase-1
    base decoder (deviation, sigma units) and vs the true state (zero-input NRMSE)."""
    dev = sig.device
    nz = es.z_obs.shape[-1]
    Zo = es.z_obs[:, es.burn:].reshape(-1, nz).to(dev)
    Xo = es.x[:, es.burn:].reshape(-1, es.x.shape[-1]).to(dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    k = torch.randperm(len(Zo), generator=g)[:n].to(dev)
    z, x = Zo[k], Xo[k]
    c0 = torch.zeros(len(z), int(dc or 0), device=dev)
    out = dec(z, c0)
    ref = ref_dec(z) if ref_dec is not None else None
    r = {"zero_input_nrmse": float(((out - x) / sig).pow(2).mean().sqrt())}
    if ref is not None:
        d = (out - ref) / sig
        r["max_abs_dev_sigma"] = float(d.abs().max())
        r["rms_dev_sigma"] = float(d.pow(2).mean().sqrt())
        r["exact"] = bool(torch.equal(out, ref))
    return r
