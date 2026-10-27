r"""Training.

Phase 1, autonomous base maps (Niazi et al. 2025, sequential PINN):
    T0   x -> z   labels z from washed-out zero-input episodes  +  PDE residual
                  R = J_T0(x) f(x,0) - A T0(x) - B h(x) on an independent Sobol set,
                  weighted by -A^{-1} (the map from residual to steady latent error)
                  and divided by ||f(x,0)|| (so points near equilibria weigh most)
    T0+  z -> x  on fresh states encoded by the frozen T0 (collocation + trajectories)
    ``phase1.pde: false`` gives the model-free variant (no f anywhere).

Phase 2, input conditioning (C1). Base maps frozen. One objective for every variant,
the deployed state error in reference-std units:
    decoder-side (none, concat, film, lora, hyper):
        L = mean || (T+(z_obs_k; c_k) - x_k) / sigma ||^2
        z_obs is the latent the deployed filter carries, so this IS the deployment error.
    latent injection:
        L = mean over a K-step rollout from zhat = T0(x_k0) with Phi(zhat, c) in the
            latent dynamics, decoded by the frozen T0+.
Model selection: best validation loss (validation episodes from an independent seed).

NLOX baseline (Woelk et al. 2026): omega(z) and T_dagger trained jointly from rollouts
started at z0 = 0, model-free.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

from .conditioning import ConditionedDecoder, ContextScale, arch_options, build_conditioner, match_width
from .monitor import TrainMonitor
from .nets import MLP, BaseMap, build_context, n_params
from .observers import LatentStep


def _sched(opt, steps, warmup):
    lam = lambda s: min(1.0, (s + 1) / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, lam)


def _stats(t):
    return t.mean(0), t.std(0).clamp(min=1e-6)


def _flat_valid(es, field, stride):
    return getattr(es, field)[:, es.burn::stride].reshape(-1, getattr(es, field).shape[-1])


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1(cfg, data, scales, des, device, seed, log=print) -> dict:
    torch.manual_seed(seed)
    p = cfg["phase1"]
    t0 = time.time()
    p1, coll = data["p1"], data["coll"]
    X = _flat_valid(p1, "x", p["stride"]).to(device)
    Z = _flat_valid(p1, "z", p["stride"]).to(device)
    xc, fc, yc = (coll[k].to(device) for k in ("x", "f0", "y"))
    nx, nz = X.shape[1], Z.shape[1]
    A = torch.as_tensor(des.A, dtype=torch.float32, device=device)
    B = torch.as_tensor(des.B, dtype=torch.float32, device=device)
    W = torch.as_tensor(-np.linalg.inv(des.A), dtype=torch.float32, device=device)

    # Output scale from the FORCED latents, which span the operating domain. Zero-input
    # labels alone can collapse to one point (a globally attracting equilibrium, e.g. the
    # bioreactor: label std 3e-4), and normalising by that forces the network to emit
    # ~1e4 to cover the domain. Labels at the equilibrium + the PDE residual over the
    # domain remain a well-posed problem; only the scaling was wrong.
    Zf = _flat_valid(data["train"], "z", p["stride"] * 4).to(device)
    T0 = BaseMap(nx, nz, p["enc_width"], p["enc_depth"], "silu",
                 _stats(torch.cat([X, xc])), _stats(torch.cat([Z, Zf]))).to(device)
    opt = torch.optim.Adam(T0.parameters(), lr=p["lr"])
    sch = _sched(opt, p["steps_enc"], 100)
    zs = T0.out.std
    bs = int(p["batch"])
    hist = []
    mon = TrainMonitor(T0.parameters(), opt)
    for s in range(int(p["steps_enc"])):
        i = torch.randint(0, len(X), (bs,), device=device)
        l_lab = ((T0(X[i]) - Z[i]) / zs).pow(2).mean()
        loss = l_lab
        l_pde = torch.zeros((), device=device)
        if p["pde"]:
            j = torch.randint(0, len(xc), (bs,), device=device)
            zj, Jf = jvp(T0, (xc[j],), (fc[j],))
            R = (Jf - zj @ A.T - yc[j] @ B.T) @ W.T
            l_pde = (R / fc[j].norm(dim=-1, keepdim=True).clamp(min=1e-6)).pow(2).sum(-1).mean()
            loss = loss + float(p["lambda_pde"]) * min(1.0, s / (0.25 * p["steps_enc"])) * l_pde
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if s % 500 == 0 or s == p["steps_enc"] - 1:
            hist.append([s, l_lab.detach().item(), l_pde.detach().item()])
            log(f"    [T0]  step {s:5d}  label {hist[-1][1]:.3e}  pde {hist[-1][2]:.3e}  "
                + mon.fmt(mon.flush(s)))
    diag_enc = mon.windows
    for q in T0.parameters():
        q.requires_grad_(False)

    # ---- decoder on fresh states encoded by the frozen T0 -------------------------
    # Each batch is half collocation (uniform over the envelope the in-bounds input tiers
    # reach) and half trajectory states (where the observer spends its time). Concatenating them instead
    # let the ~20x more numerous trajectory samples dominate, and the decoder then failed
    # off the attractor: eps_dec 4.0 sigma on quad_tank (archive: "uniform collocation
    # beats trajectory data 3-10x" for the decoder).
    sources = []
    if "coll" in p["decoder_data"]:
        sources.append(xc)
    if "traj" in p["decoder_data"]:
        sources.append(torch.cat([X, _flat_valid(data["train"], "x", p["stride"] * 4).to(device)]))
    enc = lambda A_: torch.cat([T0(A_[i:i + 65536]) for i in range(0, len(A_), 65536)])
    with torch.no_grad():
        zsrc = [enc(A_) for A_ in sources]
    Xd, Zd = torch.cat(sources), torch.cat(zsrc)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    Ti = BaseMap(nz, nx, p["dec_width"], p["dec_depth"], "tanh", _stats(Zd), _stats(Xd)).to(device)
    opt = torch.optim.Adam(Ti.parameters(), lr=p["lr"])
    sch = _sched(opt, p["steps_dec"], 100)
    dhist = []
    mon = TrainMonitor(Ti.parameters(), opt)
    per = bs // len(sources)
    for s in range(int(p["steps_dec"])):
        idx = [torch.randint(0, len(A_), (per,), device=device) for A_ in sources]
        xb = torch.cat([A_[i] for A_, i in zip(sources, idx)])
        zb = torch.cat([Z_[i] for Z_, i in zip(zsrc, idx)])
        loss = ((Ti(zb) - xb) / sig).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if s % 500 == 0 or s == p["steps_dec"] - 1:
            dhist.append([s, loss.detach().item()])
            log(f"    [T0+] step {s:5d}  recon {dhist[-1][1]:.3e}  " + mon.fmt(mon.flush(s)))
    for q in Ti.parameters():
        q.requires_grad_(False)
    return {"T0": T0, "T0inv": Ti, "wall_s": time.time() - t0,
            "hist_enc": hist, "hist_dec": dhist, "opt_enc": diag_enc, "opt_dec": mon.windows,
            "diag": phase1_diagnostics(T0, Ti, X, Z, xc, fc, yc, A, B, W, sig)}


@torch.no_grad()
def phase1_diagnostics(T0, Ti, X, Z, xc, fc, yc, A, B, W, sig, n=8192) -> dict:
    """The measurable terms of the error bound (THEORY.md, Prop. 1)."""
    g = torch.Generator(device=X.device).manual_seed(0)
    i = torch.randint(0, len(X), (n,), device=X.device, generator=g)
    j = torch.randint(0, len(xc), (n,), device=X.device, generator=g)
    eps_enc = float(((T0(X[i]) - Z[i]) / T0.out.std).pow(2).mean().sqrt())
    eps_dec = float(((Ti(T0(xc[j])) - xc[j]) / sig).pow(2).sum(-1).mean().sqrt())
    with torch.enable_grad():
        zj, Jf = jvp(T0, (xc[j],), (fc[j],))
    R = Jf - zj @ A.T - yc[j] @ B.T
    zt = T0(xc[j])
    d = torch.randn(zt.shape, generator=g, device=X.device)
    d = d / d.norm(dim=-1, keepdim=True)
    h = 1e-2 * zt.std(0).norm()
    l_dec = ((Ti(zt + h * d) - Ti(zt)) / sig).norm(dim=-1) / h
    return {"eps_enc_rel": eps_enc, "eps_dec": eps_dec,
            "eps_pde_rms": float(R.norm(dim=-1).pow(2).mean().sqrt()),
            "eps_pde_ss_rms": float((R @ W.T).norm(dim=-1).pow(2).mean().sqrt()),
            "l_dec_q95": float(torch.quantile(l_dec, 0.95)), "l_dec_mean": float(l_dec.mean())}


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

class _Batcher:
    """Episode-first sampling: draw episodes, then time indices within them."""

    def __init__(self, es, device, gen):
        self.es, self.dev, self.g = es, device, gen
        self.E, self.N1 = es.x.shape[0], es.x.shape[1]

    def episodes(self, n):
        return torch.randint(0, self.E, (n,), device=self.dev, generator=self.g)

    def times(self, n_ep, m, lo, hi):
        return torch.randint(lo, hi, (n_ep, m), device=self.dev, generator=self.g)


def _decoder_loss(cond, ctx, es, e, k, sig):
    c = ctx.seq(es.u[e]) if ctx is not None else torch.zeros(len(e), es.u.shape[1], 0, device=es.u.device)
    r = torch.arange(len(e), device=e.device)[:, None]
    z, x, cc = es.z_obs[e[:, None], k], es.x[e[:, None], k], c[r, k]
    pred = cond(z.reshape(-1, z.shape[-1]), cc.reshape(-1, cc.shape[-1]))
    return ((pred - x.reshape(-1, x.shape[-1])) / sig).pow(2).mean()


class _Graphed:
    """Loss + backward of a fixed-shape computation, replayed as ONE CUDA graph.

    Rollouts are hundreds of tiny sequential kernels, so they are launch-bound (1.7 s per
    training step for K = 256 on a loaded host); one graph replay costs ~0.12 s. The
    computation reads only STATIC tensors (the training set on the GPU, parameters, and
    index buffers the caller refreshes in place before each call), so a replay computes
    exactly what an eager call would. Gradients land in the parameters' .grad tensors
    (created inside the graph's pool: never set them to None afterwards); the optimiser
    step stays eager. ``enabled=False`` runs the same function eagerly.
    """

    def __init__(self, fn, params, enabled=True, warmup=3):
        self.fn, self.params = fn, list(params)
        self.enabled = bool(enabled) and torch.cuda.is_available() and \
            all(q.is_cuda for q in self.params)
        self.warmup, self.graph, self.loss = int(warmup), None, None

    def __call__(self):
        if not self.enabled:
            for q in self.params:
                q.grad = None
            loss = self.fn()
            loss.backward()
            return loss
        if self.graph is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):                 # warm-up: allocator, cuFFT/cuDNN plans
                for _ in range(self.warmup):
                    for q in self.params:
                        q.grad = None
                    self.fn().backward()
            torch.cuda.current_stream().wait_stream(st)
            for q in self.params:
                q.grad = None
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss = self.fn()
                self.loss.backward()
        self.graph.replay()
        return self.loss


def _rollout_loss(inj, ctx, lat, T0, Ti, es, e, k0, K, sig):
    """K-step latent rollout with injection from zhat = T0(x_k0), decoded by the frozen
    T0+ in one call after the rollout (decoding inside the loop is launch-bound)."""
    c = ctx.seq(es.u[e])
    r = torch.arange(len(e), device=e.device)
    z = T0(es.x[e, k0])
    zs = []
    for j in range(K):
        kk = k0 + j
        ck = c[r, kk]
        z = lat(z, es.y[e, kk], es.y[e, kk + 1], lambda zz: inj(zz, ck))
        zs.append(z)
    Z = torch.stack(zs, 1)                                          # (B, K, nz)
    kk = k0[:, None] + torch.arange(1, K + 1, device=e.device)[None]
    X = es.x[e[:, None], kk]
    return ((Ti(Z.reshape(-1, Z.shape[-1])) - X.reshape(-1, X.shape[-1])) / sig).pow(2).mean()


def phase2(kind, cfg, data, scales, des, base, device, seed, log=print) -> dict:
    """Train one conditioning variant. Returns {"cond", "ctx", "n_params", ...}."""
    torch.manual_seed(seed)
    t0 = time.time()
    p = cfg["phase2"]
    ccfg = p["context"]
    tr, va = data["train"].to(device), data["val"].to(device)
    T0, Ti = base["T0"], base["T0inv"]
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    nz = des.A.shape[0]
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    u_ref = float(cfg["system"].get("u_scale", 1.0))

    if kind == "none":
        return {"cond": ConditionedDecoder(Ti, "none", 0, 0), "ctx": None, "n_params": 0,
                "width": 0, "wall_s": 0.0, "hist": [], "val": float("nan")}

    ctx = build_context(ccfg["kind"], ccfg, 1, des.dt, np.linalg.eigvals(des.A), u_ref).to(device)
    dc = ctx.dim
    # budget_includes_context: the context encoder counts against the budget, which is the
    # fair rule when variants differ in their ENCODER (node C1-ctx: 5k-14k parameters)
    budget = int(p["budget"]) - (n_params(ctx) if p.get("budget_includes_context") else 0)
    arch = arch_options(p)                     # tuning knobs; {} = the original architecture
    width, npar = match_width(kind, Ti, dc, budget, int(p["rank"]), z_std, **arch)
    cond = build_conditioner(kind, Ti, dc, width, int(p["rank"]), z_std, **arch).to(device)
    ctx_scale = None
    if p.get("ctx_rms") is not None:           # shared context-scale knob (absent = unscaled, as before)
        with torch.no_grad():
            cc = torch.cat([ctx.seq(tr.u[i:i + 64])[:, tr.burn:] for i in range(0, len(tr), 64)])
            rms = float(cc.pow(2).mean().sqrt())
        ctx_scale = float(p["ctx_rms"]) / max(rms, 1e-12)
        cond = ContextScale(cond, ctx_scale).to(device)
    params = [q for q in list(cond.parameters()) + list(ctx.parameters()) if q.requires_grad]
    log(f"    [{kind}/{ccfg['kind']}] width {width}  conditioner params {npar}  "
        f"context params {n_params(ctx)}" + (f"  lr {p['lr']:g} arch {arch}" if arch else "")
        + (f"  ctx_rms {p['ctx_rms']} (scale {ctx_scale:.4g})" if ctx_scale is not None else ""))
    opt = torch.optim.Adam(params, lr=p["lr"])
    steps = int(p["steps"])
    sch = _sched(opt, steps, int(p.get("warmup", 100)))
    g = torch.Generator(device=device).manual_seed(seed)
    bt, bv = _Batcher(tr, device, g), _Batcher(va, device, torch.Generator(device=device).manual_seed(999))
    lat = LatentStep(des, device)
    N1, burn = tr.x.shape[1], tr.burn
    Be, M = int(p["episodes_per_batch"]), int(p["times_per_episode"])
    K = int(p.get("rollout_K", 256))

    # fixed validation draw, identical for every variant
    ev = torch.arange(len(va), device=device)
    kv = bv.times(len(va), M, burn, N1 if kind != "latent" else N1 - K - 1)

    def val_loss():
        with torch.no_grad():
            if kind == "latent":
                return float(sum(_rollout_loss(cond, ctx, lat, T0, Ti, va, ev[i:i + 16],
                                               kv[i:i + 16, 0], K, sig) * len(ev[i:i + 16])
                                 for i in range(0, len(ev), 16)) / len(ev))
            return float(sum(_decoder_loss(cond, ctx, va, ev[i:i + 16], kv[i:i + 16], sig) * len(ev[i:i + 16])
                             for i in range(0, len(ev), 16)) / len(ev))

    best, best_state, hist = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    every = int(p.get("val_every", 250))
    if kind == "latent":                       # static index buffers, refreshed in place
        e_buf, k_buf = bt.episodes(Be), bt.times(Be, 1, burn, N1 - K - 1)[:, 0]
        graphed = _Graphed(lambda: _rollout_loss(cond, ctx, lat, T0, Ti, tr, e_buf, k_buf, K, sig),
                           params, enabled=cfg.get("cuda_graphs", True))
    for s in range(steps):
        if kind == "latent":
            e_buf.copy_(bt.episodes(Be))
            k_buf.copy_(bt.times(Be, 1, burn, N1 - K - 1)[:, 0])
            loss = graphed()                       # forward + backward
        else:
            e = bt.episodes(Be)
            loss = _decoder_loss(cond, ctx, tr, e, bt.times(Be, M, burn, N1), sig)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if (s + 1) % every == 0 or s == steps - 1:
            v = val_loss()
            hist.append([s + 1, loss.detach().item(), v])
            log(f"    [{kind}] step {s + 1:5d}  train {hist[-1][1]:.3e}  val {v:.3e}  "
                + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:              # also guards an all-NaN history
                best = v
                best_state = (copy.deepcopy(cond.state_dict()), copy.deepcopy(ctx.state_dict()))
    cond.load_state_dict(best_state[0]); ctx.load_state_dict(best_state[1])
    for q in params:
        q.requires_grad_(False)
    return {"cond": cond, "ctx": ctx, "n_params": npar, "width": width,
            "ctx_params": n_params(ctx, trainable_only=False), "wall_s": time.time() - t0,
            "hist": hist, "val": best, "opt": mon.windows, "lr": float(p["lr"]), "rank": int(p["rank"]),
            "arch": arch, **({"ctx_rms": float(p["ctx_rms"]), "ctx_scale": ctx_scale} if ctx_scale is not None else {})}


# ---------------------------------------------------------------------------
# NLOX baseline
# ---------------------------------------------------------------------------

class _Omega(nn.Module):
    def __init__(self, nz, nu, width, z_std, u_ref=1.0):
        super().__init__()
        self.net = MLP(nz, nz * nu, width, 2, "tanh")
        self.register_buffer("z_std", z_std)
        self.nz, self.nu, self.u_ref = nz, nu, float(u_ref)

    def forward(self, z, u):
        om = self.net(z / self.z_std).view(-1, self.nz, self.nu)
        return torch.bmm(om, (u / self.u_ref)[:, :, None])[:, :, 0] * self.z_std


class _NoCtxDecoder(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, z, c):
        return self.dec(z)


def nlox(cfg, data, scales, des, device, seed, log=print) -> dict:
    torch.manual_seed(seed)
    t0 = time.time()
    p = cfg["nlox"]
    tr, va = data["train"].to(device), data["val"].to(device)
    nz, nx = des.A.shape[0], tr.x.shape[-1]
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    zf = tr.z_obs[:, tr.burn:].reshape(-1, nz)
    xf = tr.x[:, tr.burn:].reshape(-1, nx)
    om = _Omega(nz, 1, int(p["width"]), zf.std(0).clamp(min=1e-6),
                float(cfg["system"].get("u_scale", 1.0))).to(device)
    dec = BaseMap(nz, nx, int(p["dec_width"]), int(p["dec_depth"]), "tanh", _stats(zf), _stats(xf)).to(device)
    params = list(om.parameters()) + list(dec.parameters())
    opt = torch.optim.RMSprop(params, lr=float(p["lr"]))
    steps = int(p["steps"])
    sch = _sched(opt, steps, 50)
    lat = LatentStep(des, device)
    K, burn, N = int(p["K"]), tr.burn, tr.x.shape[1] - 1
    g = torch.Generator(device=device).manual_seed(seed)
    Bs = int(p["episodes_per_batch"])

    # Stateful truncated BPTT: each stream carries its latent from z0 = 0 through a whole
    # episode, K steps per optimiser step (detached between steps), and restarts on a fresh
    # episode at the end. Training therefore covers every post-washout time the observer
    # is scored on; supervising only [burn, burn + K] from z0 = 0 never reached t > 11 s
    # on vdp, while evaluation scores t >= 25 s.
    ep = torch.randint(0, len(tr), (Bs,), device=device, generator=g)
    kk = torch.zeros(Bs, dtype=torch.long, device=device)
    zst = torch.zeros(Bs, nz, device=device)
    r = torch.arange(Bs, device=device)

    @torch.no_grad()
    def validate():
        """Deployed NLOX (z0 = 0) on every validation episode, error after washout."""
        tot = 0.0
        for i in range(0, len(va), 32):
            e = torch.arange(i, min(i + 32, len(va)), device=device)
            z, zs = torch.zeros(len(e), nz, device=device), []
            for k in range(N):
                uk = va.u[e, k]
                z = lat(z, va.y[e, k], va.y[e, k + 1], lambda zz: om(zz, uk))
                if k + 1 >= burn:
                    zs.append(z)
            Z = torch.stack(zs, 1)
            X = va.x[e, burn:burn + Z.shape[1]]
            tot += ((dec(Z.reshape(-1, nz)) - X.reshape(-1, nx)) / sig).pow(2).mean().item() * len(e)
        return tot / len(va)

    out = {}

    def tbptt_loss():
        z, zs, msk = zst, [], []
        for j in range(K):
            k = kk + j
            uk = tr.u[ep, k]
            z = lat(z, tr.y[ep, k], tr.y[ep, k + 1], lambda zz: om(zz, uk))
            zs.append(z)
            msk.append(k + 1 >= burn)
        out["z"] = z
        Z, Mk = torch.stack(zs, 1), torch.stack(msk, 1)                       # (Bs, K, .)
        X = tr.x[ep[:, None], kk[:, None] + torch.arange(1, K + 1, device=device)[None]]
        err = ((dec(Z.reshape(-1, nz)) - X.reshape(-1, nx)) / sig).pow(2).mean(-1)
        w = Mk.reshape(-1).float()
        return (err * w).sum() / w.sum().clamp(min=1.0)    # all-masked window: loss 0, grad 0

    graphed = _Graphed(tbptt_loss, params, enabled=cfg.get("cuda_graphs", True))
    best, best_state, hist = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    for s in range(steps):
        done = kk + K > N                                   # restart streams that would run out
        if done.any():
            ep[done] = torch.randint(0, len(tr), (int(done.sum()),), device=device, generator=g)
            kk[done] = 0
            zst[done] = 0.0
        loss = graphed()                           # K-step rollout + backward, one graph
        mon.clip()
        opt.step()
        sch.step()
        zst.copy_(out["z"].detach())
        kk.add_(K)
        if (s + 1) % int(p["val_every"]) == 0 or s == steps - 1:
            v = validate()
            hist.append([s + 1, loss.detach().item(), v])
            log(f"    [nlox] step {s + 1:5d}  train {hist[-1][1]:.3e}  val {v:.3e}  "
                + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:              # also guards an all-NaN history
                best, best_state = v, (copy.deepcopy(om.state_dict()), copy.deepcopy(dec.state_dict()))
    om.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    for q in params:
        q.requires_grad_(False)
    return {"omega": om, "dec": _NoCtxDecoder(dec), "n_params": n_params(om, False) + n_params(dec, False),
            "wall_s": time.time() - t0, "hist": hist, "val": best, "opt": mon.windows}
