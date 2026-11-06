r"""jitter_fix: HyperKKL phase-2 training / run-time changes against output jitter (ITERATIONS.md).

Package-agnostic (works on code/qpkg and code/bpkg; import jf_env.setup(plant) first).

The HyperKKL structure is unchanged in every candidate: linear Hurwitz latent filter z' = A z + B y (phase 1
maps, never retrained) + LoRA hypernetwork-conditioned decoder x^ = T+(z; [c_sfm(u), h(z-window)]).

mods (training keys; every key absent = beat.phase2_bg itself, i.e. the base):
  window   K                 z-window length K (stride unchanged, so the span is K/20 x the base span)
  ctx_lp   {"fc"}            causal first-order low-pass on the SFM input context c, in training (exact
                             truncated convolution, tail < 1e-7) and at run time (recursion), a = exp(-2 pi fc dt)
  specnorm {"gamma"}         LoRA generator heads U_l, V_l (hypernetwork output layers) used as
                             W * min(1, gamma / ||W||_2)
  jump     {"w", "lags"}     w * mean_L E||((x^_{k+L} - x^_k) - (x_{k+L} - x_k)) / sigma||^2
  wsmooth  {"w"}             w * E sum_l ||DeltaW_l(cc_{k+1}) - DeltaW_l(cc_k)||_F^2 / (a_l o_l)
                             (time derivative of the generated rank-r decoder weight updates)
  sens     {"wc", "wz", "eps", "z_noise", "splice"}   off-manifold slope penalties (quad_ood W1b form):
                             wc E||(T+(z, c'+eps v) - T+(z, c'))/(eps sigma)||^2, c' = context of a SPLICED input
                             history (dwell U[0.3, 2.5] t_ref); wz the same in z at z' = z + z_noise z_std N(0, I)
Run-time (post-hoc on the estimate sequence, causal): outlp fc -> x~_k = a x~_{k-1} + (1 - a) x^_k, x~_0 = x^_0.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as Pz

from hyperkkl.beat import HistKKLObserver, HistoryFeatures, make_opt, phase2_bg
from hyperkkl.conditioning import build_conditioner, match_width
from hyperkkl.monitor import TrainMonitor
from hyperkkl.nets import build_context, n_params
from hyperkkl.observers import LatentStep
from hyperkkl.train import _Batcher, _sched

TRAIN_KEYS = ("window", "ctx_lp", "specnorm", "jump", "wsmooth", "sens")


def _sq(pred, x, sig):
    """((pred - x)/sig)^2 mean -- identical expression to beat._decoder_loss_bg for dense labels."""
    return ((pred - x) / sig).pow(2).mean()


# ------------------------------------------------------------------------------------------------ pieces
class SpecCap(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, W):
        s = torch.linalg.matrix_norm(W, 2)
        return W * torch.clamp(self.gamma / s.clamp(min=1e-12), max=1.0)


def head_norms(cond):
    with torch.no_grad():
        return [float(torch.linalg.matrix_norm(m.weight, 2)) for m in list(cond.U) + list(cond.V)]


def ema_coef(fc, dt):
    return math.exp(-2.0 * math.pi * float(fc) * float(dt))


def ema_seq(c: torch.Tensor, a: float, tail=1e-7) -> torch.Tensor:
    """Causal EMA along dim 1 of c (B, N1, d), y_0 = c_0: y_k = a^k c_0 + (1-a) sum_{j=1..k} a^{k-j} c_j.
    Truncated convolution (the dropped tail mass a^K < tail), left padding with c_0."""
    B, N1, d = c.shape
    K = int(min(N1, max(2, math.ceil(math.log(tail) / math.log(a))))) if a > 0 else 1
    i = torch.arange(K, device=c.device, dtype=c.dtype)
    w = (1 - a) * a ** i
    w[-1] = a ** (K - 1)                       # mass of the infinite tail on the padded c_0 / oldest sample
    x = c.permute(0, 2, 1).reshape(B * d, 1, N1)
    x = torch.cat([x[..., :1].expand(-1, -1, K - 1), x], -1)
    y = F.conv1d(x, w.flip(0).view(1, 1, K))
    return y.reshape(B, d, N1).permute(0, 2, 1)


def ema_np(xh: np.ndarray, a: float) -> np.ndarray:
    """Run-time causal EMA of an estimate trajectory (E, N1, nx) (recursion, deployable)."""
    out = np.empty_like(xh)
    out[:, 0] = xh[:, 0]
    for k in range(1, xh.shape[1]):
        out[:, k] = a * out[:, k - 1] + (1 - a) * xh[:, k]
    return out


class LPContext(nn.Module):
    """Causal low-pass on the SFM input context (training: ema_seq; deployment: recursion from y_0 = c_0)."""

    def __init__(self, inner, a):
        super().__init__()
        self.inner, self.a = inner, float(a)
        self.dim = inner.dim

    def seq(self, u):
        return ema_seq(self.inner.seq(u), self.a)

    def init_state(self, B, device):
        return (self.inner.init_state(B, device), None)

    def step(self, u_k, state):
        s_in, y = state
        c, s_in = self.inner.step(u_k, s_in)
        y = c if y is None else self.a * y + (1 - self.a) * c
        return y, (s_in, y)


def splice_u(u: torch.Tensor, dt: float, gen: torch.Generator, lo, hi) -> torch.Tensor:
    B, N1 = u.shape[:2]
    src = torch.empty(B, N1, dtype=torch.long)
    for b in range(B):
        k = 0
        while k < N1:
            n = max(1, int(round(float(torch.empty(1).uniform_(lo, hi, generator=gen)) / dt)))
            src[b, k:k + n] = int(torch.randint(0, B, (1,), generator=gen))
            k += n
    src = src.to(u.device)
    t = torch.arange(N1, device=u.device)[None].expand(B, -1)
    return u[src, t]


def gen_updates(cond, c):
    """Generated LoRA factors per layer: [(U (n, a, r), V (n, r, o))]."""
    t = cond.trunk(c)
    out = []
    for i, layer in enumerate(cond.base.layers):
        a, o = layer.in_features, layer.out_features
        out.append((cond.U[i](t).view(-1, a, cond.rank), cond.V[i](t).view(-1, cond.rank, o)))
    return out


def dW_sq(cond, c1, c2):
    """sum_l ||U2 V2 - U1 V1||_F^2 / (r a_l o_l) per row, via 2r x 2r Gram matrices."""
    tot = 0.0
    for (U1, V1), (U2, V2) in zip(gen_updates(cond, c1), gen_updates(cond, c2)):
        P = torch.cat([U2, U1], -1)
        Q = torch.cat([V2, -V1], -2)
        g = (P.transpose(1, 2) @ P) * (Q @ Q.transpose(1, 2))
        tot = tot + g.sum((-1, -2)) / (cond.rank * U1.shape[1] * V1.shape[2])
    return tot


# ------------------------------------------------------------------------------------------------ loss
def loss_jf(cond, ctx, hist, es, e, k, sig, mods, train, aux=None, terms_out=None):
    """beat._decoder_loss_bg (+ the penalty terms of mods when train). aux: {gen, z_std, dt, t_ref}.
    terms_out (dict): receives the unweighted penalty values (for the per-plant weight rule)."""
    c = ctx.seq(es.u[e])
    r = torch.arange(len(e), device=e.device)[:, None]

    def feats(kk):
        z, cc = es.z_obs[e[:, None], kk], c[r, kk]
        if hist is not None:
            h = hist.at(es.z_obs[e], None, kk)
            cc = torch.cat([cc, h], -1)
        return z.reshape(-1, z.shape[-1]), cc.reshape(-1, cc.shape[-1])

    zf, cf = feats(k)
    x = es.x[e[:, None], k].reshape(-1, es.x.shape[-1])
    pred = cond(zf, cf)
    loss = _sq(pred, x, sig)
    if not train:
        return loss
    want = terms_out is not None
    N1 = es.x.shape[1]
    jp = mods.get("jump")
    if jp and (float(jp.get("w", 0)) > 0 or want):
        tot = 0.0
        for L in jp["lags"]:
            k2 = (k + int(L)).clamp(max=N1 - 1)
            p2 = cond(*feats(k2))
            x2 = es.x[e[:, None], k2].reshape(-1, es.x.shape[-1])
            tot = tot + (((p2 - pred) - (x2 - x)) / sig).pow(2).mean()
        tot = tot / len(jp["lags"])
        if want:
            terms_out["jump"] = float(tot)
        loss = loss + float(jp.get("w", 0)) * tot
    ws = mods.get("wsmooth")
    if ws and (float(ws.get("w", 0)) > 0 or want):
        k2 = (k + 1).clamp(max=N1 - 1)
        _, cf2 = feats(k2)
        n = min(int(ws.get("rows", 2048)), cf.shape[0])
        pen = dW_sq(cond, cf[:n], cf2[:n]).mean()
        if want:
            terms_out["wsmooth"] = float(pen)
        loss = loss + float(ws.get("w", 0)) * pen
    sp = mods.get("sens")
    if sp and (float(sp.get("wc", 0)) > 0 or float(sp.get("wz", 0)) > 0 or want):
        eps = float(sp.get("eps", 0.05))
        zp, cp = zf, cf
        if sp.get("splice", True):
            us = splice_u(es.u[e], aux["dt"], aux["gen"], 0.3 * aux["t_ref"], 2.5 * aux["t_ref"])
            csp = ctx.seq(us)[r, k]
            if hist is not None:
                csp = torch.cat([csp, hist.at(es.z_obs[e], None, k)], -1)
            cp = csp.reshape(-1, csp.shape[-1])
        if sp.get("z_noise"):
            zp = zf + float(sp["z_noise"]) * aux["z_std"] * torch.randn_like(zf)
        base_p = cond(zp, cp)
        v = torch.randn_like(cp); v = v / v.norm(dim=-1, keepdim=True)
        dc = ((cond(zp, cp + eps * v) - base_p) / (eps * sig)).pow(2).sum(-1).mean()
        w = torch.randn_like(zp); w = w / w.norm(dim=-1, keepdim=True) * aux["z_std"]
        dz = ((cond(zp + eps * w, cp) - base_p) / (eps * sig)).pow(2).sum(-1).mean()
        if want:
            terms_out["sens_c"], terms_out["sens_z"] = float(dc), float(dz)
        loss = loss + float(sp.get("wc", 0)) * dc + float(sp.get("wz", 0)) * dz
    if want:
        terms_out["fit"] = float(_sq(pred, x, sig))
    return loss


# ------------------------------------------------------------------------------------------------ phase 2
def phase2_jf(cfg, data, scales, des, base, device, seed, opts: dict, log=print, mods: dict = None,
              force_custom=False) -> dict:
    """beat.phase2_bg('lora') + mods. No train key in mods (and not force_custom): phase2_bg itself."""
    mods = dict(mods or {})
    if not any(m in mods for m in TRAIN_KEYS) and not force_custom:
        return phase2_bg("lora", cfg, data, scales, des, base, device, seed, opts, log)
    kind = "lora"
    torch.manual_seed(seed)
    t0 = time.time()
    p = cfg["phase2"]
    ccfg = p["context"]
    tr, va = data["train"].to(device), data["val"].to(device)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    nz = des.A.shape[0]
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    u_ref = float(cfg["system"].get("u_scale", 1.0))
    lr = float(opts.get("lr", p["lr"]))
    steps = int(opts.get("steps", p["steps"]))
    budget = int(opts.get("budget", p["budget"]))
    rank = int(p["rank"])
    ctx = build_context(ccfg["kind"], ccfg, 1, des.dt, np.linalg.eigvals(des.A), u_ref).to(device)
    if mods.get("ctx_lp"):
        ctx = LPContext(ctx, ema_coef(mods["ctx_lp"]["fc"], des.dt)).to(device)
    hist = None
    window = int(mods.get("window", opts.get("window", 0)))
    if window:
        hist = HistoryFeatures(nz, window, int(opts.get("stride", 1)), None, z_std=z_std, zs_std=None).to(device)
    dc = ctx.dim + (hist.dim if hist is not None else 0)
    Ti0 = base["T0inv"]
    width, npar = match_width(kind, Ti0, dc, budget, rank, z_std)
    unfreeze = opts.get("base_lr_factor") is not None
    Ti = copy.deepcopy(Ti0).to(device) if unfreeze else Ti0
    cond = build_conditioner(kind, Ti, dc, width, rank, z_std).to(device)
    if mods.get("specnorm"):
        for m in list(cond.U) + list(cond.V):
            Pz.register_parametrization(m, "weight", SpecCap(mods["specnorm"]["gamma"]))
    side = [q for q in cond.parameters() if q.requires_grad] + list(ctx.parameters()) + \
        (list(hist.parameters()) if hist is not None else [])
    groups = [{"params": side, "lr": lr}]
    n_base = 0
    if unfreeze:
        for q in Ti.parameters():
            q.requires_grad_(True)
        bp = list(Ti.parameters())
        n_base = sum(q.numel() for q in bp)
        groups.append({"params": bp, "lr": lr * float(opts["base_lr_factor"])})
    params = [q for g_ in groups for q in g_["params"]]
    opt = make_opt(opts.get("optimizer", p.get("optimizer", "adam")), groups)
    n_hist = n_params(hist) if hist is not None else 0
    log(f"    [jf-{kind}] mods {mods} width {width} cond {npar} ctx {n_params(ctx)} hist {n_hist} "
        f"base-trainable {n_base} opt {opts.get('optimizer', 'adam')} lr {lr:g} steps {steps} dc {dc}")
    sch = _sched(opt, steps, int(p.get("warmup", 100)))
    g = torch.Generator(device=device).manual_seed(seed)
    aux = {"gen": torch.Generator(device="cpu").manual_seed(seed + 4242), "z_std": z_std, "dt": des.dt,
           "t_ref": float(cfg["system"].get("t_ref", 1.0))}
    bt, bv = _Batcher(tr, device, g), _Batcher(va, device, torch.Generator(device=device).manual_seed(999))
    N1, burn = tr.x.shape[1], tr.burn
    Be, M = int(p["episodes_per_batch"]), int(p["times_per_episode"])
    ev = torch.arange(len(va), device=device)
    kv = bv.times(len(va), M, burn, N1)

    def val_loss():
        with torch.no_grad():
            return float(sum(loss_jf(cond, ctx, hist, va, ev[i:i + 16], kv[i:i + 16], sig, mods, False)
                             * len(ev[i:i + 16]) for i in range(0, len(ev), 16)) / len(ev))

    best, best_state, hist_log = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    every = int(p.get("val_every", 250))
    mlist = [cond, ctx] + ([hist] if hist is not None else [])
    for s in range(steps):
        e = bt.episodes(Be)
        loss = loss_jf(cond, ctx, hist, tr, e, bt.times(Be, M, burn, N1), sig, mods, True, aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if (s + 1) % every == 0 or s == steps - 1:
            v = val_loss()
            hist_log.append([s + 1, loss.detach().item(), v])
            log(f"    [jf-{kind}] step {s + 1:5d}  train {hist_log[-1][1]:.3e}  val {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:
                best = v
                best_state = [copy.deepcopy(m.state_dict()) for m in mlist]
    for m, sd in zip(mlist, best_state):
        m.load_state_dict(sd)
    if mods.get("specnorm"):
        with torch.no_grad():
            for m in list(cond.U) + list(cond.V):
                Pz.remove_parametrizations(m, "weight", leave_parametrized=True)
    for q in cond.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(False)
    return {"cond": cond, "ctx": ctx, "hist": hist, "n_params": npar, "width": width,
            "ctx_params": n_params(ctx, trainable_only=False), "hist_params": n_hist,
            "base_dec_params_trainable": n_base,
            "trainable_total": npar + n_params(ctx, trainable_only=False) + n_hist + n_base,
            "wall_s": time.time() - t0, "hist_log": hist_log, "val": best, "steps": steps, "opt": mon.windows}


@torch.no_grad()
def penalty_values(cfg, data, scales, des, r, device, seed, mods, n_batches=8):
    """Unweighted penalty terms + fit loss of a trained model r on n_batches fixed training batches
    (generator seed 1234 + seed): the operating point of the per-plant weight rule."""
    p = cfg["phase2"]
    tr = data["train"].to(device)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    nz = des.A.shape[0]
    z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
    g = torch.Generator(device=device).manual_seed(1234 + seed)
    aux = {"gen": torch.Generator(device="cpu").manual_seed(4321 + seed), "z_std": z_std, "dt": des.dt,
           "t_ref": float(cfg["system"].get("t_ref", 1.0))}
    bt = _Batcher(tr, device, g)
    N1, burn = tr.x.shape[1], tr.burn
    Be, M = int(p["episodes_per_batch"]), int(p["times_per_episode"])
    acc = {}
    torch.manual_seed(777 + seed)
    for _ in range(n_batches):
        t = {}
        loss_jf(r["cond"], r["ctx"], r["hist"], tr, bt.episodes(Be), bt.times(Be, M, burn, N1), sig, mods, True,
                aux, terms_out=t)
        for k, v in t.items():
            acc[k] = acc.get(k, 0.0) + v / n_batches
    return acc


def make_observer(name, des, base, r, device):
    return HistKKLObserver(name, LatentStep(des, device), r["cond"], r["ctx"], r["hist"], z0_map=base["T0"])
