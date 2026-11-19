r"""The three methods of the ICLR 2026 workshop paper (HyperKKL), on this package's data,
design, phase-1 base maps and evaluation. Method names: static, dynamic, curriculum.

Shared: the phase-1 maps T0 (encoder) and T0+ (decoder) are frozen; the latent filter is
the exact zhat' = A zhat + B y; the input enters through a context c_k = E(u_0..u_k) (an
LSTM by default, phase2.context.kind), with E(0) = 0 exactly, so every method below is the
autonomous observer at zero input. Physics terms use the modeller's model f (data.model_plant)
and the residual of a latent equation is weighted by W = -A^{-1} (residual -> steady latent
error, as in phase 1) and divided by the latent scale (scales.z_scale).

static   (Sec. 3.2.1, Eq. 9-10)  zhat' = A zhat + B y + Phi(zhat, c),  xhat = T0+(zhat)
         Phi(z, c) = M([z, c]) - M([z, 0]) (conditioning.LatentInjection), base maps frozen.
         phase2.static_loss:
           rollout    (default) sum_j ||(x_{k+j} - T0+(zhat_{k+j})) / sigma||^2 over a K-step
                      rollout of Eq. 9 from zhat_k = T0(x_k): the deployed observer itself
                      (train.phase2 "latent"). The residual losses below fit Phi at T0(x),
                      but it acts at zhat, which differs from T0(x) by T0's own PDE error;
                      on the ICLR pilot (seed 0, validation NRMSE) they were 1.7-125x worse.
           pde        R = J_T0(x) f(x,u) - A T0(x) - B h(x) - Phi(T0(x), c)       (Eq. 10)
           pde_input  R = J_T0(x) [f(x,u) - f(x,0)] - Phi(T0(x), c)
                      (the input-induced part only: Bernard & Andrieu's phi(T(x)) u for an
                      input-affine plant; T0's own autonomous residual is not fitted)
           fd         R = (T0(x_{k+1}) - T0(x_k)) / dt - A T0(x_k) - B h(x_k) - Phi(T0(x_k), c_k)
                      (model-free: the time derivative along the trajectory, no f)
         pde / pde_input use trajectory states and Sobol collocation states (Eq. 10 holds for
         every x); fd needs consecutive trajectory states. phase2.latent_gain caps Lip_z(Phi).

dynamic  (Sec. 3.2.2, Eq. 8, 11-14)  xhat = T+(zhat; phi_base + U_phi a_phi(c))
         theta(c) = theta_base + U_theta a_theta(c),  phi(c) = phi_base + U_phi a_phi(c)
         (Eq. 13): two bias-free heads a(c) in R^r (MLP 2 x 128, SiLU, last layer zero at
         init), learned bases U (every weight and bias of the map, r columns), r =
         phase2.dyn_rank or system.dyn_rank. a(0) = 0, so the perturbation vanishes at zero
         input. phase2.dyn_target chooses what the decoder inverts:
           filter  (default) L = ||(x - T+(zhat; phi)) / sigma||^2 + ||(T(x; theta) - z) / s_z||^2
                             + lambda ||W R / z_scale||^2,
                   zhat = the filter latent on the measured y of the forced training trajectory
                   (z_obs), the latent the deployed observer decodes; z = the same filter on the
                   noise-free output from z(0) = 0, i.e. the exact T(x(t), t) of Appendix C,
                   Remark 3 (phase 1's labels, now under forcing; dyn_label=z_obs uses zhat).
                   Decoder and label terms on trajectory states, R also on collocation.
           enc     L = ||(x - T+(T(x; theta); phi)) / sigma||^2 + lambda ||W R / z_scale||^2
                   (Eq. 8 as written: the decoder inverts the encoder, which equals the deployed
                   zhat only where R = 0; worse than the autonomous observer on all four ICLR
                   systems in the pilot, for every dT/dt below)
           R = dT/dx f(x, u) + dT/dt - A T(x; theta) - B h(x).  phase2.dtdt chooses dT/dt:
           fd      (T(x; c_k) - T(x; c_{k-1})) / dt at fixed x                      (Eq. 14)
           fd2     (3 T(x; c_k) - 4 T(x; c_{k-1}) + T(x; c_{k-2})) / (2 dt)   (two-step, BDF2)
           jvp     one JVP over (x, c) with tangent (f, (c_k - c_{k-1}) / dt): the
                   continuous chain rule dT/dc cdot, only cdot differenced
           none    dT/dt = 0 (the autonomous PDE with a conditioned map)
         For weights interpolated linearly between samples (Appendix C), fd is the exact
         mean of dT/dt over the sample interval, jvp its first-order linearisation.

curriculum (Sec. 2.3, 3.3.2)  the training-only baseline: the autonomous observer's
         architecture (no input pathway), T0+ fine-tuned on forced episodes, T0 frozen.
         Target: the latent the deployed filter carries (z_obs), not an autoencoder target.
         Difficulty of an episode = spectral centroid of its input (t_ref units; 0 for a
         constant input). Level 1 = zero and constant inputs, levels 2..K = quantile bins of
         the centroid of the rest. Stages are cumulative (level k trains on levels 1..k, so the
         autonomous solution is replayed, not overwritten). The stage advances when the
         validation loss of the stage improves by less than cur_eps (relative) over
         cur_patience checks (every cur_every steps, after cur_min steps), or after
         cur_max * steps; the final stage (all data) runs to the step budget and selects the
         model on the full validation set.
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

from . import data as Dm
from .conditioning import build_conditioner, match_width
from .design import spectral_bounds
from .monitor import TrainMonitor
from .nets import MLP, BaseMap, build_context, n_params
from .train import _sched, _small_gain          # iclr is imported lazily by train.phase2

KINDS = ("static", "dynamic", "curriculum")
STATIC_LOSSES = ("rollout", "pde", "pde_input", "fd")
DTDT = ("fd", "fd2", "jvp", "none")
DYN_TARGETS = ("filter", "enc")


# ---------------------------------------------------------------------------
# Eq. 13: low-rank perturbation of a frozen base map
# ---------------------------------------------------------------------------

class LowRankMap(nn.Module):
    """T(x; theta_base + U a) for a per-sample coefficient a (B, r). U holds, for every
    layer, a weight basis (r, out, in) and a bias basis (r, out): its r columns span the
    perturbation subspace of Eq. 13. The perturbed layer is evaluated as
    W h + b + [(a kron h) @ U_W] + a @ U_b, one GEMM, without forming per-sample weights."""

    def __init__(self, base: BaseMap, r: int, init: float = 0.1):
        super().__init__()
        self.base, self.r = base, int(r)
        for q in base.parameters():
            q.requires_grad_(False)
        s = init / math.sqrt(self.r)
        self.Uw = nn.ParameterList(nn.Parameter(torch.randn(self.r, l.out_features, l.in_features)
                                                * float(l.weight.std()) * s) for l in base.layers)
        self.Ub = nn.ParameterList(nn.Parameter(torch.randn(self.r, l.out_features)
                                                * float(l.bias.std().clamp(min=1e-3)) * s) for l in base.layers)

    def forward(self, x, a):
        b, n, B = self.base, len(self.base.layers), x.shape[0]
        h = b.inp.norm(x)
        for j, layer in enumerate(b.layers):
            i, o = layer.in_features, layer.out_features
            Uw = self.Uw[j].permute(0, 2, 1).reshape(self.r * i, o)
            h = layer(h) + (a[:, :, None] * h[:, None, :]).reshape(B, self.r * i) @ Uw + a @ self.Ub[j]
            if j < n - 1:
                h = b.act(h)
        return b.out.denorm(h)


class DynamicHyperKKL(nn.Module):
    """The hypernetwork of Sec. 3.2.2 minus its input encoder (the shared context, built by
    phase2): two heads and the two low-rank maps. forward(z, c) is the deployed decoder."""

    def __init__(self, T0: BaseMap, Ti: BaseMap, dc: int, r: int, width: int = 128):
        super().__init__()
        self.enc, self.dec = LowRankMap(T0, r), LowRankMap(Ti, r)
        self.head_enc = MLP(dc, r, width, 2, "silu", bias=False, zero_last=True)
        self.head_dec = MLP(dc, r, width, 2, "silu", bias=False, zero_last=True)

    def encode(self, x, c):
        return self.enc(x, self.head_enc(c))

    def forward(self, z, c):
        return self.dec(z, self.head_dec(c))


class _Plain(nn.Module):
    """A decoder with the autonomous interface dec(z, c) and no input pathway."""

    def __init__(self, base: BaseMap):
        super().__init__()
        self.base = base

    def forward(self, z, c):
        return self.base(z)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _Sampler:
    """Training samples: (episode, time) pairs with the context history needed by the time
    derivatives, plus collocation states paired with the first contexts of the batch."""

    def __init__(self, es, coll_x, device, seed, lag_lo, lag_hi):
        self.es, self.cx, self.dev = es, coll_x, device
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.lo, self.hi = max(es.burn, lag_lo), es.x.shape[1] - lag_hi

    def draw(self, n_ep, m, n_coll, pool=None):
        E = len(self.es) if pool is None else len(pool)
        e = torch.randint(0, E, (n_ep,), device=self.dev, generator=self.g)
        if pool is not None:
            e = pool[e]
        k = torch.randint(self.lo, self.hi, (n_ep, m), device=self.dev, generator=self.g)
        j = torch.randint(0, len(self.cx), (n_coll,), device=self.dev, generator=self.g) if n_coll else None
        return e, k, j


def _gather(es, ctx, e, k, lags):
    """x_k, u_k, and c_{k-l} for every lag l in ``lags`` (negative l = future samples)."""
    r = torch.arange(len(e), device=e.device)[:, None]
    c = ctx.seq(es.u[e]) if ctx is not None else None
    flat = lambda t: t.reshape(-1, t.shape[-1])
    out = {"x": flat(es.x[e[:, None], k]), "u": flat(es.u[e[:, None], k]),
           "z": flat(es.z_obs[e[:, None], k]), "z_lab": flat(es.z[e[:, None], k]),
           "x_next": flat(es.x[e[:, None], (k + 1).clamp(max=es.x.shape[1] - 1)])}
    if c is not None:
        for l in lags:
            out[f"c{l}"] = flat(c[r, k - l])
    return out


def _with_coll(b, xc):
    """Append collocation states, each paired with the input sample and contexts of a
    trajectory sample (Eq. 7 and Eq. 10 hold for every x under a given input history)."""
    if xc is None:
        return b
    n = min(len(xc), len(b["x"]))
    out = {"x": torch.cat([b["x"], xc[:n]]), "u": torch.cat([b["u"], b["u"][:n]])}
    for key, v in b.items():
        if key.startswith("c"):
            out[key] = torch.cat([v, v[:n]])
    return out


def phase2(kind, cfg, data, scales, des, base, device, seed, log=print) -> dict:
    assert kind in KINDS, kind
    torch.manual_seed(seed)
    t0 = time.time()
    p = cfg["phase2"]
    tr, va = data["train"].to(device), data["val"].to(device)
    T0, Ti = base["T0"], base["T0inv"]
    plant = Dm.model_plant(cfg["system"])
    A = torch.as_tensor(des.A, dtype=torch.float32, device=device)
    B = torch.as_tensor(des.B, dtype=torch.float32, device=device)
    W = torch.as_tensor(-np.linalg.inv(des.A), dtype=torch.float32, device=device)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    zsc = float(scales["z_scale"])
    dt = float(des.dt)
    coll_x = data["coll"]["x"].to(device)
    lam = float(p.get("lambda_pde", 1.0))
    steps, every = int(p["steps"]), int(p.get("val_every", 250))
    f = lambda x, u: plant.f(x, u)
    h = lambda x: plant.h(x)
    pde = lambda R: ((R @ W.T) / zsc).pow(2).sum(-1).mean()
    extra = {}

    if kind == "curriculum":
        return _curriculum(cfg, tr, va, Ti, sig, des, device, seed, log, t0)
    if kind == "static" and p.get("static_loss", "rollout") == "rollout":
        from .train import phase2 as rollout_phase2
        return {**rollout_phase2("latent", cfg, data, scales, des, base, device, seed, log),
                "static_loss": "rollout"}

    ccfg = p["context"]
    u_ref = float(cfg["system"].get("u_scale", 1.0))
    ctx = build_context(ccfg["kind"], ccfg, 1, des.dt, np.linalg.eigvals(des.A), u_ref).to(device)
    dc = ctx.dim
    nz = des.A.shape[0]

    if kind == "static":
        loss_kind = p.get("static_loss", "pde")
        assert loss_kind in STATIC_LOSSES, loss_kind
        z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
        kappa, alpha = spectral_bounds(des.A)
        lip = float(p["latent_gain"]) * alpha / kappa if p.get("latent_gain") is not None else None
        width, npar = match_width("latent", Ti, dc, int(p["budget"]), int(p["rank"]), z_std)
        net = build_conditioner("latent", Ti, dc, width, int(p["rank"]), z_std, latent_lip=lip).to(device)
        if lip is not None:
            net.project_()
        n_coll = 0 if loss_kind == "fd" else int(p.get("n_coll_batch", 2048))
        lags = (0,)

        def loss_fn(es, e, k, j):
            b = _gather(es, ctx, e, k, lags)
            if loss_kind == "fd":
                with torch.no_grad():
                    z = T0(b["x"])
                    tgt = (T0(b["x_next"]) - z) / dt - z @ A.T - h(b["x"]) @ B.T
            else:
                b = _with_coll(b, coll_x[j] if j is not None else None)
                with torch.no_grad():
                    fx = f(b["x"], b["u"])
                    if loss_kind == "pde_input":
                        z, tgt = jvp(T0, (b["x"],), (fx - f(b["x"], torch.zeros_like(b["u"])),))
                    else:
                        z, Jf = jvp(T0, (b["x"],), (fx,))
                        tgt = Jf - z @ A.T - h(b["x"]) @ B.T
            return pde(tgt - net(z, b["c0"]))

        log(f"    [static/{ccfg['kind']}] loss {loss_kind}  width {width}  injection params {npar}  "
            f"context params {n_params(ctx)}" + (f"  lip cap {lip:.4g}" if lip is not None else ""))
        lag_lo, lag_hi = 0, 1
    else:  # dynamic
        mode = p.get("dtdt", "fd")
        assert mode in DTDT, mode
        r = int(p.get("dyn_rank") or cfg["system"].get("dyn_rank", 32))
        width = int(p.get("dyn_width", 128))
        unfreeze = bool(p.get("unfreeze_base", False))
        # unfreeze_base (reviewer NpQ8): theta_base, phi_base train jointly with the
        # hypernetwork, on private copies (the other methods keep the frozen phase-1 maps)
        net = DynamicHyperKKL(copy.deepcopy(T0) if unfreeze else T0, copy.deepcopy(Ti) if unfreeze else Ti,
                              dc, r, width).to(device)
        npar = n_params(net)
        if unfreeze:
            for q in list(net.enc.base.parameters()) + list(net.dec.base.parameters()):
                q.requires_grad_(True)
            extra.update({"unfreeze_base": True, "n_params_trained": n_params(net)})
        target = p.get("dyn_target", "filter")
        assert target in DYN_TARGETS, target
        label = p.get("dyn_label", "z")                 # encoder label: z (exact T(x(t), t), Remark 3) or z_obs
        assert label in ("z", "z_obs"), label
        n_coll = int(p.get("n_coll_batch", 1024))
        lags = {"fd": (0, 1), "fd2": (0, 1, 2), "jvp": (0, 1), "none": (0,)}[mode]
        zs = T0.out.std                                 # phase 1's label scale

        def loss_fn(es, e, k, j):
            bt = _gather(es, ctx, e, k, lags)
            n = len(bt["x"])                            # trajectory rows come first
            b = _with_coll(bt, coll_x[j] if j is not None else None)
            x, c0 = b["x"], b["c0"]
            fx = f(x, b["u"])
            if mode == "jvp":
                z, dT = jvp(net.encode, (x, c0), (fx, (c0 - b["c1"]) / dt))
            else:
                z, dT = jvp(lambda xx: net.encode(xx, c0), (x,), (fx,))
                if mode == "fd":
                    dT = dT + (z - net.encode(x, b["c1"])) / dt
                elif mode == "fd2":
                    dT = dT + (3 * z - 4 * net.encode(x, b["c1"]) + net.encode(x, b["c2"])) / (2 * dt)
            l_pde = pde(dT - z @ A.T - h(x) @ B.T)
            if target == "enc":
                return ((net(z, c0) - x) / sig).pow(2).mean() + lam * l_pde
            zl = bt["z"]
            l_rec = ((net(zl, c0[:n]) - x[:n]) / sig).pow(2).mean()
            l_lab = ((z[:n] - (bt["z_lab"] if label == "z" else zl)) / zs).pow(2).mean()
            return l_rec + l_lab + lam * l_pde

        log(f"    [dynamic/{ccfg['kind']}] target {target}  dT/dt {mode}  rank {r}  hypernet params {npar}  "
            f"context params {n_params(ctx)}  lambda {lam:g}")
        lag_lo, lag_hi = max(lags), 1
        extra.update({"dtdt": mode, "dyn_rank": r, "dyn_target": target, "dyn_label": label})

    params = [q for q in list(net.parameters()) + list(ctx.parameters()) if q.requires_grad]
    opt = torch.optim.Adam(params, lr=float(p["lr"]))
    sch = _sched(opt, steps, int(p.get("warmup", 100)))
    Be, M = int(p["episodes_per_batch"]), int(p.get("iclr_times_per_episode", 96))
    st, sv = _Sampler(tr, coll_x, device, seed, lag_lo, lag_hi), _Sampler(va, coll_x, device, 999, lag_lo, lag_hi)
    ev, kv, jv = sv.draw(len(va), M, n_coll)            # fixed validation draw

    def val_loss():
        with torch.no_grad():                           # forward-mode jvp needs no autograd
            tot = 0.0
            for i in range(0, len(ev), 16):
                jj = jv[: n_coll * len(ev[i:i + 16]) // len(ev)] if jv is not None else None
                tot += float(loss_fn(va, ev[i:i + 16], kv[i:i + 16], jj).detach()) * len(ev[i:i + 16])
            return tot / len(ev)

    best, best_state, hist = float("inf"), None, []
    mon = TrainMonitor(params, opt)
    for s in range(steps):
        e, k, j = st.draw(Be, M, n_coll)
        loss = loss_fn(tr, e, k, j)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        if getattr(net, "capped", False):
            net.project_()
        if (s + 1) % every == 0 or s == steps - 1:
            v = val_loss()
            hist.append([s + 1, loss.detach().item(), v])
            log(f"    [{kind}] step {s + 1:5d}  train {hist[-1][1]:.3e}  val {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if best_state is None or v < best:
                best = v
                best_state = (copy.deepcopy(net.state_dict()), copy.deepcopy(ctx.state_dict()))
    net.load_state_dict(best_state[0]); ctx.load_state_dict(best_state[1])
    for q in params:
        q.requires_grad_(False)
    if kind == "static":
        extra["small_gain"] = _small_gain(net, ctx, tr, kappa, alpha, lip)
        extra["static_loss"] = loss_kind
        log(f"    [static] small gain: {extra['small_gain']}")
    return {"cond": net, "ctx": ctx, "n_params": npar, "width": width,
            "ctx_params": n_params(ctx, trainable_only=False), "wall_s": time.time() - t0,
            "hist": hist, "val": best, "opt": mon.windows, "lr": float(p["lr"]), **extra}


# ---------------------------------------------------------------------------
# Curriculum baseline
# ---------------------------------------------------------------------------

def difficulty(es, t_ref: float) -> np.ndarray:
    """Spectral centroid of each episode's input after washout, in cycles per t_ref; 0 for
    an input that is constant over the episode (zero and constant families)."""
    u = es.u[:, es.burn:, 0].double().cpu().numpy()
    u = u - u.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(u, axis=1)) ** 2
    fr = np.fft.rfftfreq(u.shape[1], es.dt) * t_ref
    tot = P.sum(1)
    out = np.where(tot > 1e-12 * u.shape[1], (P * fr).sum(1) / np.maximum(tot, 1e-300), 0.0)
    out[u.std(1) < 1e-6] = 0.0
    return out


def _levels(d: np.ndarray, edges, K: int) -> np.ndarray:
    """Level 0 = constant input, 1..K-1 = centroid bins; K = 1 puts everything in one level
    (no curriculum: the same fine-tuning on all forced data, the control of the ordering)."""
    lv = np.zeros(len(d), dtype=int)
    nz = d > 0
    lv[nz] = 1 + np.searchsorted(edges, d[nz], side="right")
    return np.minimum(lv, K - 1)


def _curriculum(cfg, tr, va, Ti, sig, des, device, seed, log, t0) -> dict:
    p = cfg["phase2"]
    K = int(p.get("cur_levels", 4))                     # level 1 = constant inputs, 2..K = centroid bins
    t_ref = float(cfg["system"].get("t_ref", 1.0))
    dtr, dva = difficulty(tr, t_ref), difficulty(va, t_ref)
    edges = np.quantile(dtr[dtr > 0], np.linspace(0, 1, K)[1:-1]) if (dtr > 0).any() else np.array([])
    ltr, lva = _levels(dtr, edges, K), _levels(dva, edges, K)
    pools = [torch.as_tensor(np.flatnonzero(ltr <= k), device=device) for k in range(K)]
    vpools = [torch.as_tensor(np.flatnonzero(lva <= k), device=device) for k in range(K)]
    log(f"    [curriculum] levels (train episodes): {[int((ltr == k).sum()) for k in range(K)]}  "
        f"centroid edges {np.round(edges, 3).tolist()} cycles/t_ref")

    dec = _Plain(copy.deepcopy(Ti)).to(device)
    for q in dec.parameters():
        q.requires_grad_(True)
    params = list(dec.parameters())
    steps, every = int(p["steps"]), int(p.get("val_every", 250))
    opt = torch.optim.Adam(params, lr=float(p.get("cur_lr", p["lr"])))
    sch = _sched(opt, steps, int(p.get("warmup", 100)))
    Be, M = int(p["episodes_per_batch"]), int(p["times_per_episode"])
    g = torch.Generator(device=device).manual_seed(seed)
    gv = torch.Generator(device=device).manual_seed(999)
    N1, burn = tr.x.shape[1], tr.burn
    kv = torch.randint(burn, N1, (len(va), M), device=device, generator=gv)
    cur_every, eps = int(p.get("cur_every", 100)), float(p.get("cur_eps", 0.01))
    patience, min_s, max_s = int(p.get("cur_patience", 3)), int(p.get("cur_min", 200)), float(p.get("cur_max", 0.25))

    def loss_on(es, e, k):
        z, x = es.z_obs[e[:, None], k], es.x[e[:, None], k]
        return ((dec.base(z.reshape(-1, z.shape[-1])) - x.reshape(-1, x.shape[-1])) / sig).pow(2).mean()

    def val(pool):
        with torch.no_grad():
            return float(sum(loss_on(va, pool[i:i + 16], kv[pool[i:i + 16]]) * len(pool[i:i + 16])
                             for i in range(0, len(pool), 16)) / max(1, len(pool)))

    stage, s_stage, ref, stall = 0, 0, float("inf"), 0
    while len(pools[stage]) == 0 or len(vpools[stage]) == 0:      # a level with no episodes
        stage += 1
    stages, best, best_state, hist = [[stage, 0]], float("inf"), None, []
    mon = TrainMonitor(params, opt)
    for s in range(steps):
        pool = pools[stage]
        e = pool[torch.randint(0, len(pool), (Be,), device=device, generator=g)]
        k = torch.randint(burn, N1, (Be, M), device=device, generator=g)
        loss = loss_on(tr, e, k)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mon.clip()
        opt.step(); sch.step()
        s_stage += 1
        if stage < K - 1 and (s + 1) % cur_every == 0:
            v = val(vpools[stage])
            stall = stall + 1 if (ref - v) < eps * abs(ref) else 0
            ref = min(ref, v)
            if (s_stage >= min_s and stall >= patience) or s_stage >= max_s * steps:
                log(f"    [curriculum] step {s + 1}: level {stage + 1} -> {stage + 2} "
                    f"(val {v:.3e}, {'plateau' if stall >= patience else 'max steps'})")
                stage, s_stage, ref, stall = stage + 1, 0, float("inf"), 0
                stages.append([stage, s + 1])
        if (s + 1) % every == 0 or s == steps - 1:
            v = val(vpools[-1])
            hist.append([s + 1, loss.detach().item(), v, stage + 1])
            log(f"    [curriculum] step {s + 1:5d}  level {stage + 1}/{K}  train {hist[-1][1]:.3e}  "
                f"val(all) {v:.3e}  " + mon.fmt(mon.flush(s + 1)))
            if stage == K - 1 and (best_state is None or v < best):
                best, best_state = v, copy.deepcopy(dec.state_dict())
    if best_state is not None:
        dec.load_state_dict(best_state)
    else:
        best = val(vpools[-1])
    for q in params:
        q.requires_grad_(False)
    return {"cond": dec, "ctx": None, "n_params": n_params(dec, trainable_only=False), "width": 0,
            "ctx_params": 0, "wall_s": time.time() - t0, "hist": hist, "val": best, "opt": mon.windows,
            "lr": float(p.get("cur_lr", p["lr"])),
            "curriculum": {"stages": stages, "edges": edges.tolist(),
                           "levels_train": [int((ltr == k).sum()) for k in range(K)]}}
