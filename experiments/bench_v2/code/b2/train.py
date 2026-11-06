"""Training of the baselines (PREREG.md section 2). Selection data: data['val'] only."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from hyperkkl import baselines_data as BD
from hyperkkl.metrics import episode_metrics
from hyperkkl.observers import run
from hyperkkl.train import _Graphed, _sched

from .models import rk4_train

WARMUP = 100
CLIP = 1.0


def _clip(params):
    return torch.nn.utils.clip_grad_norm_(params, CLIP)


# ------------------------------------------------------------------------------- sequence observers

@torch.no_grad()
def seq_val_loss(net, va, burn, sig, chunk=32, state_w=None):
    """Mean over validation episodes and steps >= burn of ||(x̂ - x) / x_std||^2, from h_0 = 0."""
    tot, n = 0.0, 0
    for i in range(0, len(va), chunk):
        u, y, x = va.u[i:i + chunk], va.y[i:i + chunk], va.x[i:i + chunk]
        xh = net.seq(u, y)
        e = ((xh[:, burn:] - x[:, burn:]) / sig).pow(2)
        e = e.mean(-1) if state_w is None else (e * state_w).sum(-1) / state_w.sum()
        e = torch.nan_to_num(e, nan=1e6, posinf=1e6).clamp(max=1e6)
        tot += float(e.mean()) * len(u)
        n += len(u)
    return tot / n


def train_seq(net, tr, va, steps, lr, device, seed, log=print, Bs=32, K=256, val_every=250,
              graphs=True, state_w=None) -> dict:
    """Stateful truncated BPTT as train.nlox: Bs streams carry their hidden state through whole
    training episodes, K samples per optimiser step (detached between steps), restarted on a fresh
    episode at the end; loss on samples >= burn."""
    torch.manual_seed(seed)
    t0 = time.time()
    net = net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=float(lr))
    sch = _sched(opt, steps, WARMUP)
    sig = net.norm.xs
    N1 = tr.x.shape[1]
    burn = int(tr.burn)
    g = torch.Generator(device=device).manual_seed(int(seed))
    ep = torch.randint(0, len(tr), (Bs,), device=device, generator=g)
    kk = torch.zeros(Bs, dtype=torch.long, device=device)
    ar = torch.arange(K + 1, device=device)
    is_gru = net.kind == "gru"
    if is_gru:
        hst = net.init_state(Bs, device)
    else:
        hst = torch.zeros(Bs, net.dz, device=device)
    out = {}

    def loss_fn():
        if is_gru:
            idx = kk[:, None] + ar[None, :K]                         # samples kk .. kk+K-1
            u, y, x = tr.u[ep[:, None], idx], tr.y[ep[:, None], idx], tr.x[ep[:, None], idx]
            o, h = net.seq_norm(u, y, hst)
            out["h"] = h
        else:
            idx = kk[:, None] + ar[None, :]                          # samples kk .. kk+K
            u, y = tr.u[ep[:, None], idx], tr.y[ep[:, None], idx]
            X = net.X(u, y, kk)
            z0 = torch.where((kk == 0)[:, None], net.zeta(X[:, 0]), hst)
            o, z = net.seq_norm_from(z0, X)
            out["h"] = z
            idx = idx[:, 1:]
            x = tr.x[ep[:, None], idx]
        err = (net.norm.out_norm(x) - o).pow(2)
        err = err.mean(-1) if state_w is None else (err * state_w).sum(-1) / state_w.sum()
        w = (idx >= burn).float()
        return (err * w).sum() / w.sum().clamp(min=1.0)

    use_graph = graphs and not is_gru
    graphed = _Graphed(loss_fn, params, enabled=use_graph)
    best, best_state, hist = float("inf"), None, []
    n_bad = 0
    for s in range(steps):
        span = K if is_gru else K + 1
        done = kk + span > N1
        if done.any():
            nd = int(done.sum())
            ep[done] = torch.randint(0, len(tr), (nd,), device=device, generator=g)
            kk[done] = 0
            if is_gru:
                hst[:, done] = 0.0
            else:
                hst[done] = 0.0
        if use_graph:
            loss = graphed()
        else:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn()
            loss.backward()
        gn = _clip(params)
        if torch.isfinite(gn):
            opt.step()
        else:
            n_bad += 1
        sch.step()
        with torch.no_grad():
            hst.copy_(torch.nan_to_num(out["h"].detach(), nan=0.0))
        kk.add_(K)
        if (s + 1) % val_every == 0 or s == steps - 1:
            v = seq_val_loss(net, va, burn, sig, state_w=state_w)
            hist.append([s + 1, float(loss.detach()), v])
            log(f"    [{net.kind}] step {s + 1:5d} train {hist[-1][1]:.3e} val {v:.3e}")
            if best_state is None or v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    return {"val_loss": best, "hist": hist, "wall_s": time.time() - t0, "n_bad_steps": n_bad,
            "n_params": sum(p.numel() for p in net.parameters())}


# --------------------------------------------------------------------------------- dynamics models

def _windows_source(sets):
    xs, us = [], []
    for s in sets:
        ok = torch.isfinite(s.x).all(-1).all(-1) & torch.isfinite(s.u).all(-1).all(-1)
        xs.append(s.x[ok]); us.append(s.u[ok])
    return torch.cat(xs), torch.cat(us)


def _rollout_loss(fm, net, X, U, e, s0, K, dt, sig):
    x = X[e, s0]
    tot = 0.0
    for j in range(K):
        x = rk4_train(fm, net, x, U[e, s0 + j], dt)
        tot = tot + ((x - X[e, s0 + j + 1]) / sig).pow(2).mean()
    return tot / K


def train_dyn(fm, net, train_sets, va, dt, steps, lr, device, seed, log=print, K=32, K_warm=1,
              warm_steps=300, batch=1024, val_every=100, n_val=4096, use_graphs=True) -> dict:
    """Multi-step (K-sample RK4/ZOH rollout) training of r_theta in f = [f_known] + r_theta,
    starting at the TRUE state of each window; the first warm_steps use K_warm = 1."""
    torch.manual_seed(seed)
    t0 = time.time()
    net = net.to(device)
    X, U = _windows_source([s.to(device) for s in train_sets])
    Xv, Uv = _windows_source([va.to(device)])
    N1 = X.shape[1]
    sig = torch.as_tensor(np.asarray(net.xs.cpu()), device=device)
    g = torch.Generator(device=device).manual_seed(int(seed))
    gv = torch.Generator(device=device).manual_seed(int(seed) + 7)
    ev = torch.randint(0, len(Xv), (n_val,), device=device, generator=gv)
    sv = torch.randint(0, N1 - K, (n_val,), device=device, generator=gv)
    params = list(net.parameters())
    opt = torch.optim.Adam(params, lr=float(lr))
    sch = _sched(opt, steps, WARMUP)

    @torch.no_grad()
    def val():
        tot = 0.0
        for i in range(0, n_val, 1024):
            l = _rollout_loss(fm, net, Xv, Uv, ev[i:i + 1024], sv[i:i + 1024], K, dt, sig)
            tot += float(torch.nan_to_num(l, nan=1e6, posinf=1e6).clamp(max=1e6)) * len(ev[i:i + 1024])
        return tot / n_val

    best, best_state, hist, n_bad = float("inf"), None, [], 0
    # static index buffers (refreshed in place) so each K's loss+backward replays as one CUDA graph
    eb = torch.zeros(batch, dtype=torch.long, device=device)
    sb = torch.zeros(batch, dtype=torch.long, device=device)
    graphs = {}

    def get_graph(k):
        if k not in graphs:
            gr = _Graphed(lambda: _rollout_loss(fm, net, X, U, eb, sb, k, dt, sig), params,
                          enabled=use_graphs)
            graphs[k] = gr
        return graphs[k]

    for s in range(steps):
        k = K_warm if s < warm_steps else K
        eb.copy_(torch.randint(0, len(X), (batch,), device=device, generator=g))
        sb.copy_(torch.randint(0, N1 - k, (batch,), device=device, generator=g))
        gr = get_graph(k)
        try:
            loss = gr()
        except Exception as ex:                       # capture not possible: run eagerly
            if not gr.enabled:
                raise
            log(f"    [dyn] CUDA graph capture failed ({type(ex).__name__}: {ex}); eager")
            gr.enabled, gr.graph = False, None
            loss = gr()
        if torch.isfinite(loss):
            gn = _clip(params)
            if torch.isfinite(gn):
                opt.step()
            else:
                n_bad += 1
        else:
            n_bad += 1
        sch.step()
        if (s + 1) % val_every == 0 or s == steps - 1:
            if s + 1 < warm_steps:
                continue
            v = val()
            hist.append([s + 1, float(loss.detach()), v])
            if (s + 1) % (5 * val_every) == 0 or s == steps - 1:
                log(f"    [dyn] step {s + 1:5d} K {k} train {hist[-1][1]:.3e} val(K={K}) {v:.3e}")
            if best_state is None or v < best:
                best, best_state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net = net.cpu().eval()
    return {"val_loss": best, "hist": hist, "wall_s": time.time() - t0, "n_bad_steps": n_bad,
            "n_params": sum(p.numel() for p in net.parameters())}


def target_stats(fm, sets, dt, device, seed, n=200_000):
    """Mean / std of the one-step derivative targets the MLP's output is standardised with:
    (x1 - Phi_known(x0, u0)) / dt for the UDE (as the grey-box residual), (x1 - x0) / dt for NODE."""
    X, U, X1 = BD.transitions(sets, n, seed + 1, device)
    T = BD.residual_targets(fm, X, U, X1, dt) if fm is not None else (X1 - X) / float(dt)
    T = T[torch.isfinite(T).all(1)]
    return T.mean(0).float().cpu().numpy(), T.std(0).clamp(min=1e-6).float().cpu().numpy()


# ------------------------------------------------------------------------------- deployed selection

@torch.no_grad()
def deployed_val(ob, cfg, va, scales, device, x0, chunk=1024) -> dict:
    """Settled NRMSE (+ divergence) of the deployed observer on the validation episodes (as
    job.tune_oracle / job.validation_nrmse: centre-of-box prior, settle = 0.5 N)."""
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (va.x.shape[1] - 1))
    xh = torch.cat([run(ob, va.y[i:i + chunk], va.u[i:i + chunk], x0[i:i + chunk])
                    for i in range(0, len(va), chunk)])
    m = episode_metrics(va.x, xh, scales["x_std"], settle)
    return {"val_nrmse": float(np.mean(m["nrmse"])), "val_diverged": float(np.mean(m["diverged"]))}
