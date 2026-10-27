"""Exp 3 (information probe) + Exp 4 (decoder Jacobian) on one finished run (cached models).
Probes (fit on TRAINING episodes, k >= burn; held-out = evaluation episodes id / sequence_ood,
k >= settle): wide MLP and kNN from [z_obs, c] -> x, from z_obs windows, and from the noise-free
latent z_clean (run_latent on y_clean). c = the trained lora context (ctx.seq(u)).
usage: exp34.py NODE SYSTEM SEED DEVICE OUTDIR"""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import common as K
from hyperkkl import data as Dm

node, system, seed, device, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
torch.set_num_threads(2); torch.manual_seed(0)
t0 = time.time()
payload, cfg, scales, des, data, base, trained = K.load_run(node, system, seed, device, ["latent", "lora", "hyper"])
dev = torch.device(device)
sig = torch.tensor(scales["x_std"], device=dev); xm = torch.tensor(scales["x_mean"], device=dev)
settle_frac = float(cfg["eval"].get("settle_frac", 0.5))


def prep(es):
    es = es.to(dev)
    zc = Dm.run_latent(des, es.y_clean.double()).float()
    return es, zc


tr, ztr_c = prep(data["train"]); va, zva_c = prep(data["val"])
ev = {t: prep(Dm.eval_data(cfg, t, device=device)) for t in ("id", "sequence_ood")}
nz = tr.z_obs.shape[-1]
z_std = tr.z_obs[:, tr.burn:].reshape(-1, nz).std(0).clamp(min=1e-6)
ctx = trained["lora"]["ctx"]


@torch.no_grad()
def cseq(ctx_, u):
    return torch.cat([ctx_.seq(u[i:i + 64]) for i in range(0, len(u), 64)])


def window(z, k, stride=1):
    """[z_t, z_{t-s}, ..., z_{t-(k-1)s}] (edge-padded at t < (k-1)s)."""
    out = []
    for j in range(k):
        sh = j * stride
        out.append(torch.cat([z[:, :1].expand(-1, sh, -1), z[:, :z.shape[1] - sh]], 1) if sh else z)
    return torch.cat(out, -1)


def idx(es, lo):
    E, N1 = es.x.shape[:2]
    return lo, N1


C = {"train": cseq(ctx, tr.u), "val": cseq(ctx, va.u), **{t: cseq(ctx, e.u) for t, (e, _) in ev.items()}}
sets = {"train": (tr, ztr_c, tr.burn), "val": (va, zva_c, tr.burn),
        **{t: (e, zc, int(settle_frac * (e.x.shape[1] - 1))) for t, (e, zc) in ev.items()}}
FEATS = {"z_c": lambda z, c: torch.cat([z / z_std, c], -1),
         "zwin5_c": lambda z, c: torch.cat([window(z / z_std, 5), c], -1),
         "zwin20_c": lambda z, c: torch.cat([window(z / z_std, 20), c], -1),
         "zwin20s10_c": lambda z, c: torch.cat([window(z / z_std, 20, 10), c], -1),
         "zwin20s30_c": lambda z, c: torch.cat([window(z / z_std, 20, 30), c], -1)}


def flat(name, which, zsrc="obs", nmax=None, g=None):
    es, zc, lo = sets[which]
    z = es.z_obs if zsrc == "obs" else zc
    F = FEATS[name](z, C[which])[:, lo:]
    X = ((es.x - xm) / sig)[:, lo:]
    F, X = F.reshape(-1, F.shape[-1]), X.reshape(-1, X.shape[-1])
    if nmax and len(F) > nmax:
        i = torch.randperm(len(F), generator=g, device="cpu")[:nmax].to(dev)
        F, X = F[i], X[i]
    return F, X


def nrmse(P, X):
    return float((P - X).pow(2).mean().sqrt())


def fit_mlp(F, X, Fv, Xv, steps=6000, width=512, depth=4, bs=4096):
    torch.manual_seed(seed)
    mu, sd = F.mean(0), F.std(0).clamp(min=1e-6)
    layers, d = [], F.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]; d = width
    net = nn.Sequential(*layers, nn.Linear(d, X.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.05)
    best, state = 1e9, None
    for s in range(steps):
        i = torch.randint(0, len(F), (bs,), device=dev)
        loss = (net((F[i] - mu) / sd) - X[i]).pow(2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if (s + 1) % 500 == 0:
            with torch.no_grad():
                v = nrmse(net((Fv - mu) / sd), Xv)
            if v < best:
                best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(state)
    return lambda Q: torch.cat([net((Q[i:i + 65536] - mu) / sd) for i in range(0, len(Q), 65536)]), best


@torch.no_grad()
def knn(F, X, Q, k=10, chunk=2048):
    mu, sd = F.mean(0), F.std(0).clamp(min=1e-6)
    Fn = (F - mu) / sd
    out = []
    for i in range(0, len(Q), chunk):
        d = torch.cdist((Q[i:i + chunk] - mu) / sd, Fn)
        j = d.topk(k, largest=False).indices
        out.append(X[j].mean(1))
    return torch.cat(out)


g = torch.Generator().manual_seed(seed)
NTR, NQ = 600_000, 200_000
res = {"node": node, "system": system, "seed": seed, "decoder": {}, "probe": {}, "knn": {}, "jac": {}}

# ---- trained decoders on the same pooled samples (train / val / eval tiers) -------------
dec = {m: (trained[m]["cond"], trained[m]["ctx"]) for m in ("lora", "hyper") if m in trained}
dec["autonomous_base"] = (lambda z, c: base["T0inv"](z), None)
with torch.no_grad():
    for which, (es, zc, lo) in sets.items():
        for m, (f, cx) in dec.items():
            cc = cseq(cx, es.u) if cx is not None else None
            for zsrc, z in (("obs", es.z_obs), ("clean", zc)):
                P = []
                for i in range(0, len(es), 16):
                    zz = z[i:i + 16, lo:]; sh = zz.shape
                    c_ = cc[i:i + 16, lo:].reshape(-1, cc.shape[-1]) if cc is not None else None
                    P.append(((f(zz.reshape(-1, nz), c_) - xm) / sig).reshape(*sh[:2], -1))
                P = torch.cat(P); X = ((es.x[:, lo:] - xm) / sig)
                res["decoder"].setdefault(m, {}).setdefault(which, {})[zsrc] = nrmse(P, X)
                if zsrc == "obs":
                    keep = P
                else:      # noise-propagation part: T(z_obs) - T(z_clean)
                    res["decoder"][m][which]["noise_part"] = nrmse(keep, P)
print("decoders", json.dumps(res["decoder"]), f"{time.time() - t0:.0f}s", flush=True)

# ---- probes -------------------------------------------------------------------------------
for name in FEATS:
    for zsrc in (("obs", "clean") if name in ("z_c", "zwin20_c") else ("obs",)):
        F, X = flat(name, "train", zsrc, NTR, g)
        Fv, Xv = flat(name, "val", zsrc, 100_000, g)
        pred, vbest = fit_mlp(F, X, Fv, Xv)
        with torch.no_grad():
            r = {"train": nrmse(pred(F[:200_000]), X[:200_000]), "val": vbest}
        for t in ev:
            Q, XQ = flat(name, t, zsrc)
            with torch.no_grad():
                r[t] = nrmse(pred(Q), XQ)
        res["probe"][f"{name}|{zsrc}"] = r
        print("probe", name, zsrc, r, f"{time.time() - t0:.0f}s", flush=True)
        if name == "z_c":
            Fk, Xk = F[:300_000], X[:300_000]
            rk = {}
            for t in ev:
                Q, XQ = flat(name, t, zsrc, NQ, g)
                rk[t] = nrmse(knn(Fk, Xk, Q), XQ)
            res["knn"][f"{name}|{zsrc}"] = rk
            print("knn", name, zsrc, rk, flush=True)
        del F, X, Fv, Xv
        torch.cuda.empty_cache()

# ---- Exp 4: decoder Jacobian dx/dz (sigma units per unit z) along eval trajectories --------
for m, (f, cx) in dec.items():
    for t, (es, zc, lo) in ((t, sets[t]) for t in ev):
        E, N1 = es.x.shape[:2]
        ii = torch.randint(0, E, (2000,), generator=g).to(dev)
        kk = torch.randint(lo, N1, (2000,), generator=g).to(dev)
        z0 = es.z_obs[ii, kk]; ez = z0 - zc[ii, kk]
        c0 = cseq(cx, es.u)[ii, kk] if cx is not None else None
        fz = lambda z: (f(z, c0) - xm) / sig
        J = torch.autograd.functional.jacobian(lambda z: fz(z).sum(0), z0, vectorize=True)  # (nx, B, nz)
        J = J.permute(1, 0, 2)
        sv = torch.linalg.matrix_norm(J, ord=2)
        lin = torch.einsum("bij,bj->bi", J, ez)     # linearised noise error
        res["jac"].setdefault(m, {})[t] = {
            "spec_median": float(sv.median()), "spec_q95": float(torch.quantile(sv, 0.95)),
            "spec_q99": float(torch.quantile(sv, 0.99)),
            "ez_rms": float(ez.pow(2).sum(-1).mean().sqrt()),
            "lin_noise_nrmse": float(lin.pow(2).mean().sqrt()),
            "jac_scaled_by_zstd_median": float(torch.linalg.matrix_norm(J * z_std, ord=2).median())}
print("jac", json.dumps(res["jac"]), flush=True)
res["wall_s"] = time.time() - t0
outdir.mkdir(parents=True, exist_ok=True)
(outdir / f"{node}_{system}_s{seed}.json").write_text(json.dumps(res))
print("done", f"{time.time() - t0:.0f}s")
