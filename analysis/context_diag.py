"""Context-extrapolation diagnostics for node C1-where (where the input enters).

    .venv/bin/python -m analysis.context_diag run --system vdp --device cuda:5
    .venv/bin/python -m analysis.context_diag aggregate

Read-only with respect to the experiment tree: the C1-where artefacts are located by
recomputing the cache keys with the module hash of the code snapshot the runs were made
with (results.json 'code_snapshot'); systems.py / train.py / cache.py were edited after
C1-where, so the live module_hash() no longer addresses them. The cached evaluation and
training episodes are loaded with the same old keys (nothing is regenerated or written
into experiments/cache), and one evaluation tier per system is ALSO regenerated with the
current code and compared, to validate the amplitude-sweep episodes (generated fresh,
never cached).

Per (system, seed, method) this computes
  1. context statistics and dist(c, C_train) (nearest training context, standardised c);
  2. l_c = ||diag(1/x_std) d xhat / d c diag(c_std)||_2 for decoder-side variants and
     ||diag(1/z_std) d Phi / d c diag(c_std)||_2 for the latent injection, plus the
     latent DC-gain equivalent ||diag(1/x_std) J_dec (-A)^{-1} dPhi/dc diag(c_std)|| and
     the log-norm mu_2(A + dPhi/dz);
  3. an input-amplitude sweep s x (id-regime signals);
  4. per-coordinate context clamping to the training [q0.5%, q99.5%] box on ood.
Outputs: experiments/analysis/C1-extrapolation/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hyperkkl import data as Dm                                  # noqa: E402
from hyperkkl import signals as S                                # noqa: E402
from hyperkkl.config import digest                               # noqa: E402
from hyperkkl.evaluate import build_observers, initial_guess     # noqa: E402
from hyperkkl.job import method_cfg                              # noqa: E402
from hyperkkl.metrics import episode_metrics                     # noqa: E402
from hyperkkl.observers import KKLObserver                       # noqa: E402
from hyperkkl.systems import build_plant                         # noqa: E402

NODE = "C1-where"
SYSTEMS = ["vdp", "duffing", "bioreactor", "lorenz", "quad_tank"]
SEEDS = [0, 1, 2, 3, 4]
COND = ["latent", "concat", "film", "lora", "hyper"]
DEC = ["concat", "film", "lora", "hyper"]
TIERS = ["id", "ood", "waveform_ood"]
AMPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_SWEEP = 32
OUT = ROOT / "experiments" / "analysis" / "C1-extrapolation"
SCRATCH = OUT / "_snapshot"            # extracted old sources (read-only use)
N_TRAIN_C = 20_000                     # training contexts for the nearest-neighbour proxy
N_EVAL_C = 4096                        # eval (t >= settle) points per tier for dist
N_JAC = 1024                           # of those, points at which Jacobians are taken
DATA_MODULES = ("systems", "signals", "design", "data")
MODEL_MODULES = DATA_MODULES + ("nets", "train", "conditioning", "observers")


# ---------------------------------------------------------------------------
# Locating the C1-where artefacts
# ---------------------------------------------------------------------------

def results(system, seed) -> dict:
    return json.loads((ROOT / "experiments" / "tree" / NODE / "runs" / system / f"seed{seed}"
                       / "results.json").read_text())


def old_hashes(snapshot: str) -> tuple:
    """(data hash, model hash) with the algorithm of cache.module_hash, applied to the
    snapshot's sources and the snapshot's MODEL_MODULES (which had no 'monitor')."""
    d = SCRATCH / Path(snapshot).name.split(".")[0]
    if not d.exists():
        d.mkdir(parents=True)
        with tarfile.open(ROOT / snapshot) as tf:
            tf.extractall(d, filter="data")
    src = (d / "hyperkkl" / "cache.py").read_text()
    assert 'MODEL_MODULES = DATA_MODULES + ("nets", "train", "conditioning", "observers")' in src

    def mh(names):
        h = hashlib.sha1()
        for n in names:
            h.update((d / "hyperkkl" / f"{n}.py").read_bytes())
        return h.hexdigest()[:10]
    return mh(DATA_MODULES), mh(MODEL_MODULES)


def patch_data_hash(h_data: str):
    """data.py keys its caches with module_hash(DATA_MODULES); point it at the old hash so
    the exact C1-where episodes are loaded (all such files exist; checked by the caller)."""
    Dm.module_hash = lambda names=None: h_data


def load_models(system, seed, cfg, h_model, device) -> tuple:
    mc_dir = Dm.CACHE / "models"
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, h_model])
    p1 = mc_dir / f"{system}_p1_{k1}.pt"
    base = torch.load(p1, weights_only=False, map_location=device)
    trained, files = {}, {"p1": p1.name}
    for m in cfg["methods"]:
        kind, mc = method_cfg(cfg, m)
        k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, h_model])
        p = mc_dir / f"{system}_p2_{kind}_{k2}.pt"
        if not p.exists():
            print(f"  MISSING {p.name}", flush=True)
            continue
        r = torch.load(p, weights_only=False, map_location=device)
        for k in ("cond", "ctx"):
            if isinstance(r.get(k), torch.nn.Module):
                r[k] = r[k].to(device).eval()
        trained[m], files[m] = r, p.name
    for k in ("T0", "T0inv"):
        base[k] = base[k].to(device).eval()
    return base, trained, files


# ---------------------------------------------------------------------------
# Observers with recording and clamping
# ---------------------------------------------------------------------------

class ClampCtx(torch.nn.Module):
    """Context encoder whose output is clamped per coordinate (deployment step only)."""

    def __init__(self, ctx, lo, hi):
        super().__init__()
        self.ctx, self.lo, self.hi, self.dim = ctx, lo, hi, ctx.dim

    def init_state(self, B, device):
        return self.ctx.init_state(B, device)

    def step(self, u_k, state):
        c, st = self.ctx.step(u_k, state)
        return torch.maximum(torch.minimum(c, self.hi), self.lo), st


@torch.no_grad()
def run_rec(ob, y, u, x0, record=False):
    """observers.run, optionally also returning the latent and context sequences."""
    ob.reset(y.shape[0], x0, u[:, 0])
    xs = [ob.estimate()]
    zs, cs = ([ob.z], [ob.c]) if record else (None, None)
    for k in range(y.shape[1] - 1):
        ob.advance(y[:, k], u[:, k], y[:, k + 1], u[:, k + 1])
        xs.append(ob.estimate())
        if record:
            zs.append(ob.z); cs.append(ob.c)
    xh = torch.nan_to_num(torch.stack(xs, 1), nan=1e6, posinf=1e6, neginf=-1e6)
    if not record:
        return xh
    return xh, torch.stack(zs, 1), torch.stack(cs, 1)


def run_chunked(ob, Y, U, X0, record=False, chunk=512):
    outs = [run_rec(ob, Y[i:i + chunk], U[i:i + chunk], X0[i:i + chunk], record)
            for i in range(0, len(Y), chunk)]
    if not record:
        return torch.cat(outs)
    return tuple(torch.cat([o[j] for o in outs]) for j in range(3))


# ---------------------------------------------------------------------------
# Jacobians (per-sample; every conditioner is sample-wise, so row-wise autograd is exact)
# ---------------------------------------------------------------------------

def jac(fn, *args, wrt):
    """J[b] = d fn(args)[b] / d args[wrt][b], shape (B, d_out, d_in)."""
    args = [a.detach().clone().requires_grad_(i == wrt) for i, a in enumerate(args)]
    with torch.enable_grad():
        out = fn(*args)
        rows = [torch.autograd.grad(out[:, j].sum(), args[wrt], retain_graph=True)[0]
                for j in range(out.shape[1])]
    return torch.stack(rows, 1).detach()


def spec(M):
    return torch.linalg.matrix_norm(M, 2)


def q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if len(v) else float("nan")


def summ(v) -> dict:
    v = np.asarray(v, float)
    return {"median": q(v, 0.5), "q95": q(v, 0.95), "mean": float(np.nanmean(v)) if len(v) else float("nan"),
            "max": q(v, 1.0)}


@torch.no_grad()
def nn_dist(Q, R, chunk=2048):
    """Distance and index of the nearest row of R for each row of Q."""
    d, ix = [], []
    for i in range(0, len(Q), chunk):
        D = torch.cdist(Q[i:i + chunk], R)
        m = D.min(1)
        d.append(m.values); ix.append(m.indices)
    return torch.cat(d), torch.cat(ix)


# ---------------------------------------------------------------------------
# Amplitude sweep episodes (current code, nominal plant, never cached)
# ---------------------------------------------------------------------------

def sweep_sets(cfg, des, scales, device) -> dict:
    scfg, dcfg = cfg["system"], cfg["data"]
    plant = build_plant(scfg)
    seed = Dm.split_seed(scfg["name"], "amp_sweep", "dev")
    orig = Dm._signals
    out = {}
    for s in AMPS:
        def scaled(families, n, regime, rng, des_, scfg_, n_strata=4, _s=s):
            sigs = orig(families, n, regime, rng, des_, scfg_, n_strata)
            for g in sigs:                    # same draw for every s: paired episodes
                g.scale *= _s
                S.enforce_bounds(g, des_.dt, des_.T, scfg_.get("u_bounds"))
            return sigs
        Dm._signals = scaled
        try:
            es = Dm.generate(plant, des, scales, scfg, dict(dcfg, k_cross=1), S.families_for("id"),
                             N_SWEEP, "id", seed, device)
        finally:
            Dm._signals = orig
        out[s] = es
    return out


# ---------------------------------------------------------------------------
# One system
# ---------------------------------------------------------------------------

def run_system(system, device, seeds=SEEDS):
    t_all = time.time()
    r0 = results(system, seeds[0])
    h_data, h_model = old_hashes(r0["code_snapshot"])
    patch_data_hash(h_data)
    cfg = r0["config"]
    scales = Dm.system_scales(cfg, device)
    assert np.allclose(scales["x_std"], r0["scales"]["x_std"]), "scales mismatch"
    des = Dm.make_design(cfg["system"])
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * des.N)
    log = lambda m: print(f"[{system} {time.time() - t_all:6.0f}s] {m}", flush=True)
    log(f"old hashes data {h_data} model {h_model}; N={des.N} settle={settle}")

    ev = {t: Dm.eval_data(cfg, t, device=device) for t in TIERS}        # cached C1-where episodes
    # regenerate the ood tier with the CURRENT code (no cache) and compare
    scfg, dcfg = cfg["system"], cfg["data"]
    regen = Dm.generate(build_plant(scfg), des, scales, scfg, dict(dcfg, k_cross=1),
                        S.families_for("ood"), int(cfg["eval"]["n_episodes"]), "ood",
                        Dm.split_seed(scfg["name"], "ood", "dev", 0), device)
    regen_check = {k: float((getattr(regen, k) - getattr(ev["ood"], k)).abs().max())
                   for k in ("x", "u", "y")}
    regen_check["x_rel"] = regen_check["x"] / float(ev["ood"].x.abs().max())
    log(f"regenerated ood vs cached: {regen_check}")
    sw = sweep_sets(cfg, des, scales, device)
    u_amp = {s: float(sw[s].u.abs().amax(1).mean()) for s in AMPS}
    log(f"sweep sizes {[len(sw[s]) for s in AMPS]}  mean max|u| {u_amp}")

    blocks = [(t, ev[t]) for t in TIERS] + [(f"amp{s}", sw[s]) for s in AMPS]
    X = torch.cat([b.x for _, b in blocks]).to(device)
    Y = torch.cat([b.y for _, b in blocks]).to(device)
    U = torch.cat([b.u for _, b in blocks]).to(device)
    X0 = initial_guess(cfg, len(X), "center", 0, device)
    offs, o = {}, 0
    for name, b in blocks:
        offs[name] = (o, o + len(b)); o += len(b)

    A = torch.as_tensor(des.A, dtype=torch.float32, device=device)
    Ainv = torch.linalg.inv(-A)
    muA = float(torch.linalg.eigvalsh(0.5 * (A + A.T)).max())
    out = {"system": system, "old_hash": {"data": h_data, "model": h_model},
           "regen_check": regen_check, "sweep_u_amp": u_amp,
           "sweep_n": {str(s): len(sw[s]) for s in AMPS}, "mu2_A": muA, "alpha": des.alpha,
           "seeds": {}}
    ekf_done = None

    for seed in seeds:
        rs = results(system, seed)
        cfg = rs["config"]
        base, trained, files = load_models(system, seed, cfg, h_model, device)
        train = Dm.training_data(cfg, seed, device)["train"]
        obs = build_observers(cfg, des, scales, base, trained, None, device)
        if ekf_done is not None:
            obs.pop("ekf", None)
        R = {"files": files, "nrmse": {}, "repro": {}, "ctx": {}, "lc": {}, "clamp": {}}
        rec = {}
        for name, ob in obs.items():
            t0 = time.time()
            if name in COND:
                xh, Z, C = run_chunked(ob, Y, U, X0, record=True)
                rec[name] = (Z, C)
            else:
                xh = run_chunked(ob, Y, U, X0)
            m = episode_metrics(X, xh, sig, settle)
            R["nrmse"][name] = {b: m["nrmse"][a:e].tolist() for b, (a, e) in offs.items()}
            R["nrmse"][name + "__diverged"] = {b: float(m["diverged"][a:e].mean()) for b, (a, e) in offs.items()}
            for t in TIERS:
                stored = rs["results"][t]["clean"]["methods"].get(name)
                if stored is not None:
                    a, e = offs[t]
                    new, old = float(m["nrmse"][a:e].mean()), float(np.mean(stored["nrmse"]))
                    R["repro"].setdefault(name, {})[t] = {"new": new, "stored": old,
                                                          "rel": (new - old) / max(old, 1e-12)}
            log(f"seed {seed} {name:10s} " + " ".join(
                f"{b}={np.mean(R['nrmse'][name][b]):.4f}" for b in list(offs)[:3]) +
                f"  ({time.time() - t0:.0f}s)")
        if "ekf" in obs:
            ekf_done = R["nrmse"]["ekf"]
        worst = max(abs(v["rel"]) for d in R["repro"].values() for v in d.values())
        R["repro_worst_rel"] = worst
        log(f"seed {seed} reproduction worst |rel| = {worst:.2e}")
        if worst > 0.01:
            log("REPRODUCTION FAILED (>1%) -- stopping")
            out["seeds"][seed] = R
            out["repro_failed"] = True
            return out

        g = torch.Generator(device="cpu").manual_seed(1000 + seed)
        Ztr_all = train.z_obs.to(device)
        for name in COND:
            if name not in trained:
                continue
            r = trained[name]
            ctx = r["ctx"]
            with torch.no_grad():
                Ctr = torch.cat([ctx.seq(train.u[i:i + 64].to(device)) for i in range(0, len(train), 64)])
            Ctr = Ctr[:, train.burn:].reshape(-1, Ctr.shape[-1])
            Ztr = Ztr_all[:, train.burn:].reshape(-1, Ztr_all.shape[-1])
            pick = torch.randperm(len(Ctr), generator=g)[:N_TRAIN_C].to(device)
            pick_q = torch.randperm(len(Ctr), generator=g)[:400_000].to(device)
            c_tr, z_tr = Ctr[pick], Ztr[pick]
            c_mu, c_sd = c_tr.mean(0), c_tr.std(0).clamp(min=1e-8)
            z_mu, z_sd = z_tr.mean(0), z_tr.std(0).clamp(min=1e-8)
            Cq = Ctr[pick_q]
            lo, hi = torch.quantile(Cq, 0.005, dim=0), torch.quantile(Cq, 0.995, dim=0)
            ctr_n, ztr_n = (c_tr - c_mu) / c_sd, (z_tr - z_mu) / z_sd
            # training self-distance (leave-one-out) as the floor of the proxy
            Dself = torch.cdist(ctr_n[:2048], ctr_n)
            Dself[torch.arange(2048), torch.arange(2048)] = float("inf")
            cinfo = {"dc": int(c_tr.shape[1]), "train_c_norm": summ((ctr_n.norm(dim=1)).cpu()),
                     "train_nn_dist": summ(Dself.min(1).values.cpu()),
                     "train_c_std_raw": c_sd.cpu().tolist()}
            Z, C = rec[name]
            lcinfo = {}
            for t in ["id", "ood", "waveform_ood"] + [f"amp{s}" for s in (1.0, 2.0, 3.0)]:
                a, e = offs[t]
                Zt, Ct = Z[a:e, settle:].reshape(-1, Z.shape[-1]), C[a:e, settle:].reshape(-1, C.shape[-1])
                ii = torch.randperm(len(Ct), generator=g)[:N_EVAL_C].to(device)
                zq, cq = Zt[ii], Ct[ii]
                cqn, zqn = (cq - c_mu) / c_sd, (zq - z_mu) / z_sd
                d, nn = nn_dist(cqn, ctr_n)
                dj, _ = nn_dist(torch.cat([zqn, cqn], 1), torch.cat([ztr_n, ctr_n], 1))
                dz, _ = nn_dist(zqn, ztr_n)
                outside = ((cq < lo) | (cq > hi)).any(1).float().mean()
                cinfo[t] = {"c_norm": summ(cqn.norm(dim=1).cpu()), "dist": summ(d.cpu()),
                            "dist_joint_zc": summ(dj.cpu()), "dist_z": summ(dz.cpu()),
                            "frac_outside_box": float(outside),
                            "frac_beyond_train_q95": float((d > cinfo["train_nn_dist"]["q95"]).float().mean())}
                # Jacobians
                zj, cj = zq[:N_JAC], cq[:N_JAC]
                dcn = (cqn[:N_JAC] - ctr_n[nn[:N_JAC]])          # standardised offset to NN
                if name == "latent":
                    inj = r["cond"]
                    Jc = jac(inj, zj, cj, wrt=1)                   # (B, nz, dc), latent units
                    Jz = jac(inj, zj, cj, wrt=0)                   # (B, nz, nz)
                    zs = inj.z_std
                    Pn = Jc / zs[None, :, None] * c_sd[None, None, :]
                    Jd = jac(lambda zz: base["T0inv"](zz), zj, wrt=0)       # (B, nx, nz)
                    G = Jd / sig[None, :, None] @ Ainv @ (Jc * c_sd[None, None, :])
                    Acl = A[None] + Jz
                    G_cl = Jd / sig[None, :, None] @ torch.linalg.solve(-Acl, Jc * c_sd[None, None, :])
                    mu = torch.linalg.eigvalsh(0.5 * (Acl + Acl.transpose(1, 2))).amax(1)
                    absc = torch.linalg.eigvals(Acl).real.amax(1)
                    lcinfo[t] = {"l_phi_c": summ(spec(Pn).cpu()), "l_c_dc": summ(spec(G).cpu()),
                                 "l_c_dc_closed": summ(spec(G_cl).cpu()),
                                 "first_order_dc": summ((G @ dcn[..., None])[..., 0].norm(dim=1).cpu()),
                                 "mu2_Acl": summ(mu.cpu()), "abscissa_Acl": summ(absc.cpu()),
                                 "l_phi_z": summ(spec(Jz).cpu())}
                else:
                    dec = r["cond"]
                    Jc = jac(dec, zj, cj, wrt=1)                   # (B, nx, dc)
                    Jz = jac(dec, zj, cj, wrt=0)
                    Jn = Jc / sig[None, :, None] * c_sd[None, None, :]
                    Jzn = Jz / sig[None, :, None]
                    Jz0 = jac(lambda zz: base["T0inv"](zz), zj, wrt=0) / sig[None, :, None]
                    lcinfo[t] = {"l_c": summ(spec(Jn).cpu()),
                                 "l_c_raw": summ(spec(Jc / sig[None, :, None]).cpu()),
                                 "first_order": summ((Jn @ dcn[..., None])[..., 0].norm(dim=1).cpu()),
                                 "l_z": summ(spec(Jzn).cpu()), "l_z_base": summ(spec(Jz0).cpu())}
            R["ctx"][name], R["lc"][name] = cinfo, lcinfo

            # context clamp on id and ood
            if name in DEC:
                ob = KKLObserver(name, obs[name].lat, r["cond"], ctx=ClampCtx(ctx, lo, hi),
                                 z0_map=base["T0"])
            else:
                ob = KKLObserver(name, obs[name].lat, obs[name].dec, ctx=ClampCtx(ctx, lo, hi),
                                 injection=r["cond"], z0_map=base["T0"])
            cl = {}
            for t in ["id", "ood", "amp2.0", "amp3.0"]:
                a, e = offs[t]
                xh = run_chunked(ob, Y[a:e], U[a:e], X0[a:e])
                m = episode_metrics(X[a:e], xh, sig, settle)
                un = np.mean(R["nrmse"][name][t])
                cl[t] = {"clamped": float(m["nrmse"].mean()), "unclamped": float(un),
                         "rel": float(m["nrmse"].mean() / un - 1), "per_ep": m["nrmse"].tolist()}
            R["clamp"][name] = cl
            log(f"seed {seed} {name:7s} dist ood med {cinfo['ood']['dist']['median']:.2f} "
                f"(train nn {cinfo['train_nn_dist']['median']:.2f}) | clamp ood {cl['ood']['rel']:+.1%} "
                f"id {cl['id']['rel']:+.1%}")
        R.pop("files_obj", None)
        out["seeds"][seed] = R
        del rec
        torch.cuda.empty_cache()
    out["ekf"] = ekf_done
    return out


# ---------------------------------------------------------------------------
# Aggregation: tables, figures, report
# ---------------------------------------------------------------------------

COLORS = {"autonomous": "#8a8a85", "latent": "#2a78d6", "concat": "#eb6834", "film": "#1baf7a",
          "lora": "#eda100", "hyper": "#e87ba4", "ekf": "#222222"}
MARKERS = {"autonomous": "x", "latent": "o", "concat": "s", "film": "^", "lora": "D",
           "hyper": "v", "ekf": "*"}
LABEL = {"autonomous": "autonomous", "latent": "latent", "concat": "concat", "film": "FiLM",
         "lora": "LoRA", "hyper": "hypernet", "ekf": "EKF (model)"}
SYSLABEL = {"vdp": "Van der Pol", "duffing": "Duffing", "bioreactor": "Bioreactor",
            "lorenz": "Lorenz", "quad_tank": "Quadruple tank"}


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
        "grid.color": "#e4e4e0", "grid.linewidth": 0.5, "axes.axisbelow": True,
        "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.6,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")


def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, kendalltau
    _style()
    per = {s: json.loads((OUT / f"raw_{s}.json").read_text()) for s in SYSTEMS
           if (OUT / f"raw_{s}.json").exists()}
    systems = [s for s in SYSTEMS if s in per]
    methods_all = ["autonomous", "latent", "concat", "film", "lora", "hyper"]
    summ_out = {"systems": systems, "reproduction": {}, "regen_check": {}, "per_system": {}}

    def seedmean(s, m, block):
        v = [np.mean(per[s]["seeds"][k]["nrmse"][m][block]) for k in per[s]["seeds"]
             if m in per[s]["seeds"][k]["nrmse"]]
        return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, v

    def lcget(s, m, tier, key, stat):
        return [per[s]["seeds"][k]["lc"][m][tier][key][stat] for k in per[s]["seeds"] if m in per[s]["seeds"][k]["lc"]]

    def cget(s, m, tier, key, stat):
        return [per[s]["seeds"][k]["ctx"][m][tier][key][stat] for k in per[s]["seeds"] if m in per[s]["seeds"][k]["ctx"]]

    rows = []
    for s in systems:
        P = per[s]
        summ_out["regen_check"][s] = P["regen_check"]
        summ_out["reproduction"][s] = {k: P["seeds"][k]["repro_worst_rel"] for k in P["seeds"]}
        ps = {"nrmse": {}, "ratio_ood_id": {}, "ratio_wood_id": {}, "l_c": {}, "dist": {}, "clamp": {},
              "sweep": {}, "latent_margin": {}, "sweep_u_amp": P["sweep_u_amp"]}
        for m in methods_all:
            ps["nrmse"][m] = {t: seedmean(s, m, t)[:2] for t in TIERS}
            ps["sweep"][m] = {str(a): seedmean(s, m, f"amp{a}")[:2] for a in AMPS}
            ps["ratio_ood_id"][m] = float(np.mean(seedmean(s, m, "ood")[2]) / np.mean(seedmean(s, m, "id")[2]))
            ps["ratio_wood_id"][m] = float(np.mean(seedmean(s, m, "waveform_ood")[2]) / np.mean(seedmean(s, m, "id")[2]))
        if P.get("ekf"):
            ps["sweep"]["ekf"] = {str(a): (float(np.mean(P["ekf"][f"amp{a}"])), 0.0) for a in AMPS}
            ps["nrmse"]["ekf"] = {t: (float(np.mean(P["ekf"][t])), 0.0) for t in TIERS}
        for m in COND:
            key = "l_phi_c" if m == "latent" else "l_c"
            ps["l_c"][m] = {t: {st: float(np.mean(lcget(s, m, t, key, st))) for st in ("median", "q95")}
                            for t in ("id", "ood", "waveform_ood")}
            if m == "latent":
                ps["l_c"][m]["dc_gain"] = {t: {st: float(np.mean(lcget(s, m, t, "l_c_dc", st)))
                                               for st in ("median", "q95")} for t in ("id", "ood")}
                ps["latent_margin"] = {t: {"mu2_Acl_q95": float(np.mean(lcget(s, m, t, "mu2_Acl", "q95"))),
                                           "mu2_Acl_max": float(np.max(lcget(s, m, t, "mu2_Acl", "max"))),
                                           "abscissa_max": float(np.max(lcget(s, m, t, "abscissa_Acl", "max"))),
                                           "l_phi_z_q95": float(np.mean(lcget(s, m, t, "l_phi_z", "q95")))}
                                       for t in ("id", "ood")}
                ps["latent_margin"]["mu2_A"] = P["mu2_A"]
                ps["latent_margin"]["alpha"] = P["alpha"]
            else:
                ps["l_c"][m]["l_z_ood_q95"] = float(np.mean(lcget(s, m, "ood", "l_z", "q95")))
                ps["l_c"][m]["l_z_id_q95"] = float(np.mean(lcget(s, m, "id", "l_z", "q95")))
            ps["dist"][m] = {t: {st: float(np.mean(cget(s, m, t, "dist", st))) for st in ("median", "q95", "mean")}
                             for t in ("id", "ood", "waveform_ood")}
            ps["dist"][m]["train_nn_median"] = float(np.mean(
                [per[s]["seeds"][k]["ctx"][m]["train_nn_dist"]["median"] for k in per[s]["seeds"]]))
            for t in ("id", "ood", "waveform_ood"):
                ps["dist"][m][t]["frac_outside_box"] = float(np.mean(cget(s, m, t, "frac_outside_box", None)
                                                                     if False else [per[s]["seeds"][k]["ctx"][m][t]["frac_outside_box"] for k in per[s]["seeds"]]))
                ps["dist"][m][t]["dist_z_median"] = float(np.mean(cget(s, m, t, "dist_z", "median")))
            cl = {t: [per[s]["seeds"][k]["clamp"][m][t] for k in per[s]["seeds"]] for t in ("id", "ood", "amp2.0", "amp3.0")}
            ps["clamp"][m] = {t: {"unclamped": float(np.mean([c["unclamped"] for c in v])),
                                  "clamped": float(np.mean([c["clamped"] for c in v])),
                                  "rel_mean": float(np.mean([c["rel"] for c in v])),
                                  "rel_seeds": [c["rel"] for c in v]} for t, v in cl.items()}
            if m in DEC:
                fo = "first_order"
                lq = np.array(lcget(s, m, "ood", "l_c", "q95"))
                dd = np.array(cget(s, m, "ood", "dist", "mean"))
                rid = np.array(seedmean(s, m, "id")[2]); rood = np.array(seedmean(s, m, "ood")[2])
                rows.append({"system": s, "method": m, "log_ratio": float(np.log(rood.mean() / rid.mean())),
                             "log_lc_dist": float(np.log(np.mean(lq * dd))),
                             "log_lc": float(np.log(lq.mean())), "log_dist": float(np.log(dd.mean())),
                             "log_first_order": float(np.log(np.mean(lcget(s, m, "ood", fo, "mean")))),
                             "per_seed": [{"log_ratio": float(np.log(rood[i] / rid[i])),
                                           "log_lc_dist": float(np.log(lq[i] * dd[i]))} for i in range(len(lq))]})
        summ_out["per_system"][s] = ps

    # correlations
    def sp(x, y):
        r = spearmanr(x, y)
        return {"rho": float(r.statistic), "p": float(r.pvalue), "n": len(x)}
    lr = [r["log_ratio"] for r in rows]
    corr = {"pooled_lc_x_dist": sp([r["log_lc_dist"] for r in rows], lr),
            "pooled_lc_only": sp([r["log_lc"] for r in rows], lr),
            "pooled_dist_only": sp([r["log_dist"] for r in rows], lr),
            "pooled_first_order": sp([r["log_first_order"] for r in rows], lr),
            "per_seed_lc_x_dist": sp([p["log_lc_dist"] for r in rows for p in r["per_seed"]],
                                     [p["log_ratio"] for r in rows for p in r["per_seed"]])}
    # within-system (ranks of the 4 decoder methods), averaged Kendall tau
    within = {}
    for s in systems:
        rr = [r for r in rows if r["system"] == s]
        within[s] = float(kendalltau([r["log_lc_dist"] for r in rr], [r["log_ratio"] for r in rr]).statistic)
    corr["within_system_kendall_tau"] = within
    corr["within_system_mean_tau"] = float(np.nanmean(list(within.values())))
    # leave-lorenz-out (lorenz has no degradation)
    rr = [r for r in rows if r["system"] != "lorenz"]
    corr["pooled_lc_x_dist_no_lorenz"] = sp([r["log_lc_dist"] for r in rr], [r["log_ratio"] for r in rr])
    summ_out["correlation"] = corr
    summ_out["scatter_rows"] = rows
    (OUT / "summary.json").write_text(json.dumps(summ_out, indent=1))

    # ---------------- figures ----------------
    n = len(systems)
    # 1. amplitude sweep
    fig, axs = plt.subplots(1, n, figsize=(1.95 * n + 0.6, 2.35), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, systems):
        sw = summ_out["per_system"][s]["sweep"]
        for m in methods_all + ["ekf"]:
            if m not in sw:
                continue
            mu = np.array([sw[m][str(a)][0] for a in AMPS]); sd = np.array([sw[m][str(a)][1] for a in AMPS])
            ls = "--" if m in ("ekf", "autonomous") else "-"
            ax.plot(AMPS, mu, ls, color=COLORS[m], marker=MARKERS[m], ms=4.2, lw=1.4, label=LABEL[m],
                    mec="white", mew=0.5, zorder=3)
            if m not in ("ekf",):
                ax.fill_between(AMPS, np.maximum(mu - sd, 1e-6), mu + sd, color=COLORS[m], alpha=0.12, lw=0)
        ax.axvspan(0.0, 1.0, color="#f1f1ee", zorder=0, lw=0)
        ax.set_xlim(0.4, 3.1)
        ax.set_yscale("log")
        ax.set_title(SYSLABEL[s])
        ax.set_xlabel("input amplitude scale $s$")
    axs[0].set_ylabel("steady-state NRMSE")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig_amplitude_sweep"); plt.close(fig)

    # 2. l_c bars (id vs ood, q95) for decoder-side; latent l_phi_c and DC gain separately
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [3, 1.3]})
    x = np.arange(n); w = 0.19
    ax = axs[0]
    for i, m in enumerate(DEC):
        v_id = [summ_out["per_system"][s]["l_c"][m]["id"]["q95"] for s in systems]
        v_ood = [summ_out["per_system"][s]["l_c"][m]["ood"]["q95"] for s in systems]
        xx = x + (i - 1.5) * w
        ax.bar(xx, v_ood, w * 0.9, color=COLORS[m], label=LABEL[m], zorder=2)
        ax.scatter(xx, v_id, marker="_", s=60, color="#222222", zorder=4, lw=1.2,
                   label="id level" if i == 0 else None)
    ax.set_yscale("log"); ax.set_xticks(x, [SYSLABEL[s] for s in systems])
    ax.set_ylabel(r"$\ell_c$ (q95, ood points)" "\n" r"$\|\Sigma_x^{-1}\,\partial\hat x/\partial c\,\Sigma_c\|_2$")
    ax.set_title("Decoder-side context sensitivity")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax = axs[1]
    v1 = [summ_out["per_system"][s]["l_c"]["latent"]["dc_gain"]["ood"]["q95"] for s in systems]
    ax.bar(x, v1, 0.6, color=COLORS["latent"], zorder=2)
    ax.set_yscale("log"); ax.set_xticks(x, [SYSLABEL[s].split()[0][:8] for s in systems], rotation=30, ha="right")
    ax.set_title("Latent: DC-gain equivalent")
    ax.set_ylabel(r"$\|\Sigma_x^{-1}J_{T_0^+}(-A)^{-1}\partial_c\Phi\,\Sigma_c\|_2$")
    fig.tight_layout()
    _save(fig, "fig_lc_bars"); plt.close(fig)

    # 3. scatter
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    smark = {"vdp": "o", "duffing": "s", "bioreactor": "^", "lorenz": "D", "quad_tank": "v"}
    for r in rows:
        ax.scatter(np.exp(r["log_lc_dist"]), np.exp(r["log_ratio"]), color=COLORS[r["method"]],
                   marker=smark[r["system"]], s=38, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axhline(1, color="#999", lw=0.6, ls=":")
    ax.set_xlabel(r"$\ell_c^{q95}\times\overline{\mathrm{dist}}(c, C_{\rm train})$ (ood)")
    ax.set_ylabel("NRMSE ratio ood / id")
    c = corr["pooled_lc_x_dist"]
    ax.set_title(f"Spearman $\\rho$ = {c['rho']:.2f} (p = {c['p']:.2g}, n = {c['n']})")
    from matplotlib.lines import Line2D
    hm = [Line2D([], [], color=COLORS[m], marker="o", ls="", ms=5, label=LABEL[m]) for m in DEC]
    hs = [Line2D([], [], color="#555", marker=smark[s], ls="", ms=5, mfc="none", label=SYSLABEL[s]) for s in systems]
    ax.legend(handles=hm + hs, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, "fig_scatter_degradation"); plt.close(fig)

    # 4. clamp effect
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    for i, m in enumerate(COND):
        xx = x + (i - 2) * 0.16
        v = np.array([summ_out["per_system"][s]["clamp"][m]["ood"]["rel_mean"] for s in systems]) * 100
        sd = np.array([np.std(summ_out["per_system"][s]["clamp"][m]["ood"]["rel_seeds"], ddof=1) for s in systems]) * 100
        ax.bar(xx, v, 0.15, yerr=sd, color=COLORS[m], label=LABEL[m], zorder=2,
               error_kw={"lw": 0.6, "capsize": 1.5, "ecolor": "#444"})
    ax.axhline(0, color="#333", lw=0.6)
    ax.set_xticks(x, [SYSLABEL[s] for s in systems])
    ax.set_ylabel("ood NRMSE change\nwith context clamp (%)")
    ax.set_title("Clamping $c$ to the training box [q0.5%, q99.5%] (mean ± sd over seeds)")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    _save(fig, "fig_clamp"); plt.close(fig)

    write_report(summ_out)


def write_report(S_):
    systems = S_["systems"]
    L = []
    f = lambda v, d=3: f"{v:.{d}f}" if isinstance(v, (int, float)) and np.isfinite(v) else "-"
    L.append("# C1-where: context extrapolation diagnostics\n")
    L.append("Generated by `analysis/context_diag.py` from the cached C1-where models (no retraining). "
             "Steady-state NRMSE (t >= settle_frac T), mean over 5 seeds; l_c is the spectral norm of "
             "diag(1/x_std) dxhat/dc diag(c_std) (c standardised by training-context std), dist is the "
             "standardised Euclidean distance from each eval context (t >= settle) to its nearest of 20k training contexts.\n")
    L.append("## Reproduction\n")
    L.append("| system | worst |rel. diff| vs stored nrmse (all methods, tiers, seeds) | regenerated ood vs cached: max|dx| | max|du| |")
    L.append("|---|---|---|---|")
    for s in systems:
        r = S_["reproduction"][s]; g = S_["regen_check"][s]
        L.append(f"| {s} | {max(r.values()):.1e} | {g['x']:.1e} | {g['u']:.1e} |")
    L.append("\n## Degradation (NRMSE ratio vs id)\n")
    L.append("| system | method | id | ood | waveform_ood | ood/id | wood/id |")
    L.append("|---|---|---|---|---|---|---|")
    for s in systems:
        P = S_["per_system"][s]
        for m in ["autonomous"] + COND + (["ekf"] if "ekf" in P["nrmse"] else []):
            nr = P["nrmse"][m]
            rat = P["ratio_ood_id"].get(m, nr["ood"][0] / nr["id"][0])
            rw = P["ratio_wood_id"].get(m, nr["waveform_ood"][0] / nr["id"][0])
            L.append(f"| {s} | {m} | {f(nr['id'][0], 4)} | {f(nr['ood'][0], 4)} | {f(nr['waveform_ood'][0], 4)} | {f(rat, 2)} | {f(rw, 2)} |")
    L.append("\n## Context distance to the training set (standardised, median over eval steps; mean over seeds)\n")
    L.append("| system | method | train NN (floor) | id | waveform_ood | ood | ood q95 | ood frac outside q0.5-99.5 box | dist_z ood median |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in systems:
        for m in COND:
            d = S_["per_system"][s]["dist"][m]
            L.append(f"| {s} | {m} | {f(d['train_nn_median'], 2)} | {f(d['id']['median'], 2)} | {f(d['waveform_ood']['median'], 2)} | "
                     f"{f(d['ood']['median'], 2)} | {f(d['ood']['q95'], 2)} | {f(d['ood']['frac_outside_box'], 2)} | {f(d['ood']['dist_z_median'], 2)} |")
    L.append("\n## Context sensitivity l_c (median / q95; decoder-side: d xhat/dc; latent: d Phi/dc in z_std units and its DC-gain equivalent)\n")
    L.append("| system | method | id median | id q95 | ood median | ood q95 | wood q95 | l_z q95 id -> ood |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in systems:
        P = S_["per_system"][s]["l_c"]
        for m in COND:
            p = P[m]
            extra = f"{f(p['l_z_id_q95'], 2)} -> {f(p['l_z_ood_q95'], 2)}" if m != "latent" else "-"
            L.append(f"| {s} | {m} | {f(p['id']['median'])} | {f(p['id']['q95'])} | {f(p['ood']['median'])} | {f(p['ood']['q95'])} | {f(p['waveform_ood']['q95'])} | {extra} |")
            if m == "latent":
                g = p["dc_gain"]
                L.append(f"| {s} | latent (DC gain to xhat) | {f(g['id']['median'])} | {f(g['id']['q95'])} | {f(g['ood']['median'])} | {f(g['ood']['q95'])} | - | - |")
    L.append("\n## Latent contraction margin\n")
    L.append("| system | alpha | mu2(A) | mu2(A+dPhi/dz) q95 id | q95 ood | max ood | max Re eig(A+dPhi/dz) ood | ||dPhi/dz|| q95 ood |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in systems:
        M = S_["per_system"][s]["latent_margin"]
        L.append(f"| {s} | {f(M['alpha'], 2)} | {f(M['mu2_A'], 2)} | {f(M['id']['mu2_Acl_q95'], 2)} | {f(M['ood']['mu2_Acl_q95'], 2)} | "
                 f"{f(M['ood']['mu2_Acl_max'], 2)} | {f(M['ood']['abscissa_max'], 2)} | {f(M['ood']['l_phi_z_q95'], 2)} |")
    L.append("\n## Context clamp (relative NRMSE change, mean over seeds)\n")
    L.append("| system | method | id | ood | amp x2 | amp x3 |")
    L.append("|---|---|---|---|---|---|")
    for s in systems:
        for m in COND:
            c = S_["per_system"][s]["clamp"][m]
            L.append(f"| {s} | {m} | {c['id']['rel_mean']:+.1%} | {c['ood']['rel_mean']:+.1%} | {c['amp2.0']['rel_mean']:+.1%} | {c['amp3.0']['rel_mean']:+.1%} |")
    L.append("\n## Amplitude sweep (NRMSE, mean over seeds; s x id amplitude, id-regime waveforms)\n")
    L.append("| system | method | " + " | ".join(f"s={a}" for a in AMPS) + " |")
    L.append("|---|---|" + "---|" * len(AMPS))
    for s in systems:
        sw = S_["per_system"][s]["sweep"]
        for m in ["autonomous"] + COND + (["ekf"] if "ekf" in sw else []):
            L.append(f"| {s} | {m} | " + " | ".join(f(sw[m][str(a)][0], 4) for a in AMPS) + " |")
        ua = S_["per_system"][s]["sweep_u_amp"]
        L.append(f"| {s} | (mean max abs u) | " + " | ".join(f(ua[str(a)] if str(a) in ua else ua[a], 3) for a in AMPS) + " |")
    L.append("\n## Correlation (decoder-side variants; one point per system x method, seed means)\n")
    C = S_["correlation"]
    for k in ("pooled_lc_x_dist", "pooled_lc_only", "pooled_dist_only", "pooled_first_order",
              "per_seed_lc_x_dist", "pooled_lc_x_dist_no_lorenz"):
        L.append(f"- {k}: Spearman rho = {C[k]['rho']:.3f} (p = {C[k]['p']:.3g}, n = {C[k]['n']})")
    L.append(f"- within-system Kendall tau (4 methods each): " +
             ", ".join(f"{s} {v:+.2f}" for s, v in C["within_system_kendall_tau"].items()) +
             f"; mean {C['within_system_mean_tau']:+.2f}")
    L.append("\nFigures: fig_amplitude_sweep, fig_lc_bars, fig_scatter_degradation, fig_clamp (.png/.pdf).\n")
    (OUT / "report.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "aggregate"])
    ap.add_argument("--system")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "run":
        torch.set_num_threads(2)
        res = run_system(a.system, a.device, [int(s) for s in a.seeds.split(",")])
        tag = "" if a.seeds == "0,1,2,3,4" else "_partial"
        (OUT / f"raw_{a.system}{tag}.json").write_text(json.dumps(res))
    else:
        aggregate()


if __name__ == "__main__":
    main()
