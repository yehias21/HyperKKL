"""One bench_v2 job: train ONE method family on one cell's mixed data, score every block with the bench scorer.

    CUDA_VISIBLE_DEVICES=2 python v2run.py driven vdp realistic hk --seed 0 --lr 1e-2 [--smoke]

methods (PREREG.md section 3):
  filters     ekf, ekf_gated, ukf, ukf_gated, ekf_gated_tuned, ukf_gated_tuned (no training; tuned on val). seed 0 only
  ude1        corrected grey-box (ude5k1 style: K = 1 through RK4) + gated EKF / UKF   [b2 recipe, lr grid of 4]
  ude32       multi-step UDE (K = 32 RK4 rollouts) + gated EKF / UKF                    [b2 recipe, lr grid of 4]
  node32      neural ODE (no physics, K = 32) + gated EKF                              [b2 recipe, lr grid of 4]
  gru         2-layer GRU, ~140k parameters                                             [b2 recipe, lr grid of 4]
  ncde        neural CDE, ~140k                                                          [b2 recipe, lr grid of 4]
  p1          HyperKKL phase 1 (A3 optimiser) -> saved maps; scores the autonomous KKL observer `auto`
  hk          HyperKKL: phase 2 base (A3 + mixed) at --lr, then the default C5 (jump 3 / slope 0.3 / cap 0.5)
              -> observers hk_base, hyperkkl
  latent | concat | film   conditioning baselines (train.phase2 of the research harness, Adam, 3000 steps) at --lr
  nlox_m | dkkl_m | dkkl_inj_m   peers (experiments/kkl_peers/code/peers.py), 15000 steps, hk_budget params, --lr
Output runs/<family>/<plant>/<variant>/<method>[_lr<lr>]/seed<k>.json (+ .log), models under models/.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback

import env
import numpy as np
import torch

import v2cfg

A3_P1 = {"optimizer": "soap", "lr": 1e-2}
A3_P2 = {"optimizer": "soap", "lr": 1e-2, "base_lr_factor": 0.1, "window": 20}
C5 = {"jump": {"rho": 3.0}, "sens": {"rho": 0.3}, "specnorm": {"kappa": 0.5}}
PEER_STEPS = 15000
FAIR = {"optimizer": "soap", "lr": 1e-2, "K": 256, "Bs": 32, "val_every": 250}
B_COND_ID = ["clean", "outliers", "burst_outage", "poor_init", "input_realistic", "input_cmd_vs_applied", "input_quant"]
B_COND_OOD = ["clean", "outliers", "burst_outage", "input_realistic"]
A_COND_ID = ["clean", "noise10", "outliers", "dropout", "burst_outage", "poor_init"]


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


# ============================================================================================ cell
def run_safe(ob, Y, U, X0, chunk, log, tag):
    """_run_chunks; if a chunk raises a numerical error (e.g. a singular innovation covariance after a filter
    diverged), rerun that chunk one episode at a time and give an episode that still fails a NaN trajectory, which
    the scorer counts as DIVERGED (clipped error 100, PROTOCOL section 5). Logged; the method is unchanged."""
    from hyperkkl.evaluate import _run_chunks
    out = []
    for i in range(0, len(Y), chunk):
        sl = slice(i, i + chunk)
        try:
            out.append(_run_chunks(ob, Y[sl], U[sl], X0[sl], chunk))
        except (RuntimeError, torch._C._LinAlgError) as ex:
            log(f"  [{tag}] chunk {i}: {type(ex).__name__}: {str(ex)[:120]} -> per-episode rerun")
            for j in range(i, min(i + chunk, len(Y))):
                try:
                    out.append(_run_chunks(ob, Y[j:j + 1], U[j:j + 1], X0[j:j + 1], 1))
                except (RuntimeError, torch._C._LinAlgError):
                    out.append(torch.full((1, Y.shape[1], X0.shape[1]), float("nan"), device=Y.device))
                    log(f"  [{tag}] episode {j}: failed numerically -> scored as diverged")
    return torch.cat(out)


class Cell:
    def __init__(self, family, plant, variant, device, smoke=False):
        from hyperkkl import data as Dm
        from hyperkkl import systems_extra, systems_bounded, systems_chaos  # noqa: F401
        import build_data
        build_data.setup_cache()
        self.Dm, self.family, self.plant, self.variant, self.dev, self.smoke = Dm, family, plant, variant, device, smoke
        self.cfg = v2cfg.config(family, plant, variant)
        self.dir = env.DATA / family / plant / variant
        assert (self.dir / "MANIFEST.json").exists(), f"cell not built: {self.dir}"
        self.sc = json.loads((self.dir / "scales.json").read_text())
        self.des = Dm.make_design(self.cfg["system"])
        L = lambda n: torch.load(self.dir / f"{n}.pt", weights_only=False, map_location="cpu")
        self.L = L
        from jf_env_v2 import cat
        tr, va = cat(L("train"), L("train_event")), cat(L("val"), L("val_event"))
        if smoke:
            tr, va = tr.select(range(0, len(tr), 10)), va.select(range(0, len(va), 4))
        self.data = {"p1": L("phase1"), "train": tr, "val": va,
                     "coll": torch.load(self.dir / "coll.pt", weights_only=False)}
        self.model = Dm.model_plant(self.cfg["system"])
        self.sensors = list(self.model.sensors)
        self.nx = len(self.sc["x_std"])
        self.unm = [j for j in range(self.nx) if j not in self.sensors]
        dcfg, s = self.cfg["data"], self.cfg["system"]
        self.q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(self.sc["x_std"])
        self.r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(self.sc["y_std"])
        box = np.asarray(s["ic_box"], float)
        self.p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
        self.t_ref = float(s.get("t_ref", 1.0))
        self.settle = int(0.5 * self.des.N)

    # ------------------------------------------------------------------ evaluation blocks
    def blocks(self):
        """{name: (es, Y, U_obs, X0)} on CPU; deterministic in (cell, block)."""
        from hyperkkl.evaluate import initial_guess
        out = {}
        sp = lambda n: self.L(n)
        n_ep = 8 if self.smoke else None
        cut = (lambda es: es.select(range(min(n_ep, len(es))))) if n_ep else (lambda es: es)
        ctr = lambda es: initial_guess(self.cfg, len(es), "center", 0, "cpu")
        va = cut(self.data["val"])
        out["val/clean"] = (va, va.y, va.u, ctr(va))
        vo = cut(sp("val_ood"))
        out["val_ood/clean"] = (vo, vo.y, vo.u, ctr(vo))
        if self.family == "driven":
            plan = [("test_op_id", B_COND_ID), ("test_op_ood", B_COND_OOD)] + \
                   [(f"test_legacy_{t}", ["clean"]) for t in ("id", "shape_ood", "sequence_ood")]
        else:
            plan = [("test_id", A_COND_ID), ("test_ic_ood", ["clean"]), ("test_param_shift", ["clean"])]
        for split, conds in plan:
            es = cut(sp(split))
            for c in conds:
                out[f"{split}/{c}"] = self.condition(es, split, c)
        return out

    def condition(self, es, split, c):
        import conditions_v2 as C
        return C.apply(self, es, split, c)

    def score(self, es, xh):
        from hyperkkl_bench import metrics as M
        X = es.x.double().numpy()
        Xh = xh.double().cpu().numpy()
        m = M.episode_metrics(X, Xh, np.asarray(self.sc["x_std"]), self.settle, self.sensors)
        return {k: np.asarray(m[k]).tolist() for k in ("nrmse_unmeasured", "nrmse", "nrmse_transient", "diverged")}

    def evaluate(self, observers: dict, log, chunk=64):
        from hyperkkl.evaluate import _run_chunks
        res = {n: {} for n in observers}
        for bname, (es, Y, U, X0) in self.blocks().items():
            for n, ob in observers.items():
                t0 = time.time()
                with torch.no_grad():
                    xh = run_safe(ob, Y.to(self.dev), U.to(self.dev), X0.to(self.dev), chunk, log, f"{n} {bname}")
                res[n][bname] = self.score(es, xh)
                v = res[n][bname]
                log(f"  [{n}] {bname:34s} unm {np.mean(v['nrmse_unmeasured']):.4f} all {np.mean(v['nrmse']):.4f} "
                    f"div {np.mean(v['diverged']):.2f} ({time.time() - t0:.0f}s)")
        return res

    def val_score(self, res_one):
        """Selection score (validation only): mean of the val and val_ood settled NRMSE + divergence penalty."""
        v = [np.mean(res_one[b]["nrmse"]) + np.mean(res_one[b]["diverged"]) for b in ("val/clean", "val_ood/clean")]
        return float(np.mean(v))


# ============================================================================================ methods
def run_filters(cell, seed, lr, log):
    from hyperkkl.observers import EKF, UKF
    obs, info = {}, {}
    for kind, F in (("ekf", EKF), ("ukf", UKF)):
        obs[kind] = F(cell.model, cell.des.dt, cell.q, cell.r, cell.p0)
        obs[f"{kind}_gated"] = F(cell.model, cell.des.dt, cell.q, cell.r, cell.p0, gate=True)
    # tuned: Q / R std inflation chosen on the validation blocks (8 configs), gated
    from hyperkkl.evaluate import _run_chunks, initial_guess
    va, vo = cell.data["val"], cell.L("val_ood")
    cand = [(qa, ra) for qa in (1.0, 3.0, 10.0, 30.0) for ra in (1.0, 3.0)]
    for kind, F in (("ekf", EKF), ("ukf", UKF)):
        best = None
        for qa, ra in cand:
            ob = F(cell.model, cell.des.dt, cell.q * qa, cell.r * ra, cell.p0, gate=True)
            sc = []
            for es in (va.select(range(min(32, len(va)))), vo.select(range(min(32, len(vo))))):
                x0 = initial_guess(cell.cfg, len(es), "center", 0, "cpu")
                with torch.no_grad():
                    xh = run_safe(ob, es.y.to(cell.dev), es.u.to(cell.dev), x0.to(cell.dev), 64, log, f"tune {kind}")
                s = cell.score(es, xh)
                sc.append(np.mean(s["nrmse"]) + np.mean(s["diverged"]))
            v = float(np.mean(sc))
            log(f"  tune {kind} q x{qa} r x{ra}: val {v:.4f}")
            if best is None or v < best[0]:
                best = (v, qa, ra)
        obs[f"{kind}_gated_tuned"] = F(cell.model, cell.des.dt, cell.q * best[1], cell.r * best[2], cell.p0, gate=True)
        info[f"{kind}_gated_tuned"] = {"q_std_x": best[1], "r_std_x": best[2], "val": best[0], "configs": len(cand)}
    return obs, info


def _b2_job(cell, seed):
    import b2.models as M
    import b2.train as T
    from hyperkkl import baselines_data as BD
    from hyperkkl.evaluate import initial_guess
    from hyperkkl.observers import EKF, UKF

    class J:
        pass
    j = J()
    j.cfg, j.des, j.scales, j.data, j.device, j.seed = cell.cfg, cell.des, cell.sc, cell.data, cell.dev, seed
    j.model, j.sensors = cell.model, cell.sensors
    j.nx, j.nu, j.ny = cell.model.nx, cell.model.nu, cell.model.ny
    j.q, j.r, j.p0 = cell.q, cell.r, cell.p0
    j.u_scale = float(cell.cfg["system"].get("u_scale", 1.0))
    j.va = cell.data["val"].to(cell.dev)
    j.x0v = initial_guess(cell.cfg, len(j.va), "center", 0, cell.dev)
    j.steps = 20 if cell.smoke else 3000
    j.lrs = [1e-3] if cell.smoke else [3e-4, 1e-3, 3e-3, 1e-2]
    return j, M, T, BD, EKF, UKF


def run_dyn(cell, seed, kind, log):
    """b2.job.run_dyn, with the filters GATED (published 99.9 % chi-square rule, PITFALLS P2)."""
    j, M, T, BD, EKF, UKF = _b2_job(cell, seed)
    hybrid = kind in ("ude1", "ude32")
    K = 1 if kind == "ude1" else 32
    fm = j.model if hybrid else None
    sets = [j.data["train"], j.data["p1"]]
    W = M.solve_mlp(j.nx, j.nu, 5000)
    tm, ts = T.target_stats(j.model if hybrid else None, sets, j.des.dt, j.device, seed)
    filt = (("ekf", EKF), ("ukf", UKF)) if hybrid else (("ekf", EKF),)
    cands = {fk: [] for fk, _ in filt}
    best = {fk: None for fk, _ in filt}
    for lr in j.lrs:
        torch.manual_seed(seed)
        net = BD.ResidualMLP(j.nx, j.nu, W, j.scales.get("x_mean"), j.scales["x_std"], j.u_scale, tm, ts)
        info = T.train_dyn(fm, net, sets, j.va, j.des.dt, j.steps, lr, j.device, seed, log, K=K,
                           val_every=10 if cell.smoke else 100, warm_steps=5 if cell.smoke else 300, use_graphs=True)
        gm = M.deployed_dynamics(j.model, net, hybrid)
        for fk, F in filt:
            ob = F(gm, j.des.dt, j.q, j.r, j.p0, gate=True)
            dv = T.deployed_val(ob, j.cfg, j.va, j.scales, j.device, j.x0v)
            log(f"  {kind}+{fk} lr {lr:g}: rollout val {info['val_loss']:.4g} deployed val {dv['val_nrmse']:.4f} "
                f"div {dv['val_diverged']:.3f}")
            cands[fk].append({"lr": lr, **dv, "val_loss": info["val_loss"], "wall_s": info["wall_s"]})
            sel = min(range(len(cands[fk])), key=lambda i: cands[fk][i]["val_nrmse"] + cands[fk][i]["val_diverged"])
            if sel == len(cands[fk]) - 1:
                best[fk] = copy.deepcopy(net)
    obs, meta = {}, {}
    for fk, F in filt:
        i = min(range(len(cands[fk])), key=lambda i: cands[fk][i]["val_nrmse"] + cands[fk][i]["val_diverged"])
        name = f"{kind}_{fk}_gated"
        obs[name] = F(M.deployed_dynamics(j.model, best[fk], hybrid), j.des.dt, j.q, j.r, j.p0, gate=True)
        meta[name] = {"lr": cands[fk][i]["lr"], "n_params": sum(p.numel() for p in best[fk].parameters()),
                      "K": K, "candidates": cands[fk], "configs": len(j.lrs)}
    return obs, meta


def run_seq(cell, seed, kind, log):
    j, M, T, BD, EKF, UKF = _b2_job(cell, seed)
    tr = j.data["train"].to(j.device)
    xm = np.asarray(j.scales["x_mean"])
    norm = lambda: M._Norm(j.u_scale, xm[j.sensors], j.scales["y_std"], xm, j.scales["x_std"])
    cands, best = [], None
    lrs = j.lrs if kind == "gru" else ([1e-3] if cell.smoke else [1e-3, 3e-3])     # ncde: 2 configs (cost)
    for lr in lrs:
        torch.manual_seed(seed)
        if kind == "gru":
            H = M.solve_gru(j.nu, j.ny, j.nx, 140_000)
            net = M.GRUNet(j.nu, j.ny, j.nx, H, 2, norm())
        else:
            Wd = M.solve_ncde(j.nu, j.ny, j.nx, 32, 140_000)
            net = M.NCDENet(j.nu, j.ny, j.nx, 32, Wd, norm(), j.des.dt / cell.t_ref)
        info = T.train_seq(net, tr, j.va, j.steps, lr, j.device, seed, log, val_every=10 if cell.smoke else 250)
        ob = M.SeqObserver(kind, net)
        dv = T.deployed_val(ob, j.cfg, j.va, j.scales, j.device, j.x0v)
        log(f"  {kind} lr {lr:g}: deployed val {dv['val_nrmse']:.4f} div {dv['val_diverged']:.3f} bad {info['n_bad_steps']}")
        cands.append({"lr": lr, **dv, "val_loss": info["val_loss"], "n_bad_steps": info["n_bad_steps"]})
        sel = min(range(len(cands)), key=lambda i: cands[i]["val_nrmse"] + cands[i]["val_diverged"])
        if sel == len(cands) - 1:
            best = net
    i = min(range(len(cands)), key=lambda i: cands[i]["val_nrmse"] + cands[i]["val_diverged"])
    return {kind: M.SeqObserver(kind, best)}, {kind: {"lr": cands[i]["lr"], "candidates": cands, "configs": len(lrs),
                                                     "n_params": sum(p.numel() for p in best.parameters())}}


def p1_path(cell, seed):
    d = env.HERE / ("models_smoke" if cell.smoke else "models") / cell.family / cell.plant / cell.variant / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d / "p1.pt"


def get_p1(cell, seed, log, train=False, wait_s=6 * 3600):
    from hyperkkl.config import deep_merge
    from hyperkkl.train import phase1
    f = p1_path(cell, seed)
    if train and not f.exists():
        cfg = deep_merge(cell.cfg, {"phase1": dict(A3_P1)})
        cfg["seed"] = seed
        if cell.smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40}})
        torch.manual_seed(seed); np.random.seed(seed)
        r = phase1(cfg, cell.data, cell.sc, cell.des, cell.dev, seed, log)
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        torch.save({k: (v.cpu() if isinstance(v, torch.nn.Module) else v) for k, v in r.items()}, tmp)
        os.replace(tmp, f)
    t0 = time.time()
    while not f.exists():
        if time.time() - t0 > wait_s:
            raise TimeoutError(f"phase-1 maps {f} not produced")
        time.sleep(30)
    b = torch.load(f, weights_only=False, map_location=cell.dev)
    return {k: (v.to(cell.dev) if isinstance(v, torch.nn.Module) else v) for k, v in b.items()}


def constants(cell):
    """jf_metrics.constants on the cell's mixed training set (the per-plant rule for the C5 jump lags)."""
    tr = cell.data["train"]
    x = tr.x.numpy().astype(np.float64)
    s = np.asarray(cell.sc["x_std"], float)[cell.unm]
    X = x[:, cell.settle:, cell.unm] / s
    Xd = X - X.mean(1, keepdims=True)
    Pw = (np.abs(np.fft.rfft(Xd, axis=1)) ** 2).sum((0, 2))
    f = np.fft.rfftfreq(X.shape[1], cell.des.dt)
    cum = np.cumsum(Pw[1:]) / Pw[1:].sum()
    f95 = float(f[1:][np.searchsorted(cum, 0.95)])
    fc = float(min(5.0 * f95, 0.8 * 0.5 / cell.des.dt))
    L2 = int(np.clip(round(1.0 / (fc * cell.des.dt)), 2, 50))
    return {"fc": fc, "lags": [1, L2], "f95": f95}


def p2_opts(cell, lr):
    a = float(cell.cfg["system"]["alpha"]) * float(cell.cfg["system"].get("alpha_scale", 1.0))
    o = dict(A3_P2, lr=float(lr), stride=max(1, int(round(6.0 / (a * cell.des.dt) / 20.0))))
    if cell.smoke:
        o["steps"] = 20
    return o


def run_hk(cell, seed, lr, log):
    import jf
    from hyperkkl.config import deep_merge
    base = get_p1(cell, seed, log)
    cfg = copy.deepcopy(cell.cfg); cfg["seed"] = seed
    if cell.smoke:
        cfg = deep_merge(cfg, {"phase2": {"steps": 20, "val_every": 10}})
    opts = p2_opts(cell, lr)
    torch.manual_seed(seed); np.random.seed(seed)
    rb = jf.phase2_jf(cfg, cell.data, cell.sc, cell.des, base, cell.dev, seed, opts, log, {})
    K = constants(cell)
    probe = {"jump": {"w": 0.0, "lags": K["lags"]}, "sens": {"wc": 0.0, "wz": 0.0, "eps": 0.05, "z_noise": 0.5, "splice": True}}
    pen = jf.penalty_values(cfg, cell.data, cell.sc, cell.des, rb, cell.dev, seed, probe)
    w = lambda rho, term: rho * pen["fit"] / term if term > 1e-12 * max(pen["fit"], 1e-30) else 0.0
    # guard only: a penalty term that is exactly 0 gets weight 0. (Family A: the slope term perturbs the context
    # directly, so it is small but not 0 and the per-plant rule applies unchanged.)
    mods = {"jump": {"w": w(C5["jump"]["rho"], pen["jump"]), "lags": K["lags"]},
            "sens": {"wc": w(C5["sens"]["rho"], pen["sens_c"]), "wz": w(C5["sens"]["rho"], pen["sens_z"]),
                     "eps": 0.05, "z_noise": 0.5, "splice": True},
            "specnorm": {"gamma": C5["specnorm"]["kappa"] * float(np.median(jf.head_norms(rb["cond"])))}}
    log(f"  C5 resolved: {mods} (operating point {pen}, constants {K})")
    torch.manual_seed(seed); np.random.seed(seed)
    t0 = time.time()
    rc = jf.phase2_jf(cfg, cell.data, cell.sc, cell.des, base, cell.dev, seed, opts, log, mods)
    c5_wall = time.time() - t0
    obs = {"hk_base": jf.make_observer("hk_base", cell.des, base, rb, cell.dev),
           "hyperkkl": jf.make_observer("hyperkkl", cell.des, base, rc, cell.dev)}
    meta = {"hk_base": {"lr": lr, "n_params": rb["trainable_total"], "wall_s": rb["wall_s"], "val_loss": rb["val"]},
            "hyperkkl": {"lr": lr, "n_params": rc["trainable_total"], "wall_s": rc["wall_s"], "val_loss": rc["val"],
                         "c5_mods": mods, "c5_extra_wall_s": c5_wall, "base_wall_s": rb["wall_s"]}}
    return obs, meta


def run_p1(cell, seed, log):
    from hyperkkl.conditioning import ConditionedDecoder
    from hyperkkl.observers import KKLObserver, LatentStep
    t0 = time.time()
    base = get_p1(cell, seed, log, train=True)
    obs = {"auto": KKLObserver("auto", LatentStep(cell.des, cell.dev), ConditionedDecoder(base["T0inv"], "none", 0, 0),
                               z0_map=base["T0"])}
    return obs, {"auto": {"phase1_wall_s": time.time() - t0}}


def run_cond(cell, seed, kind, lr, log):
    from hyperkkl.config import deep_merge
    from hyperkkl.conditioning import ConditionedDecoder
    from hyperkkl.observers import KKLObserver, LatentStep
    from hyperkkl.train import phase2
    base = get_p1(cell, seed, log)
    cfg = deep_merge(cell.cfg, {"phase2": {"lr": float(lr)}})
    cfg["seed"] = seed
    if cell.smoke:
        cfg = deep_merge(cfg, {"phase2": {"steps": 20, "val_every": 10}})
    torch.manual_seed(seed); np.random.seed(seed)
    r = phase2(kind, cfg, cell.data, cell.sc, cell.des, base, cell.dev, seed, log)
    lat = LatentStep(cell.des, cell.dev)
    if kind == "latent":
        ob = KKLObserver(kind, lat, ConditionedDecoder(base["T0inv"], "none", 0, 0), ctx=r["ctx"], injection=r["cond"],
                         z0_map=base["T0"])
    else:
        ob = KKLObserver(kind, lat, r["cond"], ctx=r["ctx"], z0_map=base["T0"])
    return {kind: ob}, {kind: {"lr": lr, "n_params": r.get("n_params"), "val_loss": r.get("val"), "wall_s": r.get("wall_s")}}


def run_peer(cell, seed, m, lr, log):
    import peers
    base = get_p1(cell, seed, log)
    cfg = copy.deepcopy(cell.cfg); cfg["seed"] = seed
    opts = p2_opts(cell, 1e-2)
    bud = peers.hk_budget(cfg, cell.data, cell.des, base, opts)
    steps = 20 if cell.smoke else PEER_STEPS
    torch.manual_seed(seed); np.random.seed(seed)
    if m == "nlox_m":
        o = dict(FAIR, steps=steps, lr=float(lr), budget=bud["total"])
        if cell.smoke:
            o["val_every"] = 10
        r = peers.train_nlox(cfg, cell.data, cell.sc, cell.des, cell.dev, seed, o, log)
        obs = {m: peers.nlox_observer(r, cell.des, cell.dev, m)}
        info = {k: r[k] for k in ("n_params", "widths", "val", "wall_s")}
    else:
        o = dict(FAIR, steps=steps, lr=float(lr), budget=bud["total"], huber=0.1)
        if cell.smoke:
            o["val_every"] = 10
        r = peers.train_dkkl(cfg, cell.data, cell.sc, cell.des, cell.dev, seed, o, log, inj=(m == "dkkl_inj_m"))
        obs = peers.dkkl_observers(r, m)
        info = {"n_params": r["n_params"], "width": r["width"], "val": r["val"], "wall_s": r["wall_s"]}
    info.update(lr=lr, steps=steps, hk_budget=bud)
    return obs, {k: info for k in obs}


# ============================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("family"); ap.add_argument("plant"); ap.add_argument("variant"); ap.add_argument("method")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tag = a.method + (f"_lr{a.lr:g}" if a.lr is not None else "") + ("_smoke" if a.smoke else "")
    out = env.RUNS / a.family / a.plant / a.variant / tag
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"seed{a.seed}.json"
    logf = open(out / f"seed{a.seed}.log", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    if f.exists():
        log("done already"); return
    try:
        t0 = time.time()
        cell = Cell(a.family, a.plant, a.variant, dev, a.smoke)
        log(f"job {a.family}/{a.plant}/{a.variant} {tag} seed {a.seed}: train {len(cell.data['train'])} "
            f"val {len(cell.data['val'])} N {cell.des.N} dt {cell.des.dt:.4g}")
        m = a.method
        if m == "filters":
            obs, meta = run_filters(cell, a.seed, a.lr, log)
        elif m in ("ude1", "ude32", "node32"):
            obs, meta = run_dyn(cell, a.seed, m, log)
        elif m in ("gru", "ncde"):
            obs, meta = run_seq(cell, a.seed, m, log)
        elif m == "p1":
            obs, meta = run_p1(cell, a.seed, log)
        elif m == "hk":
            obs, meta = run_hk(cell, a.seed, a.lr, log)
        elif m in ("latent", "concat", "film"):
            obs, meta = run_cond(cell, a.seed, m, a.lr, log)
        elif m in ("nlox_m", "dkkl_m", "dkkl_inj_m"):
            obs, meta = run_peer(cell, a.seed, m, a.lr, log)
        else:
            raise KeyError(m)
        t_train = time.time() - t0
        log(f"trained in {t_train:.0f}s; evaluating {list(obs)}")
        res = cell.evaluate(obs, log)
        payload = {"family": a.family, "plant": a.plant, "variant": a.variant, "method": m, "tag": tag, "seed": a.seed,
                   "lr": a.lr, "train_s": t_train, "wall_s": time.time() - t0, "meta": meta,
                   "observers": {n: {"blocks": res[n], "val_score": cell.val_score(res[n])} for n in res},
                   "time": time.strftime("%Y-%m-%d %H:%M:%S"), "n_train": len(cell.data["train"])}
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, default=_jd))
        os.replace(tmp, f)
        log(f"done in {time.time() - t0:.0f}s")
    except Exception:
        (out / f"seed{a.seed}.error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
