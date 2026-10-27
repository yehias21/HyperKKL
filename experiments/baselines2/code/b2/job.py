"""One baselines2 job = (node, system, seed, baseline): lr grid -> selection on validation -> evaluation.

    python -m b2.job --node C3-partial --system vdp --seed 0 --baseline gru140 [--smoke]

Output: experiments/baselines2/runs[/smoke]/<node>/<system>/seed<k>/<baseline>.json (+ log.txt) and
the selected model(s) in experiments/baselines2/models/... (state dicts, for the consolidated re-run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback

import numpy as np
import torch

from . import HERE, LIVE
from hyperkkl import baselines_data as BD
from hyperkkl import data as Dm
from hyperkkl.config import deep_merge
from hyperkkl.evaluate import build_observers, evaluate, initial_guess
from hyperkkl.job import _json_default, tune_oracle
from hyperkkl.nets import n_params
from hyperkkl.observers import EKF, UKF
from hyperkkl.train import nlox as train_nlox
from hyperkkl import systems_extra  # noqa: F401  (battery, network)
try:
    from hyperkkl import systems_bounded  # noqa: F401  (cartpole_b)
except ImportError:                           # pragma: no cover
    pass

from . import models as M
from . import train as T

LRS = [3e-4, 1e-3, 3e-3, 1e-2]
STEPS = 3000
SPECS = {
    "gru140": ("gru", 140_000), "gru500": ("gru", 500_000),
    "ncde140": ("ncde", 140_000), "ncde500": ("ncde", 500_000),
    "node5k": ("node", 64), "node140k": ("node", 140_000),
    "ude5k": ("ude", 64), "ude140k": ("ude", 140_000),
    "ude5k1": ("ude1", 64),          # post-hoc control (addendum 2): one-step objective, same lr grid

    "nlox140": ("nlox", 140_000), "nlox500": ("nlox", 500_000),
    "greybox": ("greybox", None),
}
GREYBOX = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
NCDE_DZ = 32
KEEP = ("nrmse", "nrmse_unmeasured", "nrmse_measured", "diverged")


def load_cfg(node, system, seed):
    src = LIVE / "experiments" / "tree" / node / "runs" / system / f"seed{min(seed, 0) if seed > 2 else seed}" / "results.json"
    cfg = json.load(open(src))["config"]
    cfg["seed"] = seed
    cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
    return cfg, str(src)


def nlox_widths(cfg, nz, nx, budget):
    """Scale nlox.width and nlox.dec_width by a common factor so the NLOX parameter count ~ budget."""
    from hyperkkl.nets import MLP
    w0, d0, depth = int(cfg["nlox"]["width"]), int(cfg["nlox"]["dec_width"]), int(cfg["nlox"]["dec_depth"])

    def count(s):
        w, d = max(2, round(w0 * s)), max(2, round(d0 * s))
        return n_params(MLP(nz, nz, w, 2), False) + n_params(MLP(nz, nx, d, depth), False), w, d
    best = min((abs(count(s)[0] - budget), s) for s in np.arange(0.5, 6.0, 0.01))[1]
    return count(best)


class Job:
    def __init__(self, node, system, seed, baseline, device, smoke=False):
        self.node, self.system, self.seed, self.baseline, self.device, self.smoke = \
            node, system, seed, baseline, device, smoke
        self.cfg, self.cfg_src = load_cfg(node, system, seed)
        self.out = HERE / "runs" / ("smoke" if smoke else "") / node / system / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.mdir = HERE / "models" / ("smoke" if smoke else "") / node / system / f"seed{seed}"
        self.mdir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "log.txt", "a")
        torch.manual_seed(seed); np.random.seed(seed)
        self.scales = Dm.system_scales(self.cfg, device)
        self.des = Dm.make_design(self.cfg["system"])
        self.data = Dm.training_data(self.cfg, seed, device)
        sc = self.cfg["system"]
        self.model = Dm.model_plant(sc)
        _tgt = self.model
        while _tgt is not None and not hasattr(_tgt, "laplacian"):    # unwrap actuator / KnownPart
            _tgt = getattr(_tgt, "base", None)
        if hasattr(_tgt, "laplacian"):                # network: cache the (constant) Laplacian per
            _lap, _cache = _tgt.laplacian, {}        # dtype/device -> same values, CUDA-graph safe

            def laplacian(dtype=torch.float64, device=None, edges=None, _lap=_lap, _cache=_cache):
                k = (dtype, str(device), edges)
                if k not in _cache:
                    _cache[k] = _lap(dtype, device, edges)
                return _cache[k]
            _tgt.laplacian = laplacian
        self.sensors = list(self.model.sensors)
        self.nx, self.nu, self.ny = self.model.nx, self.model.nu, self.model.ny
        dcfg = self.cfg["data"]
        self.q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(self.scales["x_std"])
        self.r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(self.scales["y_std"])
        box = np.asarray(sc["ic_box"], float)
        self.p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
        self.u_scale = float(sc.get("u_scale", 1.0))
        self.va = self.data["val"].to(device)
        self.x0v = initial_guess(self.cfg, len(self.va), "center", 0, device)
        self.steps = 20 if smoke else STEPS
        self.lrs = LRS[1:2] if smoke else LRS

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def norm(self):
        xm = np.asarray(self.scales["x_mean"])
        return M._Norm(self.u_scale, xm[self.sensors], self.scales["y_std"], xm, self.scales["x_std"])

    def _select(self, cands):
        return min(range(len(cands)), key=lambda i: cands[i]["val_nrmse"] + cands[i]["val_diverged"])

    # ------------------------------------------------------------------ families
    def run_seq(self, kind, budget):
        tr = self.data["train"].to(self.device)
        cands, best_net = [], None
        for lr in self.lrs:
            torch.manual_seed(self.seed)
            if kind == "gru":
                H = M.solve_gru(self.nu, self.ny, self.nx, budget)
                net = M.GRUNet(self.nu, self.ny, self.nx, H, 2, self.norm())
                arch = {"hidden": H, "layers": 2}
            else:
                W = M.solve_ncde(self.nu, self.ny, self.nx, NCDE_DZ, budget)
                tref = float(self.cfg["system"].get("t_ref", 1.0))
                net = M.NCDENet(self.nu, self.ny, self.nx, NCDE_DZ, W, self.norm(), self.des.dt / tref)
                arch = {"dz": NCDE_DZ, "width": W, "solver": "rk4, one step per sample interval",
                        "path": "linear interpolation of (t/t_ref, u/u_scale, (y-ybar)/y_std)"}
            self.log(f"  {kind} lr {lr:g}: {sum(p.numel() for p in net.parameters())} params {arch}")
            info = T.train_seq(net, tr, self.va, self.steps, lr, self.device, self.seed, self.log,
                               val_every=10 if self.smoke else 250)
            ob = M.SeqObserver(self.baseline, net)
            dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
            self.log(f"  {kind} lr {lr:g}: val loss {info['val_loss']:.4g} deployed val nrmse "
                     f"{dv['val_nrmse']:.4f} div {dv['val_diverged']:.3f} ({info['wall_s']:.0f}s)")
            cands.append({"lr": lr, **dv, "val_loss": info["val_loss"], "wall_s": info["wall_s"],
                          "n_bad_steps": info["n_bad_steps"], "hist": info["hist"]})
            if best_net is None or self._select(cands) == len(cands) - 1:
                best_net = net
        i = self._select(cands)
        torch.save({"state": best_net.state_dict(), "arch": arch, "lr": cands[i]["lr"]},
                   self.mdir / f"{self.baseline}.pt")
        npar = sum(p.numel() for p in best_net.parameters())
        obs = {self.baseline: M.SeqObserver(self.baseline, best_net)}
        meta = {self.baseline: {"lr": cands[i]["lr"], "n_params": npar, "arch": arch, "candidates": cands}}
        return obs, meta

    def run_dyn(self, kind, size):
        hybrid = kind in ("ude", "ude1")
        Kroll = 1 if kind == "ude1" else 32
        fm = self.model if hybrid else None
        sets = [self.data["train"], self.data["p1"]]
        W = int(size) if size < 1000 else M.solve_mlp(self.nx, self.nu, size)
        tm, ts = T.target_stats(self.model if hybrid else None, sets, self.des.dt, self.device, self.seed)
        cands = {"ekf": [], "ukf": []}
        best = {"ekf": None, "ukf": None}
        for lr in self.lrs:
            torch.manual_seed(self.seed)
            net = BD.ResidualMLP(self.nx, self.nu, W, self.scales.get("x_mean"), self.scales["x_std"],
                                 self.u_scale, tm, ts)
            self.log(f"  {kind} width {W} lr {lr:g}: {sum(p.numel() for p in net.parameters())} params")
            info = T.train_dyn(fm, net, sets, self.va, self.des.dt, self.steps, lr, self.device, self.seed,
                               self.log, K=Kroll, val_every=10 if self.smoke else 100,
                               warm_steps=5 if self.smoke else 300,
                               use_graphs=True)
            gm = M.deployed_dynamics(self.model, net, hybrid)
            for fk, F in (("ekf", EKF), ("ukf", UKF)):
                ob = F(gm, self.des.dt, self.q, self.r, self.p0)
                dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
                self.log(f"  {kind}+{fk} lr {lr:g}: rollout val {info['val_loss']:.4g} deployed val nrmse "
                         f"{dv['val_nrmse']:.4f} div {dv['val_diverged']:.3f}")
                cands[fk].append({"lr": lr, **dv, "val_loss": info["val_loss"], "wall_s": info["wall_s"],
                                  "n_bad_steps": info["n_bad_steps"], "hist": info["hist"]})
                if best[fk] is None or self._select(cands[fk]) == len(cands[fk]) - 1:
                    best[fk] = net
        obs, meta = {}, {}
        for fk, F in (("ekf", EKF), ("ukf", UKF)):
            name = f"{self.baseline}_{fk}"
            i = self._select(cands[fk])
            net = best[fk]
            torch.save({"state": net.state_dict(), "width": W, "hybrid": hybrid, "lr": cands[fk][i]["lr"],
                        "t_mean": tm, "t_std": ts}, self.mdir / f"{name}.pt")
            ob = F(M.deployed_dynamics(self.model, net, hybrid), self.des.dt, self.q, self.r, self.p0)
            ob.name = name
            obs[name] = ob
            meta[name] = {"lr": cands[fk][i]["lr"], "n_params": sum(p.numel() for p in net.parameters()),
                          "arch": {"width": W, "hidden_layers": 2, "act": "silu", "hybrid": hybrid,
                                   "K": Kroll, "filter": fk}, "candidates": cands[fk]}
        return obs, meta

    def run_nlox(self, budget):
        nz = self.des.A.shape[0]
        npar, w, d = nlox_widths(self.cfg, nz, self.nx, budget)
        cands, best_res = [], None
        for lr in self.lrs:
            c2 = deep_merge(self.cfg, {"nlox": {"width": w, "dec_width": d, "lr": lr, "steps": self.steps,
                                                "val_every": 10 if self.smoke else 250}})
            self.log(f"  nlox width {w} dec_width {d} (~{npar} params) lr {lr:g}")
            res = train_nlox(c2, self.data, self.scales, self.des, self.device, self.seed, self.log)
            ob = build_observers(deep_merge(self.cfg, {"baselines": ["nlox"]}), self.des, self.scales, None, {},
                                 res, self.device)["nlox"]
            dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
            self.log(f"  nlox lr {lr:g}: val {res['val']:.4g} deployed val nrmse {dv['val_nrmse']:.4f} "
                     f"div {dv['val_diverged']:.3f} ({res['wall_s']:.0f}s)")
            cands.append({"lr": lr, **dv, "val_loss": res["val"], "wall_s": res["wall_s"], "hist": res["hist"]})
            if best_res is None or self._select(cands) == len(cands) - 1:
                best_res = res
        i = self._select(cands)
        torch.save({"omega": best_res["omega"].state_dict(), "dec": best_res["dec"].state_dict(),
                    "width": w, "dec_width": d, "lr": cands[i]["lr"]}, self.mdir / f"{self.baseline}.pt")
        ob = build_observers(deep_merge(self.cfg, {"baselines": ["nlox"]}), self.des, self.scales, None, {},
                             best_res, self.device)["nlox"]
        ob.name = self.baseline
        meta = {self.baseline: {"lr": cands[i]["lr"], "n_params": int(best_res["n_params"]),
                                "arch": {"width": w, "dec_width": d, "dec_depth": int(self.cfg["nlox"]["dec_depth"])},
                                "candidates": cands}}
        return {self.baseline: ob}, meta

    def run_greybox(self):
        c2 = deep_merge(self.cfg, {"baselines": GREYBOX})
        c2["eval"]["oracle_tuning"], ot = tune_oracle(c2, self.des, self.scales, self.data["val"], self.device,
                                                      self.log)
        fits = BD.fit_all(c2, self.data, self.scales, self.des, self.device, self.seed, self.log)
        obs = build_observers(c2, self.des, self.scales, None, {}, None, self.device, fits)
        meta = {b: {"n_params": fits["info"][b]["n_params"], "fit": {k: v for k, v in fits["info"][b].items()
                                                                      if k != "hist"},
                    "oracle_tuning": ot if b == "greybox_ekf_oracle" else None} for b in GREYBOX}
        self.cfg = c2
        return obs, meta

    # ------------------------------------------------------------------ main
    def run(self):
        t0 = time.time()
        kind, size = SPECS[self.baseline]
        self.log(f"job {self.node}/{self.system}/seed{self.seed} {self.baseline} ({kind}, {size}) "
                 f"device {self.device} smoke {self.smoke} cfg {self.cfg_src}")
        self.log(f"  data: train {len(self.data['train'])} p1 {len(self.data['p1'])} val {len(self.va)} "
                 f"N {self.des.N} dt {self.des.dt:.4g} burn {self.data['train'].burn}")
        if kind in ("gru", "ncde"):
            obs, meta = self.run_seq(kind, size)
        elif kind in ("node", "ude", "ude1"):
            obs, meta = self.run_dyn(kind, size)
        elif kind == "nlox":
            obs, meta = self.run_nlox(size)
        else:
            obs, meta = self.run_greybox()
        t_train = time.time() - t0
        cfg = self.cfg
        if self.smoke:
            cfg = deep_merge(cfg, {"eval": {"tiers": ["sequence_ood"]}})
        timing = {}
        res = evaluate(cfg, self.des, self.scales, obs, self.device, self.log, timing=timing)
        results = {}
        for tier, cs in res.items():
            blk = cs["clean"]
            results[tier] = {m: {**{k: v[k] for k in KEEP if k in v},
                                 "scorecard": {k: blk["scorecard"][m].get(k) for k in
                                               ("nrmse_mean", "nrmse_median", "nrmse_p95", "div_rate",
                                                "nrmse_unmeasured", "nrmse_state")}}
                             for m, v in blk["methods"].items()}
        payload = {"node": self.node, "system": self.system, "seed": self.seed, "baseline": self.baseline,
                   "spec": SPECS[self.baseline], "lrs": self.lrs, "steps": self.steps, "smoke": self.smoke,
                   "methods": meta, "results": results, "eval_timing": timing, "train_s": t_train,
                   "wall_s": time.time() - t0, "cfg_src": self.cfg_src, "device": self.device,
                   "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = self.out / f"{self.baseline}.json.tmp"
        tmp.write_text(json.dumps(payload, default=_json_default))
        tmp.rename(self.out / f"{self.baseline}.json")
        summ = "  ".join(f"{t[:3]}:{m}={np.mean(v['nrmse']):.4f}" for t, r in results.items() for m, v in r.items())
        self.log(f"done in {time.time() - t0:.0f}s  {summ}")
        return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", required=True, choices=sorted(SPECS))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    try:
        Job(a.node, a.system, a.seed, a.baseline, a.device, a.smoke).run()
    except Exception:
        d = HERE / "runs" / ("smoke" if a.smoke else "") / a.node / a.system / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a.baseline}.error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
