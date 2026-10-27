"""One case-study job = (case, knowledge level, seed, family): train what the family needs, evaluate on every
tier (clean) and the robustness conditions (id tier), save per-method results.

    python run_cs.py --case A --level realistic --seed 0 --family hk [--device cuda:0] [--smoke]

families
  hk       HyperKKL arms: A3 (primary: SOAP + unfrozen base decoder + z-window), A0 (old default),
           A4+D3 (secondary; realistic level only). Recipes = experiments/beat_greybox (code/pkg/beat.py).
  filters  EKF, UKF with the practitioner's model (data.model_plant), climatology (training mean).
  greybox  grey-box EKF / UKF / EKF-oracle: practitioner's model + one-step residual MLP (baselines_data).
  ude5k    UDE + EKF / UKF (practitioner's model + trajectory-trained residual, b2).
  node5k   NODE + EKF / UKF (model-free learned dynamics, b2).
  gru140   GRU observer (~140k parameters, b2).
  nlox140  NLOX (capacity-matched ~140k, b2).
Output: runs/<case>/<level>/seed<k>/<family>.json (+ log_<family>.txt); models under models/.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cs_common as C  # noqa: E402  (must be first)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.evaluate import build_observers, evaluate, initial_guess  # noqa: E402
from hyperkkl.job import tune_oracle  # noqa: E402
from hyperkkl.nets import n_params  # noqa: E402
from hyperkkl.observers import EKF, UKF, Observer  # noqa: E402
from hyperkkl.train import nlox as train_nlox, phase1  # noqa: E402

from b2cs import models as M  # noqa: E402
from b2cs import train as T  # noqa: E402

FAMILIES = ["hk", "filters", "greybox", "ude5k", "node5k", "gru140", "nlox140"]
LRS = [1e-3, 1e-2]          # reduced grid of baselines2 ({3e-4, 1e-3, 3e-3, 1e-2}); selection on validation
STEPS = 3000
KEEP = ("nrmse", "nrmse_unmeasured", "nrmse_measured", "diverged", "t_conv_steps")

# ---------------------------------------------------------------------------- HyperKKL arms (beat_greybox)
S1 = {"optimizer": "soap", "lr": 1e-2}
R1 = {"dec_width": 256}
ARM_TAGS = {"A0": set(), "A3": {"S", "P", "H"}, "A4+D3": {"S", "P", "H", "R", "D3"}}


def components(tags):
    p1, p2 = {}, {}
    if "S" in tags:
        p1.update(S1); p2.update(optimizer="soap", lr=1e-2)
    if "R" in tags:
        p1.update(R1); p2.update(budget=280000, steps=6000)
    if "P" in tags:
        p2["base_lr_factor"] = 0.1
    if "H" in tags:
        p2["window"] = 20
    if "D3" in tags:
        p2["slow_factor"] = 5.0
    return {"p1": p1, "p2": p2}


def stride_rule(cfg, des) -> int:
    a = float(cfg["system"]["alpha"]) * float(cfg["system"].get("alpha_scale", 1.0))
    return max(1, int(round(6.0 / (a * des.dt) / 20.0)))


class OmegaInj:
    """Picklable NLOX injection omega(z) u (evaluate.build_observers uses a lambda)."""

    def __init__(self, om):
        self.om = om

    def __call__(self, z, c):
        return self.om(z, c)


class Climatology(Observer):
    """x_hat = the training-set mean state, ignoring y (the no-information reference)."""
    name = "climatology"

    def __init__(self, mean):
        self.mean = torch.as_tensor(np.asarray(mean, float), dtype=torch.float32)

    def reset(self, B, x0_hat, u0):
        self.x = self.mean.to(x0_hat.device, x0_hat.dtype).expand(B, -1).clone()

    def estimate(self):
        return self.x

    def advance(self, y_k, u_k, y_k1, u_k1):
        pass


def nlox_widths(cfg, nz, nx, budget):
    """As baselines2 job.nlox_widths: scale nlox.width and dec_width by a common factor -> ~budget params."""
    from hyperkkl.nets import MLP
    w0, d0, depth = int(cfg["nlox"]["width"]), int(cfg["nlox"]["dec_width"]), int(cfg["nlox"]["dec_depth"])

    def count(s):
        w, d = max(2, round(w0 * s)), max(2, round(d0 * s))
        return n_params(MLP(nz, nz, w, 2), False) + n_params(MLP(nz, nx, d, depth), False), w, d
    best = min((abs(count(s)[0] - budget), s) for s in np.arange(0.5, 6.0, 0.01))[1]
    return count(best)


def nanmean_state(sets):
    xs = torch.cat([s.x.reshape(-1, s.x.shape[-1]) for s in sets])
    return torch.nanmean(xs, 0).tolist()


class Job:
    def __init__(self, case, level, seed, family, device, smoke=False):
        self.case, self.level, self.seed, self.family, self.device, self.smoke = case, level, seed, family, device, smoke
        cfg = C.case_cfg(case, level, seed, device)
        if smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40},
                                   "phase2": {"steps": 20, "val_every": 10},
                                   "eval": {"n_episodes": 8}})
        self.cfg = cfg
        tag = "smoke" if smoke else ""
        self.out = (C.RUNS / tag / case / level / f"seed{seed}") if smoke else C.run_dir(case, level, seed)
        self.out.mkdir(parents=True, exist_ok=True)
        self.mdir = (C.MODELS / tag / case / level / f"seed{seed}") if smoke else C.model_dir(case, level, seed)
        self.mdir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / f"log_{family}.txt", "a")
        torch.manual_seed(seed); np.random.seed(seed)
        self.scales = Dm.system_scales(cfg, device)
        self.des = Dm.make_design(cfg["system"])
        self.data = Dm.training_data(cfg, seed, device)
        self.data_dense = {k: (Dm.interp_labels(v) if k != "coll" else v) for k, v in self.data.items()}
        self.sparse = self.data_dense["train"] is not self.data["train"]
        sc = cfg["system"]
        self.model = Dm.model_plant(sc)
        self.sensors = list(self.model.sensors)
        self.nx, self.nu, self.ny = self.model.nx, self.model.nu, self.model.ny
        dcfg = cfg["data"]
        self.q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(self.scales["x_std"])
        self.r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(self.scales["y_std"]) \
            * np.asarray(dcfg["sensor"].get("noise_mult", [1.0] * self.ny), float)
        box = np.asarray(sc["ic_box"], float)
        self.p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
        self.u_scale = float(sc.get("u_scale", 1.0))
        self.va = self.data["val"].to(device)
        self.x0v = initial_guess(cfg, len(self.va), "center", 0, device)
        self.steps = 20 if smoke else STEPS
        self.lrs = LRS[:1] if smoke else LRS
        self.mh = module_hash(MODEL_MODULES)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def norm(self):
        xm = np.asarray(self.scales["x_mean"])
        return M._Norm(self.u_scale, xm[self.sensors], self.scales["y_std"], xm, self.scales["x_std"])

    def _select(self, cands):
        return min(range(len(cands)), key=lambda i: cands[i]["val_nrmse"] + cands[i]["val_diverged"])

    # ------------------------------------------------------------------------------ HyperKKL
    def run_hk(self):
        arms = ["A0", "A3"] + (["A4+D3"] if self.level == "realistic" else [])
        obs, meta = {}, {}
        des, data, sc = self.des, self.data, self.scales
        for a in arms:
            t0 = time.time()
            spec = components(ARM_TAGS[a])
            if self.smoke and "steps" in spec["p2"]:
                spec["p2"]["steps"] = 20
            if "window" in spec["p2"]:
                spec["p2"]["stride"] = stride_rule(self.cfg, des)
            cfg = deep_merge(self.cfg, {"phase1": spec["p1"]})
            k1 = digest(cfg, "system", "data", "phase1", extra=[self.seed, self.mh, "cs"])
            base = Cm.cached(self.mdir / f"hk_p1_{k1}.pt",
                             lambda: phase1(cfg, data, sc, des, self.device, self.seed, self.log))
            base = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in base.items()}
            k2 = digest(cfg, "system", "data", "phase1", "phase2", extra=[self.seed, self.mh, "cs", spec["p2"]])
            r = Cm.cached(self.mdir / f"hk_p2_{a}_{k2}.pt",
                          lambda: BG.phase2_bg("lora", cfg, data, sc, des, base, self.device, self.seed,
                                               spec["p2"], self.log))
            r = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in r.items()}
            name = f"hk_{a}"
            ob = BG.make_observer(name, des, base, r, self.device)
            v = BG.validation_nrmse(self.cfg, ob, data["val"], sc, self.device)
            obs[name] = ob
            meta[name] = {"arm": a, "spec": spec, "val": v, "diag": base["diag"],
                          "params": {"encoder": n_params(base["T0"], False), "base_decoder": n_params(base["T0inv"], False),
                                     "conditioner": r["n_params"], "context": r["ctx_params"],
                                     "history": r["hist_params"], "trainable_p2": r["trainable_total"]},
                          "phase1_wall_s": base["wall_s"], "phase2_wall_s": r["wall_s"], "train_s": time.time() - t0}
            self.log(f"  {name}: val nrmse {v['val_nrmse']:.4f} div {v['val_diverged']:.3f} "
                     f"eps_dec {base['diag']['eps_dec']:.3g} ({time.time() - t0:.0f}s)")
        return obs, meta

    # ------------------------------------------------------------------------------ filters
    def run_filters(self):
        c2 = deep_merge(self.cfg, {"baselines": ["ekf", "ukf", "ekf_oracle"]})
        # ekf_oracle: the practitioner's EKF with Q (model-error part) and R inflation tuned on validation
        c2["eval"]["oracle_tuning"], ot = tune_oracle(c2, self.des, self.scales, self.data["val"], self.device, self.log)
        obs = build_observers(c2, self.des, self.scales, None, {}, None, self.device)
        obs["climatology"] = Climatology(nanmean_state([self.data["train"], self.data["p1"]]))
        meta = {k: {"n_params": 0, "model": "practitioner (data.model_plant)"} for k in ("ekf", "ukf", "ekf_oracle")}
        meta["ekf_oracle"]["oracle_tuning"] = ot
        self.cfg = c2
        meta["climatology"] = {"n_params": 0}
        return obs, meta

    # ------------------------------------------------------------------------------ grey-box
    def run_greybox(self):
        names = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]
        c2 = deep_merge(self.cfg, {"baselines": names})
        c2["eval"]["oracle_tuning"], ot = tune_oracle(c2, self.des, self.scales, self.data["val"], self.device, self.log)
        # one-step residual regression needs a full state at every sample: interpolated assay labels
        fits = BD.fit_all(c2, self.data_dense, self.scales, self.des, self.device, self.seed, self.log)
        obs = build_observers(c2, self.des, self.scales, None, {}, None, self.device, fits)
        meta = {b: {"n_params": fits["info"][b]["n_params"],
                    "fit": {k: v for k, v in fits["info"][b].items() if k != "hist"},
                    "labels": "linear interpolation of assays" if self.sparse else "dense",
                    "oracle_tuning": ot if b == "greybox_ekf_oracle" else None} for b in names}
        self.cfg = c2
        return obs, meta

    # ------------------------------------------------------------------------------ b2 families
    def run_seq(self, budget):
        tr = self.data["train"].to(self.device)
        cands, best_net = [], None
        H = M.solve_gru(self.nu, self.ny, self.nx, budget)
        for lr in self.lrs:
            torch.manual_seed(self.seed)
            net = M.GRUNet(self.nu, self.ny, self.nx, H, 2, self.norm())
            info = T.train_seq(net, tr, self.va, self.steps, lr, self.device, self.seed, self.log,
                               val_every=10 if self.smoke else 250)
            ob = M.SeqObserver(self.family, net)
            dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
            self.log(f"  gru lr {lr:g}: val loss {info['val_loss']:.4g} deployed val nrmse {dv['val_nrmse']:.4f}")
            cands.append({"lr": lr, **dv, "val_loss": info["val_loss"], "wall_s": info["wall_s"]})
            if best_net is None or self._select(cands) == len(cands) - 1:
                best_net = net
        i = self._select(cands)
        torch.save({"state": best_net.state_dict(), "hidden": H, "lr": cands[i]["lr"]}, self.mdir / f"{self.family}.pt")
        return ({self.family: M.SeqObserver(self.family, best_net)},
                {self.family: {"lr": cands[i]["lr"], "n_params": sum(p.numel() for p in best_net.parameters()),
                               "arch": {"hidden": H, "layers": 2}, "candidates": cands,
                               "labels": "masked (sparse)" if self.sparse else "dense"}})

    def run_dyn(self, hybrid):
        Kroll = 32
        K_sparse = int(np.ceil((self.cfg["data"]["labels"]["every"] + self.cfg["data"]["labels"]["jitter"])
                               / self.des.dt)) if self.sparse else None
        fm = self.model if hybrid else None
        sets = [self.data["train"], self.data["p1"]]
        W = 64
        tm, ts = T.target_stats(fm, [self.data_dense["train"], self.data_dense["p1"]], self.des.dt, self.device, self.seed)
        cands, best = {"ekf": [], "ukf": []}, {"ekf": None, "ukf": None}
        for lr in self.lrs:
            torch.manual_seed(self.seed)
            net = BD.ResidualMLP(self.nx, self.nu, W, self.scales.get("x_mean"), self.scales["x_std"],
                                 self.u_scale, tm, ts)
            info = T.train_dyn(fm, net, sets, self.va, self.des.dt, self.steps, lr, self.device, self.seed,
                               self.log, K=Kroll, val_every=10 if self.smoke else 100,
                               warm_steps=5 if self.smoke else 300, use_graphs=True, K_sparse=K_sparse,
                               batch=256 if self.sparse else 1024, n_val=1024 if self.sparse else 4096)
            gm = M.deployed_dynamics(self.model, net, hybrid)
            for fk, F in (("ekf", EKF), ("ukf", UKF)):
                ob = F(gm, self.des.dt, self.q, self.r, self.p0)
                dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
                self.log(f"  {'ude' if hybrid else 'node'}+{fk} lr {lr:g}: rollout val {info['val_loss']:.4g} "
                         f"deployed val nrmse {dv['val_nrmse']:.4f} div {dv['val_diverged']:.3f}")
                cands[fk].append({"lr": lr, **dv, "val_loss": info["val_loss"], "wall_s": info["wall_s"]})
                if best[fk] is None or self._select(cands[fk]) == len(cands[fk]) - 1:
                    best[fk] = net
        obs, meta = {}, {}
        for fk, F in (("ekf", EKF), ("ukf", UKF)):
            name = f"{self.family}_{fk}"
            i = self._select(cands[fk])
            net = best[fk]
            torch.save({"state": net.state_dict(), "width": W, "hybrid": hybrid, "lr": cands[fk][i]["lr"],
                        "t_mean": tm, "t_std": ts}, self.mdir / f"{name}.pt")
            ob = F(M.deployed_dynamics(self.model, net, hybrid), self.des.dt, self.q, self.r, self.p0)
            ob.name = name
            obs[name] = ob
            meta[name] = {"lr": cands[fk][i]["lr"], "n_params": sum(p.numel() for p in net.parameters()),
                          "arch": {"width": W, "hybrid": hybrid, "K": K_sparse or Kroll, "filter": fk},
                          "candidates": cands[fk], "labels": "masked windows from assays" if self.sparse else "dense"}
        return obs, meta

    def run_nlox(self, budget):
        nz = self.des.A.shape[0]
        npar, w, d = nlox_widths(self.cfg, nz, self.nx, budget)
        cands, best_res = [], None
        for lr in self.lrs:
            c2 = deep_merge(self.cfg, {"nlox": {"width": w, "dec_width": d, "lr": lr, "steps": self.steps,
                                                "val_every": 10 if self.smoke else 250}})
            res = train_nlox(c2, self.data, self.scales, self.des, self.device, self.seed, self.log)
            ob = build_observers(deep_merge(self.cfg, {"baselines": ["nlox"]}), self.des, self.scales, None, {},
                                 res, self.device)["nlox"]
            dv = T.deployed_val(ob, self.cfg, self.va, self.scales, self.device, self.x0v)
            self.log(f"  nlox lr {lr:g}: val {res['val']:.4g} deployed val nrmse {dv['val_nrmse']:.4f}")
            cands.append({"lr": lr, **dv, "val_loss": res["val"], "wall_s": res["wall_s"]})
            if best_res is None or self._select(cands) == len(cands) - 1:
                best_res = res
        i = self._select(cands)
        torch.save({"omega": best_res["omega"].state_dict(), "dec": best_res["dec"].state_dict(),
                    "width": w, "dec_width": d, "lr": cands[i]["lr"]}, self.mdir / f"{self.family}.pt")
        ob = build_observers(deep_merge(self.cfg, {"baselines": ["nlox"]}), self.des, self.scales, None, {},
                             best_res, self.device)["nlox"]
        ob.name = self.family
        ob.inj = OmegaInj(best_res["omega"])
        return ({self.family: ob}, {self.family: {"lr": cands[i]["lr"], "n_params": int(best_res["n_params"]),
                                                  "arch": {"width": w, "dec_width": d}, "candidates": cands}})

    # ------------------------------------------------------------------------------ main
    def run(self):
        t0 = time.time()
        f = self.family
        self.log(f"job {self.case}/{self.level}/seed{self.seed} {f} device {self.device} smoke {self.smoke}; "
                 f"data p1 {len(self.data['p1'])} train {len(self.data['train'])} val {len(self.va)} "
                 f"N {self.des.N} dt {self.des.dt:.4g} burn {self.data['train'].burn} sparse {self.sparse}")
        if f == "hk":
            obs, meta = self.run_hk()
        elif f == "filters":
            obs, meta = self.run_filters()
        elif f == "greybox":
            obs, meta = self.run_greybox()
        elif f in ("ude5k", "node5k"):
            obs, meta = self.run_dyn(hybrid=(f == "ude5k"))
        elif f == "gru140":
            obs, meta = self.run_seq(140_000)
        elif f == "nlox140":
            obs, meta = self.run_nlox(140_000)
        else:
            raise ValueError(f)
        t_train = time.time() - t0
        if self.seed == 0:                     # deployed observers (fresh) for the CPU latency benchmark
            import pickle
            keep = {}
            for k, v in obs.items():
                try:
                    pickle.dumps(v); keep[k] = v
                except Exception as e:
                    self.log(f"  observer {k} not picklable ({e!r}); latency from an equivalent observer")
            if keep:
                torch.save(keep, self.mdir / f"obs_{f}.pt")
        timing = {}
        res = evaluate(self.cfg, self.des, self.scales, obs, self.device, self.log, timing=timing)
        results = {}
        for tier, cs in res.items():
            results[tier] = {}
            for cname, blk in cs.items():
                results[tier][cname] = {
                    m: {**{k: v[k] for k in KEEP if k in v},
                        **{k: v[k] for k in v if k.startswith("nrmse_x")},
                        "scorecard": {k: blk["scorecard"][m].get(k) for k in blk["scorecard"][m]
                                      if not isinstance(blk["scorecard"][m].get(k), (list, dict))}}
                    for m, v in blk["methods"].items()}
        payload = {"case": self.case, "level": self.level, "seed": self.seed, "family": f, "smoke": self.smoke,
                   "methods": meta, "results": results, "eval_timing": timing, "train_s": t_train,
                   "wall_s": time.time() - t0, "config": self.cfg, "device": self.device,
                   "scales": {k: self.scales[k] for k in ("x_std", "y_std", "x_mean", "t_burn", "dt", "N")},
                   "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        C.save_json(self.out / f"{f}.json", payload)
        summ = "  ".join(f"{t[:3]}:{m}={np.mean(v['clean'][m]['nrmse_unmeasured']):.4f}"
                         for t, v in results.items() for m in v["clean"])
        self.log(f"done in {time.time() - t0:.0f}s  (unmeasured NRMSE) {summ}")
        return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(C.CASES))
    ap.add_argument("--level", required=True, choices=C.LEVELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    try:
        Job(a.case, a.level, a.seed, a.family, a.device, a.smoke).run()
    except Exception:
        d = (C.RUNS / "smoke" if a.smoke else C.RUNS) / a.case / a.level / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a.family}.error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
