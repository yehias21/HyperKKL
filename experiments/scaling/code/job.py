"""One scaling-study job = (n_x, seed, unit[, data_nx]).

    python code/job.py --nx 16 --seed 0 --unit kkl:partial [--data-nx 4] [--smoke] [--device cuda:0]

Units (PREREG.md section 4):
  kkl:<level>   HyperKKL A0 (lora, Adam phase 1+2), C1 alternatives concat and latent on the SAME
                A0 base at the SAME conditioner budget, HyperKKL A3 (SOAP phase 1+2, base decoder
                trainable at lr x0.1, strided z-window context), EKF and UKF on f_known
  gb:<level>    grey-box EKF / UKF (one-step residual, baselines_data.fit_all) and UDE + EKF / UKF
                (trajectory-trained residual, b2s.train.train_dyn K = 32)
  mf:<dataset>  model-free: GRU observer, NODE + EKF, NLOX   (datasets D0 = exact/partial data,
                D30 = off30 data, DR = realistic data)
  data:<dataset> CPU only: generate scales, training and evaluation episodes
Output: runs/[fixed4/]nx{nx}/seed{k}/{unit}.json (+ log), nothing else outside experiments/scaling.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback

import common as C  # noqa: F401  (must be first)
import numpy as np
import torch

from hyperkkl import baselines_data as BD
from hyperkkl import beat as BG
from hyperkkl import data as Dm
from hyperkkl.config import deep_merge
from hyperkkl.evaluate import build_observers, evaluate, initial_guess
from hyperkkl.job import _json_default
from hyperkkl.observers import EKF, UKF
from hyperkkl.train import nlox as train_nlox
from hyperkkl.train import phase1, phase2

import b2s.models as M
import b2s.train as T
import scaleplants as SP

LR = {"gru": 1e-2, "nlox": 3e-3, "ude": 1e-2, "node": 1e-2}      # baselines2 modal lr (PREREG 4)
GB_POOL_MAX = 4_000_000
KEEP = ("nrmse", "nrmse_unmeasured", "nrmse_measured", "diverged")


def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))


class Job:
    def __init__(self, nx, seed, unit, device, data_nx=None, smoke=False):
        self.nx, self.seed, self.unit, self.device, self.smoke = nx, seed, unit, device, smoke
        self.kind, _, self.arg = unit.partition(":")
        self.data_nx = data_nx
        level = self.arg if self.kind in ("kkl", "gb") else SP.DATASET_LEVEL.get(self.arg, "exact")
        self.level = level
        cfg = SP.make_cfg(nx, level, seed, data_nx)
        if smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 30, "steps_dec": 30},
                                   "phase2": {"steps": 20, "val_every": 10},
                                   "nlox": {"steps": 20, "val_every": 10}, "eval": {"n_episodes": 8},
                                   "data_baselines": {"gb_steps": 30}})
        self.cfg = cfg
        sub = ("smoke/" if smoke else "") + (f"fixed{data_nx}/" if data_nx else "")
        self.out = C.SC / "runs" / f"{sub}nx{nx}" / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / f"{unit.replace(':', '_')}.log", "a")
        torch.manual_seed(seed); np.random.seed(seed)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    # ------------------------------------------------------------------ set-up
    def setup(self):
        t0 = time.time()
        self.scales = Dm.system_scales(self.cfg)
        self.des = Dm.make_design(self.cfg["system"])
        self.data = C.training_data(self.cfg, self.seed)
        self.t_data = time.time() - t0
        tr, p1 = self.data["train"], self.data["p1"]
        self.n_trans = int(len(tr) * (tr.x.shape[1] - 1) + len(p1) * (p1.x.shape[1] - 1))
        self.model = Dm.model_plant(self.cfg["system"])
        sc, dcfg = self.cfg["system"], self.cfg["data"]
        self.q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(self.scales["x_std"])
        self.r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(self.scales["y_std"])
        box = np.asarray(sc["ic_box"], float)
        self.p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
        self.va = self.data["val"].to(self.device)
        self.x0v = initial_guess(self.cfg, len(self.va), "center", 0, self.device)
        self.log(f"nx {self.nx} seed {self.seed} unit {self.unit} level {self.level} data_nx {self.data_nx} "
                 f"nz {self.des.A.shape[0]} N {self.des.N} dt {self.des.dt} train {len(tr)} p1 {len(p1)} "
                 f"transitions {self.n_trans} burn {tr.burn} ({self.t_data:.0f}s)")

    # ------------------------------------------------------------------ units
    def run_kkl(self):
        obs, meta = {}, {}
        cfg = self.cfg
        t0 = time.time()
        base = phase1(cfg, self.data, self.scales, self.des, self.device, self.seed, self.log)
        t_p1 = time.time() - t0
        bp = nparams(base["T0"]) + nparams(base["T0inv"])
        trained = {}
        for kind in ("lora", "concat", "latent"):
            t0 = time.time()
            r = phase2(kind, cfg, self.data, self.scales, self.des, base, self.device, self.seed, self.log)
            trained[kind] = r
            name = {"lora": "hkkl_A0", "concat": "concat", "latent": "latent"}[kind]
            meta[name] = {"params_trainable": int(r["n_params"] + r["ctx_params"] + bp),
                          "params_conditioner": int(r["n_params"]), "params_base": bp,
                          "train_s": t_p1 / 3 + time.time() - t0, "val": r["val"]}
        o = build_observers(deep_merge(cfg, {"baselines": []}), self.des, self.scales, base, trained, None,
                            self.device)
        obs.update({"hkkl_A0": o["lora"], "concat": o["concat"], "latent": o["latent"]})
        # A3 = S + P + H (beat_greybox components)
        t0 = time.time()
        c3 = deep_merge(cfg, {"phase1": {"optimizer": "soap", "lr": 1e-2}})
        base3 = phase1(c3, self.data, self.scales, self.des, self.device, self.seed, self.log)
        p2 = {"optimizer": "soap", "lr": 1e-2, "base_lr_factor": 0.1, "window": 20,
              "stride": max(1, int(round(6.0 / (self.des.alpha * self.des.dt) / 20.0)))}
        if self.smoke:
            p2["steps"] = 20
        r3 = BG.phase2_bg("lora", c3, self.data, self.scales, self.des, base3, self.device, self.seed, p2, self.log)
        obs["hkkl_A3"] = BG.make_observer("hkkl_A3", self.des, base3, r3, self.device)
        meta["hkkl_A3"] = {"params_trainable": int(r3["trainable_total"] + nparams(base3["T0"])),
                           "params_conditioner": int(r3["n_params"]), "params_hist": int(r3["hist_params"]),
                           "train_s": time.time() - t0, "val": r3["val"], "p2": p2}
        obs["ekf"] = EKF(self.model, self.des.dt, self.q, self.r, self.p0)
        obs["ukf"] = UKF(self.model, self.des.dt, self.q, self.r, self.p0)
        for k in ("ekf", "ukf"):
            meta[k] = {"params_trainable": 0, "train_s": 0.0}
        return obs, meta

    def run_gb(self):
        obs, meta = {}, {}
        n_pool = min(self.n_trans, GB_POOL_MAX)
        c2 = deep_merge(self.cfg, {"baselines": ["greybox_ekf", "greybox_ukf"],
                                   "data_baselines": {"gb_n_fit": n_pool, "n_fit": n_pool}})
        t0 = time.time()
        fits = BD.fit_all(c2, self.data, self.scales, self.des, self.device, self.seed, self.log)
        o = build_observers(c2, self.des, self.scales, None, {}, None, self.device, fits)
        for b in ("greybox_ekf", "greybox_ukf"):
            obs[b] = o[b]
            meta[b] = {"params_trainable": int(fits["info"][b]["n_params"]), "train_s": time.time() - t0,
                       "pool_transitions": n_pool}
        o2, m2 = self.run_dyn("ude", hybrid=True)
        obs.update(o2); meta.update(m2)
        return obs, meta

    def run_dyn(self, kind, hybrid):
        sets = [self.data["train"], self.data["p1"]]
        W = SP.sizes(self.nx)["dyn_width"]
        tm, ts = T.target_stats(self.model if hybrid else None, sets, self.des.dt, self.device, self.seed)
        torch.manual_seed(self.seed)
        net = BD.ResidualMLP(self.nx, 1, W, self.scales.get("x_mean"), self.scales["x_std"],
                             float(self.cfg["system"].get("u_scale", 1.0)), tm, ts)
        t0 = time.time()
        info = T.train_dyn(self.model if hybrid else None, net, sets, self.va, self.des.dt,
                           20 if self.smoke else 3000, LR[kind], self.device, self.seed, self.log, K=32,
                           val_every=10 if self.smoke else 100, warm_steps=5 if self.smoke else 300)
        gm = M.deployed_dynamics(self.model, net, hybrid)
        obs, meta = {}, {}
        filters = (("ekf", EKF), ("ukf", UKF)) if hybrid else (("ekf", EKF),)
        for fk, F in filters:
            ob = F(gm, self.des.dt, self.q, self.r, self.p0)
            ob.name = f"{kind}_{fk}"
            obs[ob.name] = ob
            meta[ob.name] = {"params_trainable": nparams(net), "train_s": time.time() - t0, "width": W,
                             "lr": LR[kind], "val_loss": info["val_loss"], "n_bad_steps": info["n_bad_steps"]}
        return obs, meta

    def run_mf(self):
        obs, meta = {}, {}
        # GRU
        tr = self.data["train"].to(self.device)
        budget = SP.sizes(self.nx)["gru_budget"]
        H = M.solve_gru(1, len(self.model.sensors), self.nx, budget)
        xm = np.asarray(self.scales["x_mean"])
        norm = M._Norm(float(self.cfg["system"].get("u_scale", 1.0)), xm[list(self.model.sensors)],
                       self.scales["y_std"], xm, self.scales["x_std"])
        torch.manual_seed(self.seed)
        net = M.GRUNet(1, len(self.model.sensors), self.nx, H, 2, norm)
        t0 = time.time()
        info = T.train_seq(net, tr, self.va, 20 if self.smoke else 3000, LR["gru"], self.device, self.seed,
                           self.log, val_every=10 if self.smoke else 250)
        obs["gru"] = M.SeqObserver("gru", net)
        meta["gru"] = {"params_trainable": nparams(net), "hidden": H, "train_s": time.time() - t0,
                       "lr": LR["gru"], "val_loss": info["val_loss"]}
        del tr
        # NODE + EKF (pre-registered budget cut: not run at n_x >= 64)
        if self.nx < 64:
            o, m = self.run_dyn("node", hybrid=False)
            obs.update(o); meta.update(m)
        # NLOX
        c2 = deep_merge(self.cfg, {"nlox": {"lr": LR["nlox"]}})
        t0 = time.time()
        res = train_nlox(c2, self.data, self.scales, self.des, self.device, self.seed, self.log)
        obs["nlox"] = build_observers(deep_merge(c2, {"baselines": ["nlox"]}), self.des, self.scales, None, {},
                                      res, self.device)["nlox"]
        meta["nlox"] = {"params_trainable": int(res["n_params"]), "train_s": time.time() - t0, "lr": LR["nlox"],
                        "width": c2["nlox"]["width"], "dec_width": c2["nlox"]["dec_width"]}
        return obs, meta

    # ------------------------------------------------------------------ main
    def run(self):
        dest = self.out / f"{self.unit.replace(':', '_')}.json"
        if dest.exists():
            self.log("exists"); return
        t0 = time.time()
        if self.kind == "data":
            self.scales = Dm.system_scales(self.cfg)
            C.training_data(self.cfg, self.seed)
            for tier in self.cfg["eval"]["tiers"]:
                Dm.eval_data(self.cfg, tier)
            self.log(f"data done ({time.time() - t0:.0f}s)")
            dest.write_text(json.dumps({"wall_s": time.time() - t0}))
            return
        self.setup()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        obs, meta = {"kkl": self.run_kkl, "gb": self.run_gb, "mf": self.run_mf}[self.kind]()
        t_train = time.time() - t0
        timing = {}
        res = evaluate(self.cfg, self.des, self.scales, obs, self.device, self.log, timing=timing)
        results = {}
        for tier, cs in res.items():
            blk = cs["clean"]
            results[tier] = {m: {k: v[k] for k in KEEP if k in v} for m, v in blk["methods"].items()}
        for m in meta:
            meta[m]["train_transitions"] = self.n_trans
            meta[m]["transitions_per_param"] = self.n_trans / max(1, meta[m]["params_trainable"])
        payload = {"nx": self.nx, "seed": self.seed, "unit": self.unit, "level": self.level,
                   "data_nx": self.data_nx or self.nx, "smoke": self.smoke, "methods": meta,
                   "results": results, "eval_timing": timing, "train_wall_s": t_train,
                   "wall_s": time.time() - t0, "device": self.device, "nz": int(self.des.A.shape[0]),
                   "sizes": self.cfg["scaling"]["sizes"], "n_train": len(self.data["train"]),
                   "n_p1": len(self.data["p1"]), "transitions": self.n_trans,
                   "peak_cuda_gb": (torch.cuda.max_memory_allocated() / 2 ** 30) if self.device.startswith("cuda") else 0,
                   "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=_json_default))
        tmp.rename(dest)
        summ = "  ".join(f"{m}={np.mean(v['nrmse_unmeasured']):.4f}" for m, v in results["sequence_ood"].items())
        self.log(f"done in {time.time() - t0:.0f}s  seq unmeasured: {summ}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unit", required=True); ap.add_argument("--data-nx", type=int, default=None)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    j = Job(a.nx, a.seed, a.unit, a.device, a.data_nx, a.smoke)
    try:
        j.run()
    except Exception:
        j.log(traceback.format_exc())
        (j.out / f"{a.unit.replace(':', '_')}.error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
