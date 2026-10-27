"""quad_tank PE study, Stage 2: one job = (node, seed, arm). Scratch copy only (PREREG.md).

    python run_qt.py --node C3-partial --seed 0 --arm X2 [--device cuda:0] [--smoke] [--no-gb]

Trains HyperKKL (campaign recipe A2: SOAP + unfrozen base) on the arm's training set, refits the
grey-box residual on the same data, and evaluates both on the unchanged C3 evaluation episodes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common_qt import Dm, LIVE, np, torch, build_plant  # noqa: E402  (read-only cache patch)
import qt_signals as QS  # noqa: E402
from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import signals as S  # noqa: E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.data import EpisodeSet  # noqa: E402
from hyperkkl.evaluate import build_observers, evaluate  # noqa: E402
from hyperkkl.train import phase1  # noqa: E402

QS.install(Dm)
OUT, MC, DD = HERE / "runs", HERE / "models", HERE / "data"
P1 = {"optimizer": "soap", "lr": 1e-2}
P2 = {"optimizer": "soap", "lr": 1e-2, "base_lr_factor": 0.1}
GB = ["greybox_ekf", "greybox_ukf", "greybox_ekf_oracle"]


def concat(a: EpisodeSet, b: EpisodeSet) -> EpisodeSet:
    assert a.burn == b.burn and a.x.shape[1] == b.x.shape[1] and abs(a.dt - b.dt) < 1e-12
    kw = {k: torch.cat([getattr(a, k), getattr(b, k)]) for k in ("x", "u", "y_clean", "y", "z", "z_obs")}
    return EpisodeSet(**kw, families=list(a.families) + list(b.families), burn=a.burn, dt=a.dt,
                      meta=list(a.meta) + list(b.meta), n_rejected=a.n_rejected + b.n_rejected)


class Job:
    def __init__(self, node, seed, arm, device, smoke=False):
        src = LIVE / "experiments" / "tree" / node / "runs" / "quad_tank" / f"seed{seed}" / "results.json"
        cfg = json.load(open(src))["config"]
        cfg["seed"] = seed
        cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
        if smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40},
                                   "phase2": {"steps": 20, "val_every": 10}})
        self.cfg, self.node, self.seed, self.arm, self.device, self.smoke = cfg, node, seed, arm, device, smoke
        self.out = OUT / ("smoke" if smoke else "") / node / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / f"{arm}.log", "a")
        self.scales = Dm.system_scales(cfg, device)
        self.des0 = Dm.make_design(cfg["system"])
        self.data0 = Dm.training_data(cfg, seed, device)
        self.mh = module_hash(MODEL_MODULES)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    # --------------------------------------------------------------- training sets
    def gen(self, part, families, n, T, seed_off):
        """New forced episodes (cached in DD, built once under a file lock)."""
        n_ = 24 if self.smoke else n
        path = DD / ("smoke_" if self.smoke else "") / f"{self.node}_s{self.seed}_{part}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if path.exists():
                return torch.load(path, weights_only=False, map_location="cpu")
            scfg, dcfg = self.cfg["system"], self.cfg["data"]
            des = Dm.make_design(dict(scfg, T=T))
            assert abs(des.dt - self.des0.dt) < 1e-12
            plant = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)))
            t0 = time.time()
            # generated on the CPU: the per-step RK4 loop is launch-bound and the shared GPUs are
            # saturated (24 episodes took 375 s on GPU); the device only changes the noise draws
            es = Dm.generate(plant, des, self.scales, scfg, dcfg, families, n_, "train",
                             100_000 * (self.seed + 1) + seed_off, "cpu")
            self.log(f"  generated {part}: {len(es)} episodes x {es.x.shape[1]} samples, rejected "
                     f"{es.n_rejected}, families {sorted(set(es.families))} ({time.time() - t0:.0f}s)")
            tmp = path.with_suffix(".tmp")
            torch.save(es, tmp)
            tmp.rename(path)
            return es

    def arm_data(self):
        d0, T0 = self.data0, float(self.cfg["system"]["T"])
        if self.arm == "X0":
            tr = d0["train"]
        elif self.arm == "X1":
            tr = self.gen("long_orig", S.TRAIN_FAMILIES, 600, 3 * T0, 21)
        elif self.arm == "X2":
            k = 12 if self.smoke else 300
            tr = concat(d0["train"].select(range(k)), self.gen("slow_T1", QS.SLOW_FAMILIES, 300, T0, 22))
        elif self.arm == "X3":
            tr = concat(self.gen("long_orig", S.TRAIN_FAMILIES, 600, 3 * T0, 21),
                        self.gen("slow_T3", QS.SLOW_FAMILIES, 600, 3 * T0, 23))
        else:
            raise ValueError(self.arm)
        assert tr.burn == d0["train"].burn
        return {"p1": d0["p1"], "train": tr, "val": d0["val"], "coll": d0["coll"]}

    # --------------------------------------------------------------- grey-box only (CPU)
    def run_gb(self):
        f = self.out / f"{self.arm}_gb.json"
        if f.exists():
            self.log("gb done already"); return
        t_all = time.time()
        data = self.arm_data()
        cfg_b = json.loads(json.dumps(self.cfg))
        cfg_b["baselines"] = list(GB)
        fits = BD.fit_all(cfg_b, data, self.scales, self.des0, "cpu", self.seed, self.log)
        gobs = build_observers(cfg_b, self.des0, self.scales, None, {}, None, "cpu", fits)
        t_fit = time.time() - t_all
        res = evaluate(self.cfg, self.des0, self.scales, {k: gobs[k] for k in GB}, "cpu", self.log)
        keys = ["nrmse", "nrmse_measured", "nrmse_unmeasured", "diverged"] + [f"nrmse_x{j}" for j in range(1, 5)]
        per = {t: {m: {k: cs["clean"]["methods"][m][k] for k in keys if k in cs["clean"]["methods"][m]}
                   for m in GB} for t, cs in res.items()}
        payload = {"node": self.node, "seed": self.seed, "arm": self.arm, "results": per,
                   "greybox_fit": fits["info"].get("greybox_ekf"),
                   "wall_s": {"fit": t_fit, "total": time.time() - t_all}, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=_jd))
        tmp.rename(f)
        for t in per:
            self.log(f"  gb {t}: " + "  ".join(f"{m} {np.mean(per[t][m]['nrmse']):.4f}" for m in GB))

    # --------------------------------------------------------------- run
    def run(self, do_gb=True):
        f = self.out / f"{self.arm}.json"
        if f.exists():
            self.log("done already"); return
        t_all = time.time()
        data = self.arm_data()
        tr = data["train"]
        self.log(f"arm {self.arm}: train {tuple(tr.x.shape)} families {len(set(tr.families))} burn {tr.burn}")
        tag = ("smoke_" if self.smoke else "") + f"{self.node}_s{self.seed}_{self.arm}"
        cfg = deep_merge(self.cfg, {"phase1": P1})
        k1 = digest(cfg, "system", "data", "phase1", extra=[self.seed, self.mh, self.node, self.arm, "qt-v1"])
        t0 = time.time()
        base = Cm.cached(MC / f"{tag}_p1_{k1}.pt",
                         lambda: phase1(cfg, data, self.scales, self.des0, self.device, self.seed, self.log))
        base = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in base.items()}
        t_p1 = time.time() - t0
        k2 = digest(cfg, "system", "data", "phase1", "phase2", extra=[self.seed, self.mh, self.node, self.arm, P2, "qt-v1"])
        t0 = time.time()
        r = Cm.cached(MC / f"{tag}_p2_{k2}.pt",
                      lambda: BG.phase2_bg("lora", cfg, data, self.scales, self.des0, base, self.device,
                                           self.seed, dict(P2), self.log))
        r = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in r.items()}
        t_p2 = time.time() - t0
        ob = BG.make_observer("hyperkkl", self.des0, base, r, self.device)
        val = BG.validation_nrmse(self.cfg, ob, data["val"], self.scales, self.device)
        self.log(f"  hyperkkl val nrmse {val['val_nrmse']:.4f} eps_dec {base['diag']['eps_dec']:.4g} "
                 f"l_dec_q95 {base['diag']['l_dec_q95']:.3g}")
        obs = {"hyperkkl": ob}
        gb_info, t_gb = None, 0.0
        if do_gb:
            t0 = time.time()
            cfg_b = json.loads(json.dumps(self.cfg))
            cfg_b["baselines"] = list(GB)
            # grey-box fit and filters on the CPU: the step-by-step filters are launch-bound and
            # the shared GPUs are saturated (one grey-box EKF pass > 4 min on GPU)
            torch.set_num_threads(4)
            fits = BD.fit_all(cfg_b, data, self.scales, self.des0, "cpu", self.seed, self.log)
            gobs = build_observers(cfg_b, self.des0, self.scales, None, {}, None, "cpu", fits)
            gb_info = fits["info"].get("greybox_ekf")
            t_gb = time.time() - t0
        t0 = time.time()
        timing = {}
        res = evaluate(self.cfg, self.des0, self.scales, obs, self.device, self.log, timing=timing)
        if do_gb:
            res_g = evaluate(self.cfg, self.des0, self.scales, {k: gobs[k] for k in GB}, "cpu", self.log, timing=timing)
            for t in res:
                res[t]["clean"]["methods"].update(res_g[t]["clean"]["methods"])
            obs.update({k: gobs[k] for k in GB})
        t_eval = time.time() - t0
        keys = ["nrmse", "nrmse_measured", "nrmse_unmeasured", "diverged"] + [f"nrmse_x{j}" for j in range(1, 5)]
        per = {t: {m: {k: cs["clean"]["methods"][m][k] for k in keys if k in cs["clean"]["methods"][m]}
                   for m in obs} for t, cs in res.items()}
        payload = {"node": self.node, "seed": self.seed, "arm": self.arm, "results": per, "val": val,
                   "train_shape": list(tr.x.shape), "train_families": sorted(set(tr.families)),
                   "n_rejected": tr.n_rejected, "diag_p1": base["diag"], "p2_val_loss": r["val"],
                   "params": {"conditioner": r["n_params"], "ctx": r["ctx_params"],
                              "base_trainable": r["base_dec_params_trainable"]},
                   "greybox_fit": gb_info, "wall_s": {"p1": t_p1, "p2": t_p2, "gb_fit": t_gb, "eval": t_eval,
                                                      "total": time.time() - t_all},
                   "code_module_hash": self.mh, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=_jd))
        tmp.rename(f)
        for t in per:
            s = "  ".join(f"{m} {np.mean(per[t][m]['nrmse']):.4f}" for m in obs)
            self.log(f"  {t}: {s}")


def _jd(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--no-gb", action="store_true")
    ap.add_argument("--gb-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    j = Job(a.node, a.seed, a.arm, a.device, a.smoke)
    try:
        if a.gb_only:
            j.run_gb()
        else:
            j.run(do_gb=not a.no_gb)
        if torch.cuda.is_available():
            j.log(f"peak CUDA memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    except Exception:
        j.log(traceback.format_exc())
        (j.out / f"{a.arm}.error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
