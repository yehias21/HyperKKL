"""One memory-design job = (node, system, seed, arms). Rules: ../PREREG.md.

    python run_md.py --node C3-partial --system quad_tank --seed 0 --arms M0,M3 [--device cuda:0] [--smoke]

M0 = campaign arm A2 (its cached models are copied from the campaign's model dir by key, so M0 is
re-evaluated, not retrained, when they exist). M1/M2 (+M4/M5 on M2) need new phase-1 episodes
(gen_p1_long, CPU, cached in ../data).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import shutil
import time
import traceback
from pathlib import Path

from common_md import CAMPAIGN, HERE, LIVE, Dm, np, run_cfg, torch  # noqa: F401  (read-only cache patch)
from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import memory as MM  # noqa: E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.conditioning import match_width  # noqa: E402
from hyperkkl.evaluate import evaluate  # noqa: E402
from hyperkkl.train import phase1  # noqa: E402

ROOT = HERE.parent
OUT, MC, DD = ROOT / "runs", ROOT / "models", ROOT / "data"
CMC = CAMPAIGN / "models"
S1 = {"optimizer": "soap", "lr": 1e-2}
P2 = {"optimizer": "soap", "lr": 1e-2, "base_lr_factor": 0.1}       # = campaign A2's p2 spec (same key order)
WSTAR = {"quad_tank": 5.0, "battery": 10.0, "vdp": 2.0, "bioreactor": 1.0}   # W* in units of 1/alpha (PREREG 2.3)
K_WIN = 20
ARMS = ["M0", "M1", "M2", "M3", "M4", "M5"]
DESIGN_OF = {"M0": None, "M1": "M1", "M2": "M2", "M3": None, "M4": "M2", "M5": "M2"}
MEM_HASH = hashlib.sha1((HERE / "pkg" / "hyperkkl" / "memory.py").read_bytes()).hexdigest()[:10]


def alpha_of(cfg):
    return float(cfg["system"]["alpha"]) * float(cfg["system"].get("alpha_scale", 1.0) or 1.0)


def stride_of(cfg, dt):
    return max(1, int(round(WSTAR[cfg["system"]["name"]] / alpha_of(cfg) / (dt * (K_WIN - 1)))))


class Job:
    def __init__(self, node, system, seed, device, smoke=False):
        cfg = run_cfg(node, system, seed)
        cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
        if smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40},
                                   "phase2": {"steps": 20, "val_every": 10}})
            cfg["eval"]["n_episodes"] = cfg["eval"]["n_episodes"]   # same episodes (cache key)
        self.cfg, self.node, self.system, self.seed, self.device, self.smoke = cfg, node, system, seed, device, smoke
        self.out = OUT / ("smoke" if smoke else "") / node / system / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "log.txt", "a")
        self.scales = Dm.system_scales(cfg, device)
        self.des0 = Dm.make_design(cfg["system"])
        self.data0 = Dm.training_data(cfg, seed, device)
        self.mh = module_hash(MODEL_MODULES)
        self.cfg1 = deep_merge(cfg, {"phase1": S1})
        self._d = {}

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    def tag(self):
        return ("smoke_" if self.smoke else "") + f"{self.node}_{self.system}_s{self.seed}"

    def x_center(self):
        box = np.asarray(self.cfg["system"]["ic_box"], float)
        return torch.as_tensor(box.mean(1), dtype=torch.float32, device=self.device)[None]

    # ------------------------------------------------------------------ campaign models (read by key)
    def _cached(self, name, build):
        path = MC / name
        src = CMC / name
        if not path.exists() and src.exists() and not self.smoke:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, path.with_suffix(".copy"))
            path.with_suffix(".copy").rename(path)
            self.log(f"  copied campaign model {name}")
        t0 = time.time()
        r = Cm.cached(path, build)
        return {k: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for k, v in r.items()}, time.time() - t0

    def base_default(self):
        k = digest(self.cfg1, "system", "data", "phase1", extra=[self.seed, self.mh, self.node, {}])
        return self._cached(f"{self.tag()}_p1_{k}.pt",
                            lambda: phase1(self.cfg1, self.data0, self.scales, self.des0, self.device, self.seed, self.log))

    # ------------------------------------------------------------------ slow designs
    def p1_long(self, T_p1):
        path = DD / ("smoke_" if self.smoke else "") / f"{self.node}_{self.system}_s{self.seed}_p1long.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(path) + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if path.exists():
                return torch.load(path, weights_only=False, map_location="cpu")
            t0 = time.time()
            nthr = torch.get_num_threads()
            torch.set_num_threads(4)
            cfg_g = deep_merge(self.cfg, {"data": {"n_p1": 24}}) if self.smoke else self.cfg
            es = MM.gen_p1_long(cfg_g, self.scales, self.des0, self.seed, T_p1)
            torch.set_num_threads(nthr)
            self.log(f"  generated long p1: {tuple(es.x.shape)} rejected {es.n_rejected} ({time.time() - t0:.0f}s)")
            tmp = path.with_suffix(".tmp")
            torch.save(es, tmp)
            tmp.rename(path)
            return es

    def slow(self, which):
        """(des, base, data2, sc) for design M1 / M2: long p1, phase 1, phase-2 data from z0 = T0(x_c)."""
        if which in self._d:
            return self._d[which]
        eps = float(self.cfg["data"].get("eps_label", 1e-3))
        T = self.des0.dt * (self.data0["train"].x.shape[1] - 1)
        tb0 = float(self.scales["t_burn"])
        d2 = MM.m_design(self.cfg, self.des0, "M2")
        tb2 = MM.washout(d2, self.scales, MM.z_scale_of(self.data0["train"], d2), eps)
        T_p1 = 1.05 * tb2 + (T - tb0)
        des = d2 if which == "M2" else MM.m_design(self.cfg, self.des0, which)
        chk = BG.design_checks(des)
        if not chk["ok_relaxed"]:
            raise RuntimeError(f"design gate failed {chk}")
        zsc = MM.z_scale_of(self.data0["train"], des)
        tb = MM.washout(des, self.scales, zsc, eps)
        burn = int(math.ceil(tb / des.dt))
        sc = dict(self.scales, t_burn=tb, z_scale=zsc)
        p1l = self.p1_long(T_p1)
        dev = self.device
        data1 = {"coll": self.data0["coll"], "p1": MM.relabel_from(p1l, des, burn, None, dev),
                 "train": MM.relabel_from(self.data0["train"], des, self.data0["train"].burn, None, dev),
                 "val": MM.relabel_from(self.data0["val"], des, self.data0["val"].burn, None, dev)}
        self.log(f"  design {which}: alpha {des.alpha:.4g} t_burn {tb:.1f} burn {burn} p1 {tuple(data1['p1'].x.shape)} "
                 f"(labelled {data1['p1'].x.shape[1] - burn} samples/ep) T_p1 {T_p1:.1f}")
        k = digest(self.cfg1, "system", "data", "phase1", extra=[self.seed, self.mh, self.node, which, MEM_HASH, "md-v1"])
        base, t_b = self._cached(f"{self.tag()}_p1md_{which}_{k}.pt",
                                 lambda: phase1(self.cfg1, data1, sc, des, dev, self.seed, self.log))
        with torch.no_grad():
            z0 = base["T0"](self.x_center())[0].detach().cpu()
        data2 = {"coll": self.data0["coll"], "p1": data1["p1"],
                 "train": MM.relabel_from(self.data0["train"], des, self.data0["train"].burn, z0, dev),
                 "val": MM.relabel_from(self.data0["val"], des, self.data0["val"].burn, z0, dev)}
        self._d[which] = (des, base, data2, sc, z0, t_b)
        return self._d[which]

    # ------------------------------------------------------------------ arms
    def train_arm(self, name):
        dname = DESIGN_OF[name]
        if dname is None:
            des, data, sc = self.des0, self.data0, self.scales
            base, t_b = self.base_default()
            z0 = None
        else:
            des, base, data, sc, z0, t_b = self.slow(dname)
        p2 = dict(P2)
        z_std = data["train"].z_obs[:, data["train"].burn:].reshape(-1, des.A.shape[0]).std(0).clamp(min=1e-6)
        npar0 = match_width("lora", base["T0inv"], 16, int(self.cfg["phase2"]["budget"]), int(self.cfg["phase2"]["rank"]),
                            z_std.to(self.device))[1]
        window = K_WIN if name in ("M3", "M5") else 0
        warp = name == "M4"
        extra = MM.extra_params(des.A.shape[0], window, warp)
        steps_over = {"steps": 20} if self.smoke else {}
        if name == "M0":
            cfg2 = self.cfg1
            k2 = digest(cfg2, "system", "data", "phase1", "phase2", extra=[self.seed, self.mh, self.node, {}, p2])
            r, t_2 = self._cached(f"{self.tag()}_p2_{k2}.pt",
                                  lambda: BG.phase2_bg("lora", cfg2, data, sc, des, base, self.device, self.seed,
                                                       dict(p2, **steps_over), self.log))
            r.setdefault("warp", None); r.setdefault("warp_params", 0)
        else:
            opts = dict(p2, budget=npar0 - extra, **steps_over)
            if window:
                opts.update(window=window, stride=stride_of(self.cfg, des.dt))
            if warp:
                opts.update(warp=True, warp_rate=alpha_of(self.cfg), z0=z0.tolist())
            k2 = digest(self.cfg1, "system", "data", "phase1", "phase2",
                        extra=[self.seed, self.mh, self.node, name, {k: v for k, v in opts.items() if k != "z0"},
                               MEM_HASH, "md-v1"])
            r, t_2 = self._cached(f"{self.tag()}_p2md_{name}_{k2}.pt",
                                  lambda: MM.phase2_md(self.cfg1, data, sc, des, base, self.device, self.seed, opts, self.log))
        ob = MM.make_observer_md(name, des, base, r, self.device)
        v = BG.validation_nrmse(self.cfg, ob, data["val"], sc, self.device)
        g = BG.noise_gain(base["T0inv"], data["val"], sc, self.device)
        info = {"val": v, "noise_gain_base": g, "diag": base["diag"], "design_alpha": float(des.alpha),
                "poles_re": sorted((-np.linalg.eigvals(des.A).real).tolist()), "nz": int(des.A.shape[0]),
                "phase1_wall_s": base.get("wall_s"), "phase2_wall_s": r.get("wall_s"),
                "load_or_train_s": {"p1": t_b, "p2": t_2},
                "params": {"encoder": sum(q.numel() for q in base["T0"].parameters()),
                           "base_decoder": sum(q.numel() for q in base["T0inv"].parameters()),
                           "conditioner": r["n_params"], "context": r["ctx_params"], "history": r.get("hist_params", 0),
                           "warp": r.get("warp_params", 0), "base_decoder_trainable_p2": r["base_dec_params_trainable"],
                           "phase2_trainable_total": r["trainable_total"], "width": r["width"], "m0_conditioner": npar0},
                "steps": {"p1_enc": int(self.cfg1["phase1"]["steps_enc"]), "p1_dec": int(self.cfg1["phase1"]["steps_dec"]),
                          "p2": int(r["steps"])},
                "p2_val_loss": r["val"], "window": window,
                "stride": stride_of(self.cfg, des.dt) if window else 0, "burn_p2": int(data["train"].burn)}
        if warp:
            with torch.no_grad():
                va = data["val"].to(self.device)
                c = r["ctx"].seq(va.u[:16])
                s = r["warp"].s_seq(c)[:, va.burn:]
                info["warp_s"] = {"mean": float(s.mean()), "q05": float(torch.quantile(s.flatten()[::7], 0.05)),
                                  "q95": float(torch.quantile(s.flatten()[::7], 0.95)),
                                  "min": float(s.min()), "max": float(s.max())}
        self.log(f"  arm {name}: val nrmse {v['val_nrmse']:.4f} total p2 params {r['trainable_total']} "
                 f"eps_dec {base['diag']['eps_dec']:.4g} l_dec_q95 {base['diag']['l_dec_q95']:.3g}")
        return ob, info

    def run_arms(self, arms):
        todo = [a for a in arms if not (self.out / f"{a}.json").exists()]
        if not todo:
            self.log("all arms done"); return
        obs, infos = {}, {}
        for a in todo:
            t0 = time.time()
            obs[a], infos[a] = self.train_arm(a)
            infos[a]["train_total_s"] = time.time() - t0
        t0 = time.time()
        timing = {}
        res = evaluate(self.cfg, self.des0, self.scales, obs, self.device, self.log, timing=timing)
        t_eval = time.time() - t0
        nx = len(self.cfg["system"]["ic_box"])
        keys = ["nrmse", "nrmse_measured", "nrmse_unmeasured", "diverged"] + [f"nrmse_x{j}" for j in range(1, nx + 1)]
        for a in todo:
            per = {}
            for tier, cs in res.items():
                m = cs["clean"]["methods"][a]
                per[tier] = {k: m[k] for k in keys if k in m}
            payload = {"node": self.node, "system": self.system, "seed": self.seed, "arm": a, "info": infos[a],
                       "results": per, "eval_timing": timing.get(a), "eval_wall_s_job": t_eval,
                       "code_module_hash": self.mh, "memory_hash": MEM_HASH, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tmp = self.out / f"{a}.json.tmp"
            tmp.write_text(json.dumps(payload, default=_jd))
            tmp.rename(self.out / f"{a}.json")
            self.log(f"  saved {a}: seq {np.mean(per['sequence_ood']['nrmse']):.4f} shape {np.mean(per['shape_ood']['nrmse']):.4f} "
                     f"id {np.mean(per['id']['nrmse']):.4f}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arms", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--p1only", action="store_true", help="only generate the long phase-1 episodes (CPU)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    j = Job(a.node, a.system, a.seed, a.device, a.smoke)
    try:
        if a.p1only:
            eps = float(j.cfg["data"].get("eps_label", 1e-3))
            T = j.des0.dt * (j.data0["train"].x.shape[1] - 1)
            d2 = MM.m_design(j.cfg, j.des0, "M2")
            tb2 = MM.washout(d2, j.scales, MM.z_scale_of(j.data0["train"], d2), eps)
            j.p1_long(1.05 * tb2 + (T - float(j.scales["t_burn"])))
        if a.arms:
            j.run_arms([x for x in a.arms.split(",") if x])
        if torch.cuda.is_available():
            j.log(f"peak CUDA memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    except Exception:
        j.log(traceback.format_exc())
        (j.out / "error.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
