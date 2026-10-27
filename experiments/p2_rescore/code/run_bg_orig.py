"""One beat-the-grey-box job = (node, system, seed, arms). Scratch copy only.

    python run_bg.py --node C3-partial --system vdp --seed 0 --arms A0,A1 [--device cuda:0] [--smoke]
    python run_bg.py --node C3-partial --system vdp --seed 0 --d2select

Isolation: the package is the scratch copy (pkg/); models and outputs go under this directory.
Data / evaluation episodes are read READ-ONLY from the live cache (a miss raises: no rebuild),
the run config is the one stored in the tree run's results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "pkg"))
LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
LIVE_CACHE = LIVE / "experiments" / "cache"
OUT = HERE / "runs"
MC = HERE / "models"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import beat as BG  # noqa: E402
from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.evaluate import evaluate  # noqa: E402
from hyperkkl.train import phase1  # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402

assert Dm.CACHE.resolve() != LIVE_CACHE.resolve(), "refusing to run against the live cache"


def _readonly(path, build, **kw):
    path = Path(path)
    rel = path.relative_to(Dm.CACHE)
    live = LIVE_CACHE / rel
    if live.exists():
        return torch.load(live, weights_only=False, map_location="cpu")
    raise FileNotFoundError(f"data not in the live cache (no rebuild by design): {rel}")


Dm._cached = _readonly

# ---------------------------------------------------------------------------- arms
S1 = {"optimizer": "soap", "lr": 1e-2}
R1 = {"dec_width": 256}


def stride_rule(cfg, des) -> int:
    a = float(cfg["system"]["alpha"]) * float(cfg["system"].get("alpha_scale", 1.0))
    return max(1, int(round(6.0 / (a * des.dt) / 20.0)))


def d1_nz(cfg) -> int:
    nx = len(cfg["system"]["ic_box"])
    return 2 * (2 * nx + 1) if cfg["system"]["name"] in ("quad_tank", "battery") else 2 * nx + 2


def components(tags: set, cfg) -> dict:
    """Arm spec from component tags: S, P, H, R, D1, D2(alpha_scale, spread), D3."""
    p1, p2 = {}, {}
    if "S" in tags:
        p1.update(S1); p2.update(optimizer="soap", lr=1e-2)
    if "R" in tags:
        p1.update(R1); p2.update(budget=280000, steps=6000)
    if "P" in tags:
        p2["base_lr_factor"] = 0.1
    if "H" in tags:
        p2["window"] = 20                    # stride filled in at run time (needs dt)
    if "D3" in tags:
        p2["slow_factor"] = 5.0
    design = {}
    if "D1" in tags:
        design["nz"] = d1_nz(cfg)
    return {"p1": p1, "p2": p2, "design": design}


ARM_TAGS = {
    "A0": set(), "A1": {"S"}, "A2": {"S", "P"}, "A3": {"S", "P", "H"}, "A4": {"S", "P", "H", "R"},
    "A4-S": {"P", "H", "R"}, "A4-P": {"S", "H", "R"}, "A4-H": {"S", "P", "R"},
    "D1": {"D1"}, "D2": {"D2"}, "D3": {"D3"},
}


def arm_tags(name: str) -> set:
    """'A3+D1+D3' -> A3's tags plus D1, D3."""
    head, *extra = name.split("+")
    return set(ARM_TAGS[head]) | set(extra)


D2_GRID = [(a, s) for a in (0.5, 1.0, 2.0) for s in (1.0, 4.0)]


def d2_choice(node, system, smoke=False):
    f = OUT / ("smoke" if smoke else "") / node / system / "d2_selection.json"
    if not f.exists():
        raise RuntimeError(f"D2 selection missing for {node}/{system}: run --d2select on seed 0 first")
    return json.load(open(f))["chosen"]


# ---------------------------------------------------------------------------- job
class Job:
    def __init__(self, node, system, seed, device, smoke=False):
        src = LIVE / "experiments" / "tree" / node / "runs" / system / f"seed{seed}" / "results.json"
        cfg = json.load(open(src))["config"]
        cfg["seed"] = seed
        cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
        if smoke:
            cfg = deep_merge(cfg, {"phase1": {"steps_enc": 40, "steps_dec": 40},
                                   "phase2": {"steps": 20, "val_every": 10}})
        self.cfg, self.node, self.system, self.seed, self.device, self.smoke = cfg, node, system, seed, device, smoke
        self.out = OUT / ("smoke" if smoke else "") / node / system / f"seed{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.out / "log.txt", "a")
        self.scales = Dm.system_scales(cfg, device)
        self.des0 = Dm.make_design(cfg["system"])
        self.data0 = Dm.training_data(cfg, seed, device)
        self.mh = module_hash(MODEL_MODULES)
        self._designs = {}

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n"); self.logf.flush()

    # design -> (des, data, scales)
    def design(self, spec: dict):
        key = json.dumps(spec, sort_keys=True)
        if key not in self._designs:
            if not spec:
                self._designs[key] = (self.des0, self.data0, self.scales)
            else:
                des = BG.custom_design(self.cfg["system"], self.des0.dt, self.des0.N, nz=spec.get("nz"),
                                       alpha_scale=spec.get("alpha_scale", 1.0), spread=spec.get("spread", 1.0))
                chk = BG.design_checks(des)
                if not chk["ok_relaxed"]:
                    raise RuntimeError(f"design gate (relaxed) failed: {chk}")
                data, sc = BG.redesign(self.data0, des, self.scales, float(self.cfg["data"].get("eps_label", 1e-3)))
                self.log(f"  design {spec}: nz {des.A.shape[0]} alpha {des.alpha:.4g} max|lam|dt "
                         f"{chk['max_lam_dt']:.3f} t_burn {sc['t_burn']:.2f}s burn {data['train'].burn}")
                self._designs[key] = (des, data, sc)
        return self._designs[key]

    def tag(self):
        return ("smoke_" if self.smoke else "") + f"{self.node}_{self.system}_s{self.seed}"

    def base(self, design_spec, p1o):
        des, data, sc = self.design(design_spec)
        cfg = deep_merge(self.cfg, {"phase1": p1o})
        k = digest(cfg, "system", "data", "phase1", extra=[self.seed, self.mh, self.node, design_spec])
        t0 = time.time()
        b = Cm.cached(MC / f"{self.tag()}_p1_{k}.pt", lambda: phase1(cfg, data, sc, des, self.device, self.seed, self.log))
        b = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in b.items()}
        return b, time.time() - t0

    def train_arm(self, name, design_spec=None):
        tags = arm_tags(name)
        spec = components(tags, self.cfg)
        if "D2" in tags:
            spec["design"].update(d2_choice(self.node, self.system, self.smoke))
            if spec["design"].get("alpha_scale") == 1.0 and spec["design"].get("spread") == 1.0:
                spec["design"].pop("alpha_scale"); spec["design"].pop("spread")
        if design_spec is not None:
            spec["design"] = design_spec
        des, data, sc = self.design(spec["design"])
        if self.smoke and "steps" in spec["p2"]:
            spec["p2"]["steps"] = 20
        if "window" in spec["p2"]:
            spec["p2"]["stride"] = stride_rule(self.cfg, des)
        base, t_b = self.base(spec["design"], spec["p1"])
        cfg = deep_merge(self.cfg, {"phase1": spec["p1"]})
        k2 = digest(cfg, "system", "data", "phase1", "phase2",
                    extra=[self.seed, self.mh, self.node, spec["design"], spec["p2"]])
        t0 = time.time()
        r = Cm.cached(MC / f"{self.tag()}_p2_{k2}.pt",
                      lambda: BG.phase2_bg("lora", cfg, data, sc, des, base, self.device, self.seed, spec["p2"], self.log))
        r = {kk: (v.to(self.device) if isinstance(v, torch.nn.Module) else v) for kk, v in r.items()}
        ob = BG.make_observer(name, des, base, r, self.device)
        v = BG.validation_nrmse(self.cfg, ob, data["val"], sc, self.device)
        g = BG.noise_gain(base["T0inv"], data["val"], sc, self.device)
        ng_dep = None
        info = {"spec": spec, "tags": sorted(tags), "val": v, "noise_gain_base": g, "diag": base["diag"],
                "phase1_wall_s": base["wall_s"], "phase2_wall_s": r["wall_s"],
                "load_or_train_s": {"p1": t_b, "p2": time.time() - t0},
                "params": {"encoder": sum(q.numel() for q in base["T0"].parameters()),
                           "base_decoder": sum(q.numel() for q in base["T0inv"].parameters()),
                           "conditioner": r["n_params"], "context": r["ctx_params"], "history": r["hist_params"],
                           "base_decoder_trainable_p2": r["base_dec_params_trainable"],
                           "phase2_trainable_total": r["trainable_total"], "width": r["width"]},
                "steps": {"p1_enc": int(cfg["phase1"]["steps_enc"]), "p1_dec": int(cfg["phase1"]["steps_dec"]),
                          "p2": int(r["steps"])},
                "p2_val_loss": r["val"], "nz": int(des.A.shape[0]), "alpha": float(des.alpha),
                "burn": int(data["train"].burn), "design_spec": spec["design"]}
        self.log(f"  arm {name}: val nrmse {v['val_nrmse']:.4f}  noise gain {g:.4f}  eps_dec {base['diag']['eps_dec']:.4g} "
                 f"l_dec_q95 {base['diag']['l_dec_q95']:.3g}")
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
        for a in todo:
            per = {}
            for tier, cs in res.items():
                m = cs["clean"]["methods"][a]
                per[tier] = {"nrmse": m["nrmse"], "nrmse_unmeasured": m.get("nrmse_unmeasured"),
                             "diverged": m["diverged"], "scorecard": cs["clean"]["scorecard"][a]}
            payload = {"node": self.node, "system": self.system, "seed": self.seed, "arm": a, "info": infos[a],
                       "results": per, "eval_timing": timing.get(a), "eval_wall_s_job": t_eval,
                       "code_module_hash": self.mh, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            tmp = self.out / f"{a}.json.tmp"
            tmp.write_text(json.dumps(payload, default=_jd))
            tmp.rename(self.out / f"{a}.json")
            self.log(f"  saved {a}: seq {np.mean(per['sequence_ood']['nrmse']):.4f} shape {np.mean(per['shape_ood']['nrmse']):.4f} "
                     f"id {np.mean(per['id']['nrmse']):.4f}")

    def d2select(self):
        f = self.out.parent / "d2_selection.json"
        if f.exists():
            self.log("d2 selection exists"); return
        rows = []
        for a, s in D2_GRID:
            dspec = {} if (a == 1.0 and s == 1.0) else {"alpha_scale": a, "spread": s}
            try:
                ob, info = self.train_arm("A0", design_spec=dspec)
                rows.append({"alpha_scale": a, "spread": s, "val_nrmse": info["val"]["val_nrmse"],
                             "val_diverged": info["val"]["val_diverged"], "noise_gain": info["noise_gain_base"],
                             "l_dec_q95": info["diag"]["l_dec_q95"], "eps_dec": info["diag"]["eps_dec"],
                             "burn": info["burn"], "ok": True})
            except Exception as e:                     # e.g. washout longer than the episode
                self.log(f"  D2 candidate {a},{s} infeasible: {e!r}")
                rows.append({"alpha_scale": a, "spread": s, "ok": False, "error": repr(e)})
        d = [r for r in rows if r["alpha_scale"] == 1.0 and r["spread"] == 1.0][0]
        acc = [r for r in rows if r["ok"] and r["val_nrmse"] <= 1.10 * d["val_nrmse"]]
        best = min(acc, key=lambda r: r["noise_gain"])
        if best["noise_gain"] >= 0.98 * d["noise_gain"]:
            best = d
        out = {"node": self.node, "system": self.system, "seed": self.seed, "rows": rows,
               "chosen": {"alpha_scale": best["alpha_scale"], "spread": best["spread"]},
               "rule": "acceptable: val_nrmse <= 1.10 x default; min noise_gain; within 2% of default -> default",
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out, indent=1))
        self.log(f"  D2 chosen {out['chosen']}")


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
    ap.add_argument("--node", required=True); ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--arms", default="")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--d2select", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    j = Job(a.node, a.system, a.seed, a.device, a.smoke)
    try:
        if a.d2select:
            j.d2select()
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
