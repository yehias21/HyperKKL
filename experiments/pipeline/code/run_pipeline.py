"""One pipeline-study job = (scenario, system, seed). Reads the live caches, writes only under
experiments/pipeline/.  Usage:
    python run_pipeline.py --scenario C3-partial --system vdp --seed 0 --device cuda:0 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hyperkkl import cache as Cm  # noqa: E402
from hyperkkl import data as Dm  # noqa: E402
from hyperkkl import baselines_data as BD  # noqa: E402
from hyperkkl import pipeline as PL  # noqa: E402
from hyperkkl.cache import MODEL_MODULES, module_hash  # noqa: E402
from hyperkkl.conditioning import ConditionedDecoder  # noqa: E402
from hyperkkl.config import deep_merge, digest  # noqa: E402
from hyperkkl.evaluate import build_observers, evaluate  # noqa: E402
from hyperkkl.observers import KKLObserver, LatentStep  # noqa: E402
from hyperkkl.train import phase1, phase2  # noqa: E402
from hyperkkl import systems_extra  # noqa: F401,E402

LIVE = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
LIVE_CACHE = LIVE / "experiments" / "cache"
OUT = LIVE / "experiments" / "pipeline"
PIPE_CACHE = OUT / "cache"
SCENARIOS = {"C3-partial": "C3-partial", "C3-mismatch30": "C3-mismatch30", "C1-where": "C1-where"}
P4_SCENARIOS = ("C3-partial", "C3-mismatch30")
OWN_DATA_BASELINES = {"C3-mismatch30": ["koopman_kf", "greybox_ekf", "greybox_ukf"]}   # tree has none merged
KINDS = ("lora", "hyper")

# ---- caches: read the live cache, write misses under experiments/pipeline/cache only ----------
Dm.CACHE = LIVE_CACHE
_orig_cached = Cm.cached


def _redirect(path):
    path = Path(path)
    if path.exists():
        return path
    try:
        rel = path.relative_to(LIVE_CACHE)
    except ValueError:
        rel = path.relative_to(PIPE_CACHE) if str(path).startswith(str(PIPE_CACHE)) else Path(path.name)
    return PIPE_CACHE / rel


def cached(path, build, **kw):
    return _orig_cached(_redirect(path), build, **kw)


Dm._cached = cached


def _to(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def run(scenario, system, seed, device, smoke=False):
    src = LIVE / "experiments" / "tree" / scenario / "runs" / system / f"seed{seed}" / "results.json"
    tree_res = json.load(open(src))
    cfg = tree_res["config"]
    cfg["seed"] = seed
    out_dir = OUT / ("smoke" if smoke else "runs") / scenario / system / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    if smoke:
        cfg = deep_merge(cfg, {"phase1": {"steps_enc": 60, "steps_dec": 60}, "phase2": {"steps": 30, "val_every": 15},
                               "eval": {"n_episodes": 64}})
    cfg["eval"] = deep_merge(cfg["eval"], {"conditions": ["clean"], "save_traj": False, "memory_probe": False})
    torch.manual_seed(seed); np.random.seed(seed)
    t_all = time.time()
    log(f"pipeline job {scenario} {system} seed {seed} device {device} smoke={smoke}")
    scales = Dm.system_scales(cfg, device)
    des = Dm.make_design(cfg["system"])
    data = Dm.training_data(cfg, seed, device)
    mh = module_hash(MODEL_MODULES)
    MC = LIVE_CACHE / "models"
    walls, info, coll = {}, {}, {}

    def timed(name, fn):
        t = time.time(); r = fn(); walls[name] = time.time() - t; return r

    # ---- P0: phase 1 + frozen phase 2 (live cache; trains only on a miss, into the pipeline cache)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    p1_path = MC / f"{system}_p1_{k1}.pt"
    p1_hit = p1_path.exists()
    base = _to(timed("P0.phase1", lambda: cached(p1_path, lambda: phase1(cfg, data, scales, des, device, seed, log))), device)
    trained = {}
    for kind in KINDS:
        k2 = digest(cfg, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
        pth = MC / f"{system}_p2_{kind}_{k2}.pt"
        hit = pth.exists()
        trained[f"P0-{kind}"] = _to(timed(f"P0.p2.{kind}", lambda: cached(
            pth, lambda: phase2(kind, cfg, data, scales, des, base, device, seed, log))), device)
        info[f"P0-{kind}"] = {"cache_hit": hit, "n_params": trained[f"P0-{kind}"]["n_params"],
                              "trainable_total": trained[f"P0-{kind}"]["n_params"] + trained[f"P0-{kind}"].get("ctx_params", 0),
                              "wall_s_train": trained[f"P0-{kind}"]["wall_s"], "val": trained[f"P0-{kind}"]["val"]}
    info["phase1_P0"] = {"cache_hit": p1_hit, "wall_s_train": base["wall_s"], "diag": base["diag"]}
    tag = "smoke_" if smoke else ""
    PC = PIPE_CACHE / "models"

    # ---- P1, P3
    for kind in KINDS:
        key = digest(cfg, "system", "data", "phase1", "phase2", extra=[seed, kind, mh, "P1", PL.BASE_LR_FACTOR])
        trained[f"P1-{kind}"] = _to(timed(f"P1.{kind}", lambda: cached(
            PC / f"{tag}{system}_P1_{kind}_{key}.pt",
            lambda: PL.phase2_variant("P1", kind, cfg, data, scales, des, base, device, seed, log))), device)
    key = digest(cfg, "system", "data", "phase1", "phase2", extra=[seed, mh, "P3", PL.BASE_LR_FACTOR])
    trained["P3"] = _to(timed("P3", lambda: cached(
        PC / f"{tag}{system}_P3_{key}.pt",
        lambda: PL.phase2_variant("P3", "uncond", cfg, data, scales, des, base, device, seed, log))), device)

    # ---- P2
    jsteps = PL.compute_plan("P2", cfg)["joint_steps"]
    for kind in KINDS:
        key = digest(cfg, "system", "data", "phase1", "phase2", extra=[seed, kind, mh, "P2", jsteps])
        trained[f"P2-{kind}"] = _to(timed(f"P2.{kind}", lambda: cached(
            PC / f"{tag}{system}_P2_{kind}_{key}.pt",
            lambda: PL.joint(kind, cfg, data, scales, des, device, seed, log))), device)

    # ---- data-driven fits: greybox residual for P4 (+ own baselines where the tree has none)
    fits, own = None, []
    if scenario in P4_SCENARIOS:
        own = OWN_DATA_BASELINES.get(scenario, [])
        names = sorted(set(["greybox_ekf"] + own))
        fits = timed("greybox_fit", lambda: BD.fit_all(dict(cfg, baselines=names), data, scales, des, device, seed, log))
        info["data_fits"] = fits["info"]
        data4 = PL.greybox_collocation(data, fits["greybox"]["model"])
        key = digest(cfg, "system", "data", "phase1", extra=[seed, mh, "P4", fits["info"]["greybox_ekf"]["val_loss"]])
        base4 = _to(timed("P4.phase1", lambda: cached(PC / f"{tag}{system}_P4p1_{key}.pt",
                                                        lambda: phase1(cfg, data4, scales, des, device, seed, log))), device)
        info["phase1_P4"] = {"wall_s_train": base4["wall_s"], "diag": base4["diag"],
                             "f0_rel_change": float(((data4["coll"]["f0"] - data["coll"]["f0"]).norm(dim=-1)
                                                     / data["coll"]["f0"].norm(dim=-1).clamp(min=1e-9)).median())}
        for kind in KINDS:
            key2 = digest(cfg, "system", "data", "phase1", "phase2", extra=[seed, kind, mh, "P4", key])
            trained[f"P4-{kind}"] = _to(timed(f"P4.p2.{kind}", lambda: cached(
                PC / f"{tag}{system}_P4p2_{kind}_{key2}.pt",
                lambda: phase2(kind, cfg, data, scales, des, base4, device, seed, log))), device)

    # ---- observers
    lat = LatentStep(des, device)
    obs = {"autonomous": KKLObserver("autonomous", lat, ConditionedDecoder(base["T0inv"], "none", 0, 0), z0_map=base["T0"])}
    for name, r in trained.items():
        z0 = r["T0"] if name.startswith("P2") else (base4["T0"] if name.startswith("P4") else base["T0"])
        obs[name] = KKLObserver(name, lat, r["cond"], ctx=r["ctx"], z0_map=z0)
    if scenario in P4_SCENARIOS:
        obs["P4-autonomous"] = KKLObserver("P4-autonomous", lat, ConditionedDecoder(base4["T0inv"], "none", 0, 0),
                                           z0_map=base4["T0"])
    if own:
        bo = build_observers(dict(cfg, baselines=own), des, scales, None, {}, None, device, fits)
        obs.update(bo)

    # ---- collapse diagnostics (zero-input episodes)
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float32, device=device)
    p1es = data["p1"]
    Ti0 = base["T0inv"]
    for name, r in trained.items():
        dc = r["ctx"].dim if r.get("ctx") is not None else 0
        own_base = getattr(r["cond"], "base", None)
        coll[name] = {"vs_phase1_base": PL.collapse(r["cond"], Ti0 if not name.startswith("P4") else base4["T0inv"],
                                                    p1es, sig, dc=dc)}
        if own_base is not None and name.split("-")[0] in ("P1", "P2") and hasattr(r["cond"], "kind") \
                and r["cond"].kind in ("lora", "hyper"):
            coll[name]["vs_own_base"] = PL.collapse(r["cond"], own_base, p1es, sig, dc=dc)
    coll["autonomous"] = PL.collapse(ConditionedDecoder(Ti0, "none", 0, 0), Ti0, p1es, sig, dc=0)

    for name, r in trained.items():
        if name.startswith("P0"):
            continue
        info[name] = {k: r.get(k) for k in ("n_params", "width", "ctx_params", "base_dec_params", "enc_params",
                                            "trainable_total", "wall_s", "val", "steps", "hist", "diag")}
    for name in ("P4-lora", "P4-hyper"):
        if name in info and info[name].get("trainable_total") is None:
            info[name]["trainable_total"] = info[name]["n_params"] + (info[name].get("ctx_params") or 0)
    t_eval = time.time()
    res = evaluate(cfg, des, scales, obs, device, log)
    walls["eval"] = time.time() - t_eval

    # ---- P0 reproduction check against the tree run (same episodes, same models)
    repro = {}
    for kind in KINDS:
        for tier in res:
            a = np.asarray(res[tier]["clean"]["methods"][f"P0-{kind}"]["nrmse"])
            b = np.asarray(tree_res["results"][tier]["clean"]["methods"][kind]["nrmse"])
            repro[f"{tier}/{kind}"] = float(np.max(np.abs(a - b) / np.maximum(b, 1e-12)))
    log(f"  P0 reproduction (max rel episode diff vs tree): {json.dumps({k: round(v, 6) for k, v in repro.items()})}")
    plans = {a: PL.compute_plan(a, cfg) for a in PL.ARMS}
    slim = {t: {"clean": {"families": res[t]["clean"]["families"],
                          "methods": {m: {k: v for k, v in d.items() if k in ("nrmse", "nrmse_unmeasured", "diverged", "max_nerr")}
                                      for m, d in res[t]["clean"]["methods"].items()}}} for t in res}
    payload = {"scenario": scenario, "system": system, "seed": seed, "device": device, "smoke": smoke,
               "source_run": str(src.relative_to(LIVE)), "results": slim, "info": info, "collapse": coll,
               "compute_plan": plans, "wall_s": walls, "total_wall_s": time.time() - t_all, "p0_repro": repro,
               "torch": torch.__version__, "host": os.uname().nodename}
    tmp = out_dir / "results.json.tmp"
    tmp.write_text(json.dumps(payload, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.rename(out_dir / "results.json")
    log(f"done in {time.time() - t_all:.0f}s  walls {json.dumps({k: round(v) for k, v in walls.items()})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS))
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    allowed = {"5", "6", "7"}
    until = os.environ.get("PIPELINE_EARLY_GPU_UNTIL", "")
    if until:                     # user update: GPUs 0-4 allowed until 05:45 CET on 2026-10-26 only
        import datetime as _dt
        lim = min(_dt.datetime.fromisoformat(until),
                  _dt.datetime(2026, 10, 26, 5, 45, tzinfo=_dt.timezone(_dt.timedelta(hours=1))))
        if _dt.datetime.now(_dt.timezone.utc) < lim:
            allowed |= {"0", "1", "2", "3", "4"}
    if a.device.startswith("cuda") and (not vis or any(v.strip() not in allowed for v in vis.split(","))):
        sys.exit(f"refusing to run: CUDA_VISIBLE_DEVICES={vis!r} must be within {sorted(allowed)}")
    torch.set_num_threads(2)
    try:
        run(a.scenario, a.system, a.seed, a.device, a.smoke)
    except Exception:
        d = OUT / ("smoke" if a.smoke else "runs") / a.scenario / a.system / f"seed{a.seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
