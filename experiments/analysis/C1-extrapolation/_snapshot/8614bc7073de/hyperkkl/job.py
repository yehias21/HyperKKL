"""One job = one (experiment node, system, seed): train what is missing, evaluate, save.

    python -m hyperkkl.job --node C1-where --system vdp --seed 0 [--device cuda:0]

Caching (experiments/cache/models): phase-1 maps are keyed by (system, data, phase1,
seed); each phase-2 method by (system, data, phase1, phase2, method, seed). A child node
that only changes evaluation conditions therefore re-trains nothing, and two nodes that
share a phase-1 configuration share identical base maps.

Output: experiments/tree/<node>/runs/<system>/seed<k>/results.json (+ log.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import data as Dm
from . import tree as Tr
from .cache import MODEL_MODULES, module_hash
from .cache import cached as _cached_model
from .config import ROOT, deep_merge, digest, load, parse_value
from .evaluate import build_observers, costs, evaluate
from .train import nlox, phase1, phase2

MODEL_CACHE = Dm.CACHE / "models"
PKG = Path(__file__).resolve().parent


RESULT_MODULES = ("cache", "conditioning", "config", "data", "design", "evaluate", "job",
                  "metrics", "nets", "observers", "signals", "systems", "train")


def code_hash() -> str:
    """Hash of the modules that can change a number (not the tree/report/scheduler code)."""
    h = hashlib.sha1()
    for m in RESULT_MODULES:
        p = PKG / f"{m}.py"
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:12]


def snapshot_code(ch: str) -> str:
    """Keep one tarball of the package per code hash, so any run can be re-created."""
    import tarfile
    out = ROOT / "experiments" / "code" / f"{ch}.tar.gz"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".tmp{np.random.randint(1 << 30)}")
        with tarfile.open(tmp, "w:gz") as tf:
            for p in sorted(PKG.glob("*.py")):
                tf.add(p, arcname=f"hyperkkl/{p.name}")
            for p in sorted((ROOT / "configs").rglob("*.yaml")):
                tf.add(p, arcname=str(p.relative_to(ROOT)))
        tmp.rename(out)
    return str(out.relative_to(ROOT))


def _to(obj, device):
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def method_cfg(cfg: dict, method: str) -> tuple:
    """'lora' | 'lora@gru' | 'lora@window:window=20,dim=8' -> (kind, config).

    The part after '@' overrides phase2.context.kind for this method only; the part
    after ':' overrides other phase2.context keys, so one node can sweep a context
    parameter across its methods."""
    kind, _, rest = method.partition("@")
    kind = "none" if kind == "autonomous" else kind
    ctx, _, kvs = rest.partition(":")
    over = {}
    if ctx:
        over["kind"] = ctx
    for kv in filter(None, kvs.split(",")):
        k, _, v = kv.partition("=")
        over[k] = parse_value(v)
    return kind, (deep_merge(cfg, {"phase2": {"context": over}}) if over else cfg)


def run_job(node_id: str, system: str, seed: int, device: str) -> dict:
    overrides = Tr.resolved_overrides(node_id)
    cfg = load(system, overrides)
    cfg["seed"] = seed
    run_dir = Tr.run_dir(node_id, system, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = open(run_dir / "log.txt", "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t_start = time.time()
    ch = code_hash()
    snap = snapshot_code(ch)        # at start: code edited while the job runs must not enter it
    torch.manual_seed(seed); np.random.seed(seed)
    log(f"node {node_id}  system {system}  seed {seed}  device {device}  code {ch}")
    scales = Dm.system_scales(cfg, device)
    des = Dm.make_design(cfg["system"])
    if not scales["design_check"]["ok"]:
        raise RuntimeError(f"design gate failed: {scales['design_check']}")
    log(f"  design: nz={des.A.shape[0]} alpha={des.alpha} dt={des.dt:.4f} N={des.N} "
        f"t_burn={scales['t_burn']:.2f}s memory={scales['memory_s']:.2f}s")
    data = Dm.training_data(cfg, seed, device)
    log(f"  data: p1={len(data['p1'])} train={len(data['train'])} val={len(data['val'])} "
        f"rejected={data['train'].n_rejected}")

    mh = module_hash(MODEL_MODULES)
    k1 = digest(cfg, "system", "data", "phase1", extra=[seed, mh])
    base = _to(_cached_model(MODEL_CACHE / f"{system}_p1_{k1}.pt",
                             lambda: phase1(cfg, data, scales, des, device, seed, log)), device)
    log(f"  phase1 diag: {json.dumps({k: round(v, 5) for k, v in base['diag'].items()})}")

    trained, train_info = {}, {}
    for m in cfg["methods"]:
        kind, mc = method_cfg(cfg, m)
        k2 = digest(mc, "system", "data", "phase1", "phase2", extra=[seed, kind, mh])
        r = _to(_cached_model(MODEL_CACHE / f"{system}_p2_{kind}_{k2}.pt",
                              lambda: phase2(kind, mc, data, scales, des, base, device, seed, log)), device)
        trained[m] = r
        train_info[m] = {k: r.get(k) for k in ("n_params", "width", "ctx_params", "wall_s", "val", "hist")}
    nl = None
    if "nlox" in cfg["baselines"]:
        k3 = digest(cfg, "system", "data", "nlox", extra=[seed, mh])
        nl = _to(_cached_model(MODEL_CACHE / f"{system}_nlox_{k3}.pt",
                               lambda: nlox(cfg, data, scales, des, device, seed, log)), device)
        train_info["nlox"] = {k: nl.get(k) for k in ("n_params", "wall_s", "val", "hist")}

    pf_tuning = None
    if "pf" in cfg["baselines"]:
        cfg["eval"]["pf_schedule"], pf_tuning = tune_pf(cfg, des, scales, data["val"], device, log)
    obs = build_observers(cfg, des, scales, base, trained, nl, device)
    res = evaluate(cfg, des, scales, obs, device, log)
    traj = {f"{t}/{c}": r.pop("_traj") for t, cs in res.items() for c, r in cs.items() if "_traj" in r}
    if traj:
        np.savez_compressed(run_dir / "traj.npz", **{f"{k}/{m}": v for k, d in traj.items() for m, v in d.items()})
    payload = {
        "node": node_id, "system": system, "seed": seed, "code": ch, "code_snapshot": snap,
        "host": platform.node(), "device": device, "wall_s": time.time() - t_start,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "scales": scales, "phase1": {"diag": base["diag"], "wall_s": base["wall_s"]},
        "train": train_info, "pf_tuning": pf_tuning,
        "cost": _costs(cfg, des, scales, base, trained, nl, data, device),
        "results": res, "config": cfg,
    }
    tmp = run_dir / "results.json.tmp"
    tmp.write_text(json.dumps(payload, default=_json_default))
    tmp.rename(run_dir / "results.json")
    log(f"done in {time.time() - t_start:.0f}s")
    return payload


def tune_pf(cfg, des, scales, val, device, log) -> tuple:
    """The particle filter's proposal inflation, chosen per (system, seed) on the
    VALIDATION episodes (never on evaluation data) from eval.pf_grid. A hard-coded 3x
    failed on lorenz (4/8 episodes diverged) while annealing 30 -> 1 tracked it."""
    from .evaluate import build_observers, initial_guess
    from .metrics import episode_metrics
    from .observers import run
    es = val.to(device)
    x0 = initial_guess(cfg, len(es), "center", 0, device)
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (es.x.shape[1] - 1))
    scores = []
    for g in cfg["eval"]["pf_grid"]:
        c = deep_merge(cfg, {"baselines": ["pf"], "eval": {"pf_schedule": g}})
        pf = build_observers(c, des, scales, None, {}, None, device)["pf"]
        xh = torch.cat([run(pf, es.y[i:i + 32], es.u[i:i + 32], x0[i:i + 32]) for i in range(0, len(es), 32)])
        m = episode_metrics(es.x, xh, scales["x_std"], settle)
        score = float(np.mean(m["nrmse"])) + float(np.mean(m["diverged"]))   # divergence dominates
        scores.append({**g, "val_nrmse": float(np.mean(m["nrmse"])), "val_diverged": float(np.mean(m["diverged"]))})
        log(f"  pf tuning {g}: val nrmse {scores[-1]['val_nrmse']:.4f} diverged {scores[-1]['val_diverged']:.2f}")
    best = min(range(len(scores)), key=lambda i: scores[i]["val_nrmse"] + scores[i]["val_diverged"])
    return dict(cfg["eval"]["pf_grid"][best]), scores


def _costs(cfg, des, scales, base, trained, nl, data, device) -> dict:
    """Online cost: GPU at batch 64, CPU (1 thread) at batch 1 -- the embedded setting.
    Runs last, because moving the modules to the CPU is in place."""
    nx, ny = data["train"].x.shape[-1], data["train"].y.shape[-1]
    gpu = costs(build_observers(cfg, des, scales, base, trained, nl, device), nx, ny, device)
    torch.set_num_threads(1)
    cpu = costs(build_observers(cfg, des, scales, _to(base, "cpu"), _to(trained, "cpu"),
                                _to(nl, "cpu"), "cpu"), nx, ny, "cpu")
    return {m: {"us_gpu_b64": gpu[m].get("us_b64"), "us_cpu_b1": cpu[m].get("us_b1")} for m in gpu}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    torch.set_num_threads(2)
    try:
        run_job(a.node, a.system, a.seed, a.device)
    except Exception:
        d = Tr.run_dir(a.node, a.system, a.seed)
        d.mkdir(parents=True, exist_ok=True)
        (d / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
