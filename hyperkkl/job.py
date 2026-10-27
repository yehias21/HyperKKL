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

from . import baselines_data as BD
from . import data as Dm
from . import tree as Tr
from .cache import MODEL_MODULES, module_hash
from .cache import cached as _cached_model
from .config import ROOT, deep_merge, digest, load, parse_value
from .evaluate import ORACLE, build_observers, costs, evaluate, memory_probe
from .train import nlox, phase1, phase2
from . import systems_extra  # noqa: F401,E402  (battery, network: a run records its code, RESULT_MODULES)

MODEL_CACHE = Dm.CACHE / "models"
PKG = Path(__file__).resolve().parent


RESULT_MODULES = ("baselines_data", "cache", "conditioning", "config", "data", "design", "evaluate", "job", "monitor",
                  "metrics", "nets", "observers", "signals", "systems", "systems_extra", "train")


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


# phase-2 keys a method string may set in brackets: kind[key=value,...]
METHOD_KEYS = ("lr", "rank", "steps", "warmup", "budget", "budget_includes_context", "rollout_K",
               "episodes_per_batch", "times_per_episode", "gen_layers", "film_depth", "concat_depth",
               "latent_depth", "ctx_rms")


def _kv(s: str, what: str) -> dict:
    out = {}
    for kv in filter(None, s.split(",")):
        k, eq, v = kv.partition("=")
        if not eq or not k.strip():
            raise ValueError(f"bad {what} override {kv!r} (expected key=value)")
        out[k.strip()] = parse_value(v)
    return out


def method_cfg(cfg: dict, method: str) -> tuple:
    """'lora' | 'lora@gru' | 'lora@window:window=20,dim=8' | 'hyper[lr=0.001,gen_layers=2]'
    | 'lora[rank=16]@gru:dim=8' -> (kind, config).

    The part after '@' overrides phase2.context.kind for this method only; the part
    after ':' overrides other phase2.context keys, so one node can sweep a context
    parameter across its methods. The bracketed part right after the kind overrides
    phase-2 keys (METHOD_KEYS only, e.g. a per-method learning rate or architecture knob:
    the C1 tuning study). Without brackets the config is returned unchanged (same object)."""
    head, _, rest = method.partition("@")
    kind, br, p2s = head.partition("[")
    if br:
        if not p2s.endswith("]"):
            raise ValueError(f"method {method!r}: unclosed '['")
        p2 = _kv(p2s[:-1], "phase-2")
        bad = sorted(set(p2) - set(METHOD_KEYS))
        if bad:
            raise ValueError(f"method {method!r}: unknown phase-2 key(s) {bad}; allowed {METHOD_KEYS}")
    else:
        p2 = {}
    kind = "none" if kind == "autonomous" else kind
    ctx, _, kvs = rest.partition(":")
    over = _kv(kvs, "context")
    if ctx:
        over["kind"] = ctx
    if over:
        p2 = {**p2, "context": over}
    return kind, (deep_merge(cfg, {"phase2": p2}) if p2 else cfg)


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
        train_info[m] = {k: r.get(k) for k in ("n_params", "width", "ctx_params", "wall_s", "val", "hist", "opt")}
        for k in ("lr", "rank", "arch", "ctx_rms", "ctx_scale"):           # recorded by phase 2 since the tuning study
            if r.get(k) is not None:
                train_info[m][k] = r[k]
    if cfg["eval"].get("val_nrmse"):           # deployed-observer NRMSE on data.val (selection)
        for m, v in validation_nrmse(cfg, des, scales, base, trained, data["val"], device, log).items():
            train_info[m].update(v)
    nl = None
    if "nlox" in cfg["baselines"]:
        k3 = digest(cfg, "system", "data", "nlox", extra=[seed, mh])
        nl = _to(_cached_model(MODEL_CACHE / f"{system}_nlox_{k3}.pt",
                               lambda: nlox(cfg, data, scales, des, device, seed, log)), device)
        train_info["nlox"] = {k: nl.get(k) for k in ("n_params", "wall_s", "val", "hist", "opt")}

    pf_tuning = None
    if "pf" in cfg["baselines"]:
        cfg["eval"]["pf_schedule"], pf_tuning = tune_pf(cfg, des, scales, data["val"], device, log)
    oracle_tuning = None
    if any(b in ORACLE or b == "greybox_ekf_oracle" for b in cfg["baselines"]):
        cfg["eval"]["oracle_tuning"], oracle_tuning = tune_oracle(cfg, des, scales, data["val"], device, log)
    data_fits = None
    if BD.wanted(cfg):                  # data-driven baselines: fitted once, held in memory only
        t_f = time.time()
        data_fits = BD.fit_all(cfg, data, scales, des, device, seed, log)
        train_info.update(data_fits["info"])
        log(f"  data-driven baselines fitted in {time.time() - t_f:.0f}s")
    obs = build_observers(cfg, des, scales, base, trained, nl, device, data_fits)
    t_eval, ev_timing = time.time(), {}
    res = evaluate(cfg, des, scales, obs, device, log, timing=ev_timing)
    t_eval = time.time() - t_eval
    probe = None
    if cfg["eval"].get("memory_probe"):
        t_p = time.time()
        probe = memory_probe(cfg, des, scales, trained, device, log)
        ev_timing["_memory_probe_s"] = time.time() - t_p
    coverage = state_coverage(cfg, data, scales, device, log)
    traj = {f"{t}/{c}": r.pop("_traj") for t, cs in res.items() for c, r in cs.items() if "_traj" in r}
    if traj:
        np.savez_compressed(run_dir / "traj.npz", **{f"{k}/{m}": v for k, d in traj.items() for m, v in d.items()})
    payload = {
        "node": node_id, "system": system, "seed": seed, "code": ch, "code_snapshot": snap,
        "host": platform.node(), "device": device, "wall_s": time.time() - t_start,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "numpy": np.__version__, "cuda": torch.version.cuda},
        "scales": scales, "phase1": {"diag": base["diag"], "wall_s": base["wall_s"],
                   "opt_enc": base.get("opt_enc"), "opt_dec": base.get("opt_dec")},
        "train": train_info, "pf_tuning": pf_tuning, "oracle_tuning": oracle_tuning,
        "cost": _costs(cfg, des, scales, base, trained, nl, data, device, data_fits),
        "results": res, "state_coverage": coverage, "config": cfg,
        "eval_wall_s": t_eval, "eval_timing": ev_timing, "memory_probe": probe,
    }
    tmp = run_dir / "results.json.tmp"
    tmp.write_text(json.dumps(payload, default=_json_default))
    tmp.rename(run_dir / "results.json")
    log(f"done in {time.time() - t_start:.0f}s")
    return payload


def validation_nrmse(cfg, des, scales, base, trained, val, device, log) -> dict:
    """Deployed-observer error of every learned method on the phase-2 VALIDATION episodes
    (data.val; never evaluation data): the exact latent filter from the centre-of-box prior,
    decoded through the trained conditioner, scored by the settled NRMSE of
    metrics.episode_metrics (divergence clamped), exactly as a clean evaluation block.
    Used for model selection (experiments/tuning/PROTOCOL.md)."""
    from .evaluate import _run_chunks, build_observers, initial_guess
    from .metrics import episode_metrics
    es = val.to(device)
    x0 = initial_guess(cfg, len(es), "center", 0, device)
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (es.x.shape[1] - 1))
    obs = build_observers(deep_merge(cfg, {"baselines": []}), des, scales, base, trained, None, device)
    out = {}
    with torch.no_grad():
        for m, ob in obs.items():
            xh = _run_chunks(ob, es.y, es.u, x0, int(cfg["eval"].get("chunk", 1024)))
            em = episode_metrics(es.x, xh, scales["x_std"], settle)
            out[m] = {"val_nrmse": float(np.mean(em["nrmse"])), "val_diverged": float(np.mean(em["diverged"])),
                      "val_n": len(es)}
            log(f"  val nrmse {m:32s} {out[m]['val_nrmse']:.4f}")
    return out


def state_coverage(cfg, data, scales, device, log) -> dict:
    """Per evaluation tier: how many evaluation states (after burn-in) lie outside the
    states seen in training (metrics.state_coverage). Inputs are in-bounds in every tier,
    so this measures how far a shape / sequence shift moves the STATE distribution."""
    from .metrics import state_coverage as cov
    tr = data["train"]
    out = {}
    for tier in cfg["eval"]["tiers"]:
        es = Dm.eval_data(cfg, tier, device=device)
        out[tier] = cov(tr.x, es.x[:, es.burn:], scales["x_std"], seed=0)
        log(f"  state coverage {tier:12s}: outside train box {100 * out[tier]['outside_box']:.2f}% "
            f"(per state {[round(100 * v, 2) for v in out[tier]['outside_box_j']]}), "
            f"kNN > train q99 {100 * out[tier]['outside_knn']:.2f}%  n_rej {es.n_rejected}")
    return out


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


def tune_oracle(cfg, des, scales, val, device, log, n_max: int = 32) -> tuple:
    """Oracle-filter tuning, per (system, seed) on VALIDATION episodes (as tune_pf): every
    (q_inflate, r_inflate) of eval.oracle_grid is scored by the settled NRMSE (+ divergence
    rate) of ekf_oracle under the job's own sensor and plant (mismatch); the choice is then used
    by ekf_oracle, ukf_oracle and ekf_oracle_huber in every condition block (each block's
    sensor still sets R through oracle_noise). q_inflate multiplies the model-error variance
    only, so with an exact model (knowledge full, no mismatch) only r_inflate is searched."""
    from .evaluate import build_observers, initial_guess
    from .metrics import episode_metrics
    from .observers import run
    grid = cfg["eval"].get("oracle_grid") or {"q_inflate": [1.0], "r_inflate": [1.0]}
    exact = not float(cfg["system"].get("mismatch", 0.0)) and cfg["system"].get("knowledge", "full") == "full"
    qs = [1.0] if exact else [float(q) for q in grid.get("q_inflate", [1.0])]
    pts = [{"q_inflate": q, "r_inflate": float(r)} for q in qs for r in grid.get("r_inflate", [1.0])]
    es = val.select(range(min(n_max, len(val)))).to(device)
    x0 = initial_guess(cfg, len(es), "center", 0, device)
    settle = int(float(cfg["eval"].get("settle_frac", 0.5)) * (es.x.shape[1] - 1))
    scores = []
    for g in pts:
        c = deep_merge(cfg, {"baselines": ["ekf_oracle"], "eval": {"oracle_tuning": g}})
        f = build_observers(c, des, scales, None, {}, None, device)["ekf_oracle"]
        xh = torch.cat([run(f, es.y[i:i + 32], es.u[i:i + 32], x0[i:i + 32]) for i in range(0, len(es), 32)])
        m = episode_metrics(es.x, xh, scales["x_std"], settle)
        scores.append({**g, "val_nrmse": float(np.mean(m["nrmse"])), "val_diverged": float(np.mean(m["diverged"]))})
        log(f"  oracle tuning {g}: val nrmse {scores[-1]['val_nrmse']:.4f} diverged {scores[-1]['val_diverged']:.2f}")
    best = min(range(len(scores)), key=lambda i: scores[i]["val_nrmse"] + scores[i]["val_diverged"])
    return dict(pts[best]), {"choice": dict(pts[best]), "scores": scores, "n_val": len(es)}


def _costs(cfg, des, scales, base, trained, nl, data, device, data_fits=None) -> dict:
    """Online cost: GPU at batch 64, CPU (1 thread) at batch 1 -- the embedded setting.
    Runs last, because moving the modules to the CPU is in place."""
    nx, ny = data["train"].x.shape[-1], data["train"].y.shape[-1]
    gpu = costs(build_observers(cfg, des, scales, base, trained, nl, device, data_fits), nx, ny, device)
    torch.set_num_threads(1)
    cpu = costs(build_observers(cfg, des, scales, _to(base, "cpu"), _to(trained, "cpu"),
                                _to(nl, "cpu"), "cpu", data_fits), nx, ny, "cpu")
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
