"""The standard evaluation: fit once, score on every tier x condition, measure cost, report.

    from hyperkkl_bench import evaluate
    rep = evaluate(est, "vdp", "realistic")                # fits on train/val, scores test tiers
    print(rep.table()); rep.save("vdp_realistic.json")

Rules enforced here (PROTOCOL.md): the estimator is fitted ONLY on ``train`` (+ ``val`` for model
selection); it is never shown test data or x of the test episodes; every test episode starts
from the same prior (centre of the IC box, std = box width / sqrt 12) unless the condition is
``poor_init``; the same fitted estimator faces every condition.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import conditions as C
from . import metrics as M
from .context import build_context
from .cost import measure as measure_cost
from .data import TEST_TIERS, load, metadata
from .estimator import run_episodes
from .version import BENCH_VERSION, DATA_VERSION, SCORER_VERSION

HEADLINE = "nrmse_unmeasured"


@dataclass
class Report:
    meta: dict
    results: dict                          # results[split][condition] = {summary, ci, per_episode, calibration}
    cost: dict = field(default_factory=dict)

    def to_dict(self, per_episode: bool = True) -> dict:
        res = {s: {c: {k: v for k, v in b.items() if per_episode or k != "per_episode"}
                   for c, b in cs.items()} for s, cs in self.results.items()}
        return {"meta": self.meta, "results": res, "cost": self.cost}

    def save(self, path, per_episode: bool = True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(per_episode), indent=1, default=_js))
        return path

    @classmethod
    def load(cls, path) -> "Report":
        d = json.loads(Path(path).read_text())
        return cls(d["meta"], d["results"], d.get("cost", {}))

    def value(self, split="test_sequence_ood", condition="clean", metric=HEADLINE):
        return self.results[split][condition]["summary"].get(metric)

    def table(self, metric: str = HEADLINE) -> str:
        m = self.meta
        conds = sorted({c for cs in self.results.values() for c in cs}, key=list(C.CONDITIONS).index)
        lines = [f"### {m['estimator']} on {m['plant']} / {m['knowledge']}  "
                 f"(hyperkkl-bench {m['bench_version']}, {m['data_version']})", "",
                 f"settled {metric} (mean over episodes, 95% episode-bootstrap CI); div = diverged fraction", "",
                 "| split | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for s, cs in self.results.items():
            cells = []
            for c in conds:
                b = cs.get(c)
                if b is None:
                    cells.append("")
                    continue
                v, ci = b["summary"].get(metric), b["ci"].get(metric, [None, None])
                d = b["summary"].get("div_rate", 0.0)
                cell = "n/a" if v is None else f"{v:.4f}"
                if ci[0] is not None:
                    cell += f" [{ci[0]:.4f}, {ci[1]:.4f}]"
                if d:
                    cell += f" div {d:.2f}"
                cells.append(cell)
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        cal = [(s, cs["clean"]["calibration"]) for s, cs in self.results.items()
               if "clean" in cs and cs["clean"].get("calibration")]
        if cal:
            lines += ["", "| calibration (clean) | coverage68 | coverage95 | ANEES | NEES in 95% |", "|---|---|---|---|---|"]
            for s, c in cal:
                f = lambda v: "n/a" if v is None else f"{v:.3f}"
                lines.append(f"| {s} | {f(c['coverage68'])} | {f(c['coverage95'])} | {f(c['anees'])} | {f(c['nees_in_95'])} |")
        if self.cost:
            c = self.cost
            lines += ["", f"cost: params {c.get('n_params')}, state {c.get('state_bytes')} B, "
                          f"FLOPs/step {c.get('flops_per_step')} ({c.get('flops_source')}), "
                          f"latency {c.get('latency_us_median', float('nan')):.1f} us median (batch 1, 1 CPU thread, Python), "
                          f"MCU estimate {c.get('mcu_us_est', float('nan')):.1f} us"
                          + ("" if c.get("within_latency_budget") is None else
                             f"; within budget: latency {c.get('within_latency_budget')}, memory {c.get('within_memory_budget')}")]
        if self.meta.get("input_ood_gap"):
            g = self.meta["input_ood_gap"]
            lines += ["", "input-OOD gap (ratio to test_id, clean): " +
                      ", ".join(f"{k} {v:.3f}" for k, v in g.items() if v is not None)]
        return "\n".join(lines)


def _js(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


class _threads:
    def __init__(self, n):
        self.n = n

    def __enter__(self):
        try:
            import torch
        except ImportError:
            self.n = None
            return
        if self.n:
            self.old = torch.get_num_threads()
            torch.set_num_threads(self.n)

    def __exit__(self, *a):
        if self.n:
            import torch
            torch.set_num_threads(self.old)


def score_block(X, Xh, P, meta, dt, cond_info=None) -> dict:
    settle = int(meta["settle_index"])
    m = M.episode_metrics(X, Xh, meta["x_std"], settle, meta["sensors"])
    summ = M.summarize(m)
    ci = {k: M.bootstrap_ci(m[k]) for k in ("nrmse", "nrmse_unmeasured", "nrmse_measured")}
    b = {"summary": summ, "ci": ci, "per_episode": {k: np.asarray(v).tolist() for k, v in m.items()},
         "n_episodes": int(len(X))}
    if P is not None:
        b["calibration"] = M.calibration(X, Xh, P, settle, m["diverged"])
    if cond_info:
        b["condition"] = cond_info
    return b


def evaluate(estimator, plant: str, knowledge: str = "exact", splits=None,
             conditions=("clean", "outliers", "burst_outage", "poor_init"), fit: bool = True,
             train_n: int | None = None, train_seed: int = 0, cost: bool = True,
             n_episodes: int | None = None, root=None, extra=None, verbose: bool = True,
             budget_us=None, budget_bytes=None, threads: int | None = 1) -> Report:
    """Fit (unless fit=False) and score ``estimator``.

    splits      default: the three public test tiers. 'val_ood' / 'val' allowed for model selection.
    conditions  subset of conditions.CONDITIONS (robustness conditions run on the test tiers).
    train_n     data-efficiency hook: fit on a nested subset of n training episodes (seed train_seed).
    n_episodes  score only the first n episodes of each split (smoke tests; not a reportable number).
    threads     torch intra-op threads while RUNNING the estimator (default 1: small batched filters are
                5-10x slower with many threads on a busy machine); None = leave unchanged. fit() is
                not affected.
    """
    meta = metadata(plant, root)
    ctx = build_context(plant, knowledge, root)
    splits = list(splits or [f"test_{t}" for t in TEST_TIERS])
    log = print if verbose else (lambda *a, **k: None)
    t0 = time.time()
    if fit:
        tr = load(plant, knowledge, "train", n=train_n, seed=train_seed, root=root)
        va = load(plant, knowledge, "val", root=root)
        log(f"[hkb] fitting {getattr(estimator, 'name', type(estimator).__name__)} on {plant}/{knowledge}: "
            f"{len(tr)} train, {len(va)} val episodes")
        estimator.fit(tr, va, ctx)
        del tr
    elif not hasattr(estimator, "context"):
        estimator.context = ctx
    t_fit = time.time() - t0
    results = {}
    es = None
    for s in splits:
        es = load(plant, knowledge, s, root=root, extra=extra)
        if n_episodes:
            es = es.subset(np.arange(min(n_episodes, len(es))))
        results[s] = {}
        conds = conditions if s.startswith("test_") else ("clean",)
        clean_med = None
        for c in conds:
            t1 = time.time()
            Y, X0, P0, info = C.apply(c, es, meta)
            with _threads(threads):
                Xh, P = run_episodes(estimator, es.u.astype(np.float64), Y, X0, P0)
            b = score_block(es.x, Xh, P, meta, es.dt, info)
            if c == "clean":
                en, _ = M._norm_err(es.x, Xh, meta["x_std"])
                clean_med = float(np.median(np.sqrt((en[:, int(meta["settle_index"]):] ** 2).mean(-1))))
            if c == "poor_init" and clean_med is not None:
                tc = M.conv_time(es.x, Xh, meta["x_std"], es.dt, float(meta["t_ref"]),
                                 max(2 * clean_med, M.CONV_BAND))
                b["summary"]["t_conv_median_s"] = float(np.nanmedian(tc)) if np.isfinite(tc).any() else None
                b["summary"]["conv_censored"] = float(np.mean(~np.isfinite(tc)))
            results[s][c] = b
            log(f"[hkb]   {s:18s} {c:13s} {HEADLINE} {b['summary'].get(HEADLINE, float('nan')):.4f}  "
                f"nrmse {b['summary']['nrmse']:.4f}  div {b['summary']['div_rate']:.2f}  ({time.time() - t1:.0f}s)")
    rep_meta = {"estimator": getattr(estimator, "name", type(estimator).__name__), "plant": plant,
                "knowledge": knowledge, "bench_version": BENCH_VERSION, "data_version": DATA_VERSION,
                "scorer_version": SCORER_VERSION, "train_n": train_n, "train_seed": train_seed,
                "fit_s": t_fit, "n_episodes_override": n_episodes,
                "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "fit_info": getattr(estimator, "fit_info", None)}
    if "test_id" in results:
        base = results["test_id"]["clean"]["summary"].get(HEADLINE)
        gap = {s: (results[s]["clean"]["summary"].get(HEADLINE) / base if base else None)
               for s in ("test_shape_ood", "test_sequence_ood") if s in results}
        if gap:
            rep_meta["input_ood_gap"] = gap
    cst = measure_cost(estimator, es, budget_us=budget_us, budget_bytes=budget_bytes) if cost and es is not None else {}
    return Report(rep_meta, results, cst)


def data_efficiency(make_estimator, plant, knowledge, sizes=(50, 150, 600), seeds=(0,), **kw) -> dict:
    """Fit a fresh estimator on nested subsets of the training split and score the clean test
    tiers. Returns {size: {seed: {split: headline}}}."""
    out = {}
    for n in sizes:
        out[n] = {}
        for sd in seeds:
            rep = evaluate(make_estimator(sd), plant, knowledge, conditions=("clean",), train_n=n,
                           train_seed=sd, cost=False, **kw)
            out[n][sd] = {s: rep.value(s) for s in rep.results}
    return out
