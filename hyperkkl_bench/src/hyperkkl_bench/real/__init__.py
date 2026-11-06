"""Real-data tracks: state estimation of real input-driven systems.

    import hyperkkl_bench.real as hr
    hr.list_real()                                   # ['battery_pf', ...]
    hr.fetch("battery_pf")                           # download from the original source (SHA-256 checked)
    hr.prepare("battery_pf")                         # raw -> frozen episodes (deterministic)
    train = hr.load("battery_pf", "train")           # RealEpisodeSet (EpisodeSet + prior, label_mask, meta)
    rep = hr.evaluate(est, "battery_pf")             # fit on train/val, score every test split
    print(rep.table())

The raw data are NEVER redistributed by this package: ``fetch`` downloads them from the original
host (or copies a local copy, ``from_local=``) and checks every file against its recorded SHA-256.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fetch import fetch  # noqa: F401
from .registry import DATASETS, list_real, module, spec  # noqa: F401


def prepare(dataset: str, **kw) -> dict:
    return module(dataset).prepare(**kw)


def load(dataset: str, split: str = "train", **kw):
    return module(dataset).load(split, **kw)


def splits(dataset: str) -> list:
    return module(dataset).splits()


def context(dataset: str, **kw) -> dict:
    return module(dataset).context(**kw)


def test_splits(dataset: str) -> list:
    return [s for s in splits(dataset) if s.startswith("test")]


@dataclass
class RealReport:
    meta: dict
    results: dict
    cost: dict = field(default_factory=dict)

    def headline(self):
        return self.meta.get("headline")

    def to_dict(self, per_episode=True):
        res = {s: {k: v for k, v in b.items() if per_episode or k != "per_episode"} for s, b in self.results.items()}
        return {"meta": self.meta, "results": res, "cost": self.cost}

    def save(self, path, per_episode=True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(per_episode), indent=1, default=float))
        return path

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["meta"], d["results"], d.get("cost", {}))

    def table(self) -> str:
        m = self.meta
        lines = [f"### {m['estimator']} on {m['dataset']}  (hyperkkl-bench {m['bench_version']}, scorer {m['scorer']})", "",
                 f"{m['metric_about']}", "", "| split | " + m["headline_metric"] + " | n |", "|---|---|---|"]
        for s, b in self.results.items():
            v = b["summary"].get(m["headline_metric"])
            lines.append(f"| {s} | {v:.4g} | {b['summary'].get('n_episodes')} |")
        if m.get("headline") is not None:
            lines.append(f"| **headline** | **{m['headline']:.4g}** | |")
        return "\n".join(lines)


def evaluate(est, dataset: str, splits_=None, fit: bool = True, n_episodes: int | None = None,
             verbose: bool = False, threads: int = 1) -> RealReport:
    """Fit ``est`` on the track's train / val, then run every test split (or ``splits_``) from the
    track's prior and score it with the track's scorer. A split whose condition differs (e.g. the
    0 degC splits of battery_pf) gets its own context: the estimator is re-fitted ONLY if it defines
    ``refit_for_condition = True`` (model-based filters swap the model; learned methods keep their fit)."""
    import torch
    from ..estimator import run_episodes
    from ..version import BENCH_VERSION, REAL_VERSION
    torch.set_num_threads(threads)
    mod = module(dataset)
    train, val = mod.load("train"), mod.load("val")
    ctx0 = mod.context()
    t0 = time.time()
    if fit:
        est.fit(train, val, ctx0)
    fit_s = time.time() - t0
    cond0 = getattr(mod, "condition_of", lambda _s: "default")("val")
    results, cond_now = {}, cond0
    for s in splits_ or test_splits(dataset):
        es = mod.load(s)
        if n_episodes is not None and n_episodes < len(es):
            es = es.subset(np.arange(n_episodes))
        cond = getattr(mod, "condition_of", lambda _s: "default")(s)
        if cond != cond_now:
            if hasattr(est, "set_context"):
                est.set_context(mod.context(condition=cond))
            elif getattr(est, "refit_for_condition", False):
                est.fit(train, val, mod.context(condition=cond))
            cond_now = cond
        t1 = time.time()
        Xh, _ = run_episodes(est, es.u.astype(np.float64), es.y.astype(np.float64),
                             np.asarray(es.prior, np.float64), np.tile(es.prior_std, (len(es), 1)))
        results[s] = mod.score(Xh, es)
        results[s]["summary"]["wall_s"] = time.time() - t1
        if verbose:
            print(f"  {dataset}/{s}: {mod.HEADLINE} {results[s]['summary'][mod.HEADLINE]:.4g} ({time.time() - t1:.0f}s)", flush=True)
    if cond_now != cond0:
        if hasattr(est, "set_context"):
            est.set_context(ctx0)
        elif getattr(est, "refit_for_condition", False):
            est.fit(train, val, ctx0)
    try:
        head = mod.headline(results)
    except KeyError:
        head = None
    meta = {"estimator": getattr(est, "name", type(est).__name__), "dataset": dataset, "bench_version": BENCH_VERSION,
            "real_version": REAL_VERSION,
            "data_version": getattr(mod, "PREP_VERSION", ""), "scorer": mod.SCORER, "headline_metric": mod.HEADLINE,
            "headline": head, "metric_about": getattr(mod, "METRIC_ABOUT", ""), "fit_s": fit_s,
            "fit_info": getattr(est, "fit_info", None), "n_episodes_limit": n_episodes}
    return RealReport(meta, results)
