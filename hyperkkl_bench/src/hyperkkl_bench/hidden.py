"""Hidden test (labels withheld) for a future leaderboard.

File layout
    <extra root>/data/<plant>/<variant>/hidden_<tier>.parquet      PUBLIC: episode, family, signal, dt, u, y
                                                                   (no x, no x0)
    <labels root>/data/<plant>/<variant>/hidden_<tier>.parquet     PRIVATE: episode, x0, x
    <labels root>/SECRET                                           PRIVATE: seed secret of the generator
    <submission>/<plant>/<knowledge>/hidden_<tier>.npz             x_hat (E, N+1, n_x) float32, episode (E,)
    <submission>/SUBMISSION.json                                   estimator name, versions, cells

Workflow
    participant:  hkb.hidden.predict(est, plant, knowledge, "submission/")   (after est.fit on train/val)
    host:         hkb.hidden.score_hidden("submission/", labels_root=...)   -> per-cell report
The public tier generators are those of the public test tiers (same families, ranges, plant,
noise); only the seeds are secret. The hidden numbers are therefore comparable in expectation to
the public test numbers, but a method cannot have been tuned on these episodes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import conditions as C
from .data import HIDDEN_SPLITS, load, metadata
from .estimator import run_episodes
from .evaluate import _threads, score_block
from .paths import labels_root
from .registry import variant_of
from .version import BENCH_VERSION, DATA_VERSION, SCORER_VERSION


def predict(est, plant: str, knowledge: str, out_dir, splits=HIDDEN_SPLITS, extra=None) -> list:
    """Run a FITTED estimator on the hidden splits and write the submission files."""
    meta = metadata(plant)
    out, written = Path(out_dir), []
    for s in splits:
        es = load(plant, knowledge, s, extra=extra)
        assert es.x is None, "hidden split must not carry labels"
        x0, p0 = C.prior(meta, len(es), "center")
        with _threads(1):
            Xh, _ = run_episodes(est, es.u.astype(np.float64), es.y.astype(np.float64), x0, p0)
        p = out / plant / knowledge / f"{s}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, x_hat=Xh.astype(np.float32), episode=es.episode_ids)
        written.append(str(p))
    man = out / "SUBMISSION.json"
    d = json.loads(man.read_text()) if man.exists() else {"cells": []}
    d.update({"estimator": getattr(est, "name", type(est).__name__), "bench_version": BENCH_VERSION,
              "data_version": DATA_VERSION, "time": time.strftime("%Y-%m-%d %H:%M:%S %z")})
    d["cells"] = sorted(set(map(tuple, d["cells"])) | {(plant, knowledge)})
    man.write_text(json.dumps(d, indent=1))
    return written


def _labels(plant, knowledge, split, root):
    import pyarrow.parquet as pq
    from .data import _stack
    p = labels_root(root) / "data" / plant / variant_of(knowledge) / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"hidden labels not available at {p} (only the leaderboard host has them)")
    tb = pq.read_table(p)
    nx = int(metadata(plant)["n_x"])
    return _stack(tb.column("x"), nx), np.asarray(tb.column("episode").to_numpy())


def score_hidden(sub_dir, labels=None) -> dict:
    """Score a submission folder against the private labels. Invalid cells (missing episodes,
    wrong shape) are reported as errors, never silently skipped."""
    sub = Path(sub_dir)
    man = json.loads((sub / "SUBMISSION.json").read_text())
    out = {"estimator": man.get("estimator"), "bench_version": BENCH_VERSION, "scorer_version": SCORER_VERSION,
           "submission_bench_version": man.get("bench_version"), "cells": {}}
    for plant, knowledge in man["cells"]:
        meta = metadata(plant)
        cell = {}
        for s in HIDDEN_SPLITS:
            f = sub / plant / knowledge / f"{s}.npz"
            if not f.exists():
                cell[s] = {"error": "missing"}
                continue
            d = np.load(f)
            X, ep = _labels(plant, knowledge, s, labels)
            Xh = d["x_hat"].astype(np.float64)
            if Xh.shape != X.shape or not np.array_equal(d["episode"], ep):
                cell[s] = {"error": f"shape/episode mismatch: got {Xh.shape}, expected {X.shape}"}
                continue
            b = score_block(X, Xh, None, meta, float(meta["dt"]))
            b.pop("per_episode")
            cell[s] = b
        out["cells"][f"{plant}/{knowledge}"] = cell
    return out
