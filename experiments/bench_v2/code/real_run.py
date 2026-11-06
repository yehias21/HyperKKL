"""One bench_v2 real-track job through the benchmark's own real-data API (hkb_snapshot.real.evaluate: the track's
train / val, its priors, its scorer), on equal data for every method.

    python real_run.py battery_pf ude1_ekf_gated --seed 0        (CPU methods)
    CUDA_VISIBLE_DEVICES=3 python real_run.py battery_pf hyperkkl_a3 --seed 0

methods: ekf_gated, ukf_gated, ekf_gated_tuned, ukf_gated_tuned (deterministic, seed 0), ude1_{ekf,ukf}_gated
(corrected grey-box, K = 1), ude32_{ekf,ukf}_gated (multi-step K = 32), node32_ekf_gated, gru (2 x 64 GRU),
hyperkkl_a3 (HyperKKLTrain: A3 recipe, GPU subprocess; C5 is not available in the real-data trainer, stated).
Output runs/real/<dataset>/<method>/seed<k>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import env
os.environ.setdefault("HYPERKKL_BENCH_REAL", str(env.BENCH / "data_real"))
import numpy as np  # noqa: E402


def _RealTuned(kind, name):
    """hyperkkl_bench TunedKF selects on val from context['prior_mean'], which the real tracks do not provide (they
    give a per-episode prior; FINDING for the port). Same grid and score, each val episode starts from its own track
    prior (es.prior), exactly as the scorer starts the test episodes."""
    import numpy as np
    import torch
    from hyperkkl_bench.baselines.filters import TunedKF

    class RealTunedKF(TunedKF):
        def fit(self, train, val, context):
            if getattr(val, "prior", None) is None:
                return super().fit(train, val, context)
            from hyperkkl_bench.metrics import episode_metrics
            self.context = context
            meta = context["meta"]
            idx = np.arange(min(self.n_val, len(val)))
            sub = val.subset(idx)
            X0 = np.asarray(sub.prior, np.float64)
            P0 = np.tile(np.asarray(sub.prior_std, np.float64), (len(sub), 1))
            lab = np.isfinite(sub.x).all(axis=(0, 1))
            sens = set(meta["sensors"])
            unm = [j for j in range(sub.x.shape[-1]) if j not in sens and lab[j]]
            scores = {}
            with torch.no_grad():
                for q in self.q_grid:
                    for r in self.r_grid:
                        f = self._make(q, r)
                        f.fit(train, val, context)
                        xh, _ = f.run_batch(sub.u.astype(float), sub.y.astype(float), X0, P0)
                        m = episode_metrics(np.nan_to_num(sub.x[..., unm]), xh[..., unm],
                                            np.asarray(meta["x_std"], float)[unm], int(meta["settle_index"]), [])
                        scores[f"q{q}_r{r}"] = float(np.mean(m["nrmse"]))
            best = min(scores, key=scores.get)
            q, r = (float(t[1:]) for t in best.split("_"))
            self.inner = self._make(q, r)
            self.inner.fit(train, val, context)
            self.fit_info = {"chosen": {"q_inflate": q, "r_inflate": r}, "val_scores": scores, "n_val": len(sub),
                             "note": "per-episode track prior (bench TunedKF needs prior_mean)"}
    return RealTunedKF(kind, gate=True, name=name)


def make(name, seed):
    from hyperkkl_bench.baselines import REGISTRY, UDEFilter, GRUObserver
    if name in ("ekf_gated_tuned", "ukf_gated_tuned"):
        return _RealTuned(name.split("_")[0], name=name)
    if name in ("ekf_gated", "ukf_gated"):
        return REGISTRY[name]()
    if name.startswith(("ude1_", "ude32_", "node32_")):
        K = 1 if name.startswith("ude1_") else 32
        kind = "ukf" if "_ukf_" in name else "ekf"
        return UDEFilter(hybrid=not name.startswith("node"), kind=kind, gate=True, K=K, seed=seed, name=name)
    if name == "gru":
        return GRUObserver(hidden=64, layers=2, steps=3000, seed=seed, name="gru")
    if name == "hyperkkl_a3":
        from hyperkkl_bench.baselines.hyperkkl_train import HyperKKLTrain
        g = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        return HyperKKLTrain(seed=seed, device="cuda:0" if g else "cpu", gpu=g or None)
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("method"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=None)
    a = ap.parse_args()
    out = env.RUNS / "real" / a.dataset / (a.method + ("_smoke" if a.n_episodes else ""))
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"seed{a.seed}.json"
    if f.exists():
        print("done already"); return
    try:
        env.check_imports()
        import hyperkkl_bench.real as hr
        t0 = time.time()
        est = make(a.method, a.seed)
        rep = hr.evaluate(est, a.dataset, n_episodes=a.n_episodes, verbose=True, threads=2)
        d = {"meta": rep.meta, "results": rep.results, "seed": a.seed, "method": a.method, "wall_s": time.time() - t0}
        tmp = f.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(d, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, f)
        print(f"done {a.dataset} {a.method} seed {a.seed}: headline {rep.meta.get('headline')} ({time.time() - t0:.0f}s)")
    except Exception:
        (out / f"seed{a.seed}.error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
