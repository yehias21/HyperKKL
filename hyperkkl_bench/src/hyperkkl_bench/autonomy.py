"""Inclusion check 4a (PROTOCOL v2): non-trivial unforced dynamics (family A).

    from hyperkkl_bench.autonomy import check_autonomous
    rep = check_autonomous(system_config, dt, T, perturb=(0.0, 0.3))   # {'pass': bool, 'per_perturb': {...}}

As implemented (and as used for the v2 family decision, experiments/bench_v2/code/scan_autonomy.py): simulate at u = 0
from n_ic ICs of the IC box for `horizon` x T. Reference scale per state: sd_j = std of |x_j| pooled over ICs and the
last 80 % of the run. Pass when (1) max_j median_ICs [time-std of x_j over the last 20 %] / sd_j >= min_rel (0.05),
(2) the fraction of ICs whose largest late time-std exceeds 5 % of max_j sd_j is >= min_frac (0.9), and (3) that
late quantity divided by the same quantity over 20-40 % of the run is >= min_ratio (0.5). Uses only the bench plants.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _rk4(plant, x, u, h):
    a1 = plant.f(x, u); a2 = plant.f(x + h / 2 * a1, u); a3 = plant.f(x + h / 2 * a2, u); a4 = plant.f(x + h * a3, u)
    return x + h / 6 * (a1 + 2 * a2 + 2 * a3 + a4)


def scan(system_config, dt, T, perturb=0.0, n_ic=64, horizon=5.0, seed=0, max_steps=40000):
    from hyperkkl_bench.plants import build_plant
    pl = build_plant(system_config, perturb=perturb)
    box = np.asarray(system_config["ic_box"], float)
    rng = np.random.RandomState(seed)
    x0 = box[:, 0] + (box[:, 1] - box[:, 0]) * rng.rand(n_ic, len(box))
    N = min(int(math.ceil(horizon * T / dt)), max_steps)
    x = torch.as_tensor(x0, dtype=torch.float64)
    u = torch.zeros(n_ic, 1, dtype=torch.float64)
    ev = max(1, N // 2000)
    xs = []
    with torch.no_grad():
        for k in range(N):
            x = _rk4(pl, x, u, dt)
            if k % ev == 0:
                xs.append(x.clone())
    X = torch.stack(xs, 1).numpy()
    fin = np.isfinite(X).all((1, 2))
    X = X[fin]
    M = X.shape[1]
    early, late = X[:, int(.2 * M):int(.4 * M)], X[:, int(.8 * M):]
    sd = np.maximum(np.abs(X[:, int(.2 * M):]).std((0, 1)), 1e-9)
    ls, es_ = np.median(late.std(1) / sd, 0), np.median(early.std(1) / sd, 0)
    return {"finite": int(fin.sum()), "late_std_max": float(ls.max()), "ratio": float(ls.max() / max(es_.max(), 1e-12)),
            "frac_active": float((late.std(1).max(1) / sd.max() > 0.05).mean()), "T_sim": N * dt}


def check_autonomous(system_config, dt, T, perturb=(0.0,), min_rel=0.05, min_frac=0.9, min_ratio=0.5, **kw) -> dict:
    out = {}
    for p in perturb:
        r = scan(system_config, dt, T, perturb=p, **kw)
        r["pass"] = bool(r["late_std_max"] >= min_rel and r["frac_active"] >= min_frac and r["ratio"] >= min_ratio)
        out[p] = r
    return {"pass": all(r["pass"] for r in out.values()), "per_perturb": out,
            "rule": f"late std >= {min_rel} x ref for >= {min_frac:.0%} of ICs, late/early >= {min_ratio}, u = 0"}
