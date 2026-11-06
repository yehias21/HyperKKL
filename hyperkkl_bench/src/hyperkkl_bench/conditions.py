"""Robustness conditions (configs/base.yaml ``conditions`` of the paper), applied on top of the
frozen test episodes. Deterministic in (plant, knowledge, split, condition). No estimator is told
which condition it faces: it must use the same fitted model and the nominal settings.

    clean          the stored y; prior = centre of the IC box
    outliers       2 % of samples (per output) get +-12 y_std added (random sign)
    burst_outage   the sensor freezes at k0 = round(0.6 N) and repeats y[k0] for 2 t_ref
                   (sample-and-hold outage; all outputs)
    poor_init      prior mean = a far corner: centre + 0.9 half-width x random sign per state

Paper equivalence: ``outliers`` and ``burst_outage`` in the paper were drawn by re-measuring the
clean output with the merged sensor model; here they are drawn on top of the stored noisy y with
an independent seeded stream (same distribution, different realisation), so the absolute numbers
differ from the paper's tables at the episode level but not in expectation.
"""

from __future__ import annotations

import hashlib

import numpy as np

CONDITIONS = {
    "clean": {},
    "outliers": {"outlier_rate": 0.02, "outlier_scale": 12.0},
    "burst_outage": {"burst": {"start_frac": 0.6, "len_tref": 2.0}},
    "poor_init": {"init": "poor"},
}


def _seed(*keys) -> int:
    return int(hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest(), 16) % (2 ** 31)


def prior(meta: dict, n: int, kind: str = "center", seed: int = 0):
    box = np.asarray(meta["ic_box"], float)
    ctr, half = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    if kind == "center":
        x0 = np.tile(ctr, (n, 1))
    elif kind == "poor":
        x0 = ctr + 0.9 * half * np.random.RandomState(seed).choice([-1.0, 1.0], size=(n, len(ctr)))
    else:
        raise ValueError(kind)
    p0 = np.tile((box[:, 1] - box[:, 0]) / np.sqrt(12.0), (n, 1))     # prior std (uniform box)
    return x0, p0


def burst_window(burst: dict, N1: int, dt: float, t_ref: float) -> tuple:
    k0 = int(round(float(burst.get("start_frac", 0.6)) * (N1 - 1)))
    k0 = min(max(k0, 0), N1 - 1)
    L = int(round(float(burst.get("len_tref", 2.0)) * float(t_ref) / float(dt)))
    return k0, max(0, min(L, N1 - 1 - k0))


def spec_for(cond: str, meta: dict) -> dict:
    """The condition as applied to this plant. Deployment-track plants carry their own physically
    motivated versions in metadata['robust_conditions'] (casestudies: 1 % outliers at 8 sigma; GPS /
    concentration-sensor loss on channel 0 for 5 / 1.5 t_ref from 0.5 T)."""
    if cond not in CONDITIONS:
        raise ValueError(f"unknown condition {cond!r}; choose from {list(CONDITIONS)}")
    rc = meta.get("robust_conditions")
    if rc and cond == "outliers" and "outliers" in rc:
        return dict(rc["outliers"]["sensor"])
    if rc and cond == "burst_outage":
        for k, v in rc.items():
            if "burst" in (v.get("sensor") or {}):
                return {"burst": dict(v["sensor"]["burst"]), "name": k}
    return CONDITIONS[cond]


def apply(cond: str, es, meta: dict):
    """-> (Y, X0_prior, P0_prior_std, info) for a condition on an EpisodeSet."""
    spec = spec_for(cond, meta)
    s = _seed(es.plant, es.knowledge, es.split, cond)
    Y = es.y.astype(np.float64).copy()
    E, N1, ny = Y.shape
    ys = np.asarray(meta["y_std"], float)
    info = {}
    if "outlier_rate" in spec:
        rng = np.random.RandomState(s)
        m = rng.uniform(size=Y.shape) < spec["outlier_rate"]
        sgn = np.where(rng.uniform(size=Y.shape) < 0.5, -1.0, 1.0)
        Y = np.where(m, Y + spec["outlier_scale"] * ys * sgn, Y)
        info["outlier_fraction"] = float(m.mean())
    if "burst" in spec:
        k0, L = burst_window(spec["burst"], N1, es.dt, float(meta.get("t_ref", 1.0)))
        ch = spec["burst"].get("channels")
        ch = list(range(ny)) if ch is None else list(ch)
        if L > 0:
            Y[:, k0 + 1:k0 + L + 1, ch] = Y[:, k0:k0 + 1, ch]
        info["burst"] = {"k0": k0, "len": L, "channels": ch}
    x0, p0 = prior(meta, E, spec.get("init", "center"), s)
    return Y, x0, p0, info
