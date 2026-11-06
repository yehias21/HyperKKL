"""D1 'sequence-diverse' training-input generator (PREREG.md section 1). New code; the package is unchanged.

Each episode signal is a concatenation of i.i.d. segments until T (all times in t_ref units):
  p = 0.55  family segment: a TRAIN_FAMILIES member (not zero) drawn by the unchanged
            signals.sample_signal(fam, "train", ...) over its own horizon, duration log-U[3, 12]
  p = 0.45  event segment, uniform over
            rest  : v = 0 for U[1.5, 4]
            pulse : 1-3 pulses from base b: level U[-1, 1], width U[0.5, 1.5], back to b for U[0.5, 1.5]
            ramp  : linear b -> U[-1, 1] over U[1.5, 4], hold U[1, 3]
v(t) in [-1, 1] by construction; applied u = scale * v (then signals.enforce_bounds as the package).
"""
from __future__ import annotations

import numpy as np

from hyperkkl import signals as S

P_FAMILY = 0.55
FAMILY_DUR = (3.0, 12.0)          # log-uniform
REST_DUR = (1.5, 4.0)
PULSE_N = (1, 3)
PULSE_W = (0.5, 1.5)
PULSE_GAP = (0.5, 1.5)
RAMP_DUR = (1.5, 4.0)
RAMP_HOLD = (1.0, 3.0)
SEG_FAMILIES = [f for f in S.TRAIN_FAMILIES if f != "zero"]


class CompositeSignal:
    """Duck-types signals.Signal for data.generate / enforce_bounds: __call__, family, params, scale, describe."""
    family = "composite"

    def __init__(self, pieces, T, scale=1.0, segments=None):
        self.pieces = pieces            # list of (t0, t1, kind, obj); obj: Signal (time-shifted) or (knots, vals)
        self.T = float(T)
        self.scale = float(scale)
        self.params = {"segments": segments or []}

    def unit(self, t):
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t)
        starts = np.array([p[0] for p in self.pieces])
        idx = np.clip(np.searchsorted(starts, t, side="right") - 1, 0, len(self.pieces) - 1)
        for j, (t0, t1, kind, obj) in enumerate(self.pieces):
            m = idx == j
            if not m.any():
                continue
            if kind == "family":
                sub = obj
                out[m] = sub(t[m] - t0) / sub.scale
            else:
                kn, kv = obj
                out[m] = np.interp(t[m], kn, kv)
        return out

    def __call__(self, t):
        return self.scale * self.unit(t)

    def describe(self):
        return {"family": self.family, "scale": self.scale, "segments": self.params["segments"],
                **({"bound_rescale": self.params["bound_rescale"]} if "bound_rescale" in self.params else {})}


def sample_composite(rng: np.random.RandomState, dt: float, T: float, scale: float = 1.0,
                     t_ref: float = 1.0) -> CompositeSignal:
    tr = float(t_ref)
    t, b = 0.0, 0.0                     # current time, current level (end of previous segment)
    pieces, segs = [], []
    while t < T - 1e-12:
        if rng.uniform() < P_FAMILY:
            fam = SEG_FAMILIES[rng.randint(len(SEG_FAMILIES))]
            d = float(np.exp(rng.uniform(np.log(FAMILY_DUR[0]), np.log(FAMILY_DUR[1])))) * tr
            sub = S.sample_signal(fam, "train", rng, dt, d, stratum=None, scale=1.0, t_ref=tr)
            t1 = min(t + d, T)
            pieces.append((t, t1, "family", sub))
            segs.append({"kind": "family", "family": fam, "t0": t, "t1": t1})
            b = float(sub(np.array([t1 - t]))[0])
            t = t1
            continue
        kind = ("rest", "pulse", "ramp")[rng.randint(3)]
        if kind == "rest":
            d = rng.uniform(*REST_DUR) * tr
            kn, kv = [t, t + d], [0.0, 0.0]
            b = 0.0
        elif kind == "pulse":
            n = rng.randint(PULSE_N[0], PULSE_N[1] + 1)
            kn, kv, tt = [], [], t
            for _ in range(n):
                lv, w, g = rng.uniform(-1, 1), rng.uniform(*PULSE_W) * tr, rng.uniform(*PULSE_GAP) * tr
                # rectangular: hold lv on [tt, tt + w), then b on [tt + w, tt + w + g)
                kn += [tt, tt + w - 1e-9 * tr, tt + w, tt + w + g - 1e-9 * tr]
                kv += [lv, lv, b, b]
                tt += w + g
            d = tt - t
        else:
            lv, r, h = rng.uniform(-1, 1), rng.uniform(*RAMP_DUR) * tr, rng.uniform(*RAMP_HOLD) * tr
            kn, kv = [t, t + r, t + r + h], [b, lv, lv]
            b, d = lv, r + h
        t1 = min(t + d, T)
        pieces.append((t, t1, "event", (np.asarray(kn), np.asarray(kv))))
        segs.append({"kind": kind, "t0": t, "t1": t1, "knots": [float(k) for k in kn],
                     "vals": [float(v) for v in kv]})
        t = t1
    return CompositeSignal(pieces, T, scale, segs)


def d1_signals(families, n, regime, rng, des, scfg, n_strata=4):
    """Drop-in for hyperkkl.data._signals (signature identical; families / regime / n_strata ignored)."""
    T = des.T
    out = []
    for _ in range(n):
        s = sample_composite(rng, des.dt, T, scale=float(scfg.get("u_scale", 1.0)),
                             t_ref=float(scfg.get("t_ref", 1.0)))
        out.append(S.enforce_bounds(s, des.dt, T, scfg.get("u_bounds")))
    return out
