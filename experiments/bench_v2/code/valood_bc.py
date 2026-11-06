"""Validation-OOD input generator (battery_deep). FROZEN before Stage 1 (see VALOOD.md).

Four families, none of which is used by training (signals.TRAIN_FAMILIES), shape_ood (chirp,
filtered_noise, triangle, sawtooth) or sequence_ood (stop-and-go 'drive'); all times in t_ref units,
v(t) in [-1, 1] by construction, applied u = u_scale * v:

  regen_decay  events every U[2, 6]: jump to a ~ U[-1, 1], exponential relaxation to a base
               b ~ U[-0.3, 0.3] with time constant U[0.3, 2]
  gated_sine   sinusoid (A ~ U[0.3, 1], w ~ U[0.5, 3]) switched on for U[2, 6], off (v = 0) for U[1, 4]
  spline       monotone cubic (PCHIP) through knots spaced U[1, 4] with values U[-1, 1]
  urban_cycle  idle (0, U[0.5, 3]) -> accelerate (linear to U[0.4, 1] over U[0.3, 1]) -> cruise (hold
               U[1, 4]) -> regen brake (level U[-1, -0.3] for U[0.3, 1.5]) -> idle ...

Switch / kink times are snapped to the integration grid (as signals._pwc / _pwl) so no RK4 step
straddles a jump.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

FAMILIES = ["regen_decay", "gated_sine", "spline", "urban_cycle"]
SEED = 777_001          # frozen
N_EPISODES = 128        # 32 per family


def _snap(t, dt):
    return float(np.round(t / dt) * dt)


class SegSignal:
    """Duck-types signals.Signal: piecewise analytic segments starting at grid-snapped times."""

    def __init__(self, family, starts, fns, scale=1.0, params=None):
        self.family, self.starts, self.fns = family, np.asarray(starts, float), fns
        self.scale = float(scale)
        self.params = dict(params or {})

    def unit(self, t):
        t = np.asarray(t, dtype=np.float64)
        idx = np.clip(np.searchsorted(self.starts, t, side="right") - 1, 0, len(self.fns) - 1)
        out = np.zeros_like(t)
        for j, f in enumerate(self.fns):
            m = idx == j
            if m.any():
                out[m] = f(t[m])
        return out

    def __call__(self, t):
        return self.scale * self.unit(t)

    def describe(self):
        return {"family": self.family, "scale": self.scale,
                **{k: v for k, v in self.params.items() if not k.startswith("_")}}


def _const(v):
    return lambda t: np.full_like(t, v)


def sample(family, rng, dt, T, scale=1.0, t_ref=1.0):
    tr = float(t_ref)
    starts, fns, t = [], [], 0.0
    if family == "regen_decay":
        while t < T:
            a, b, tau = rng.uniform(-1, 1), rng.uniform(-0.3, 0.3), rng.uniform(0.3, 2.0) * tr
            t0 = _snap(t, dt)
            starts.append(t0)
            fns.append(lambda tt, a=a, b=b, tau=tau, t0=t0: b + (a - b) * np.exp(-(tt - t0) / tau))
            t = t0 + max(rng.uniform(2.0, 6.0) * tr, dt)
    elif family == "gated_sine":
        on = bool(rng.randint(2))
        while t < T:
            t0 = _snap(t, dt)
            starts.append(t0)
            if on:
                A, w, ph = rng.uniform(0.3, 1.0), rng.uniform(0.5, 3.0) / tr, rng.uniform(0, 2 * np.pi)
                fns.append(lambda tt, A=A, w=w, ph=ph: A * np.sin(w * tt + ph))
                d = rng.uniform(2.0, 6.0) * tr
            else:
                fns.append(_const(0.0))
                d = rng.uniform(1.0, 4.0) * tr
            t = t0 + max(d, dt)
            on = not on
    elif family == "spline":
        kt, kv = [0.0], [rng.uniform(-1, 1)]
        while kt[-1] < T + 4 * tr:
            kt.append(kt[-1] + rng.uniform(1.0, 4.0) * tr)
            kv.append(rng.uniform(-1, 1))
        p = PchipInterpolator(np.asarray(kt), np.asarray(kv))   # monotone: stays in [min, max] of knots
        starts, fns = [0.0], [lambda tt, p=p: np.clip(p(tt), -1.0, 1.0)]
    elif family == "urban_cycle":
        while t < T:
            # idle
            t0 = _snap(t, dt); starts.append(t0); fns.append(_const(0.0))
            t = t0 + max(rng.uniform(0.5, 3.0) * tr, dt)
            # accelerate: linear ramp 0 -> L
            L, r = rng.uniform(0.4, 1.0), max(rng.uniform(0.3, 1.0) * tr, dt)
            t0 = _snap(t, dt); t1 = _snap(t0 + r, dt)
            t1 = max(t1, t0 + dt)
            starts.append(t0); fns.append(lambda tt, t0=t0, t1=t1, L=L: L * np.clip((tt - t0) / (t1 - t0), 0, 1))
            # cruise
            starts.append(t1); fns.append(_const(L))
            t = t1 + max(rng.uniform(1.0, 4.0) * tr, dt)
            # regen brake
            t0 = _snap(t, dt); bl = rng.uniform(-1.0, -0.3)
            starts.append(t0); fns.append(_const(bl))
            t = t0 + max(rng.uniform(0.3, 1.5) * tr, dt)
    else:
        raise ValueError(family)
    # drop duplicate starts (keep the last one in force)
    st = np.asarray(starts)
    keep = np.concatenate([np.diff(st) > 0.5 * dt, [True]]) if len(st) > 1 else np.array([True])
    st = st[keep]; fns = [f for f, k in zip(fns, keep) if k]
    st[0] = 0.0
    return SegSignal(family, st, fns, scale, {"n_seg": len(fns)})


def valood_signals(families, n, regime, rng, des, scfg, n_strata=4):
    """Drop-in for hyperkkl.data._signals (families / regime ignored): FAMILIES round-robin."""
    from hyperkkl import signals as S
    out = []
    for r in range(n):
        s = sample(FAMILIES[r % len(FAMILIES)], rng, des.dt, des.T, float(scfg.get("u_scale", 1.0)),
                   float(scfg.get("t_ref", 1.0)))
        out.append(S.enforce_bounds(s, des.dt, des.T, scfg.get("u_bounds")))
    return out
