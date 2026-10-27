"""Slow-band input families for the quad_tank PE study (Stage 2, arm X2/X3).

Kept OUTSIDE the package's data modules (signals.py / data.py are unchanged, so the live-cache
keys of the original episodes stay valid). Every family realises v(t) in [-1, 1] by
construction (knot / level values in [-1, 1], linear interpolation or sums of sinusoids whose
amplitudes plus offset are <= 1), the same amplitude box as the training families; applied
input u = u_scale * v as signals.sample_signal.

TAU = 49 s (quad_tank config: median slow time constant; slowest linear mode 84 s).
    slow_staircase  levels U[-1, 1] (first level stratified), dwell U[1, 3] TAU, held (ZOH)
    slow_ramp       piecewise-linear random walk: knot values U[-1, 1], knot spacing U[1, 3] TAU
                    (each segment is a long ramp between two uniform levels)
    slow_multisine  offset + sum_i a_i sin(w_i t + phi_i), n in {2, 3, 4}, w_i U[0.2, 1] / TAU
                    rad/s (all below the slow corner 1/TAU), amplitudes Dirichlet * A, A U[0.1, 1]
                    (stratified), offset U[-1, 1] * (1 - A)
"""
from __future__ import annotations

import numpy as np

from hyperkkl import signals as S

TAU = 49.0
SLOW_FAMILIES = ["slow_staircase", "slow_ramp", "slow_multisine"]


def sample_slow(family, rng, dt, T, stratum=None, n_strata=4, scale=1.0):
    st = lambda lo, hi: S._strat(rng, lo, hi, stratum, n_strata)
    if family == "slow_staircase":
        times, vals, t = [0.0], [st(-1.0, 1.0)], 0.0
        while True:
            t += float(rng.uniform(1.0, 3.0)) * TAU
            if t >= T:
                break
            times.append(t); vals.append(float(rng.uniform(-1.0, 1.0)))
        ts, lv = S._pwc(times, vals, dt)
        return S.Signal(family, {"n_seg": len(lv)}, ts, lv, scale)
    if family == "slow_ramp":
        times = [-float(rng.uniform(0.0, 1.0)) * TAU]
        vals = [st(-1.0, 1.0)]
        while times[-1] < T:
            times.append(times[-1] + float(rng.uniform(1.0, 3.0)) * TAU)
            vals.append(float(rng.uniform(-1.0, 1.0)))
        kt, kv = S._pwl(times, vals, dt)
        return S.Signal(family, {"n_knots": len(kt)}, scale=scale, knots=kt, knot_vals=kv)
    if family == "slow_multisine":
        n = int(rng.randint(2, 5))
        a = st(0.1, 1.0)
        amps = rng.dirichlet(np.ones(n)) * a
        freqs = rng.uniform(0.2, 1.0, size=n) / TAU
        ph = rng.uniform(0, 2 * np.pi, size=n)
        off = float(rng.uniform(-1.0, 1.0)) * (1.0 - a)
        # stored as knots every 1 s (linear interpolation error < 1e-4 at w <= 1/TAU); the knot
        # values lie in [-1, 1], so the realised signal does too
        tk = np.arange(0.0, T + 1.0 + 1e-9, 1.0)
        vk = off + sum(ai * np.sin(wi * tk + pi) for ai, wi, pi in zip(amps, freqs, ph))
        kt, kv = S._pwl(tk, vk, dt)
        return S.Signal(family, {"A": a, "amps": amps.tolist(), "freqs": freqs.tolist(),
                                 "phases": ph.tolist(), "offset": off}, scale=scale, knots=kt, knot_vals=kv)
    raise ValueError(family)


def install(Dm):
    """Route the slow family names through sample_slow inside data.generate (monkeypatch of the
    module-level _signals; the original families are drawn exactly as before)."""
    orig = Dm._signals

    def _signals(families, n, regime, rng, des, scfg, n_strata=4):
        if not any(f in SLOW_FAMILIES for f in families):
            return orig(families, n, regime, rng, des, scfg, n_strata)
        out = []
        for r in range(n):
            fam = families[r % len(families)]
            kw = dict(stratum=r // len(families), n_strata=n_strata, scale=float(scfg.get("u_scale", 1.0)))
            if fam in SLOW_FAMILIES:
                s = sample_slow(fam, rng, des.dt, des.T, **kw)
            else:
                s = S.sample_signal(fam, regime, rng, des.dt, des.T, t_ref=float(scfg.get("t_ref", 1.0)), **kw)
            out.append(S.enforce_bounds(s, des.dt, des.T, scfg.get("u_bounds")))
        return out

    Dm._signals = _signals
