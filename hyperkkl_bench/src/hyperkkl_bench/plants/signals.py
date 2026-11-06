"""Input signal families and the evaluation regimes (input protocol v2, 2026-10-25).

Out-of-distribution means what it means for a deployed estimator: the input is BOUNDED
by fixed actuator limits (a pedal travels between released and fully pressed), and what
changes in deployment is its SHAPE or its SEQUENCE (a different driver on a different
road), never its amplitude. Therefore

  * every family, in every regime, realises v(t) in [-1, 1] by construction (no
    clipping); the applied input is u(t) = u_scale * v(t), and plants that declare
    ``u_bounds`` are additionally rescaled (never clipped) to stay inside them;
  * training covers the whole box: constant / staircase levels are uniform on [-1, 1],
    periodic families carry an offset drawn inside [-(1 - A), 1 - A], so they reach the
    edges;
  * all time-like parameters are in units of the plant's reference time ``t_ref``
    (seconds per signal time unit, configs/systems/*.yaml): a frequency w in RANGES is
    applied as w / t_ref, a dwell / ramp time / correlation time as dwell * t_ref.
    Fractions of the horizon T (step time, ramp fraction) are unchanged.

Regimes (the evaluation tiers of configs/base.yaml ``eval.tiers``)
    train / id     the training families (TRAIN_FAMILIES) at the "train" ranges; id uses
                   independent seeds (and drops the zero input)
    shape_ood      waveform families NEVER used in training (chirp, filtered_noise,
                   triangle, sawtooth): same box, same frequency band (in t_ref units)
    sequence_ood   the 'drive' family (pedal-like event sequence), which IS a training
                   family, with a different event sequence: same level grid and box,
                   different transition structure and timing (RANGES["drive"])

The 'drive' family: a level on the grid LEVELS (9 levels in [-1, 1]) is held for a dwell,
then the next level is reached by a linear ramp (press / release). The next level is
drawn uniformly among the grid levels whose distance from the current one lies in
``delta``:

    train          'smooth driver':  |delta| in [0.25, 0.5]  dwell U[2, 6]   ramp U[0.5, 1.5]
    sequence_ood   'stop-and-go':    |delta| in [1.0, 2.0]   dwell U[0.5, 2] ramp U[0.2, 0.5]

(times in t_ref). Both start at a uniform grid level and both can visit every level, so
the marginal level RANGE is identical ([-1, 1], same grid); what differs is the order of
the levels (small steps vs large jumps between distant levels, e.g. full press <->
release), the dwell between events (shorter: stop-and-go) and the press / release rate.

Piecewise-constant families carry explicit switch times snapped to the integration
grid; piecewise-linear families (drive, triangle, sawtooth) carry knots snapped to the
grid (consecutive knots at least one step apart), so no RK4 step straddles a switch or a
kink and the integrator keeps its order (tests/test_verify_data.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# offset = offset_frac * (1 - A): a periodic family with amplitude A stays inside [-1, 1]
OFFSET = (-1.0, 1.0)

RANGES = {
    # ---- training families (regime 'train'; 'id' uses the same ranges) --------------
    "constant":       {"train": {"A": (-1.0, 1.0)}},
    "step":           {"train": {"A": (-1.0, 1.0), "t_step": (0.15, 0.5)}},   # t_step: fraction of T
    "staircase":      {"train": {"A": (-1.0, 1.0), "dwell": (2.0, 8.0)}},
    "prbs":           {"train": {"A": (0.1, 1.0), "offset_frac": OFFSET, "dwell": (0.5, 4.0)}},
    "square":         {"train": {"A": (0.1, 1.0), "w": (0.5, 2.0), "duty": (0.3, 0.7),
                                 "offset_frac": OFFSET}},
    "sinusoid":       {"train": {"A": (0.1, 1.0), "w": (0.5, 3.0), "offset_frac": OFFSET}},
    "multisine":      {"train": {"A": (0.1, 1.0), "w": (0.3, 3.0), "n_comp": (3, 6),
                                 "offset_frac": OFFSET}},                  # A = sum of amplitudes
    "ramp":           {"train": {"A": (-1.0, 1.0), "frac": (0.3, 1.0)}},   # frac: of T
    "am":             {"train": {"A": (0.1, 1.0), "w_fast": (2.0, 5.0),     # A = peak A (1 + m)
                                 "w_slow": (0.05, 0.3), "m": (0.4, 1.0)}},
    "drive":          {"train":        {"n_levels": 9, "delta": (0.25, 0.5),
                                        "dwell": (2.0, 6.0), "ramp": (0.5, 1.5)},
                       "sequence_ood": {"n_levels": 9, "delta": (1.0, 2.0),
                                        "dwell": (0.5, 2.0), "ramp": (0.2, 0.5)}},
    # ---- held-out waveforms (regime 'shape_ood'), inside the training band ----------
    "chirp":          {"shape_ood": {"A": (0.1, 1.0), "w0": (0.5, 1.5), "w1": (1.5, 3.0),
                                     "offset_frac": OFFSET}},
    "filtered_noise": {"shape_ood": {"A": (0.1, 1.0), "tau": (0.5, 5.0), "offset_frac": OFFSET}},
    "triangle":       {"shape_ood": {"A": (0.1, 1.0), "w": (0.5, 2.0), "offset_frac": OFFSET}},
    "sawtooth":       {"shape_ood": {"A": (0.1, 1.0), "w": (0.5, 2.0), "offset_frac": OFFSET}},
}
TIME_KEYS = {"dwell", "ramp", "tau"}                 # multiplied by t_ref
FREQ_KEYS = {"w", "w0", "w1", "w_fast", "w_slow"}    # divided by t_ref

TRAIN_FAMILIES = ["zero", "constant", "step", "staircase", "prbs", "square",
                  "sinusoid", "multisine", "ramp", "am", "drive"]
SHAPE_OOD_FAMILIES = ["chirp", "filtered_noise", "triangle", "sawtooth"]
SEQUENCE_OOD_FAMILIES = ["drive"]
TIERS = ("id", "shape_ood", "sequence_ood")
PIECEWISE_CONSTANT = {"zero", "constant", "step", "staircase", "prbs", "square"}
PIECEWISE_LINEAR = {"drive", "triangle", "sawtooth"}

assert not set(TRAIN_FAMILIES) & set(SHAPE_OOD_FAMILIES), "held-out waveforms leak"
assert set(SEQUENCE_OOD_FAMILIES) <= set(TRAIN_FAMILIES), "sequence_ood must reuse a training family"


def families_for(tier: str) -> list:
    return {"train": TRAIN_FAMILIES, "id": [f for f in TRAIN_FAMILIES if f != "zero"],
            "shape_ood": SHAPE_OOD_FAMILIES, "sequence_ood": SEQUENCE_OOD_FAMILIES}[tier]


def range_key(family: str, regime: str) -> str:
    """Which RANGES entry a (family, regime) draws from; raises for a family the regime
    does not use (e.g. a held-out waveform in training)."""
    key = "train" if regime in ("train", "id") else regime
    if family != "zero" and key not in RANGES.get(family, {}):
        raise ValueError(f"family {family!r} has no {regime!r} regime")
    return key


def regime_table() -> dict:
    """The reviewable protocol: {tier: {family: ranges (t_ref units)}}."""
    return {t: {f: RANGES.get(f, {}).get(range_key(f, t), {}) for f in families_for(t)}
            for t in ("train",) + TIERS}


@dataclass
class Signal:
    """One fully specified input realisation u(t) (scalar input)."""
    family: str
    params: dict = field(default_factory=dict)
    switch_times: np.ndarray = None      # piecewise-constant: grid-aligned left endpoints
    levels: np.ndarray = None
    scale: float = 1.0
    knots: np.ndarray = None             # piecewise-linear: grid-aligned knot times
    knot_vals: np.ndarray = None

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self.switch_times is not None:
            i = np.searchsorted(self.switch_times, t, side="right") - 1
            return self.scale * self.levels[np.clip(i, 0, len(self.levels) - 1)]
        if self.knots is not None:
            return self.scale * np.interp(t, self.knots, self.knot_vals)
        q, f = self.params, self.family
        if f == "sinusoid":
            v = q["A"] * np.sin(q["w"] * t + q["phi"]) + q["offset"]
        elif f == "multisine":
            v = q["offset"] + sum(a * np.sin(w * t + p)
                                  for a, w, p in zip(q["amps"], q["freqs"], q["phases"]))
        elif f == "ramp":
            v = q["A"] * np.clip(t / max(q["frac"] * q["T"], 1e-9), 0.0, 1.0)
        elif f == "am":
            v = q["A"] / (1 + q["m"]) * (1 + q["m"] * np.sin(q["w_slow"] * t + q["ph_s"])) \
                * np.sin(q["w_fast"] * t + q["ph_f"])
        elif f == "chirp":
            k = (q["w1"] - q["w0"]) / max(q["T"], 1e-9)
            v = q["A"] * np.sin(q["w0"] * t + 0.5 * k * t ** 2 + q["phi"]) + q["offset"]
        elif f == "filtered_noise":
            v = np.interp(t, q["_t"], q["_v"])
        else:
            raise ValueError(f)
        return self.scale * v

    def describe(self) -> dict:
        p = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in self.params.items() if not k.startswith("_")}
        return {"family": self.family, "scale": self.scale, **p}


def _strat(rng, lo, hi, stratum, n_strata):
    """Stratified uniform: draw inside one of n_strata equal bins of [lo, hi]."""
    if stratum is None or n_strata <= 1:
        return float(rng.uniform(lo, hi))
    w = (hi - lo) / n_strata
    k = stratum % n_strata
    return float(rng.uniform(lo + k * w, lo + (k + 1) * w))


def _pwc(times, levels, dt):
    st = np.round(np.asarray(times, float) / dt) * dt
    st[0] = 0.0
    # switches snapped onto the same grid point: the LAST level is the one in force
    keep = np.concatenate([np.diff(st) > 0.5 * dt, [True]])
    return st[keep], np.asarray(levels, float)[keep]


def _pwl(times, vals, dt):
    """Knots of a piecewise-linear signal snapped to the grid, at least one step apart.

    Knots before t = dt / 2 (a random phase) are replaced by the interpolated value at 0. The
    value at each knot is kept, so the realised extremes are the knot values: a signal
    whose knot values lie in [-1, 1] stays in [-1, 1] at every time."""
    times, vals = np.asarray(times, float), np.asarray(vals, float)
    v0 = float(np.interp(0.0, times, vals))
    m = times >= 0.5 * dt
    times, vals = np.concatenate([[0.0], times[m]]), np.concatenate([[v0], vals[m]])
    idx = np.round(times / dt).astype(np.int64)
    idx[0] = 0
    for k in range(1, len(idx)):
        idx[k] = max(idx[k], idx[k - 1] + 1)
    return idx * dt, vals


def drive_events(sig: Signal) -> dict:
    """Event statistics of one 'drive' realisation (t_ref units): levels, |delta level|,
    dwell and ramp durations as SAMPLED (before grid snapping)."""
    q = sig.params
    lv = np.asarray(q["_levels"])
    return {"levels": lv, "delta": np.abs(np.diff(lv)), "dwell": np.asarray(q["_dwell"]),
            "ramp": np.asarray(q["_ramp"])}


def sample_signal(family: str, regime: str, rng: np.random.RandomState, dt: float,
                  T: float, stratum: int = None, n_strata: int = 4,
                  scale: float = 1.0, t_ref: float = 1.0) -> Signal:
    """Draw one realisation. ``regime`` in {train, id, shape_ood, sequence_ood};
    ``t_ref`` = seconds per signal time unit (frequencies / t_ref, durations * t_ref)."""
    R = RANGES.get(family, {}).get(range_key(family, regime), {})
    tr = float(t_ref)

    def conv(key, v):
        return v * tr if key in TIME_KEYS else (v / tr if key in FREQ_KEYS else v)
    s = lambda key: conv(key, _strat(rng, *R[key], stratum, n_strata))
    # a second stratified parameter uses the NEXT digit of the stratum, so (A, w) tile an
    # n_strata x n_strata grid; sharing one index gave corr(A, w) = 0.93 in training
    s2 = lambda key: conv(key, _strat(rng, *R[key], None if stratum is None else stratum // n_strata,
                                      n_strata))
    u = lambda key: conv(key, float(rng.uniform(*R[key])))
    off = lambda a: u("offset_frac") * (1.0 - a)        # |a| + |offset| <= 1

    if family == "zero":
        st, lv = _pwc([0.0], [0.0], dt)
        return Signal("zero", {}, st, lv, scale)
    if family == "constant":
        a = s("A")
        st, lv = _pwc([0.0], [a], dt)
        return Signal(family, {"A": a}, st, lv, scale)
    if family == "step":
        a, ts = s("A"), u("t_step") * T
        st, lv = _pwc([0.0, ts], [0.0, a], dt)
        return Signal(family, {"A": a, "t_step": float(st[-1])}, st, lv, scale)
    if family in ("staircase", "prbs"):
        times, vals, t = [0.0], [], 0.0
        if family == "prbs":
            a0 = s("A")
            o, sign = off(a0), rng.choice([-1.0, 1.0])
        while t < T:
            if family == "staircase":     # only the first level is stratified: a staircase
                vals.append(s("A") if not vals else u("A"))             # spans the whole range
            else:
                vals.append(o + a0 * sign)
                sign = -sign
            t += u("dwell")
            if t < T:
                times.append(t)
        st, lv = _pwc(times, vals[:len(times)], dt)
        p = {"n_seg": len(lv)} if family == "staircase" else {"A": a0, "offset": o, "n_seg": len(lv)}
        return Signal(family, p, st, lv, scale)
    if family == "square":
        a, w, duty = s("A"), s2("w"), u("duty")
        o = off(a)
        period = 2 * np.pi / w
        times, vals, t = [], [], -rng.uniform(0.0, 1.0) * period
        while t < T:
            times.append(max(t, 0.0)); vals.append(a + o)
            t += duty * period
            if t < T:
                times.append(max(t, 0.0)); vals.append(-a + o)
            t += (1 - duty) * period
        st, lv = _pwc(times, vals, dt)
        return Signal(family, {"A": a, "w": w, "duty": duty, "offset": o}, st, lv, scale)
    if family == "sinusoid":
        a, w = s("A"), s2("w")
        return Signal(family, {"A": a, "w": w, "phi": rng.uniform(0, 2 * np.pi),
                               "offset": off(a)}, scale=scale)
    if family == "multisine":
        n = int(rng.randint(R["n_comp"][0], R["n_comp"][1] + 1))
        a = s("A")
        return Signal(family, {"A": a, "amps": rng.dirichlet(np.ones(n)) * a,
                               "freqs": rng.uniform(*R["w"], size=n) / tr,
                               "phases": rng.uniform(0, 2 * np.pi, size=n),
                               "offset": off(a)}, scale=scale)
    if family == "ramp":
        return Signal(family, {"A": s("A"), "frac": u("frac"), "T": T}, scale=scale)
    if family == "am":
        return Signal(family, {"A": s("A"), "w_fast": u("w_fast"), "w_slow": u("w_slow"),
                               "m": u("m"), "ph_s": rng.uniform(0, 2 * np.pi),
                               "ph_f": rng.uniform(0, 2 * np.pi)}, scale=scale)
    if family == "drive":
        L = np.linspace(-1.0, 1.0, int(R["n_levels"]))
        dlo, dhi = R["delta"]
        i = int(stratum % len(L)) if stratum is not None else int(rng.randint(len(L)))
        times, vals, t = [0.0], [L[i]], 0.0
        seq, dwells, ramps = [L[i]], [], []
        while t < T:
            d = float(rng.uniform(*R["dwell"]))
            t += d * tr
            times.append(t); vals.append(L[i])
            gap = np.abs(L - L[i])
            cand = np.flatnonzero((gap >= dlo - 1e-9) & (gap <= dhi + 1e-9))
            i = int(rng.choice(cand))
            r = float(rng.uniform(*R["ramp"]))
            t += r * tr
            times.append(t); vals.append(L[i])
            seq.append(L[i]); dwells.append(d); ramps.append(r)
        kt, kv = _pwl(times, vals, dt)
        return Signal(family, {"n_events": len(ramps), "t_ref": tr, "_levels": seq,
                               "_dwell": dwells, "_ramp": ramps}, scale=scale, knots=kt, knot_vals=kv)
    if family in ("triangle", "sawtooth"):
        a, w = s("A"), s2("w")
        o = off(a)
        period = 2 * np.pi / w
        t0 = -rng.uniform(0.0, 1.0) * period
        sgn = 1.0 if family == "triangle" else float(rng.choice([-1.0, 1.0]))   # rising / falling
        n = int(np.ceil((T - t0) / period)) + 2
        times, vals = [], []
        for k in range(n):
            tk = t0 + k * period
            if family == "triangle":
                times += [tk, tk + period / 2]; vals += [o - a, o + a]
            else:                         # linear rise over the period, flyback in one step
                times += [tk, tk + period - dt]; vals += [o - sgn * a, o + sgn * a]
        kt, kv = _pwl(times, vals, dt)
        p = {"A": a, "w": w, "offset": o}
        if family == "sawtooth":
            p["direction"] = sgn
        return Signal(family, p, scale=scale, knots=kt, knot_vals=kv)
    if family == "chirp":
        a = s("A")
        return Signal(family, {"A": a, "w0": u("w0"), "w1": u("w1"), "T": T,
                               "phi": rng.uniform(0, 2 * np.pi), "offset": off(a)}, scale=scale)
    if family == "filtered_noise":
        a, tau = s("A"), u("tau")
        o = off(a)
        n = int(np.ceil(T / dt)) + 2
        ph = np.exp(-dt / tau)
        v = np.zeros(n)
        for i in range(1, n):                       # Ornstein-Uhlenbeck, exact discretisation
            v[i] = v[i - 1] * ph + np.sqrt(1 - ph ** 2) * rng.randn()
        v = a * v / max(np.abs(v).max(), 1e-12) + o
        return Signal(family, {"A": a, "tau": tau, "offset": o, "_t": np.arange(n) * dt, "_v": v},
                      scale=scale)
    raise ValueError(f"unknown signal family {family}")


def enforce_bounds(sig: Signal, dt: float, T: float, bounds) -> Signal:
    """Rescale so the applied input stays inside the plant's admissible interval."""
    if not bounds:
        return sig
    lo, hi = float(bounds[0]), float(bounds[1])
    v = sig(np.arange(0.0, T + dt, dt / 2))         # every RK4 stage time, not only the grid
    f = 1.0
    if v.min() < lo < 0:
        f = min(f, lo / v.min())
    if v.max() > hi > 0:
        f = min(f, hi / v.max())
    if f < 1.0:
        sig.scale *= f
        sig.params["bound_rescale"] = f
    return sig


def stage_inputs(sigs: list, t: np.ndarray) -> np.ndarray:
    """u at the three distinct RK4 stage times of every step, shape (N, 3, n_ep).

    [u(t_k^+), u(t_k + h/2), u(t_{k+1}^-)]: because switches sit on grid points, the
    one-sided offsets pick the value inside the step for piecewise-constant inputs.
    """
    h = t[1] - t[0]
    eps = 1e-9 * h
    out = np.empty((len(t) - 1, 3, len(sigs)))
    for j, s in enumerate(sigs):
        out[:, 0, j] = s(t[:-1] + eps)
        out[:, 1, j] = s(t[:-1] + 0.5 * h)
        out[:, 2, j] = s(t[1:] - eps)
    return out
