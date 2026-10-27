"""Input signal families and the three evaluation regimes.

Regimes
    train / id     shared parameter ranges, independent seeds
    ood            DISJOINT amplitude / frequency ranges (|A| in [1.5, 2.5] vs <= 1)
    waveform_ood   families never used in training (chirp, filtered noise),
                   drawn at training amplitudes

Piecewise-constant families carry explicit switch times snapped to the integration
grid, so no RK4 step straddles a discontinuity and the integrator keeps its order.
All amplitudes are multiplied by the plant's admissible scale ``u_scale`` and, where a
plant declares ``u_bounds``, a realisation is rescaled (never clipped) to stay inside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RANGES = {
    "constant":       {"train": {"A": (-1.0, 1.0)},
                       "ood":   {"A": (1.5, 2.5)}},
    "step":           {"train": {"A": (-1.0, 1.0), "t_step": (0.15, 0.5)},
                       "ood":   {"A": (1.5, 2.5), "t_step": (0.15, 0.5)}},
    "staircase":      {"train": {"A": (-1.0, 1.0), "dwell": (2.0, 8.0)},
                       "ood":   {"A": (1.5, 2.5), "dwell": (0.3, 1.0)}},
    "prbs":           {"train": {"A": (0.2, 1.0), "dwell": (0.5, 4.0)},
                       "ood":   {"A": (1.5, 2.5), "dwell": (0.1, 0.4)}},
    "square":         {"train": {"A": (0.1, 1.0), "w": (0.5, 2.0), "duty": (0.3, 0.7),
                                 "offset": (-0.8, 0.8)},
                       "ood":   {"A": (1.5, 2.5), "w": (3.0, 6.0), "duty": (0.1, 0.25),
                                 "offset": (-0.8, 0.8)}},
    "sinusoid":       {"train": {"A": (0.1, 1.0), "w": (0.5, 3.0), "offset": (-0.8, 0.8)},
                       "ood":   {"A": (1.5, 2.5), "w": (4.0, 8.0), "offset": (-0.8, 0.8)}},
    "multisine":      {"train": {"A": (0.1, 1.0), "w": (0.3, 3.0), "n_comp": (3, 6),
                                 "offset": (-0.8, 0.8)},
                       "ood":   {"A": (1.5, 2.5), "w": (4.0, 8.0), "n_comp": (3, 6),
                                 "offset": (-0.8, 0.8)}},
    "ramp":           {"train": {"A": (-1.0, 1.0), "frac": (0.3, 1.0)},
                       "ood":   {"A": (1.5, 2.5), "frac": (0.15, 0.5)}},
    "am":             {"train": {"A": (0.1, 1.0), "w_fast": (2.0, 5.0),
                                 "w_slow": (0.05, 0.3), "m": (0.4, 1.0)},
                       "ood":   {"A": (1.5, 2.5), "w_fast": (6.0, 12.0),
                                 "w_slow": (0.02, 0.1), "m": (0.4, 1.0)}},
    "chirp":          {"train": {"A": (0.1, 1.0), "w0": (0.2, 1.0), "w1": (1.0, 3.0)}},
    "filtered_noise": {"train": {"A": (0.1, 1.0), "tau": (0.5, 5.0)}},
}

TRAIN_FAMILIES = ["zero", "constant", "step", "staircase", "prbs", "square",
                  "sinusoid", "multisine", "ramp", "am"]
OOD_FAMILIES = ["constant", "ramp", "sinusoid", "am", "prbs"]
WAVEFORM_OOD_FAMILIES = ["chirp", "filtered_noise"]
PIECEWISE_CONSTANT = {"zero", "constant", "step", "staircase", "prbs", "square"}

assert not set(TRAIN_FAMILIES) & set(WAVEFORM_OOD_FAMILIES), "held-out waveforms leak"


def families_for(tier: str) -> list:
    return {"train": TRAIN_FAMILIES, "id": [f for f in TRAIN_FAMILIES if f != "zero"],
            "ood": OOD_FAMILIES, "waveform_ood": WAVEFORM_OOD_FAMILIES}[tier]


@dataclass
class Signal:
    """One fully specified input realisation u(t) (scalar input)."""
    family: str
    params: dict = field(default_factory=dict)
    switch_times: np.ndarray = None      # piecewise-constant: grid-aligned left endpoints
    levels: np.ndarray = None
    scale: float = 1.0

    def __call__(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self.switch_times is not None:
            i = np.searchsorted(self.switch_times, t, side="right") - 1
            return self.scale * self.levels[np.clip(i, 0, len(self.levels) - 1)]
        q, f = self.params, self.family
        if f == "sinusoid":
            v = q["A"] * np.sin(q["w"] * t + q["phi"]) + q["offset"]
        elif f == "multisine":
            v = q["offset"] + sum(a * np.sin(w * t + p)
                                  for a, w, p in zip(q["amps"], q["freqs"], q["phases"]))
        elif f == "ramp":
            v = q["A"] * np.clip(t / max(q["frac"] * q["T"], 1e-9), 0.0, 1.0)
        elif f == "am":
            v = q["A"] * (1 + q["m"] * np.sin(q["w_slow"] * t + q["ph_s"])) \
                * np.sin(q["w_fast"] * t + q["ph_f"])
        elif f == "chirp":
            k = (q["w1"] - q["w0"]) / max(q["T"], 1e-9)
            v = q["A"] * np.sin(q["w0"] * t + 0.5 * k * t ** 2 + q["phi"])
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


def sample_signal(family: str, regime: str, rng: np.random.RandomState, dt: float,
                  T: float, stratum: int = None, n_strata: int = 4,
                  scale: float = 1.0) -> Signal:
    """Draw one realisation. ``regime`` in {train, id, ood, waveform_ood}."""
    R = RANGES.get(family, {}).get("ood" if regime == "ood" else "train", {})
    sgn = (lambda: rng.choice([-1.0, 1.0])) if regime == "ood" else (lambda: 1.0)
    s = lambda key: _strat(rng, *R[key], stratum, n_strata)
    # a second stratified parameter uses the NEXT digit of the stratum, so (A, w) tile an
    # n_strata x n_strata grid; sharing one index gave corr(A, w) = 0.93 in training
    s2 = lambda key: _strat(rng, *R[key], None if stratum is None else stratum // n_strata, n_strata)
    u = lambda key: float(rng.uniform(*R[key]))

    if family == "zero":
        st, lv = _pwc([0.0], [0.0], dt)
        return Signal("zero", {}, st, lv, scale)
    if family == "constant":
        a = s("A") * sgn()
        st, lv = _pwc([0.0], [a], dt)
        return Signal(family, {"A": a}, st, lv, scale)
    if family == "step":
        a, ts = s("A") * sgn(), u("t_step") * T
        st, lv = _pwc([0.0, ts], [0.0, a], dt)
        return Signal(family, {"A": a, "t_step": float(st[-1])}, st, lv, scale)
    if family in ("staircase", "prbs"):
        times, vals, t = [0.0], [], 0.0
        a0, sign = s("A"), rng.choice([-1.0, 1.0])
        while t < T:
            if family == "staircase":     # only the first level is stratified: a staircase
                vals.append((s("A") if not vals else u("A")) * sgn())   # spans the whole range
            else:
                vals.append(a0 * sign)
                sign = -sign
            t += u("dwell")
            if t < T:
                times.append(t)
        st, lv = _pwc(times, vals[:len(times)], dt)
        return Signal(family, {"n_seg": len(lv)}, st, lv, scale)
    if family == "square":
        a, w, duty, off = s("A"), s2("w"), u("duty"), u("offset")
        period = 2 * np.pi / w
        times, vals, t = [], [], -rng.uniform(0.0, 1.0) * period
        while t < T:
            times.append(max(t, 0.0)); vals.append(a + off)
            t += duty * period
            if t < T:
                times.append(max(t, 0.0)); vals.append(-a + off)
            t += (1 - duty) * period
        st, lv = _pwc(times, vals, dt)
        return Signal(family, {"A": a, "w": w, "duty": duty, "offset": off}, st, lv, scale)
    if family == "sinusoid":
        return Signal(family, {"A": s("A"), "w": s2("w"), "phi": rng.uniform(0, 2 * np.pi),
                               "offset": u("offset")}, scale=scale)
    if family == "multisine":
        n = int(rng.randint(R["n_comp"][0], R["n_comp"][1] + 1))
        a = s("A")
        return Signal(family, {"A": a, "amps": rng.dirichlet(np.ones(n)) * a,
                               "freqs": rng.uniform(*R["w"], size=n),
                               "phases": rng.uniform(0, 2 * np.pi, size=n),
                               "offset": u("offset")}, scale=scale)
    if family == "ramp":
        return Signal(family, {"A": s("A") * sgn(), "frac": u("frac"), "T": T}, scale=scale)
    if family == "am":
        return Signal(family, {"A": s("A"), "w_fast": u("w_fast"), "w_slow": u("w_slow"),
                               "m": u("m"), "ph_s": rng.uniform(0, 2 * np.pi),
                               "ph_f": rng.uniform(0, 2 * np.pi)}, scale=scale)
    if family == "chirp":
        return Signal(family, {"A": s("A"), "w0": u("w0"), "w1": u("w1"), "T": T,
                               "phi": rng.uniform(0, 2 * np.pi)}, scale=scale)
    if family == "filtered_noise":
        a, tau = s("A"), u("tau")
        n = int(np.ceil(T / dt)) + 2
        ph = np.exp(-dt / tau)
        v = np.zeros(n)
        for i in range(1, n):                       # Ornstein-Uhlenbeck, exact discretisation
            v[i] = v[i - 1] * ph + np.sqrt(1 - ph ** 2) * rng.randn()
        v = a * v / max(np.abs(v).max(), 1e-12)
        return Signal(family, {"A": a, "tau": tau, "_t": np.arange(n) * dt, "_v": v},
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
