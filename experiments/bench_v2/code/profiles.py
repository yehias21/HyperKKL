"""Realistic open-loop OPERATING-PROFILE input generators for the HyperKKL benchmark (proposal; standalone).

Not wired into hyperkkl/ or hyperkkl_bench/. numpy only. See INPUT_REVIEW.md for the rationale and sources.

Design (one pipeline for every plant):

    operator  ->  command c[k]  ->  ACTUATOR (deadband/stiction -> rate limit -> lag -> clip -> quantise)  ->  applied u[k]
                                                                                             |
                                       optional INPUT RECORD imperfections (noise, bias, delay, quantisation,
                                       aggregation, dropout-hold)  ->  what the observer is told  u_obs[k]

* Times are in the plant's NATIVE time unit (battery and cstr_exo: minutes; bioreactor: hours; quad_tank,
  highway_traffic, quad_planar, cartpole, two_link_arm: seconds). For the dimensionless plants (vdp, duffing, fhn,
  network/fhnring, lorenz, rossler, lorenz96) times are in units of the benchmark's t_ref, as in signals.py.
* Units of u are the units the plant's f(x, u) receives (benchmark units, see SPECS[...]['unit']); ``physical()``
  converts to the physical actuator quantity (e.g. bioreactor D = 0.3 + u [1/h], quad_tank pump-1 voltage 3 + u [V]).
* The applied-input box [lo, hi] is the LEGACY box of each plant (u_scale x [-1, 1], plus u_bounds), so the state
  envelope / phase-1 domain stay comparable; every proposed box was checked against the real actuator (INPUT_REVIEW.md).
* train vs ood: DIFFERENT OPERATORS / ROUTES / SCHEDULES on the same device with the SAME actuator and box. OOD is never
  "larger amplitude": the same ``Actuator`` clips both, and tests check that both stay inside the same box.
* Sampled on the integration grid t_k = k dt and applied by zero-order hold, which is exactly how
  hyperkkl.data.integrate applies any input (u_k = s(t_k) on [t_k, t_k+1)). ``SampledSignal`` duck-types
  hyperkkl.signals.Signal so a later integration can pass it to data.generate unchanged.

Entry points:  generate(plant, split, seed, T=None, dt=None, operator=None) -> Profile
               observed_input(u, dt, Imperfection, span, seed) -> u_obs
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------------------------------------------------
# Actuator and input-record models
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Actuator:
    lo: float                 # applied-input bounds (benchmark units)
    hi: float
    rate: float = math.inf    # max |du/dt| (units per native time unit)
    tau: float = 0.0          # first-order lag (native time unit): pump spin-up, valve positioner, platoon dispersion
    quant: float = 0.0        # actuator resolution (DAC / pump-speed step / positioner resolution), 0 = none
    deadband: float = 0.0     # stiction: the actuator moves only once |command - held| > deadband (valves)


def apply_actuator(cmd: np.ndarray, dt: float, act: Actuator) -> np.ndarray:
    """Command -> applied input. Guarantees lo <= u <= hi and |u[k+1] - u[k]| <= rate dt + quant."""
    cmd = np.clip(np.asarray(cmd, float), act.lo, act.hi)
    n = len(cmd)
    out = np.empty(n)
    held = cmd[0]
    x = held                                    # rate-limited signal
    y = held                                    # lagged signal
    a = 1.0 - math.exp(-dt / act.tau) if act.tau > 0 else 1.0
    step = act.rate * dt
    prev = None
    for k in range(n):
        c = cmd[k]
        if act.deadband <= 0 or abs(c - held) > act.deadband:
            held = c
        x += min(max(held - x, -step), step)
        y += a * (x - y)
        v = min(max(y, act.lo), act.hi)
        if act.quant > 0:
            v = act.lo + round((v - act.lo) / act.quant) * act.quant
            v = min(max(v, act.lo), act.hi)
        if prev is not None and math.isfinite(step):      # quantisation may add at most one quantum
            v = min(max(v, prev - step - act.quant), prev + step + act.quant)
        out[k] = prev = v
    return out


@dataclass(frozen=True)
class Imperfection:
    """What the OBSERVER is told about the input (the plant always receives the applied u).
    Relative quantities are fractions of the actuator span (hi - lo)."""
    noise: float = 0.0        # std, fraction of span
    noise_tau: float = 0.0    # 0 = white per sample; > 0 = coloured (OU) with this correlation time
    bias: float = 0.0         # constant offset, fraction of span (current-sensor offset, flow-meter zero error)
    delay: float = 0.0        # transport / communication delay, native time unit (rounded to samples)
    quant: float = 0.0        # resolution of the logged input (absolute units), 0 = none
    aggregate: float = 0.0    # the input is only known as block means over this window (traffic counts), causal
    dropout: float = 0.0      # probability that a logged sample is lost and the previous one is held
    use_command: bool = False # the observer gets the COMMAND, not the applied (rate-limited / lagged) input


def observed_input(u: np.ndarray, dt: float, imp: Imperfection, span: float, seed: int,
                   cmd: np.ndarray = None) -> np.ndarray:
    rng = np.random.RandomState(seed + 7919)
    v = np.asarray(cmd if (imp.use_command and cmd is not None) else u, float).copy()
    n = len(v)
    if imp.aggregate > 0:                              # block means, known only at the END of each window
        w = max(1, int(round(imp.aggregate / dt)))
        out = np.empty(n)
        out[:w] = v[0]
        for s in range(0, n, w):
            blk = v[s:s + w].mean()
            out[s + w:s + 2 * w] = blk
        v = out
    d = int(round(imp.delay / dt))
    if d > 0:
        v = np.concatenate([np.full(d, v[0]), v[:-d]])
    if imp.bias:
        v = v + imp.bias * span
    if imp.noise > 0:
        if imp.noise_tau > 0:
            phi = math.exp(-dt / imp.noise_tau)
            e = np.empty(n)
            e[0] = rng.randn()
            g = rng.randn(n)
            for k in range(1, n):
                e[k] = phi * e[k - 1] + math.sqrt(1 - phi * phi) * g[k]
        else:
            e = rng.randn(n)
        v = v + imp.noise * span * e
    if imp.quant > 0:
        v = np.round(v / imp.quant) * imp.quant
    if imp.dropout > 0:
        lost = rng.rand(n) < imp.dropout
        lost[0] = False
        for k in np.flatnonzero(lost):
            v[k] = v[k - 1]
    return v


# ---------------------------------------------------------------------------------------------------------
# Building blocks (all return arrays on the grid t)
# ---------------------------------------------------------------------------------------------------------

def _hold(t, times, vals):
    """Piecewise constant: vals[i] on [times[i], times[i+1])."""
    i = np.searchsorted(np.asarray(times), t, side="right") - 1
    return np.asarray(vals, float)[np.clip(i, 0, len(vals) - 1)]


def _ou(rng, n, dt, tau, sigma):
    phi = math.exp(-dt / tau)
    g = rng.randn(n)
    e = np.empty(n)
    e[0] = sigma * g[0]
    s = sigma * math.sqrt(1 - phi * phi)
    for k in range(1, n):
        e[k] = phi * e[k - 1] + s * g[k]
    return e


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def _moves(t, rng, start, targets_fn, move_fn, dwell_fn, shape="minjerk", vmax=None, amax=None):
    """Point-to-point moves: hold, then move to the next target. shape: 'minjerk' (duration from move_fn) or
    'trapezoid' (duration from vmax / amax). Returns (signal, event_times)."""
    T = t[-1]
    out = np.full_like(t, start)
    tc, pos, ev = 0.0, start, []
    while tc < T:
        tc += dwell_fn()
        if tc >= T:
            break
        tgt = targets_fn(pos)
        d = abs(tgt - pos)
        if shape == "trapezoid":
            ta = vmax / amax
            dur = 2 * math.sqrt(d / amax) if d < vmax * ta else d / vmax + ta
        else:
            dur = move_fn()
        m = t >= tc
        s = (t[m] - tc) / max(dur, 1e-9)
        if shape == "trapezoid":
            prof = _trap_profile(s, d, dur, vmax, amax)
        else:
            prof = _minjerk(s)
        out[m] = pos + (tgt - pos) * prof
        ev.append(tc)
        pos, tc = tgt, tc + dur
    return out, ev


def _trap_profile(s, d, dur, vmax, amax):
    """Normalised position of a trapezoidal-velocity move (s = t / dur)."""
    if d <= 0:
        return np.ones_like(s)
    tt = np.clip(s, 0, 1) * dur
    ta = min(vmax / amax, dur / 2)
    v = amax * ta
    x = np.where(tt < ta, 0.5 * amax * tt ** 2,
                 np.where(tt < dur - ta, 0.5 * amax * ta ** 2 + v * (tt - ta),
                          d - 0.5 * amax * (dur - tt) ** 2))
    return np.clip(x / d, 0, 1)


def _schedule(t, rng, level_fn, dwell_fn, start=None):
    """Set-point schedule: levels held for dwells (transition shaping is left to the actuator)."""
    T = t[-1]
    times, vals, tc = [0.0], [level_fn(None) if start is None else start], 0.0
    while True:
        tc += dwell_fn()
        if tc >= T:
            break
        times.append(tc)
        vals.append(level_fn(vals[-1]))
    return _hold(t, times, vals), times[1:]


def _fixed_rms(x, rms):
    s = float(np.sqrt(np.mean(x ** 2)))
    return x * (rms / s) if s > 0 else x


# ---------------------------------------------------------------------------------------------------------
# Operators per plant. Each: f(rng, t, P) -> (command array, event times). P = the plant spec (dict).
# ---------------------------------------------------------------------------------------------------------

U = lambda rng, a, b: float(rng.uniform(a, b))


# ---- forced oscillators (vdp, duffing): electrodynamic-shaker / signal-generator tests; zero mean, no DC -----

def op_stepped_sine(rng, t, P):
    """Stepped-sine test: frequency stepped through a band around resonance, constant drive level per step,
    raised-cosine amplitude taper at each step (standard practice to limit transients)."""
    tr, A = P["t_ref"], U(rng, 0.3, 0.9)
    dwell = U(rng, 6, 12) * tr
    n = int(math.ceil(t[-1] / dwell)) + 1
    w = np.sort(rng.uniform(0.3, 2.5, n)) / tr
    if rng.rand() < 0.5:
        w = w[::-1]
    k = np.minimum((t // dwell).astype(int), n - 1)
    loc = (t % dwell) / dwell
    taper = np.minimum(1.0, np.minimum(loc, 1 - loc) / 0.1)
    phase = np.cumsum(w[k]) * (t[1] - t[0])
    return A * taper * np.sin(phase), list(np.arange(1, n) * dwell)


def op_random_band(rng, t, P):
    """Band-limited random excitation at a FIXED drive level (RMS set by the test engineer), zero mean: the
    difference of two OU processes (band-pass), no per-episode max-normalisation."""
    tr, dt = P["t_ref"], t[1] - t[0]
    lo_tau, hi_tau = U(rng, 2, 5) * tr, U(rng, 0.2, 0.6) * tr
    x = _ou(rng, len(t), dt, hi_tau, 1.0) - _ou(rng, len(t), dt, lo_tau, 1.0)
    return _fixed_rms(x, U(rng, 0.2, 0.4)), []


def op_swept_sine(rng, t, P):
    """Swept-sine (sine-sweep) test, up then down, constant level: the other standard shaker protocol."""
    tr, A, T = P["t_ref"], U(rng, 0.3, 0.9), t[-1]
    w0, w1 = U(rng, 0.3, 0.6) / tr, U(rng, 1.8, 2.5) / tr
    s = 1 - np.abs(2 * t / T - 1)                         # 0 -> 1 -> 0
    w = w0 + (w1 - w0) * s
    phase = np.cumsum(w) * (t[1] - t[0])
    return A * np.minimum(1.0, t / (2 * tr)) * np.sin(phase), [T / 2]


def op_burst_random(rng, t, P):
    """Burst-random: random bursts of fixed RMS, each followed by a quiet window (modal-test practice)."""
    tr, dt = P["t_ref"], t[1] - t[0]
    x = _fixed_rms(_ou(rng, len(t), dt, U(rng, 0.2, 0.5) * tr, 1.0), U(rng, 0.25, 0.4))
    on, tc, ev, gate = U(rng, 4, 8) * tr, 0.0, [], np.zeros_like(t)
    while tc < t[-1]:
        gate[(t >= tc) & (t < tc + on)] = 1.0
        ev.append(tc)
        tc += on + U(rng, 3, 8) * tr
    return x * gate, ev


# ---- neurons (fhn, network, fhnring): current-clamp stimulus protocols -----------------------------------

def op_long_square(rng, t, P):
    """Long-square current steps from rest, an increasing / decreasing series (the standard f-I protocol)."""
    tr = P["t_ref"]
    amps = np.linspace(-0.9, 0.9, 7)
    order = amps if rng.rand() < 0.5 else amps[::-1]
    times, vals, tc, i = [0.0], [0.0], U(rng, 2, 6) * tr, 0
    while tc < t[-1]:
        times += [tc, tc + U(rng, 8, 15) * tr]
        vals += [float(order[i % len(order)] + rng.uniform(-0.05, 0.05)), 0.0]
        tc = times[-1] + U(rng, 5, 10) * tr
        i += 1
    return _hold(t, times, vals), times[1::2]


def op_ramp_protocol(rng, t, P):
    """Slow current ramps from rest to a peak and back (rheobase protocol), then rest."""
    tr = P["t_ref"]
    times, vals, tc = [0.0], [0.0], U(rng, 2, 5) * tr
    while tc < t[-1]:
        pk, r = U(rng, -0.6, 1.0), U(rng, 8, 20) * tr
        times += [tc, tc + r, tc + r + U(rng, 0, 3) * tr, tc + 2 * r]
        vals += [0.0, pk, pk, 0.0]
        tc = times[-1] + U(rng, 3, 8) * tr
    return np.interp(t, times, vals), times[1::4]


def op_synaptic_noise(rng, t, P):
    """In-vivo-like stimulation: Ornstein-Uhlenbeck 'synaptic' current around a mean drive that switches between a
    few states (fixed RMS per state; no max-normalisation)."""
    tr, dt = P["t_ref"], t[1] - t[0]
    mean, ev = _schedule(t, rng, lambda _: U(rng, -0.3, 0.5), lambda: U(rng, 15, 30) * tr)
    return mean + _ou(rng, len(t), dt, U(rng, 0.5, 2) * tr, U(rng, 0.15, 0.3)), ev


def op_pulse_train(rng, t, P):
    """Trains of short current pulses at a fixed stimulation rate, separated by rest (rhythmic stimulation)."""
    tr = P["t_ref"]
    out, tc, ev = np.zeros_like(t), U(rng, 2, 5) * tr, []
    while tc < t[-1]:
        per, wid, amp = U(rng, 2.5, 8) * tr, U(rng, 0.4, 1.0) * tr, U(rng, 0.5, 1.0)
        dur = U(rng, 15, 30) * tr
        ph = (t - tc) % per
        m = (t >= tc) & (t < tc + dur) & (ph < wid)
        out[m] = amp
        ev.append(tc)
        tc += dur + U(rng, 8, 15) * tr
    return out, ev


# ---- bioreactor (u = dilution-rate change, D = 0.3 + u [1/h]) -----------------------------------------------

def op_chemostat_steps(rng, t, P):
    """Chemostat run: D held for 3-6 residence times (1/D ~ 2-7 h) at each set-point of a grid, then changed."""
    grid = np.linspace(P["lo"], P["hi"], 7)
    return _schedule(t, rng, lambda prev: float(rng.choice(grid if prev is None else grid[np.abs(grid - prev) > 1e-9])),
                     lambda: U(rng, 10, 20))


def op_shift_up_down(rng, t, P):
    """Shift-up / shift-down experiment: alternate between two dilution rates."""
    a, b = U(rng, P["lo"], 0.0), U(rng, 0.0, P["hi"])
    lv = [a, b] if rng.rand() < 0.5 else [b, a]
    times, vals, tc, i = [0.0], [lv[0]], 0.0, 1
    while True:
        tc += U(rng, 10, 25)
        if tc >= t[-1]:
            break
        times.append(tc); vals.append(lv[i % 2]); i += 1
    return _hold(t, times, vals), times[1:]


def op_astat(rng, t, P):
    """A-stat (accelerostat) cultivation: D increased linearly at a constant acceleration after a chemostat phase,
    optionally followed by a ramp down (de-stat)."""
    T, lo, hi = t[-1], P["lo"], P["hi"]
    t0 = U(rng, 0.1, 0.3) * T
    acc = U(rng, 0.004, 0.012)                     # 1/h^2
    d0 = U(rng, lo, 0.5 * (lo + hi))
    up = d0 + acc * np.clip(t - t0, 0, None)
    if rng.rand() < 0.5:                           # de-stat after a peak
        tp = t0 + (hi - d0) / acc * U(rng, 0.6, 1.0)
        peak = d0 + acc * (tp - t0)
        up = np.where(t < tp, up, peak - acc * (t - tp))
    return np.clip(up, lo, hi), [t0]


def op_feed_interruption(rng, t, P):
    """Feed-bottle change / pump fault: nominal operation, the feed pump drops to its minimum for 1-4 h, restart."""
    lvl = U(rng, -0.05, 0.1)
    times, vals, tc = [0.0], [lvl], 0.0
    while True:
        tc += U(rng, 15, 35)
        if tc >= t[-1]:
            break
        times += [tc, tc + U(rng, 1, 4)]
        vals += [P["lo"], lvl + U(rng, -0.05, 0.05)]
        tc = times[-1]
    return _hold(t, times, vals), times[1::2]


# ---- quad_tank (u = pump-voltage change, v1 = 3 + u, v2 = 3 - u/2 [V]) ---------------------------------------

def op_tank_small_moves(rng, t, P):
    """Lab operator nudging the pump set-point: +-0.1..0.5 V moves held 1-3 min (>= one tank time constant)."""
    def lvl(prev):
        if prev is None:
            return U(rng, -0.5, 0.5)
        return float(np.clip(prev + rng.choice([-1, 1]) * U(rng, 0.1, 0.5), P["lo"], P["hi"]))
    return _schedule(t, rng, lvl, lambda: U(rng, 60, 180))


def op_tank_recipe(rng, t, P):
    """Recipe of 3-5 operating points visited in a random order, each held 2-4 min."""
    pts = rng.uniform(-1.0, 1.0, int(rng.randint(3, 6)))
    return _schedule(t, rng, lambda prev: float(rng.choice(pts)), lambda: U(rng, 120, 240))


def op_tank_grade_change(rng, t, P):
    """Grade change: slow monotone transfer between two production points over 3-10 min, then hold."""
    T = t[-1]
    a, b = U(rng, -0.9, -0.2), U(rng, 0.2, 0.9)
    if rng.rand() < 0.5:
        a, b = b, a
    t0, dur = U(rng, 0.05, 0.4) * T, U(rng, 180, 600)
    return a + (b - a) * np.clip((t - t0) / dur, 0, 1), [t0]


def op_tank_pump_trip(rng, t, P):
    """Pump trip and restart: steady operation, the pump command falls to the minimum for 20-60 s, then restored."""
    lvl = U(rng, -0.4, 0.6)
    times, vals, tc = [0.0], [lvl], 0.0
    while True:
        tc += U(rng, 60, 150)
        if tc >= t[-1]:
            break
        times += [tc, tc + U(rng, 20, 60)]
        vals += [P["lo"], lvl]
        tc = times[-1]
    return _hold(t, times, vals), times[1::2]


# ---- highway_traffic (u = on-ramp inflow change [veh/s]; ramp flow = 0.3 + u) ----------------------------------

def _poisson_rate(rng, lam, t, window=60.0):
    """Vehicle-count fluctuation: the realised ramp flow over detector windows is Poisson(lam * window) / window."""
    dt = t[1] - t[0]
    w = max(1, int(round(window / dt)))
    out = np.empty_like(lam)
    for s in range(0, len(t), w):
        lm = max(float(lam[s:s + w].mean()), 0.0)
        out[s:s + w] = rng.poisson(lm * window) / window
    return out


def op_peak_demand(rng, t, P):
    """Time-of-day demand: ramp demand drifts toward the peak level (logistic onset), with Poisson count noise."""
    T, base = t[-1], 0.3
    off, peak = U(rng, 0.17, 0.3), U(rng, 0.3, 0.45)
    if rng.rand() < 0.5:
        off, peak = peak, off                      # shoulder of the peak: falling demand
    tc, w = U(rng, 0.2, 0.8) * T, U(rng, 20, 60)
    lam = off + (peak - off) / (1 + np.exp(-(t - tc) / w))
    return _poisson_rate(rng, lam, t) - base, [tc]


def op_metering_plan(rng, t, P):
    """Fixed-time (pre-timed) ramp-metering plan: the metering rate is a table entry held 1-3 min."""
    rates = np.array([600, 750, 900, 1050, 1200, 1350, 1500]) / 3600.0     # veh/s
    lam, ev = _schedule(t, rng, lambda prev: float(rng.choice(rates)), lambda: U(rng, 60, 180))
    return _poisson_rate(rng, lam, t) - 0.3, ev


def op_signal_platoons(rng, t, P):
    """Arrivals released by an upstream signalised intersection: platoons every cycle (60-120 s), green 30-50 %."""
    C, g = U(rng, 60, 120), U(rng, 0.3, 0.5)
    mean, ph = U(rng, 0.22, 0.38), U(rng, 0, 1)
    in_green = ((t / C + ph) % 1.0) < g
    p = U(rng, 0.08, 0.15)                         # platoon contrast; the cycle mean stays `mean`
    lam = np.where(in_green, mean + p * (1 - g), mean - p * g)
    return _poisson_rate(rng, lam, t, window=15.0) - 0.3, list(np.arange(0, t[-1], C))


def op_event_letout(rng, t, P):
    """Event let-out (stadium / car park): demand surges within ~1 min, holds, then decays."""
    T = t[-1]
    t0, r, hold, dec = U(rng, 0.1, 0.5) * T, U(rng, 30, 90), U(rng, 30, 90), U(rng, 40, 120)
    b, pk = U(rng, 0.17, 0.27), U(rng, 0.38, 0.45)
    lam = np.interp(t, [0, t0, t0 + r, t0 + r + hold, t0 + r + hold + dec, T + 1], [b, b, pk, pk, b, b])
    return _poisson_rate(rng, lam, t) - 0.3, [t0]


# ---- battery (u = cell current [A], + = discharge; time unit = min) ---------------------------------------------

DRIVERS = {   # micro-trip parameters: target speed [m/s], accel / decel [m/s^2], idle / cruise [s], P(stop at end)
    "urban":      dict(v=(6, 15), a=(0.6, 1.4), b=(0.8, 1.8), idle=(5, 40), cruise=(10, 60), p_stop=0.7, sv=0.6,
                      p_hard=0.15),
    "suburban":   dict(v=(12, 22), a=(0.5, 1.2), b=(0.6, 1.4), idle=(5, 30), cruise=(30, 120), p_stop=0.4, sv=0.8,
                      p_hard=0.15),
    "highway_aggr": dict(v=(22, 34), a=(1.5, 3.0), b=(1.5, 3.5), idle=(2, 10), cruise=(15, 80), p_stop=0.1, sv=1.5),
    "congested":  dict(v=(2, 8), a=(1.0, 2.2), b=(1.2, 2.5), idle=(2, 15), cruise=(3, 15), p_stop=0.8, sv=0.3),
}
VEH = dict(m=1600.0, crr=0.010, cda=0.65, rho=1.2, eta=0.90, eta_regen=0.65, aux=600.0, p_1c=60e3)


def drive_speed(rng, T_s, style, dt_s=1.0):
    """Synthetic drive cycle (1 Hz): micro-trips idle -> accelerate -> cruise (OU speed fluctuation) -> decelerate."""
    d = DRIVERS[style]
    n = int(math.ceil(T_s / dt_s)) + 1
    v = np.zeros(n)
    k, cur = 0, 0.0
    while k < n:
        if cur < 0.1:                                   # idle
            m = int(U(rng, *d["idle"]) / dt_s)
            v[k:k + m] = 0.0; k += m
        tgt, a = U(rng, *d["v"]), U(rng, *d["a"])
        if rng.rand() < d.get("p_hard", 0.0):           # occasional full-power acceleration (merge, overtake)
            a = U(rng, 2.0, 3.0)
        while k < n and cur < tgt:                      # accelerate (accel fades near the target speed)
            cur = min(tgt, cur + a * dt_s * max(0.3, 1 - cur / (tgt + 1e-9) * 0.5)); v[k] = cur; k += 1
        m = int(U(rng, *d["cruise"]) / dt_s)
        if k < n and cur > tgt:                         # slower target: brake to it first
            while k < n and cur > tgt:
                cur = max(tgt, cur - U(rng, *d["b"]) * dt_s); v[k] = cur; k += 1
        e = _ou(rng, m, dt_s, 8.0, d["sv"]) if m > 0 else np.zeros(0)
        e = e - e[0] if m > 0 else e                    # speed fluctuation starts from the reached speed
        v[k:k + m] = np.clip(tgt + e, 0, d["v"][1] + 2.0)[:max(0, min(m, n - k))]; k += m
        cur = v[min(k, n) - 1]
        nxt = 0.0 if rng.rand() < d["p_stop"] else U(rng, *d["v"])
        b = U(rng, *d["b"])
        while k < n and cur > nxt:
            cur = max(nxt, cur - b * dt_s); v[k] = cur; k += 1
    v = v[:n]
    for k in range(1, n):                               # traction / braking limit |a| <= 3.5 m/s^2
        v[k] = min(max(v[k], v[k - 1] - 3.5 * dt_s), v[k - 1] + 3.5 * dt_s)
    return v


def speed_to_current(v, dt_s=1.0, i_1c=0.85):
    """Road-load power -> cell current with a fixed pack scaling (1 C = VEH['p_1c'] pack power)."""
    a = np.gradient(v, dt_s)
    F = VEH["m"] * a + VEH["crr"] * VEH["m"] * 9.81 * (v > 0.1) + 0.5 * VEH["rho"] * VEH["cda"] * v ** 2
    Pw = F * v
    Pb = np.where(Pw >= 0, Pw / VEH["eta"], Pw * VEH["eta_regen"]) + VEH["aux"]
    return Pb / VEH["p_1c"] * i_1c


def _drive_op(style):
    def op(rng, t, P):
        T_s = t[-1] * 60.0
        i = speed_to_current(drive_speed(rng, T_s + 2, style))
        return np.interp(t * 60.0, np.arange(len(i)), i), []
    op.__name__ = f"op_drive_{style}"
    op.__doc__ = f"Synthetic {style} drive cycle (micro-trip model) -> road-load power -> cell current."
    return op


def op_charge_rest(rng, t, P):
    """Charger session: rest, constant-current charge at 0.3-1 C, CV-like exponential taper, rest."""
    T, i1c = t[-1], 0.85
    t0, tcc, tau = U(rng, 0, 0.2) * T, U(rng, 0.3, 0.6) * T, U(rng, 2, 6)
    icc = -U(rng, 0.3, 1.0) * i1c
    i = np.where(t < t0, 0.0, np.where(t < t0 + tcc, icc, icc * np.exp(-(t - t0 - tcc) / tau)))
    return i, [t0, t0 + tcc]


# ---- cstr_exo (u = coolant-flow change, fraction of nominal; time unit = min) ---------------------------------

def op_cstr_shift_moves(rng, t, P):
    """Day-shift operator: small manual coolant moves (+-2..10 %) every 30-120 min."""
    def lvl(prev):
        if prev is None:
            return U(rng, -0.3, 0.3)
        return float(np.clip(prev + rng.choice([-1, 1]) * U(rng, 0.02, 0.10), P["lo"], P["hi"]))
    return _schedule(t, rng, lvl, lambda: U(rng, 30, 120))


def op_cstr_production_plan(rng, t, P):
    """Production plan: 2-4 coolant set-points per 8-h shift (rate changes), held 1-3 h."""
    return _schedule(t, rng, lambda prev: U(rng, -0.5, 0.5), lambda: U(rng, 60, 180))


def op_cstr_campaign_ramp(rng, t, P):
    """Campaign change: slow ramp of the coolant flow to a new level over 1-2 h, then hold."""
    T = t[-1]
    a, b = U(rng, -0.4, 0.4), U(rng, -0.4, 0.4)
    t0, dur = U(rng, 0.1, 0.5) * T, U(rng, 60, 120)
    return a + (b - a) * np.clip((t - t0) / dur, 0, 1), [t0]


def op_cstr_alarm_response(rng, t, P):
    """Operator response to a high-temperature alarm: coolant opened wide (+30..50 %) for 10-30 min, then
    returned stepwise to the previous setting."""
    lvl = U(rng, -0.3, 0.1)
    times, vals, tc = [0.0], [lvl], 0.0
    while True:
        tc += U(rng, 90, 200)
        if tc >= t[-1]:
            break
        hi = U(rng, 0.3, 0.5)
        times += [tc, tc + U(rng, 10, 30), tc + U(rng, 35, 50)]
        vals += [hi, 0.5 * (hi + lvl), lvl]
        tc = times[-1]
    return _hold(t, times, vals), times[1::3]


# ---- quad_planar (u = path set-point r [m]) --------------------------------------------------------------------

def op_survey_sweeps(rng, t, P):
    """Survey mission: back-and-forth sweeps between waypoints, trapezoidal velocity, hover at each waypoint."""
    vmax, amax = U(rng, 0.5, 1.5), U(rng, 0.5, 1.5)
    side = [1.0]
    def tgt(pos):
        side[0] = -side[0]
        return side[0] * U(rng, 1.5, 3.0)
    return _moves(t, rng, U(rng, -1, 1), tgt, None, lambda: U(rng, 1, 4), "trapezoid", vmax, amax)


def op_inspection_hops(rng, t, P):
    """Inspection: short hops (0.3-1 m) along the structure with long look-and-hold hovers."""
    vmax, amax = U(rng, 0.3, 0.8), U(rng, 0.5, 1.0)
    def tgt(pos):
        s = rng.choice([-1, 1]) if abs(pos) < 2.4 else -np.sign(pos)
        return float(np.clip(pos + s * U(rng, 0.3, 1.0), -3, 3))
    return _moves(t, rng, U(rng, -2, 2), tgt, None, lambda: U(rng, 3, 8), "trapezoid", vmax, amax)


def op_manual_stick(rng, t, P):
    """Manual RC pilot in position-hold mode: the stick commands a velocity (OU, deflections inside a stick
    deadzone are ignored), integrated to the set-point and stopped at the geofence |r| <= 3 m."""
    dt = t[1] - t[0]
    st = _ou(rng, len(t), dt, U(rng, 0.8, 2.5), 1.0)
    st = np.where(np.abs(st) < 0.3, 0.0, st)
    vmax = U(rng, 0.8, 1.6)
    r = np.empty_like(t)
    r[0] = U(rng, -1, 1)
    for k in range(1, len(t)):
        r[k] = np.clip(r[k - 1] + vmax * np.tanh(st[k]) * dt, -3, 3)
    return r, []


# ---- cartpole / cartpole_b (u = exogenous cart command, force units f_scale = 5 N / unit) -----------------------

def op_cart_reposition(rng, t, P):
    """Operator repositions the cart: minimum-jerk moves between random stations, move 1-3 s, dwell 1-5 s."""
    return _moves(t, rng, U(rng, -0.5, 0.5), lambda pos: U(rng, -0.9, 0.9), lambda: U(rng, 1, 3),
                  lambda: U(rng, 1, 5))


def op_cart_shuttle(rng, t, P):
    """Shuttle duty: periodic transfer between two fixed stations with a fixed takt (move 0.6-1.2 s, dwell 0.5-2 s)."""
    a, b = U(rng, -0.9, -0.2), U(rng, 0.2, 0.9)
    mv, dw = U(rng, 0.6, 1.2), U(rng, 0.5, 2.0)
    st = [a]
    def tgt(pos):
        st[0] = b if st[0] == a else a
        return st[0]
    return _moves(t, rng, a, tgt, lambda: mv * U(rng, 0.95, 1.05), lambda: dw * U(rng, 0.9, 1.1))


# ---- two_link_arm(_b) (u = shoulder torque [N m]) -----------------------------------------------------------------

def op_fourier_excitation(rng, t, P):
    """Periodic Fourier-series excitation trajectory (robot identification practice), 5 harmonics of a base
    frequency, coefficients scaled so sum |a| <= peak (never clipped)."""
    wf, H = U(rng, 0.15, 0.4), 5
    a, b = rng.randn(H), rng.randn(H)
    x = sum(a[h] * np.sin((h + 1) * wf * t) + b[h] * np.cos((h + 1) * wf * t) for h in range(H))
    pk = U(rng, 0.5, 0.95)
    return x * pk / (np.abs(a).sum() + np.abs(b).sum()), []


def _bump(t, t0, w, amp):
    s = (t - t0) / w
    return amp * np.where((s >= 0) & (s <= 1), np.sin(np.pi * np.clip(s, 0, 1)) ** 2, 0.0)


def op_pick_place(rng, t, P):
    """Pick-and-place feedforward: accelerate bump, coast, brake bump of opposite sign, hold; random timing."""
    out, tc, ev = np.zeros_like(t), U(rng, 0.5, 2), []
    while tc < t[-1]:
        w, amp, coast = U(rng, 0.5, 1.5), U(rng, 0.3, 1.0) * rng.choice([-1, 1]), U(rng, 0.2, 1.5)
        out += _bump(t, tc, w, amp) + _bump(t, tc + w + coast, w, -amp * U(rng, 0.7, 1.0))
        ev.append(tc)
        tc += 2 * w + coast + U(rng, 1.5, 5)
    return np.clip(out, -1, 1), ev


def op_task_cycle(rng, t, P):
    """Repetitive work cell: the SAME short torque pattern repeated every cycle (palletising takt 3-6 s)."""
    cyc = U(rng, 3, 6)
    w = U(rng, 0.4, 0.8)
    amp1, amp2 = U(rng, 0.4, 0.9) * rng.choice([-1, 1]), U(rng, 0.3, 0.8) * rng.choice([-1, 1])
    loc = t % cyc
    x = _bump(loc, 0.0, w, amp1) + _bump(loc, w, w, -amp1) + _bump(loc, 2.2 * w, w, amp2) + _bump(loc, 3.2 * w, w, -amp2)
    return np.clip(x, -1, 1), list(np.arange(0, t[-1], cyc))


def op_teleop(rng, t, P):
    """Teleoperation: a human on a joystick, smooth (OU of the torque rate) with physiological tremor (8-12 Hz is
    beyond the sample rate and omitted); fixed RMS."""
    dt = t[1] - t[0]
    rate = _ou(rng, len(t), dt, U(rng, 0.3, 1.0), 1.0)
    x, lam = np.empty_like(t), math.exp(-dt / U(rng, 2.0, 5.0))      # leaky integral of the rate: zero mean
    x[0] = 0.0
    for k in range(1, len(t)):
        x[k] = lam * x[k - 1] + rate[k] * dt
    return np.clip(_fixed_rms(x, U(rng, 0.25, 0.4)), -1, 1), []


# ---- chaotic / forced models (lorenz, rossler, lorenz96): forcing scenarios ---------------------------------------

def op_climatology(rng, t, P):
    """Climatological forcing: slow 'seasonal' cycle (period 0.5-2 T) + red-noise anomalies (fixed RMS)."""
    tr, dt, T = P["t_ref"], t[1] - t[0], t[-1]
    A, per = U(rng, 0.2, 0.5), U(rng, 0.5, 2.0) * T
    mean = U(rng, -0.3, 0.3)
    x = mean + A * np.sin(2 * np.pi * t / per + U(rng, 0, 2 * np.pi)) + _ou(rng, len(t), dt, U(rng, 1, 4) * tr,
                                                                            U(rng, 0.1, 0.2))
    return P["hi"] * x, []


def op_regime_shifts(rng, t, P):
    """Persistent forcing regimes: forcing held at a level for long spells, with red-noise anomalies."""
    tr, dt = P["t_ref"], t[1] - t[0]
    lvl, ev = _schedule(t, rng, lambda _: U(rng, -0.6, 0.6), lambda: U(rng, 6, 15) * tr)
    return P["hi"] * (lvl + _ou(rng, len(t), dt, U(rng, 1, 4) * tr, U(rng, 0.1, 0.2))), ev


def op_scenario_trend(rng, t, P):
    """Scenario run: linear forcing trend over the episode + red-noise anomalies."""
    tr, dt, T = P["t_ref"], t[1] - t[0], t[-1]
    a, b = U(rng, -0.7, 0.0), U(rng, 0.0, 0.7)
    if rng.rand() < 0.5:
        a, b = b, a
    return P["hi"] * (a + (b - a) * t / T + _ou(rng, len(t), dt, U(rng, 1, 4) * tr, U(rng, 0.1, 0.2))), []


def op_volcanic_pulses(rng, t, P):
    """Abrupt negative forcing pulses (volcanic-like) with exponential recovery, on top of a base forcing."""
    tr, dt = P["t_ref"], t[1] - t[0]
    x, tc, ev = np.full_like(t, U(rng, -0.2, 0.4)), U(rng, 1, 6) * tr, []
    while tc < t[-1]:
        x += np.where(t >= tc, -U(rng, 0.4, 0.9) * np.exp(-(t - tc) / (U(rng, 2, 6) * tr)), 0.0)
        ev.append(tc)
        tc += U(rng, 8, 20) * tr
    return P["hi"] * (x + _ou(rng, len(t), dt, 2 * tr, 0.1)), ev


# ---------------------------------------------------------------------------------------------------------
# Plant specifications (legacy grid T, dt, t_ref, box from configs/systems/*.yaml and the case-study configs)
# ---------------------------------------------------------------------------------------------------------

def _spec(unit, physical, time_unit, T, dt, t_ref, lo, hi, act, train, ood, imperf, notes=""):
    return dict(unit=unit, physical=physical, time_unit=time_unit, T=T, dt=dt, t_ref=t_ref, lo=lo, hi=hi,
                act=act, train=train, ood=ood, imperf=imperf, notes=notes)


OSC_TRAIN, OSC_OOD = [op_stepped_sine, op_random_band], [op_swept_sine, op_burst_random]
NEU_TRAIN, NEU_OOD = [op_long_square, op_ramp_protocol], [op_synaptic_noise, op_pulse_train]
CHA_TRAIN, CHA_OOD = [op_climatology, op_regime_shifts], [op_scenario_trend, op_volcanic_pulses]
DAQ = lambda span: Imperfection(noise=0.005, quant=span / 4096, delay=0.0)        # 12-bit logged command

SPECS = {
    "vdp": _spec("forcing (dimensionless)", (0.0, 1.0, "u"), "t_ref", 50, 0.0206, 1.0, -1, 1,
                 Actuator(-1, 1, rate=10.0, tau=0.02), OSC_TRAIN, OSC_OOD, DAQ(2)),
    "duffing": _spec("shaker force (dimensionless)", (0.0, 1.0, "u"), "t_ref", 50, 0.0103, 1.0, -1, 1,
                     Actuator(-1, 1, rate=10.0, tau=0.02), OSC_TRAIN, OSC_OOD, DAQ(2)),
    "fhn": _spec("injected current (dimensionless)", (0.0, 1.0, "u"), "t_ref", 100, 0.0618, 2.0, -1, 1,
                 Actuator(-1, 1, rate=20.0, tau=0.05), NEU_TRAIN, NEU_OOD, DAQ(2)),
    "network": _spec("injected current, node 0", (0.0, 1.0, "u"), "t_ref", 100, 0.0435, 1.5, -1, 1,
                     Actuator(-1, 1, rate=20.0, tau=0.05), NEU_TRAIN, NEU_OOD, DAQ(2)),
    "bioreactor": _spec("dilution-rate change [1/h]", (0.3, 1.0, "D [1/h]"), "h", 100, 0.0618, 1.0, -0.15, 0.15,
                        Actuator(-0.15, 0.15, rate=1.0, tau=0.0, quant=0.003),
                        [op_chemostat_steps, op_shift_up_down], [op_astat, op_feed_interruption],
                        Imperfection(noise=0.01, bias=0.01, quant=0.003),
                        "peristaltic feed pump; D held for several residence times"),
    "quad_tank": _spec("pump-voltage change [V]", (3.0, 1.0, "v1 [V]"), "s", 300, 0.05, 15.0, -1, 1,
                       Actuator(-1, 1, rate=0.5, tau=1.0, quant=10 / 4096),
                       [op_tank_small_moves, op_tank_recipe], [op_tank_grade_change, op_tank_pump_trip],
                       Imperfection(noise=0.002, quant=10 / 4096, delay=0.1)),
    # box widened from the legacy +-0.15 veh/s (540-1620 veh/h) to 180-1800 veh/h (0.05-0.5 veh/s): one-lane ramp
    # from nearly closed to capacity; minute-count fluctuations alone exceed the legacy box (INPUT_REVIEW.md)
    "highway_traffic": _spec("on-ramp inflow change [veh/s]", (0.3, 3600.0, "ramp flow [veh/h]"), "s", 200, 0.0261,
                             10.0, -0.25, 0.2, Actuator(-0.25, 0.2, rate=0.01, tau=10.0),
                             [op_peak_demand, op_metering_plan], [op_signal_platoons, op_event_letout],
                             Imperfection(aggregate=20.0)),
    "battery": _spec("cell current [A], + discharge", (0.0, 1.0, "I [A]"), "min", 30, 0.01483, 1.0, -0.85, 0.85,
                     Actuator(-0.85, 0.85, rate=2 * 0.85 * 60, tau=0.1 / 60),
                     [_drive_op("urban"), _drive_op("suburban"), op_charge_rest],
                     [_drive_op("highway_aggr"), _drive_op("congested")],
                     Imperfection(noise=0.005, bias=0.003, quant=0.001)),
    "cstr_exo": _spec("coolant-flow change [fraction]", (1.0, 100.0, "F_c [% of nominal]"), "min", 480, 1 / 6, 20.0,
                      -0.5, 0.5, Actuator(-0.5, 0.5, rate=1.0, tau=0.2, quant=0.001, deadband=0.005),
                      [op_cstr_shift_moves, op_cstr_production_plan], [op_cstr_campaign_ramp, op_cstr_alarm_response],
                      Imperfection(noise=0.01, use_command=True)),
    "quad_planar": _spec("path set-point r [m]", (0.0, 1.0, "r [m]"), "s", 20, 0.01, 1.0, -3, 3,
                         Actuator(-3, 3, rate=2.0), [op_survey_sweeps], [op_inspection_hops, op_manual_stick],
                         Imperfection(delay=0.02)),
    "cartpole": _spec("cart command [x 5 N]", (0.0, 5.0, "F_cmd [N]"), "s", 30, 0.01725, 1.0, -1, 1,
                      Actuator(-1, 1, rate=4.0, tau=0.01), [op_cart_reposition], [op_cart_shuttle], DAQ(2)),
    "two_link_arm": _spec("shoulder torque [N m]", (0.0, 1.0, "tau [N m]"), "s", 100, 0.01294, 2.0, -1, 1,
                          Actuator(-1, 1, rate=10.0, tau=0.005), [op_fourier_excitation, op_pick_place],
                          [op_task_cycle, op_teleop], DAQ(2)),
    "lorenz": _spec("forcing (dimensionless)", (0.0, 1.0, "u"), "t_ref", 50, 0.01113, 1.0, -1, 1,
                    Actuator(-1, 1), CHA_TRAIN, CHA_OOD, Imperfection()),
    "rossler": _spec("forcing (dimensionless)", (0.0, 1.0, "u"), "t_ref", 160, 0.02225, 8.0, -1, 1,
                     Actuator(-1, 1), CHA_TRAIN, CHA_OOD, Imperfection()),
    "lorenz96": _spec("forcing anomaly on nodes 0/5/10/15", (8.0, 1.0, "F_i"), "t_ref", 24, 0.00187, 1.0, -4, 4,
                      Actuator(-4, 4), CHA_TRAIN, CHA_OOD, Imperfection()),
}
SPECS["cartpole_b"] = dict(SPECS["cartpole"])
SPECS["two_link_arm_b"] = dict(SPECS["two_link_arm"])
SPECS["fhnring"] = dict(SPECS["network"], T=100, dt=0.05)


def physical(plant: str, u: np.ndarray) -> np.ndarray:
    off, sc, _ = SPECS[plant]["physical"]
    return (off + np.asarray(u)) * sc


# ---------------------------------------------------------------------------------------------------------
# Profile + Signal adapter
# ---------------------------------------------------------------------------------------------------------

class SampledSignal:
    """Duck-types hyperkkl.signals.Signal: zero-order hold of grid samples (what data.integrate applies)."""

    def __init__(self, family, u, dt, params=None):
        self.family, self.u, self.dt = family, np.asarray(u, float), float(dt)
        self.params, self.scale = dict(params or {}), 1.0

    def __call__(self, t):
        k = np.floor(np.asarray(t, float) / self.dt + 1e-6).astype(np.int64)
        return self.scale * self.u[np.clip(k, 0, len(self.u) - 1)]

    def describe(self):
        return {"family": self.family, "scale": self.scale, **{k: v for k, v in self.params.items()
                                                               if not k.startswith("_")}}


@dataclass
class Profile:
    plant: str
    split: str
    operator: str
    t: np.ndarray
    cmd: np.ndarray
    u: np.ndarray
    events: list = field(default_factory=list)
    seed: int = 0

    def signal(self) -> SampledSignal:
        return SampledSignal(f"profile:{self.operator}", self.u, self.t[1] - self.t[0],
                             {"split": self.split, "seed": self.seed})


def operators(plant: str, split: str) -> list:
    return SPECS[plant]["train" if split in ("train", "id", "val") else "ood"]


def generate(plant: str, split: str, seed: int, T: float = None, dt: float = None, operator: str = None) -> Profile:
    """One episode. split in {train, id, val} (training operators) or {ood} (held-out operators)."""
    P = dict(SPECS[plant])
    T = float(P["T"] if T is None else T)
    dt = float(P["dt"] if dt is None else dt)
    rng = np.random.RandomState(seed)
    ops = operators(plant, split)
    op = ops[rng.randint(len(ops))] if operator is None else next(o for o in ops if o.__name__ == operator)
    t = np.arange(int(math.ceil(T / dt)) + 1) * dt
    cmd, ev = op(rng, t, P)
    cmd = np.asarray(cmd, float)
    u = apply_actuator(cmd, dt, P["act"])
    return Profile(plant, split, op.__name__, t, cmd, u, list(map(float, ev)), seed)


def describe_specs() -> dict:
    out = {}
    for k, P in SPECS.items():
        out[k] = {"unit": P["unit"], "time_unit": P["time_unit"], "box": [P["lo"], P["hi"]],
                  "actuator": asdict(P["act"]), "train": [o.__name__ for o in P["train"]],
                  "ood": [o.__name__ for o in P["ood"]], "record": asdict(P["imperf"])}
    return out
