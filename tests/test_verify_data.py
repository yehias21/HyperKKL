"""Numerical verification of signals.py, design.py and data.py against their docstrings
and THEORY.md.

    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest tests/test_verify_data.py -q -s

Expensive artefacts (system scales) go to the content-addressed experiments/cache;
training_data / eval_data are built into a temporary cache so they can be rebuilt from
scratch (independence of the eval set from the training seed is tested on fresh builds).
"""

from __future__ import annotations

import copy
import hashlib
import math

import numpy as np
import pytest
import torch
from scipy.linalg import expm

from hyperkkl import data as Dm
from hyperkkl import design as D
from hyperkkl import signals as S
from hyperkkl.config import load, system_names
from hyperkkl.systems import Plant, build_plant

torch.set_num_threads(4)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMALL = {"data.n_p1": 16, "data.n_train": 30, "data.n_val": 12, "data.n_coll": 256,
         "eval.n_episodes": 12}
REPORT = {}


def _note(key, val):
    REPORT[key] = val
    print(f"\n[verify] {key}: {val}")


def _h(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes()).hexdigest()


# ===========================================================================
# signals.py
# ===========================================================================

STRATIFIED = {"constant": ["A"], "step": ["A"], "prbs": ["A"], "square": ["A", "w"],
              "sinusoid": ["A", "w"], "multisine": ["A"], "ramp": ["A"], "am": ["A"],
              "chirp": ["A"], "filtered_noise": ["A"], "triangle": ["A", "w"],
              "sawtooth": ["A", "w"]}
REGIMES = ("train", "id", "shape_ood", "sequence_ood")


def _regimes(fam):
    return [r for r in REGIMES if fam in S.families_for(r)]


def _in(v, lo, hi, tol=1e-12):
    return lo - tol <= v <= hi + tol


def _check_offset(q):
    assert abs(q["offset"]) <= 1.0 - abs(q["A"]) + 1e-12, q


@pytest.mark.parametrize("fam", sorted(S.RANGES))
def test_parameter_ranges_per_regime(fam):
    dt, T = 0.02, 60.0
    rng = np.random.RandomState(10)
    for regime in _regimes(fam):
        R = S.RANGES[fam][S.range_key(fam, regime)]
        for i in range(200):
            s = S.sample_signal(fam, regime, rng, dt, T, stratum=i, n_strata=4)
            q = s.params
            for key, (lo, hi) in ((k, v) for k, v in R.items() if isinstance(v, tuple)):
                if key not in q or key == "n_comp":
                    continue
                v = q[key]
                if key == "t_step":
                    v = v / T
                    lo, hi = lo - dt / T, hi + dt / T
                assert _in(v, lo, hi), (fam, regime, key, v, lo, hi)
            if "offset_frac" in R:
                _check_offset(q)
            if fam == "staircase":
                assert np.all(np.abs(s.levels) <= 1.0), s.levels
                d = np.diff(s.switch_times)
                dl, dh = R["dwell"]
                assert np.all((d >= dl - dt - 1e-9) & (d <= dh + dt + 1e-9)), (d.min(), d.max())
            if fam == "prbs":
                dev = s.levels - q["offset"]
                assert np.allclose(np.abs(dev), q["A"]) and _in(q["A"], *R["A"])
                assert np.all(np.sign(dev[1:]) == -np.sign(dev[:-1]))
                d = np.diff(s.switch_times)
                dl, dh = R["dwell"]
                assert np.all((d >= dl - dt - 1e-9) & (d <= dh + dt + 1e-9)), (d.min(), d.max())
            if fam == "square":
                hi_, lo_ = q["A"] + q["offset"], -q["A"] + q["offset"]
                assert np.all(np.isclose(s.levels, hi_) | np.isclose(s.levels, lo_))
                per = 2 * np.pi / q["w"]
                d = np.diff(s.switch_times)[1:]          # interior segments only
                lv = s.levels[1:-1]
                dh = d[np.isclose(lv, hi_)]
                dlw = d[np.isclose(lv, lo_)]
                assert np.all(np.abs(dh - q["duty"] * per) <= dt + 1e-9)
                assert np.all(np.abs(dlw - (1 - q["duty"]) * per) <= dt + 1e-9)
            if fam == "multisine":
                assert 3 <= len(q["amps"]) <= 6
                assert abs(np.sum(q["amps"]) - q["A"]) < 1e-12
                assert np.all((q["freqs"] >= R["w"][0]) & (q["freqs"] <= R["w"][1]))
            if fam == "am":
                t = np.arange(0, T, dt)
                assert np.abs(s(t)).max() <= q["A"] + 1e-12          # A is the peak, A(1+m) form
            if fam == "filtered_noise":
                assert abs(np.abs(q["_v"] - q["offset"]).max() - q["A"]) < 1e-12
            if fam == "chirp":
                t = np.linspace(0, T, 7)
                k = (q["w1"] - q["w0"]) / T
                ref = q["A"] * np.sin(q["w0"] * t + 0.5 * k * t ** 2 + q["phi"]) + q["offset"]
                assert np.allclose(s(t), ref)
            if fam in ("triangle", "sawtooth"):
                a, o = q["A"], q["offset"]
                inner = s.knot_vals[1:]                  # the first knot is the phase value at 0
                assert np.all(np.isclose(inner, o + a) | np.isclose(inner, o - a))
                d = np.diff(s.knots)[2:-1]                # interior segments only
                per = 2 * np.pi / q["w"]
                if fam == "triangle":
                    assert np.all(np.abs(d - per / 2) <= dt + 1e-9)
                else:                                     # rise over the period, flyback 1 step
                    assert np.all((np.abs(d - (per - dt)) <= dt + 1e-9) | np.isclose(d, dt))
            if fam == "drive":
                ev = S.drive_events(s)
                L = np.linspace(-1, 1, R["n_levels"])
                assert np.all(np.isin(np.round(ev["levels"], 12), np.round(L, 12)))
                assert np.all((ev["delta"] >= R["delta"][0] - 1e-9) & (ev["delta"] <= R["delta"][1] + 1e-9))
                assert np.all((ev["dwell"] >= R["dwell"][0]) & (ev["dwell"] <= R["dwell"][1]))
                assert np.all((ev["ramp"] >= R["ramp"][0]) & (ev["ramp"] <= R["ramp"][1]))
                assert np.all(np.isin(np.round(s.knot_vals, 12), np.round(L, 12)))
                assert s.knots[-1] >= T - dt


@pytest.mark.parametrize("dt", [0.013, 0.0206, 0.05])
def test_every_family_inside_the_box_at_every_rk4_stage(dt):
    """Input protocol v2: v(t) in [-1, 1] for every family and regime, by construction
    (no clipping anywhere), at all three RK4 stage times of every step; with u_scale
    the applied input is u_scale * v."""
    T = 40.0
    N = int(math.ceil(T / dt))
    t = np.arange(N + 1) * dt
    rng = np.random.RandomState(31)
    worst = {}
    for regime in REGIMES:
        fams = S.families_for(regime)
        for r in range(30 * len(fams)):
            fam = fams[r % len(fams)]
            s = S.sample_signal(fam, regime, rng, dt, N * dt, stratum=r // len(fams), scale=2.5)
            v = S.stage_inputs([s], t)[:, :, 0] / 2.5
            worst[(regime, fam)] = max(worst.get((regime, fam), 0.0), float(np.abs(v).max()))
    bad = {k: v for k, v in worst.items() if v > 1.0 + 1e-12}
    assert not bad, bad
    _note(f"dt={dt}: peak |v| per (regime, family)", {f"{k[0]}/{k[1]}": round(v, 3) for k, v in worst.items()})
    # periodic families with A near 1 and every level family reach the edges of the box
    for (regime, fam), v in worst.items():
        if fam != "zero" and regime in ("train", "id"):
            assert v > 0.9, (regime, fam, v)


def test_training_inputs_cover_the_box():
    """The union of training realisations covers [-0.95, 0.95] densely (every 0.05 bin
    holds >= 0.3 % of the samples, i.e. >= 15 % of the uniform share) and never leaves
    [-1, 1] at any RK4 stage time; both edges are reached."""
    dt, T = 0.02, 50.0
    t = np.arange(int(T / dt) + 1) * dt
    rng = np.random.RandomState(32)
    fams = S.TRAIN_FAMILIES
    vals = np.concatenate([S.stage_inputs([S.sample_signal(fams[r % len(fams)], "train", rng, dt, T,
                                                           stratum=r // len(fams))], t).ravel()
                           for r in range(40 * len(fams))])
    assert np.abs(vals).max() <= 1.0 + 1e-12
    assert vals.max() > 0.999 and vals.min() < -0.999
    h, _ = np.histogram(vals, bins=38, range=(-0.95, 0.95))
    frac = h / len(vals)
    _note("training |v| coverage: min / max bin share over [-0.95, 0.95] (uniform 0.026)",
          (round(float(frac.min()), 4), round(float(frac.max()), 4)))
    assert frac.min() > 0.003, frac


def test_shape_ood_families_disjoint_and_in_band():
    """shape_ood waveforms never occur in training (or id), and their time scales lie inside
    the training band (in t_ref units): frequencies within the union of the training
    frequency ranges, the OU correlation time within the training dwell range."""
    held = set(S.SHAPE_OOD_FAMILIES)
    assert held == {"chirp", "filtered_noise", "triangle", "sawtooth"}
    assert not held & set(S.families_for("train")) and not held & set(S.families_for("id"))
    assert set(S.families_for("shape_ood")) == held and S.families_for("sequence_ood") == ["drive"]
    assert "zero" not in S.families_for("id") and "drive" in S.families_for("train")
    w_tr = [v for f in S.TRAIN_FAMILIES for k, v in S.RANGES.get(f, {}).get("train", {}).items()
            if k == "w"]
    wlo, whi = min(r[0] for r in w_tr), max(r[1] for r in w_tr)
    for f in held:
        R = S.RANGES[f]["shape_ood"]
        for k in ("w", "w0", "w1"):
            if k in R:
                assert wlo <= R[k][0] and R[k][1] <= whi, (f, k)
        assert "train" not in S.RANGES[f]
    dw = [v for f in S.TRAIN_FAMILIES for k, v in S.RANGES.get(f, {}).get("train", {}).items() if k == "dwell"]
    tau = S.RANGES["filtered_noise"]["shape_ood"]["tau"]
    assert min(r[0] for r in dw) <= tau[0] and tau[1] <= max(r[1] for r in dw)
    tab = S.regime_table()
    assert set(tab) == {"train", "id", "shape_ood", "sequence_ood"}
    assert tab["sequence_ood"]["drive"] == S.RANGES["drive"]["sequence_ood"]


def test_drive_train_vs_sequence_ood_statistics():
    """The two driver regimes share the level grid and range; they differ in transition
    structure (mean |delta level| 0.25-0.5 vs >= 1) and timing (dwell U[2, 6] vs U[0.5, 2],
    ramp U[0.5, 1.5] vs U[0.2, 0.5], t_ref units)."""
    dt, T = 0.02, 200.0
    st = {}
    for regime in ("train", "sequence_ood"):
        rng = np.random.RandomState(33)
        ev = [S.drive_events(S.sample_signal("drive", regime, rng, dt, T, stratum=i)) for i in range(200)]
        cat = lambda k: np.concatenate([e[k] for e in ev])
        st[regime] = {k: cat(k) for k in ("levels", "delta", "dwell", "ramp")}
    tr, so = st["train"], st["sequence_ood"]
    _note("drive: mean |dlevel|, mean dwell, mean ramp (train, sequence_ood)",
          {k: (round(float(tr[k].mean()), 3), round(float(so[k].mean()), 3)) for k in ("delta", "dwell", "ramp")})
    L = np.round(np.linspace(-1, 1, 9), 12)
    assert set(np.round(tr["levels"], 12)) == set(L) == set(np.round(so["levels"], 12))   # same grid, all visited
    assert tr["levels"].min() == so["levels"].min() == -1.0 and tr["levels"].max() == so["levels"].max() == 1.0
    assert tr["delta"].max() <= 0.5 + 1e-9 and so["delta"].min() >= 1.0 - 1e-9
    assert 0.3 < tr["delta"].mean() < 0.45 and so["delta"].mean() > 1.2
    assert abs(tr["dwell"].mean() - 4.0) < 0.1 and abs(so["dwell"].mean() - 1.25) < 0.05
    assert tr["dwell"].min() >= 2.0 and so["dwell"].max() <= 2.0
    assert tr["ramp"].min() >= 0.5 and so["ramp"].max() <= 0.5
    from scipy.stats import ks_2samp
    assert ks_2samp(tr["dwell"], so["dwell"]).pvalue < 1e-6


def test_t_ref_scaling():
    """Time-like parameters are in units of t_ref: with the same draws, a frequency scales
    as 1/t_ref and a dwell / ramp / correlation time as t_ref; with T and dt scaled too, the
    whole realisation is time-rescaled, u_k(k t) = u_1(t), for every family."""
    k, dt, T = 7.0, 0.02, 50.0
    tm = (np.arange(int(T / dt)) + 0.5) * dt          # mid-step: no ties at switch instants
    for f in S.TRAIN_FAMILIES + S.SHAPE_OOD_FAMILIES:
        reg = "train" if f in S.TRAIN_FAMILIES else "shape_ood"
        a = S.sample_signal(f, reg, np.random.RandomState(3), dt, T, stratum=5)
        b = S.sample_signal(f, reg, np.random.RandomState(3), k * dt, k * T, stratum=5, t_ref=k)
        assert np.abs(a(tm) - b(k * tm)).max() < 1e-9, f
        for key in S.FREQ_KEYS & set(a.params):
            assert np.isclose(b.params[key], a.params[key] / k), (f, key)
        for key in S.TIME_KEYS & set(a.params):
            assert np.isclose(b.params[key], a.params[key] * k), (f, key)
    a = S.sample_signal("staircase", "train", np.random.RandomState(4), dt, 400.0)
    b = S.sample_signal("staircase", "train", np.random.RandomState(4), dt, 400.0, t_ref=3.0)
    da, db = np.diff(a.switch_times), np.diff(b.switch_times)
    assert np.all((da >= 2 - dt) & (da <= 8 + dt)) and np.all((db >= 6 - dt) & (db <= 24 + dt))
    a = S.sample_signal("drive", "train", np.random.RandomState(5), dt, 400.0)
    b = S.sample_signal("drive", "train", np.random.RandomState(5), dt, 400.0, t_ref=3.0)
    assert np.allclose(S.drive_events(a)["dwell"][:5], S.drive_events(b)["dwell"][:5])     # t_ref units
    ka, kb = a.knots[:6], b.knots[:6]
    assert np.allclose(kb, 3.0 * ka, atol=2 * dt)                                          # seconds
    cfg = load("quad_tank")
    sigs = Dm._signals(["sinusoid"], 8, "train", np.random.RandomState(6), Dm.make_design(cfg["system"]),
                       cfg["system"])
    t_ref = float(cfg["system"]["t_ref"])
    assert t_ref > 1 and all(0.5 / t_ref - 1e-12 <= s.params["w"] <= 3.0 / t_ref + 1e-12 for s in sigs)


@pytest.mark.parametrize("dt", [0.013, 0.02060113295832983, 0.06180339887498948, 0.1])
def test_switch_times_exactly_on_grid(dt):
    T = 40.0
    N = int(math.ceil(T / dt))
    grid = np.arange(N + 1) * dt                    # exactly Design.t
    rng = np.random.RandomState(4)
    for fam in sorted(S.PIECEWISE_CONSTANT):
        for regime in ("train", "id"):
            for i in range(30):
                s = S.sample_signal(fam, regime, rng, dt, N * dt, stratum=i)
                st = s.switch_times
                assert st[0] == 0.0 and np.all(np.diff(st) > 0.5 * dt), fam
                assert np.all(np.isin(st[st <= grid[-1]], grid)), (fam, st[~np.isin(st, grid)][:3])
                assert len(st) == len(s.levels)
    for fam in sorted(S.PIECEWISE_LINEAR):           # kinks of ramp segments on the grid too
        for regime in _regimes(fam):
            for i in range(30):
                s = S.sample_signal(fam, regime, rng, dt, N * dt, stratum=i)
                kt = s.knots
                assert kt[0] == 0.0 and np.all(np.diff(kt) > 0.5 * dt), fam
                assert np.all(np.isin(kt[kt <= grid[-1]], grid)), (fam, kt[~np.isin(kt, grid)][:3])
                assert len(kt) == len(s.knot_vals)


def test_pwc_keeps_the_level_in_force_after_coincident_switches():
    """Switches that snap onto the same grid point create zero-length segments; the level
    in force afterwards is the LAST one."""
    st, lv = S._pwc([0.0, 0.0, 0.31, 0.32, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0], 0.1)
    assert np.allclose(st, [0.0, 0.3, 1.0]) and np.allclose(lv, [2.0, 4.0, 5.0]), (st, lv)


def test_square_wave_random_phase():
    """The square wave's phase is uniform, so it starts in its high phase with probability
    duty; the starting level must not always be the high one."""
    rng = np.random.RandomState(0)
    hi, duty = [], []
    for i in range(2000):
        s = S.sample_signal("square", "train", rng, 0.02, 50.0, stratum=i)
        hi.append(abs(s.levels[0] - (s.params["A"] + s.params["offset"])) < 1e-12)
        duty.append(s.params["duty"])
    f, e = float(np.mean(hi)), float(np.mean(duty))
    _note("square: P(start high) vs E[duty]", (round(f, 3), round(e, 3)))
    assert abs(f - e) < 0.04, (f, e)


def test_enforce_bounds_rescales_never_clips():
    cfg = load("bioreactor")
    scfg = cfg["system"]
    des = Dm.make_design(scfg)
    lo, hi = scfg["u_bounds"]
    n_res, worst, frac_rescaled = 0, 0.0, {}
    for regime in ("train", "shape_ood", "sequence_ood"):
        fams = S.families_for(regime)
        for r in range(150):
            fam = fams[r % len(fams)]
            seed = 1000 * len(regime) + r
            raw = S.sample_signal(fam, regime, np.random.RandomState(seed), des.dt, des.T,
                                  stratum=r // len(fams), scale=scfg["u_scale"])
            ref = copy.deepcopy(raw)
            s = S.enforce_bounds(raw, des.dt, des.T, scfg["u_bounds"])
            st = S.stage_inputs([s], des.t)[:, :, 0]          # every value the plant sees
            v_ref = S.stage_inputs([ref], des.t)[:, :, 0]
            f = s.params.get("bound_rescale", 1.0)
            assert f <= 1.0
            assert np.allclose(st, f * v_ref, rtol=0, atol=1e-15), "not a pure rescale"
            viol = max(st.max() - hi, lo - st.min(), 0.0)
            worst = max(worst, viol)
            if f < 1.0:
                n_res += 1
                frac_rescaled[(regime, fam)] = frac_rescaled.get((regime, fam), 0) + 1
                # rescaled to touch the bound, not shrunk further
                touch = min(abs(st.max() - hi), abs(st.min() - lo))
                assert touch < 1e-9, (regime, fam, touch)
            else:
                assert v_ref.max() <= hi + 1e-12 and v_ref.min() >= lo - 1e-12
    _note("bioreactor enforce_bounds: worst bound violation at RK4 stage times", worst)
    _note("bioreactor rescaled signals per (regime, family) of 150/len(fams)",
          {f"{k[0]}/{k[1]}": v for k, v in sorted(frac_rescaled.items())})
    assert worst < 1e-9, worst
    s = S.sample_signal("constant", "train", np.random.RandomState(1), 0.05, 10.0, 0)
    a = s.scale
    assert S.enforce_bounds(s, 0.05, 10.0, []).scale == a        # no bounds: untouched


def test_filtered_noise_is_ou(monkeypatch):
    """Exact OU discretisation: lag-1 autocorrelation e^{-dt/tau} (not EM's 1 - dt/tau),
    max |u - offset| = A, linear between samples."""
    dt = 0.05
    for i, tau in enumerate([0.5, 2.0, 5.0]):
        monkeypatch.setitem(S.RANGES["filtered_noise"]["shape_ood"], "tau", (tau, tau))
        s = S.sample_signal("filtered_noise", "shape_ood", np.random.RandomState(20 + i),
                            dt, 20000.0)
        assert s.params["tau"] == tau
        th = dt / tau
        v = s.params["_v"][int(20 * tau / dt):]      # drop the start-at-zero transient
        x = v - v.mean()
        r1 = float((x[1:] * x[:-1]).mean() / (x * x).mean())
        se = math.sqrt((1 - r1 ** 2) / len(x))
        _note(f"filtered_noise tau={tau}: lag-1 corr (measured, EM 1-th, exact e^-th, se)",
              (round(r1, 5), round(1 - th, 5), round(math.exp(-th), 5), round(se, 5)))
        assert abs(r1 - math.exp(-th)) < 4 * se, (r1, math.exp(-th), se)
        o = s.params["offset"]
        assert abs(np.abs(s.params["_v"] - o).max() - s.params["A"]) < 1e-12
        tm = (np.arange(10) + 0.5) * dt
        assert np.allclose(s(tm), 0.5 * (s.params["_v"][:10] + s.params["_v"][1:11]))
        assert s.params["_v"][0] == o               # starts at the offset (non-stationary start)


def test_stage_inputs_inside_step_values():
    dt, N = 0.1, 50
    t = np.arange(N + 1) * dt
    st, lv = S._pwc([0.0, 2.0, 3.3], [1.0, -2.0, 5.0], dt)
    sig = S.Signal("step", {}, st, lv, 1.0)
    out = S.stage_inputs([sig], t)[:, :, 0]
    k = int(round(2.0 / dt))
    assert np.all(out[:k] == 1.0) and np.all(out[k:33] == -2.0) and np.all(out[33:] == 5.0)
    assert out[k - 1, 2] == 1.0 and out[k, 0] == -2.0
    # every piecewise-constant family: constant inside every step
    rng = np.random.RandomState(5)
    for fam in sorted(S.PIECEWISE_CONSTANT):
        for regime in ("train", "id"):
            s = S.sample_signal(fam, regime, rng, 0.02, 30.0, stratum=1)
            tt = np.arange(1501) * 0.02
            o = S.stage_inputs([s], tt)[:, :, 0]
            assert np.all(o[:, 0] == o[:, 1]) and np.all(o[:, 1] == o[:, 2]), fam
            assert np.all(o[:, 0] == s(tt[:-1])), fam       # u array = value applied on the step
    s = S.sample_signal("sinusoid", "train", rng, 0.02, 30.0)
    tt = np.arange(1501) * 0.02
    o = S.stage_inputs([s], tt)[:, :, 0]
    assert np.allclose(o[:, 1], s(tt[:-1] + 0.01), atol=1e-12)
    assert np.allclose(o[:, 2], s(tt[1:]), atol=1e-9)


def test_stratification():
    """_signals: signal r has family r % F and stratum r // F. The first stratified
    parameter (A) uses bin stratum % 4, a second one (w) bin (stratum // 4) % 4, so (A, w)
    tile a 4 x 4 grid instead of moving together; staircases stratify only their first
    level, drive its starting level (grid level stratum % 9)."""
    cfg = load("vdp")
    des = Dm.make_design(cfg["system"])
    for regime in ("train", "shape_ood", "sequence_ood"):
        fams = S.families_for(regime)
        n = 16 * len(fams)
        sigs = Dm._signals(fams, n, regime, np.random.RandomState(6), des, cfg["system"])
        for r, s in enumerate(sigs):
            assert s.family == fams[r % len(fams)]
            st = r // len(fams)
            R = S.RANGES.get(s.family, {}).get(S.range_key(s.family, regime), {})
            for i, key in enumerate(STRATIFIED.get(s.family, [])):
                k = st % 4 if i == 0 else (st // 4) % 4
                lo, hi = R[key]
                w = (hi - lo) / 4
                v = s.params[key]
                assert _in(v, lo + k * w, lo + (k + 1) * w), (regime, s.family, key, v, k)
            if s.family == "staircase":
                lo, hi = R["A"]
                w, k = (hi - lo) / 4, st % 4
                a = s.levels
                assert _in(a[0], lo + k * w, lo + (k + 1) * w)
                assert np.all((a >= lo - 1e-12) & (a <= hi + 1e-12))
            if s.family == "drive":
                assert np.isclose(s.knot_vals[0], np.linspace(-1, 1, 9)[st % 9])
    # the shared-stratum coupling is gone, and a staircase spans the whole range
    sigs = Dm._signals(["sinusoid"], 400, "train", np.random.RandomState(7), des, cfg["system"])
    A = np.array([s.params["A"] for s in sigs]); w = np.array([s.params["w"] for s in sigs])
    corr = float(np.corrcoef(A, w)[0, 1])
    _note("train sinusoid corr(A, w)", round(corr, 3))
    assert abs(corr) < 0.2, corr                          # was 0.93 with one shared index
    sigs = Dm._signals(["staircase"], 400, "train", np.random.RandomState(8), des, cfg["system"])
    span = np.array([s.levels.max() - s.levels.min() for s in sigs])
    _note("train staircase max level span within an episode (range is 2.0)", round(float(span.max()), 3))
    assert span.max() > 1.5                               # was <= 0.5 (one stratum bin)


# ===========================================================================
# design.py
# ===========================================================================

DESIGNS = [(1, 2.0, 1), (2, 0.7, 1), (4, 1.0, 1), (5, 1.5, 1), (5, 3.0, 1), (6, 0.5, 2),
           (7, 2.0, 1), (12, 0.5, 3)]


@pytest.mark.parametrize("nz,alpha,ny", DESIGNS)
def test_butterworth_design(nz, alpha, ny):
    p = D.butterworth_poles(nz, alpha)
    assert abs(p.real.max() + alpha) < 1e-12 * max(1, alpha)
    assert np.all(p.real < 0)
    assert np.allclose(np.abs(p), alpha / np.sin(np.pi / (2 * nz)))
    assert max(np.abs(np.conj(q) - p).min() for q in p) < 1e-12 * np.abs(p).max()   # conj-closed
    A, B = D.butterworth_AB(nz, alpha, ny)
    ev = np.linalg.eigvals(A)
    assert max(min(abs(e - q) for q in p) for e in ev) < 1e-12 * max(1, np.abs(p).max())
    assert max(min(abs(e - q) for q in ev) for e in p) < 1e-12 * max(1, np.abs(p).max())
    assert np.abs(A @ A.T - A.T @ A).max() < 1e-12 * np.abs(A).max() ** 2      # normal
    for t in (0.1, 1.0, 3.0 / alpha):
        assert abs(np.linalg.norm(expm(A * t), 2) - np.exp(-alpha * t)) < 1e-12
    kappa, a = D.spectral_bounds(A)
    assert abs(kappa - 1.0) < 1e-10 and abs(a - alpha) < 1e-12
    dt = D.choose_dt(A)
    assert abs(np.abs(ev).max() * dt - 0.1) < 1e-14
    assert B.shape == (nz, ny)
    if ny > 1:
        assert np.linalg.matrix_rank(B) == ny
    chk = D.validate(A, B, dt, alpha)
    assert chk["ok"], chk


def test_real_jordan_arbitrary_conjugate_closed_spectrum():
    rng = np.random.RandomState(0)
    c = -rng.uniform(0.1, 2, 3) + 1j * rng.uniform(0.1, 3, 3)
    poles = np.concatenate([c, [-0.7, -1.3], np.conj(c)])
    rng.shuffle(poles)
    A = D.real_jordan(poles)
    ev = np.linalg.eigvals(A)
    assert np.allclose(np.sort_complex(ev), np.sort_complex(poles), atol=1e-12)
    assert np.abs(A @ A.T - A.T @ A).max() < 1e-12


def _van_loan(A, B, dt):
    """Exact map of z' = A z + B y(s) + g with y(s) = y0 + s b on [0, dt]."""
    nz, ny = B.shape
    n = nz + 2 * ny + nz
    M = np.zeros((n, n))
    M[:nz, :nz] = A
    M[:nz, nz:nz + ny] = B                      # z' += B y
    M[nz:nz + ny, nz + ny:nz + 2 * ny] = np.eye(ny)   # y' = b
    M[:nz, nz + 2 * ny:] = np.eye(nz)           # z' += g
    E = expm(M * dt)
    Phi, Gy, Gb, Gg = (E[:nz, :nz], E[:nz, nz:nz + ny], E[:nz, nz + ny:nz + 2 * ny],
                       E[:nz, nz + 2 * ny:])
    return {"Ad": Phi, "W0": Gy - Gb / dt, "W1": Gb / dt, "Wg": Gg}


@pytest.mark.parametrize("nz,alpha,ny", DESIGNS)
def test_latent_step_exact(nz, alpha, ny):
    A, B = D.butterworth_AB(nz, alpha, ny)
    for dt in (D.choose_dt(A), 0.5 * D.choose_dt(A), 0.3 / alpha):
        st = D.latent_step(A, B, dt)
        ref = _van_loan(A, B, dt)
        for k in ("Ad", "W0", "W1", "Wg"):
            err = np.abs(st[k] - ref[k]).max() / max(np.abs(ref[k]).max(), 1e-300)
            assert err < 1e-13, (k, dt, err)
        assert np.abs(st["Wg"] - np.linalg.solve(A, st["Ad"] - np.eye(nz))).max() < 1e-13
        # W0 + W1 = Wg B (a constant y), and y piecewise-linear over many steps is exact
        assert np.allclose(st["W0"] + st["W1"], st["Wg"] @ B, atol=1e-14)


def test_washout_and_memory_match_tail_bound():
    kappa, alpha, Bn, yb, zs, eps = 1.0, 1.5, math.sqrt(5), 3.1, 0.6, 1e-3
    for z0n in (0.0, 2.0):
        t = D.washout_time(kappa, alpha, Bn, yb, zs, eps, z0n)
        lhs = kappa * math.exp(-alpha * t) * (z0n + kappa * Bn * yb / alpha)
        assert abs(lhs - eps * zs) < 1e-12 * eps * zs
    k2 = 1.7
    t = D.washout_time(k2, alpha, Bn, yb, zs, eps)
    assert abs(k2 * math.exp(-alpha * t) * k2 * Bn * yb / alpha - eps * zs) < 1e-15
    L = D.memory_length(kappa, alpha, Bn, yb, zs, eps)
    tail = kappa * Bn * yb * math.exp(-alpha * L) / alpha
    assert abs(tail - eps * zs) < 1e-15
    # Lemma 1's eps(L) compares TWO maps (factor 2): at memory_s it equals 2 eps z_scale
    lemma1 = 2 * kappa * Bn * yb * math.exp(-alpha * L) / alpha
    assert abs(lemma1 / (eps * zs) - 2.0) < 1e-9
    assert D.washout_time(1, 1, 1, 1e-9, 1.0) == 0.0 and D.memory_length(1, 1, 1, 1e-9, 1.0) == 0.0


def test_washout_semantics_on_the_deployed_filter():
    """A latent initialised anywhere in the ball of radius ||B|| y_bar / alpha differs from
    the z0 = 0 latent by at most eps z_scale after t_burn (kappa = 1)."""
    A, B = D.butterworth_AB(5, 1.5, 1)
    dt = D.choose_dt(A)
    yb, zs, eps = 3.0, 0.5, 1e-3
    tb = D.washout_time(1.0, 1.5, np.linalg.norm(B, 2), yb, zs, eps)
    burn = int(math.ceil(tb / dt))
    des = Dm.Design(A, B, dt, burn + 5, 1.5, D.latent_step(A, B, dt))
    y = torch.randn(64, burn + 6, 1, dtype=torch.float64)
    z0 = torch.randn(64, 5, dtype=torch.float64)
    z0 = z0 / z0.norm(dim=1, keepdim=True) * np.linalg.norm(B, 2) * yb / 1.5
    z0[0] = 0.0
    z0[0, 0] = np.linalg.norm(B, 2) * yb / 1.5      # slowest mode (block 0): worst case
    d = (Dm.run_latent(des, y, z0) - Dm.run_latent(des, y))[:, burn].norm(dim=1)
    assert d.max() <= eps * zs * (1 + 1e-9), (float(d.max()), eps * zs)
    assert d[0] >= eps * zs * math.exp(-1.5 * dt) * (1 - 1e-9)


def test_sylvester_reference():
    A, B = D.butterworth_AB(5, 1.0)
    rng = np.random.RandomState(0)
    for _ in range(5):
        F = rng.randn(3, 3)
        C = rng.randn(1, 3)
        P = D.sylvester_reference(F, C, A, B)
        assert np.abs(P @ F - A @ P - B @ C).max() < 1e-10 * max(1, np.abs(P).max())
    with pytest.raises(ValueError):
        D.sylvester_reference(A[:2, :2], np.ones((1, 2)), A, B)
    # linear plant: z - P x decays like e^{At}(z0 - P x0)  (the exact KKL map)
    F = np.array([[0.0, 1.0], [-2.0, -0.3]])
    C = np.array([[1.0, 0.0]])
    P = D.sylvester_reference(F, C, A, B)
    M = np.block([[F, np.zeros((2, 5))], [B @ C, A]])
    w0 = np.concatenate([[0.7, -0.4], np.zeros(5)])
    w = expm(M * 4.0) @ w0
    e = w[2:] - P @ w[:2]
    assert np.allclose(e, expm(A * 4.0) @ (-P @ w0[:2]), atol=1e-12)


def test_validate_refuses_bad_designs():
    A, B = D.butterworth_AB(5, 1.5, 1)
    dt = D.choose_dt(A)
    assert D.validate(A, B, dt, 1.5)["ok"]
    assert not D.validate(A, B, 1.5 * dt, 1.5)["checks"]["resolution"]
    assert not D.validate(A, B, dt, 1.0)["checks"]["alpha_matches"]
    J = np.diag([-1.5, -2, -2.5, -3, -3.5]) + np.diag([3.0] * 4, 1)          # non-normal
    r = D.validate(J, B, D.choose_dt(J), 1.5)
    assert not r["checks"]["normal_kappa_1"] and r["kappa"] > 1.0
    B0 = B.copy(); B0[4] = 0.0                                               # real pole unexcited
    assert not D.validate(A, B0, dt, 1.5)["checks"]["controllable"]
    for name in system_names():
        des = Dm.make_design(load(name)["system"])
        assert D.validate(des.A, des.B, des.dt, des.alpha)["ok"], name


# ===========================================================================
# data.py: integrator, latent filter, sensor / input models
# ===========================================================================

def _des(A, B, dt, T, alpha):
    return Dm.Design(A, B, dt, int(round(T / dt)), alpha, D.latent_step(A, B, dt))


def _richardson(plant, A, B, alpha, sig, x0, T, dts):
    finals = []
    for dt in dts:
        x, z, _ = Dm.integrate(plant, _des(A, B, dt, T, alpha), x0, [sig], device="cpu")
        finals.append(torch.cat([x[:, -1], z[:, -1]], 1).numpy())
    e1 = np.abs(finals[0] - finals[1]).max()
    e2 = np.abs(finals[1] - finals[2]).max()
    return np.log2(e1 / e2), e1, e2


def test_joint_rk4_fourth_order_on_vdp():
    scfg = load("vdp")["system"]
    plant = build_plant(scfg)                      # nonlinear, with saturating actuator
    A, B = D.butterworth_AB(5, 1.5, 1)
    T, dts = 10.0, (0.04, 0.02, 0.01)
    x0 = np.array([[1.3, -0.8], [-2.0, 2.5]])
    rng = np.random.RandomState(11)
    sin = S.sample_signal("sinusoid", "train", rng, dts[0], T, stratum=3, scale=1.0)
    sq = S.Signal("square", {}, *S._pwc([0.0, 1.0, 2.36, 4.0, 5.72, 8.0],
                                         [0.9, -0.7, 0.9, -0.7, 0.9, -0.7], dts[0]), 1.0)
    assert np.all(np.isin(sq.switch_times, np.arange(int(T / dts[0]) + 1) * dts[0]))
    # The plant input is a zero-order hold on the INTEGRATION grid, so refining dt also
    # refines the input: a smooth family converges at first order to the continuous-input
    # solution (documented semantics). Holding the same sampled input fixed -- its ZOH on
    # the coarsest grid, a piecewise-constant signal with switches on every finer grid --
    # isolates the integrator, which must be fourth order.
    tg = np.arange(int(round(T / dts[0]))) * dts[0]
    sin_zoh = S.Signal("sinusoid_zoh", {}, *S._pwc(tg, sin(tg) / sin.scale, dts[0]), sin.scale)
    res = {}
    for name, sig in (("sinusoid_zoh", sin_zoh), ("square", sq), ("sinusoid_refined_input", sin)):
        res[name] = _richardson(plant, A, B, 1.5, sig, x0, T, dts)
    # negative control: switches OFF the grid -> RK4 degrades to first order
    off = S.Signal("square", {}, np.array([0.0, 1.013, 2.371, 4.007, 5.729]),
                   np.array([0.9, -0.7, 0.9, -0.7, 0.9]), 1.0)
    res["square_off_grid"] = _richardson(plant, A, B, 1.5, off, x0, T, dts)
    _note("vdp RK4 observed order (order, e(dt,dt/2), e(dt/2,dt/4))",
          {k: (round(v[0], 2), float(f"{v[1]:.2e}"), float(f"{v[2]:.2e}")) for k, v in res.items()})
    assert res["sinusoid_zoh"][0] > 3.7 and res["square"][0] > 3.7, res
    assert 0.8 < res["sinusoid_refined_input"][0] < 1.3, res      # ZOH sampling error, O(dt)
    assert res["square_off_grid"][0] < 2.0, res


def _rk4_continuous(plant, sig, x0, T, dt):
    """Classical RK4 with the CONTINUOUS input at its true stage times (S.stage_inputs)."""
    N = int(round(T / dt))
    U = torch.as_tensor(S.stage_inputs([sig], np.arange(N + 1) * dt)[:, :, 0])
    x = torch.as_tensor(x0, dtype=torch.float64)
    f = lambda xx, uu: plant.f(xx, uu.expand(len(xx), 1))
    for k in range(N):
        u1, u2, u3 = U[k, 0:1], U[k, 1:2], U[k, 2:3]
        k1 = f(x, u1); k2 = f(x + dt / 2 * k1, u2); k3 = f(x + dt / 2 * k2, u2); k4 = f(x + dt * k3, u3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x.numpy()


def test_rk4_fourth_order_through_drive_kinks():
    """'drive' is piecewise linear with kinks at the ends of every press / release ramp.
    Knots are snapped to the grid, so (i) the deployed ZOH integrator on a fixed sampled
    drive input is 4th order (as for every other family), and (ii) even with the
    continuous input at the true RK4 stage times no step straddles a kink: 4th order.
    Negative control: the same kinks moved OFF the grid degrade (ii) to 3rd order (a
    straddled kink -- continuous input, jump in its slope -- costs O(h^3) per kink;
    measured 3.04 vs 4.04)."""
    scfg = load("vdp")["system"]
    plant = build_plant(scfg)
    A, B = D.butterworth_AB(5, 1.5, 1)
    T, dts = 10.0, (0.04, 0.02, 0.01)
    x0 = np.array([[1.3, -0.8], [-2.0, 2.5]])
    sig = S.sample_signal("drive", "sequence_ood", np.random.RandomState(13), dts[0], T, stratum=2)
    n_kinks = int(((sig.knots > 0) & (sig.knots < T)).sum())
    assert n_kinks >= 8 and np.all(np.isin(sig.knots[sig.knots <= T], np.arange(int(T / dts[0]) + 1) * dts[0]))
    tg = np.arange(int(round(T / dts[0]))) * dts[0]
    zoh = S.Signal("drive_zoh", {}, *S._pwc(tg, sig(tg), dts[0]), 1.0)
    res = {"zoh_integrator": _richardson(plant, A, B, 1.5, zoh, x0, T, dts)}

    def order(sg):
        f = [_rk4_continuous(plant, sg, x0, T, dt) for dt in dts]
        e1, e2 = np.abs(f[0] - f[1]).max(), np.abs(f[1] - f[2]).max()
        return np.log2(e1 / e2), e1, e2
    res["continuous_on_grid"] = order(sig)
    off = S.Signal("drive_off", {}, scale=1.0, knots=sig.knots + np.r_[0.0, np.full(len(sig.knots) - 1, 0.013)],
                   knot_vals=sig.knot_vals)
    res["continuous_off_grid"] = order(off)
    _note(f"vdp RK4 order through {n_kinks} drive kinks (order, e(dt,dt/2), e(dt/2,dt/4))",
          {k: (round(v[0], 2), float(f"{v[1]:.2e}"), float(f"{v[2]:.2e}")) for k, v in res.items()})
    assert res["zoh_integrator"][0] > 3.7 and res["continuous_on_grid"][0] > 3.7, res
    assert res["continuous_off_grid"][0] < 3.5, res


def test_z_is_the_latent_filter_of_h_x():
    """z from the joint RK4 vs the exact discrete filter on y_clean = h(x): the only
    difference is the filter's linear interpolation of y, O(dt^2)."""
    scfg = load("vdp")["system"]
    plant = build_plant(scfg)
    A, B = D.butterworth_AB(5, 1.5, 1)
    dt0 = D.choose_dt(A)
    T = 30.0
    rng = np.random.RandomState(12)
    sig = S.sample_signal("multisine", "train", rng, dt0, T, stratum=2)
    x0 = np.array([[1.0, 0.5], [-1.5, 2.0], [0.3, -2.4]])
    errs = []
    for dt in (dt0, dt0 / 2):
        des = _des(A, B, dt, T, 1.5)
        x, z, _ = Dm.integrate(plant, des, x0, [sig] * 3, device="cpu")
        zo = Dm.run_latent(des, plant.h(x.reshape(-1, 2)).reshape(3, -1, 1))
        b = des.N // 3
        errs.append(float((zo[:, b:] - z[:, b:]).norm() / z[:, b:].norm()))
    _note("relative |z_obs(y_clean) - z| at dt, dt/2, ratio", (errs[0], errs[1], errs[0] / errs[1]))
    assert errs[0] < 1e-4 and 3.5 < errs[0] / errs[1] < 4.5, errs


class _Still(Plant):
    nx = 2
    PARAMS = {}

    def f(self, x, u):
        return 0 * x


def test_process_noise_increments():
    A, B = D.butterworth_AB(3, 1.0, 1)
    dt = 0.05
    des = _des(A, B, dt, 50 * dt, 1.0)
    sp = np.array([0.3, 0.05])
    x0 = np.zeros((4000, 2))
    x, _, _ = Dm.integrate(_Still(), des, x0, [S.sample_signal("zero", "train", None, dt, 1.0)] * 4000,
                           sigma_p=sp, seed=3, device=DEV)
    d = torch.diff(x, dim=1).reshape(-1, 2).cpu().numpy()
    ratio = d.var(0) / (sp ** 2 * dt)
    dd = torch.diff(x, dim=1).cpu().numpy()
    lag1 = float(np.mean(dd[:, 1:, 0] * dd[:, :-1, 0]) / np.mean(dd[:, :, 0] ** 2))
    _note("process noise: var(increment) / (sigma_p^2 dt), lag-1 corr", (ratio.round(4).tolist(), round(lag1, 4)))
    assert np.all(np.abs(ratio - 1) < 0.02) and abs(lag1) < 0.01
    assert np.abs(d.mean(0)).max() < 4 * sp.max() * math.sqrt(dt / len(d))


def _measure_case(sensor, E=256, N1=2001, seed=5):
    t = torch.arange(N1, dtype=torch.float64) * 0.0173
    yc = torch.stack([torch.sin(1.3 * t + 0.1 * e) for e in range(E)])[..., None]
    yc = torch.cat([yc, 3 * torch.cos(0.7 * yc)], -1)                 # two outputs
    ys = [0.5, 2.0]
    return yc, Dm.measure(yc, sensor, ys, seed), torch.tensor(ys, dtype=torch.float64)


def test_measure_statistics():
    out = {}
    yc, y, ys = _measure_case({"noise": 0.1})
    r = (y - yc) / ys
    out["white std/0.1"] = (r.std((0, 1)) / 0.1).tolist()
    assert torch.allclose(r.std((0, 1)), torch.full((2,), 0.1, dtype=torch.float64), rtol=0.01)
    assert torch.equal(y, Dm.measure(yc, {"noise": 0.1}, [0.5, 2.0], 5))
    assert not torch.equal(y, Dm.measure(yc, {"noise": 0.1}, [0.5, 2.0], 6))
    # colored AR(1)
    yc, y, ys = _measure_case({"noise": 0.1, "kind": "colored", "rho": 0.95})
    r = (y - yc) / ys / 0.1
    lag1 = (r[:, 1:] * r[:, :-1]).mean((0, 1)) / (r * r).mean((0, 1))
    out["colored marginal std, lag1"] = (r.std((0, 1)).tolist(), lag1.tolist())
    assert torch.allclose(r.std((0, 1)), torch.ones(2, dtype=torch.float64), rtol=0.03)
    assert torch.allclose(lag1, torch.full((2,), 0.95, dtype=torch.float64), atol=0.005)
    # outliers
    yc, y, ys = _measure_case({"outlier_rate": 0.02, "outlier_scale": 12.0})
    r = (y - yc) / ys
    m = r.abs() > 1e-9
    out["outlier rate, |size|/ys, frac +"] = (float(m.double().mean()), float(r[m].abs().mean()),
                                               float((r[m] > 0).double().mean()))
    assert abs(m.double().mean() - 0.02) < 0.002
    assert torch.allclose(r[m].abs(), torch.full_like(r[m], 12.0))
    assert abs((r[m] > 0).double().mean() - 0.5) < 0.03
    # quantization
    yc, y, ys = _measure_case({"quant": 0.25})
    q = 0.25 * ys
    k = y / q
    out["quant max |k-round(k)|, max |err|/q"] = (float((k - k.round()).abs().max()),
                                                  float(((y - yc) / q).abs().max()))
    assert (k - k.round()).abs().max() < 1e-9 and ((y - yc) / q).abs().max() <= 0.5 + 1e-9
    # drift: random walk normalised so that its max |.| is exactly drift * y_std
    yc, y, ys = _measure_case({"drift": 0.3})
    r = (y - yc) / ys
    amp = r.abs().amax(1)
    out["drift max|.|/(0.3 y_std) range"] = (float(amp.min() / 0.3), float(amp.max() / 0.3))
    assert torch.allclose(amp, torch.full_like(amp, 0.3))
    inc = torch.diff(r, dim=1)
    assert abs(float((inc[:, 1:] * inc[:, :-1]).mean() / (inc * inc).mean())) < 0.01   # RW increments
    # dropout = sample-and-hold of the previous delivered value, shared by all outputs
    yc, y, ys = _measure_case({"dropout": 0.1})
    held = (y[:, 1:] == y[:, :-1])
    out["dropout hold rate"] = float(held[..., 0].double().mean())
    assert torch.equal(held[..., 0], held[..., 1])
    assert abs(held[..., 0].double().mean() - 0.1) < 0.005
    assert torch.equal(y[:, 0], yc[:, 0])
    assert torch.equal(Dm.measure(yc, {}, [0.5, 2.0], 0), yc)
    _note("measure()", out)


def test_run_latent_is_the_latent_step_recursion():
    A, B = D.butterworth_AB(6, 0.5, 2)
    dt = D.choose_dt(A)
    des = _des(A, B, dt, 40 * dt, 0.5)
    rng = np.random.RandomState(0)
    y = rng.randn(3, 41, 2)
    z0 = rng.randn(3, 6)
    st = des.step
    z = z0.copy()
    ref = [z]
    for k in range(40):
        z = z @ st["Ad"].T + y[:, k] @ st["W0"].T + y[:, k + 1] @ st["W1"].T
        ref.append(z)
    ref = np.stack(ref, 1)
    got = Dm.run_latent(des, torch.as_tensor(y), torch.as_tensor(z0)).numpy()
    assert np.abs(got - ref).max() < 1e-12
    got0 = Dm.run_latent(des, torch.as_tensor(y)).numpy()
    assert np.all(got0[:, 0] == 0)
    # and it is exact against Van Loan for y linear on every step
    vl = _van_loan(A, B, dt)
    z = z0.copy()
    for k in range(40):
        z = vl["Ad"] @ z.T
        z = z.T + y[:, k] @ vl["W0"].T + y[:, k + 1] @ vl["W1"].T
    assert np.abs(got[:, -1] - z).max() < 1e-12


def test_corrupt_input():
    u = torch.arange(20, dtype=torch.float64).reshape(1, 20, 1).repeat(2, 1, 1) + 1.0
    d = Dm.corrupt_input(u, {"delay": 3}, 1.0, 0)
    assert d.shape == u.shape
    assert torch.all(d[:, :3] == u[:, :1]) and torch.equal(d[:, 3:], u[:, :-3])
    b = Dm.corrupt_input(u, {"bias": 0.1}, 2.0, 0)
    assert torch.allclose(b - u, torch.full_like(u, 0.2))
    b = Dm.corrupt_input(u, {"bias": -0.1}, 2.0, 0)
    assert torch.allclose(b - u, torch.full_like(u, -0.2))
    db = Dm.corrupt_input(u, {"delay": 2, "bias": 0.5}, 1.0, 0)
    assert torch.allclose(db[:, 2:], u[:, :-2] + 0.5) and torch.allclose(db[:, :2], u[:, :1] + 0.5)
    U = torch.zeros(64, 5000, 1, dtype=torch.float64)
    n = Dm.corrupt_input(U, {"noise": 0.1}, 0.15, 4)
    assert abs(float(n.std()) / (0.1 * 0.15) - 1) < 0.01
    assert torch.equal(n, Dm.corrupt_input(U, {"noise": 0.1}, 0.15, 4))
    assert torch.equal(Dm.corrupt_input(u, {}, 1.0, 0), u)


# ===========================================================================
# data.py: generate, system_scales, training_data, eval_data
# ===========================================================================

@pytest.fixture(scope="module")
def vdp_scales():
    cfg = load("vdp", [SMALL])
    return cfg, Dm.system_scales(cfg, DEV)


def test_generate_crossed_design_and_bookkeeping(vdp_scales):
    cfg, sc = vdp_scales
    scfg, dcfg = cfg["system"], copy.deepcopy(cfg["data"])
    dcfg["reject_scale"] = 100.0                      # no rejections in this test
    des = Dm.make_design(scfg)
    plant = build_plant(scfg)
    n, k = 20, int(dcfg["k_cross"])
    fams = S.TRAIN_FAMILIES
    es = Dm.generate(plant, des, sc, scfg, dcfg, fams, n, "train", 77, DEV)
    n_ic = math.ceil(n / k)
    assert len(es) == n and es.n_rejected == 0
    assert es.burn == math.ceil(sc["t_burn"] / des.dt)
    assert es.families == [fams[(i // k) % len(fams)] for i in range(n)]
    assert [m["family"] for m in es.meta] == es.families
    ics = np.array([m["ic"] for m in es.meta])
    assert np.allclose(es.x[:, 0].numpy(), ics, atol=1e-6)
    hu = [_h(es.u[i]) for i in range(n)]
    hic = [tuple(np.round(ics[i], 12)) for i in range(n)]
    for s in range(n_ic):                           # each signal: k distinct ICs
        grp = [i for i in range(n) if i // k == s]
        assert len({hu[i] for i in grp}) == 1
        assert len({hic[i] for i in grp}) == len(grp)
    for c in range(n_ic):                           # each IC: ~k distinct signals
        grp = [i for i in range(n) if i % n_ic == c]
        assert len({hic[i] for i in grp}) == 1
        assert len({i // k for i in grp}) == len(grp) and len(grp) in (k - 1, k)
    nz_sig = [i for i in range(n) if es.families[i] != "zero"]
    assert len({hu[i] for i in nz_sig}) == len({i // k for i in nz_sig})
    assert torch.all(es.u[[i for i in range(n) if es.families[i] == "zero"]] == 0)
    # y / z_obs bookkeeping
    y = Dm.measure(es.y_clean.double().to(DEV), dcfg["sensor"], sc["y_std"], 77).cpu()
    assert torch.allclose(y.float(), es.y, atol=1e-5)
    if DEV != "cpu":                                # noise streams depend on the device TYPE
        yc = Dm.measure(es.y_clean.double(), dcfg["sensor"], sc["y_std"], 77)
        _note("measure(): same seed, cpu vs cuda noise identical?", bool(torch.allclose(yc, y)))
    assert torch.allclose(Dm.run_latent(des, es.y.double()).float(), es.z_obs, atol=1e-4)
    rel = float((es.z_obs[:, es.burn:] - es.z[:, es.burn:]).norm() / es.z[:, es.burn:].norm())
    _note("generate: rel |z_obs - z| after burn (sensor noise 3%)", rel)
    assert rel < 0.05
    assert torch.allclose(es.y_clean, es.x[..., scfg["sensors"]])
    # process noise uses sigma_p = noise_proc * x_std; process_noise=False switches it off
    seen = []
    orig = Dm.integrate

    def spy(*a, **kw):
        seen.append(a[4] if len(a) > 4 else kw.get("sigma_p"))
        return orig(*a, **kw)
    Dm.integrate = spy
    try:
        Dm.generate(plant, des, sc, scfg, dcfg, ["zero"], 3, "train", 1, DEV)
        Dm.generate(plant, des, sc, scfg, dcfg, ["zero"], 3, "train", 1, DEV, process_noise=False)
    finally:
        Dm.integrate = orig
    assert np.allclose(seen[0], float(dcfg["noise_proc"]) * np.asarray(sc["x_std"])) and seen[1] is None
    from collections import Counter
    per_fam = lambda nn, fl: dict(Counter(fl[r % len(fl)] for r in range(math.ceil(nn / k))))
    _note("crossed design: distinct signals per family at default sizes",
          {"train(600)": per_fam(600, S.TRAIN_FAMILIES), "val(64)": per_fam(64, S.TRAIN_FAMILIES),
           "eval id(64)": per_fam(64, S.families_for("id")),
           "eval shape_ood(64)": per_fam(64, S.families_for("shape_ood")),
           "eval sequence_ood(64)": per_fam(64, S.families_for("sequence_ood"))})


def test_generate_redraws_rejected_episodes(vdp_scales):
    cfg, sc = vdp_scales
    scfg = cfg["system"]
    des = Dm.make_design(scfg)
    plant = build_plant(scfg)
    n = 24
    d_ok = dict(cfg["data"], reject_scale=100.0)
    d_tight = dict(cfg["data"], reject_scale=0.8)
    a = Dm.generate(plant, des, sc, scfg, d_ok, S.TRAIN_FAMILIES, n, "train", 5, DEV)
    b = Dm.generate(plant, des, sc, scfg, d_tight, S.TRAIN_FAMILIES, n, "train", 5, DEV)
    dom = np.asarray(sc["domain"])
    ctr, half = dom.mean(1), (dom[:, 1] - dom[:, 0]) / 2
    lo, hi = ctr - 0.8 * half, ctr + 0.8 * half
    _note("redraw: n_rejected, kept of 24", (b.n_rejected, len(b)))
    assert b.n_rejected > 0
    box = np.asarray(scfg["ic_box"])
    ia = {i: m for i, m in enumerate(a.meta)}
    ub = {_h(b.u[j]): j for j in range(len(b))}
    xb = b.x.numpy()
    assert np.all(xb >= lo - 1e-5) and np.all(xb <= hi + 1e-5)       # kept ones are inside
    # the kept set is a subset of the original pairing: same signals, ICs redrawn in the box
    fam_a = a.families
    n_redrawn = 0
    for j, m in enumerate(b.meta):
        ic = np.array(m["ic"])
        assert np.all(ic >= box[:, 0]) and np.all(ic <= box[:, 1])
        cand = [i for i in range(n) if _h(a.u[i]) == _h(b.u[j]) and fam_a[i] == b.families[j]]
        assert cand, "signal not from the original draw"
        if not any(np.allclose(ic, ia[i]["ic"]) for i in cand):
            n_redrawn += 1
    assert b.n_rejected >= n_redrawn + (n - len(b))
    assert len(b) + (n - len(b)) == n


@pytest.mark.parametrize("name", ["vdp", "bioreactor", "quad_tank"])
def test_system_scales_sanity(name):
    cfg = load(name)
    scfg, dcfg = cfg["system"], cfg["data"]
    sc = Dm.system_scales(cfg, DEV)
    des = Dm.make_design(scfg)
    box = np.asarray(scfg["ic_box"], float)
    dom = np.asarray(sc["domain"])
    assert np.all(dom[:, 0] <= box[:, 0]) and np.all(dom[:, 1] >= box[:, 1])
    assert sc["dt"] == des.dt and sc["N"] == des.N and sc["design_check"]["ok"]
    assert sc["n_probe_diverged"] == 0
    xs, ys = np.asarray(sc["x_std"]), np.asarray(sc["y_std"])
    assert np.all(xs > 1e-6) and np.allclose(ys, xs[scfg["sensors"]], rtol=1e-12)
    Bn = float(np.linalg.norm(des.B, 2))
    eps = float(dcfg["eps_label"])
    assert abs(sc["t_burn"] - D.washout_time(1.0, des.alpha, Bn, sc["y_bar"], sc["z_scale"], eps)) < 1e-6
    assert abs(sc["memory_s"] - D.memory_length(1.0, des.alpha, Bn, sc["y_bar"], sc["z_scale"], eps)) < 1e-6
    # replay the envelope probe over the three in-bounds tiers: domain, y_bar = max ||h(x)||
    # (A2), z_scale
    plant = build_plant(scfg)
    rng = np.random.RandomState(12345)
    xz = [Dm.integrate(plant, des, Dm._sobol(box, n, sd), Dm._signals(S.families_for(t), n, t, rng, des, scfg),
                       device=DEV)[:2] for t, n, sd in (("id", 96, 1), ("shape_ood", 48, 3), ("sequence_ood", 48, 4))]
    x, z = torch.cat([a for a, _ in xz]), torch.cat([b for _, b in xz])
    xf = x.reshape(-1, x.shape[-1])
    lo, hi = xf.min(0).values.cpu().numpy(), xf.max(0).values.cpu().numpy()
    c, h = (lo + hi) / 2, (hi - lo) / 2
    sf = float(dcfg["envelope_safety"])
    assert np.allclose(dom[:, 0], np.minimum(c - sf * h, box[:, 0]), rtol=1e-6, atol=1e-9)
    assert np.allclose(dom[:, 1], np.maximum(c + sf * h, box[:, 1]), rtol=1e-6, atol=1e-9)
    yb2 = float(plant.h(xf).norm(dim=-1).max())
    ybinf = float(plant.h(xf).abs().max())
    zs = float(z[:, z.shape[1] // 2:].pow(2).sum(-1).mean().sqrt())
    _note(f"{name} scales: y_bar stored / max||h||_2 / max|h|_inf, t_burn, memory_s, rejected",
          (sc["y_bar"], yb2, ybinf, sc["t_burn"], sc["memory_s"]))
    assert abs(zs - sc["z_scale"]) < 1e-6 * zs
    assert abs(sc["y_bar"] - yb2) < 1e-6 * yb2, "y_bar must bound the Euclidean norm of h (A2)"
    # replay the ID probe: x_std after burn
    xr, _, _ = Dm.integrate(plant, des, Dm._sobol(box, 64, 2),
                            Dm._signals(S.families_for("id"), 64, "id", rng, des, scfg), device=DEV)
    b = int(math.ceil(sc["t_burn"] / des.dt))
    xr = xr[:, b:].reshape(-1, xr.shape[-1])
    assert np.allclose(xr.std(0).cpu().numpy(), xs, rtol=1e-6)


@pytest.fixture(scope="module")
def built(vdp_scales, tmp_path_factory):
    """training_data for two seeds and eval_data for every tier, built from scratch in a
    temporary cache; eval rebuilt in a second fresh cache under other global RNG states."""
    cfg, sc = vdp_scales
    mp = pytest.MonkeyPatch()
    mp.setattr(Dm, "system_scales", lambda c, device="cpu": sc)
    tiers = list(S.TIERS)
    try:
        mp.setattr(Dm, "CACHE", tmp_path_factory.mktemp("cacheA"))
        torch.manual_seed(0); np.random.seed(0)
        c0 = dict(cfg, seed=0)
        tr0 = Dm.training_data(c0, 0, DEV)
        ev0 = {t: Dm.eval_data(c0, t, device=DEV) for t in tiers}
        mp.setattr(Dm, "CACHE", tmp_path_factory.mktemp("cacheB"))
        torch.manual_seed(999); np.random.seed(999)
        c1 = dict(cfg, seed=1)
        tr1 = Dm.training_data(c1, 1, DEV)
        ev1 = {t: Dm.eval_data(c1, t, device=DEV) for t in tiers}
    finally:
        mp.undo()
    return {"cfg": cfg, "sc": sc, "tr0": tr0, "tr1": tr1, "ev0": ev0, "ev1": ev1}


def test_eval_independent_of_training_seed(built):
    for t, a in built["ev0"].items():
        b = built["ev1"][t]
        for k in ("x", "u", "y_clean", "y", "z", "z_obs"):
            assert torch.equal(getattr(a, k), getattr(b, k)), (t, k)
        assert a.families == b.families and a.meta == b.meta and a.n_rejected == b.n_rejected
        assert set(a.families) <= set(S.families_for(t)) and len(a) <= 12
    _note("eval sets: (len, n_rejected)", {t: (len(e), e.n_rejected) for t, e in built["ev0"].items()})


def test_no_input_shared_between_train_val_eval(built):
    def hs(es):
        return {_h(es.u[i]) for i in range(len(es)) if es.families[i] != "zero"}
    sets = {"train0": hs(built["tr0"]["train"]), "val0": hs(built["tr0"]["val"]),
            "train1": hs(built["tr1"]["train"]), "val1": hs(built["tr1"]["val"])}
    sets.update({f"eval_{t}": hs(es) for t, es in built["ev0"].items()})
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not sets[a] & sets[b], (a, b)
    ics = {k: {tuple(np.round(m["ic"], 9)) for m in es.meta}
           for k, es in [("train0", built["tr0"]["train"]), ("val0", built["tr0"]["val"]),
                         ("p1", built["tr0"]["p1"])] + [(t, e) for t, e in built["ev0"].items()]}
    for i, a in enumerate(sorted(ics)):
        for b in sorted(ics)[i + 1:]:
            assert not ics[a] & ics[b], (a, b)
    _note("distinct non-zero input arrays per set", {k: len(v) for k, v in sets.items()})
    fams_tr = set(built["tr0"]["train"].families) | set(built["tr0"]["val"].families)
    assert not fams_tr & set(S.SHAPE_OOD_FAMILIES)


def test_seed_streams_never_collide():
    train = {100_000 * (s + 1) + j for s in range(100) for j in range(1, 5)}
    ev = [Dm.split_seed(n, t, sp, r) for n in system_names() for t in S.TIERS
          for sp in ("dev", "test") for r in range(5)]
    assert len(set(ev)) == len(ev) and not set(ev) & train and 12345 not in ev


def test_training_data_contents(built):
    cfg, sc = built["cfg"], built["sc"]
    d = built["tr0"]
    p1, tr, va, coll = d["p1"], d["train"], d["val"], d["coll"]
    assert (len(p1), len(tr), len(va)) == (16, 30, 12)
    assert torch.all(p1.u == 0) and set(p1.families) == {"zero"}
    dom = np.asarray(sc["domain"])
    ic = np.array([m["ic"] for m in p1.meta])
    assert np.all(ic >= dom[:, 0]) and np.all(ic <= dom[:, 1])
    n_distinct = len({tuple(np.round(r, 12)) for r in ic})
    _note("p1: distinct initial conditions of 16 zero-input episodes", n_distinct)
    assert n_distinct == len(p1), "zero-input episodes duplicate their ICs (crossing is moot)"
    box = np.asarray(cfg["system"]["ic_box"])
    for es in (tr, va):
        ic = np.array([m["ic"] for m in es.meta])
        assert np.all(ic >= box[:, 0]) and np.all(ic <= box[:, 1])
    xc = coll["x"].double()
    assert xc.shape == (256, 2)
    assert np.all(xc.numpy() >= dom[:, 0] - 1e-6) and np.all(xc.numpy() <= dom[:, 1] + 1e-6)
    nominal = build_plant(cfg["system"])
    assert torch.allclose(coll["f0"].double(), nominal.f(xc, torch.zeros(256, 1, dtype=xc.dtype)), atol=1e-4)
    assert torch.allclose(coll["y"].double(), nominal.h(xc), atol=1e-6)
    assert not torch.equal(built["tr0"]["train"].u, built["tr1"]["train"].u)


def test_with_sensor_changes_only_y_and_z_obs(built):
    es = built["ev0"]["shape_ood"]
    des = Dm.make_design(built["cfg"]["system"])
    sc = built["sc"]
    w = es.with_sensor(des, {"noise": 0.1, "outlier_rate": 0.02}, sc["y_std"], 3)
    for k in ("x", "u", "y_clean", "z"):
        assert torch.equal(getattr(w, k), getattr(es, k)), k
    assert not torch.equal(w.y, es.y) and not torch.equal(w.z_obs, es.z_obs)
    assert w.families == es.families and w.burn == es.burn and w.meta == es.meta
    assert torch.allclose(w.z_obs.double(), Dm.run_latent(des, w.y.double()), atol=1e-5)
    c = es.with_sensor(des, {}, sc["y_std"], 3)
    assert torch.equal(c.y, es.y_clean)
    r = ((w.y.double() - w.y_clean.double()) / torch.tensor(sc["y_std"], dtype=torch.float64))
    r = r[r.abs() < 3]
    assert abs(float(r.std()) - 0.1) < 0.005
