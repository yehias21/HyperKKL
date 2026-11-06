"""Unit tests for profiles.py (CPU, numpy only).  Run:  .venv/bin/python -m pytest experiments/input_review -q"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import profiles as Pf  # noqa: E402

CASES = [(p, sp, op.__name__) for p in Pf.SPECS for sp in ("train", "ood") for op in Pf.operators(p, sp)]
SEEDS = (0, 1, 2)


@pytest.mark.parametrize("plant,split,op", CASES)
def test_bounds_rate_finite(plant, split, op):
    P = Pf.SPECS[plant]
    a = P["act"]
    for s in SEEDS:
        pr = Pf.generate(plant, split, s, operator=op)
        u = pr.u
        assert np.isfinite(u).all() and np.isfinite(pr.cmd).all()
        assert len(u) == len(pr.t) == int(math.ceil(P["T"] / P["dt"])) + 1
        assert u.min() >= a.lo - 1e-12 and u.max() <= a.hi + 1e-12
        if math.isfinite(a.rate):
            assert np.abs(np.diff(u)).max() <= a.rate * P["dt"] + a.quant + 1e-9
        if a.quant > 0:
            k = (u - a.lo) / a.quant
            assert np.all(np.isclose(k, np.round(k), atol=1e-6) | np.isclose(u, a.hi) | np.isclose(u, a.lo))


@pytest.mark.parametrize("plant", list(Pf.SPECS))
def test_reproducible_and_seed_dependent(plant):
    for sp in ("train", "ood"):
        a, b, c = Pf.generate(plant, sp, 5), Pf.generate(plant, sp, 5), Pf.generate(plant, sp, 6)
        assert a.operator == b.operator and np.array_equal(a.u, b.u) and np.array_equal(a.cmd, b.cmd)
        assert not np.array_equal(a.u, c.u)


@pytest.mark.parametrize("plant", list(Pf.SPECS))
def test_train_and_ood_operators_disjoint_same_box(plant):
    tr = {o.__name__ for o in Pf.operators(plant, "train")}
    od = {o.__name__ for o in Pf.operators(plant, "ood")}
    assert tr and od and not tr & od
    # OOD is a different operator, never a larger amplitude: the OOD range stays inside the box TRAINING reaches
    # (to 10 % of the span) over 30 seeds
    P = Pf.SPECS[plant]
    span = P["hi"] - P["lo"]
    ut = np.concatenate([Pf.generate(plant, "train", s).u for s in range(30)])
    uo = np.concatenate([Pf.generate(plant, "ood", s).u for s in range(30)])
    assert uo.min() >= ut.min() - 0.1 * span and uo.max() <= ut.max() + 0.1 * span


MIN_DWELL = {  # operator -> minimum spacing between consecutive set-point events (native time unit)
    "op_chemostat_steps": 10.0, "op_shift_up_down": 10.0, "op_tank_small_moves": 60.0, "op_tank_recipe": 120.0,
    "op_metering_plan": 60.0, "op_cstr_shift_moves": 30.0, "op_cstr_production_plan": 60.0,
}


@pytest.mark.parametrize("op", sorted(MIN_DWELL))
def test_min_dwell(op):
    plant = next(p for p, sp, o in CASES if o == op)
    split = next(sp for p, sp, o in CASES if o == op)
    for s in range(10):
        pr = Pf.generate(plant, split, s, operator=op)
        ev = np.asarray([0.0] + pr.events)
        assert np.diff(ev).min() >= MIN_DWELL[op] - 1e-9


def test_actuator_rate_lag_deadband_quant():
    dt = 0.1
    a = Pf.Actuator(-1, 1, rate=0.5, tau=0.0)
    u = Pf.apply_actuator(np.r_[np.zeros(5), np.ones(50)], dt, a)
    assert np.allclose(np.diff(u)[4:14], 0.05) and u[-1] == 1.0          # ramp at the rate, then the target
    a = Pf.Actuator(-1, 1, tau=1.0)
    u = Pf.apply_actuator(np.r_[0.0, np.ones(200)], dt, a)
    assert abs(u[10] - (1 - math.exp(-1.0))) < 1e-9                     # exact first-order lag (ZOH)
    a = Pf.Actuator(-1, 1, deadband=0.1)
    u = Pf.apply_actuator(np.r_[0.0, 0.05, 0.08, 0.2, 0.25], dt, a)
    assert list(u) == [0.0, 0.0, 0.0, 0.2, 0.2]                         # stiction: moves only past the deadband
    a = Pf.Actuator(-1, 1, quant=0.25)
    u = Pf.apply_actuator(np.array([0.1, 0.13, 0.4, 2.0, -3.0]), dt, a)
    assert list(u) == [0.0, 0.25, 0.5, 1.0, -1.0]


def test_observed_input_imperfections():
    dt, n = 0.1, 200
    u = np.sin(np.arange(n) * dt)
    v = Pf.observed_input(u, dt, Pf.Imperfection(delay=0.5), span=2.0, seed=0)
    assert np.array_equal(v[5:], u[:-5]) and np.all(v[:5] == u[0])
    v = Pf.observed_input(u, dt, Pf.Imperfection(quant=0.1), span=2.0, seed=0)
    assert np.allclose(v / 0.1, np.round(v / 0.1))
    v = Pf.observed_input(u, dt, Pf.Imperfection(bias=0.05), span=2.0, seed=0)
    assert np.allclose(v - u, 0.1)
    i1 = Pf.Imperfection(noise=0.01, noise_tau=1.0)
    a, b = Pf.observed_input(u, dt, i1, 2.0, 3), Pf.observed_input(u, dt, i1, 2.0, 3)
    assert np.array_equal(a, b) and abs(np.std(a - u) - 0.02) < 0.015
    # aggregation is causal: the block mean of window j is known only from the start of window j + 1
    v = Pf.observed_input(u, dt, Pf.Imperfection(aggregate=1.0), 2.0, 0)
    assert np.isclose(v[15], u[0:10].mean()) and np.all(v[:10] == u[0])
    v = Pf.observed_input(u, dt, Pf.Imperfection(dropout=0.3), 2.0, 0)
    held = v[1:] == v[:-1]
    assert 0.15 < held.mean() < 0.45
    cmd = np.round(u)
    v = Pf.observed_input(u, dt, Pf.Imperfection(use_command=True), 2.0, 0, cmd=cmd)
    assert np.array_equal(v, cmd)


def test_sampled_signal_is_zero_order_hold():
    pr = Pf.generate("quad_tank", "train", 0)
    s = pr.signal()
    t = pr.t
    assert np.array_equal(s(t), pr.u)
    assert np.array_equal(s(t[:-1] + 0.5 * (t[1] - t[0])), pr.u[:-1])       # mid-step: the held value
    assert s.describe()["family"].startswith("profile:")


@pytest.mark.parametrize("plant,lo_phys", [("bioreactor", 0.0), ("quad_tank", 0.0), ("highway_traffic", 0.0),
                                           ("cstr_exo", 0.0)])
def test_physical_non_negative(plant, lo_phys):
    """Flows, voltages and dilution rates the profiles command are physically admissible (>= 0)."""
    for sp in ("train", "ood"):
        for s in range(5):
            assert Pf.physical(plant, Pf.generate(plant, sp, s).u).min() >= lo_phys


def test_battery_drive_cycle_realism():
    rng = np.random.RandomState(0)
    for style, vmax in (("urban", 17.0), ("highway_aggr", 36.0)):
        v = Pf.drive_speed(rng, 1800, style)
        assert v.min() >= 0 and v.max() <= vmax
        assert np.abs(np.diff(v)).max() <= 3.5 + 1e-9                    # |accel| <= 3.5 m/s^2 at 1 Hz
    urban = Pf.drive_speed(np.random.RandomState(1), 1800, "urban")
    assert (urban < 0.1).mean() > 0.1                                     # urban driving has idle time
