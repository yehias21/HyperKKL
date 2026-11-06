"""v2 operating profiles = experiments/input_review/profiles.py (copied unchanged as profiles.py) + DESIGN.md section 3:

* two_link_arm_v2: torque box +-12 N m (rate 120 N m/s, lag 5 ms); the unit-box operators are scaled by 12.
* highway_v2: the review's widened box (u in [-0.25, 0.2], i.e. 180-1800 veh/h) is the v2 plant's box.
* op_astat: no cited acceleration; ramp between two box levels over 10-40 h (benchmark choice, uncited).
* lorenz96: the legacy anomaly box +-4 is kept.
"""
from __future__ import annotations

import math

import numpy as np

from . import profiles as P0
from .profiles import Actuator, Imperfection, apply_actuator, observed_input, SampledSignal, Profile  # noqa: F401

SPECS = dict(P0.SPECS)
ARM_TAU = 12.0


def _scaled(op, g):
    def f(rng, t, P):
        c, ev = op(rng, t, P)
        return g * np.asarray(c, float), ev
    f.__name__ = op.__name__
    return f


def op_astat(rng, t, P):
    """A-stat: chemostat phase, then D ramps linearly to a higher box level over 10-40 h (optionally back down).
    Ramp duration is a benchmark choice (uncited)."""
    T, lo, hi = t[-1], P["lo"], P["hi"]
    t0 = float(rng.uniform(0.1, 0.3)) * T
    d0 = float(rng.uniform(lo, 0.5 * (lo + hi)))
    d1 = float(rng.uniform(0.5 * (lo + hi), hi))
    dur = float(rng.uniform(10.0, 40.0))
    up = d0 + (d1 - d0) * np.clip((t - t0) / dur, 0, 1)
    if rng.rand() < 0.5:
        tp = t0 + dur + float(rng.uniform(5, 20))
        up = np.where(t < tp, up, d1 - (d1 - d0) * np.clip((t - tp) / dur, 0, 1))
    return np.clip(up, lo, hi), [t0]


bio = dict(SPECS["bioreactor"])
bio["ood"] = [op_astat if o.__name__ == "op_astat" else o for o in bio["ood"]]
SPECS["bioreactor"] = bio

arm = dict(SPECS["two_link_arm_b"])
arm.update(lo=-ARM_TAU, hi=ARM_TAU, act=Actuator(-ARM_TAU, ARM_TAU, rate=10.0 * ARM_TAU, tau=0.005),
           train=[_scaled(o, ARM_TAU) for o in arm["train"]], ood=[_scaled(o, ARM_TAU) for o in arm["ood"]],
           imperf=P0.DAQ(2 * ARM_TAU), unit="shoulder torque [N m]")
SPECS["two_link_arm_v2"] = arm
SPECS["highway_v2"] = dict(SPECS["highway_traffic"])

UNCITED = {"vdp": "stepped / swept sine and band-random shaker tests", "duffing": "same as vdp",
           "cartpole_b": "minimum-jerk repositioning moves", "two_link_arm_v2": "minimum-jerk pick-and-place bumps",
           "bioreactor": "A-stat ramp duration"}


def operators(plant, split):
    return SPECS[plant]["train" if split in ("train", "id", "val") else "ood"]


def generate(plant, split, seed, T=None, dt=None, operator=None) -> Profile:
    Pp = dict(SPECS[plant])
    T = float(Pp["T"] if T is None else T)
    dt = float(Pp["dt"] if dt is None else dt)
    rng = np.random.RandomState(seed)
    ops = operators(plant, split)
    op = ops[rng.randint(len(ops))] if operator is None else next(o for o in ops if o.__name__ == operator)
    t = np.arange(int(math.ceil(T / dt)) + 1) * dt
    cmd, ev = op(rng, t, Pp)
    cmd = np.asarray(cmd, float)
    u = apply_actuator(cmd, dt, Pp["act"])
    return Profile(plant, split, op.__name__, t, cmd, u, list(map(float, ev)), seed)
