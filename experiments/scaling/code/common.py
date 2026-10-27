"""Shared set-up for every scaling-study script: path order, cache redirection, CPU data.

Import this FIRST. It puts the scratch package copy (code/pkg = the beat_greybox A3 package,
byte-identical, see code/PKG_SOURCE.txt) on sys.path, registers the fhnring{N} plants and the
bank design (scaleplants.py), redirects the data cache to experiments/scaling/cache, and makes
every data artefact (scales, training and evaluation episodes) generated on the CPU with one
thread (the shared GPUs are time-sliced; launch-bound integration is 10-30x faster on 1 CPU
thread here) so that one copy serves every GPU job.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent            # experiments/scaling/code
SC = HERE.parent                                  # experiments/scaling
LIVE = SC.parent.parent
sys.path.insert(0, str(HERE / "pkg"))
sys.path.insert(1, str(HERE))

import torch  # noqa: E402

torch.set_num_threads(1)

from hyperkkl import data as Dm  # noqa: E402

assert Path(Dm.__file__).resolve().is_relative_to(HERE / "pkg"), Dm.__file__
import scaleplants as SP  # noqa: E402,F401  (registers plants, patches make_design)

CACHE = SC / "cache_rev3"          # cache/, cache_rev2/: plant rev 1 / rev 2 artefacts, unused
Dm.CACHE = CACHE
assert Dm.CACHE.resolve() != (LIVE / "experiments" / "cache").resolve()

_orig_scales, _orig_eval, _orig_train = Dm.system_scales, Dm.eval_data, Dm.training_data


def system_scales(cfg, device="cpu"):
    return _orig_scales(cfg, "cpu")


def eval_data(cfg, tier, plant_shift=0.0, device="cpu", horizon=1.0, n_episodes=None):
    return _orig_eval(cfg, tier, plant_shift=plant_shift, device="cpu", horizon=horizon, n_episodes=n_episodes)


Dm.system_scales = system_scales
Dm.eval_data = eval_data


def training_data(cfg, seed):
    """Episodes generated on CPU and cached per (real plant, sensor, data sizes, seed): the
    modeller's knowledge only enters the collocation set (f_known), recomputed here, so the
    exact and partial levels share the very same episodes."""
    c = {**cfg, "system": {k: v for k, v in cfg["system"].items() if k != "knowledge"}}
    d = _orig_train(c, seed, "cpu")
    nominal = Dm.model_plant(cfg["system"])
    xt = d["coll"]["x"].double()
    coll = {"x": d["coll"]["x"], "f0": nominal.f(xt, torch.zeros(len(xt), 1, dtype=xt.dtype)).float(),
            "y": nominal.h(xt).float()}
    return {**d, "coll": coll}


def dubai_now() -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return f"{datetime.now(ZoneInfo('Asia/Dubai')):%Y-%m-%d %H:%M}"


def status(msg: str):
    with open(SC / "STATUS.md", "a") as f:
        f.write(f"- {dubai_now()} Dubai: {msg}\n")
