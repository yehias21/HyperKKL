"""Cell configurations of bench v2.

    cfg = v2cfg.config("driven", "battery", "realistic")      # research-style config dict (system/data/phase1/...)

Base configs are the stored run configs of the v1 export (experiments/tree/<node>/runs/<system>/seed0/results.json),
so the v1 plants keep their v1 design (A, B, dt, N), sensors, envelope and noise. New plants:
  lorenz96, lorenz96_n10   l96_default load_cfg (chaos_deep forced Lorenz-96)
  two_link_arm_v2          TEST-C3-B two_link_arm_b + torque box +-12 N m (u_scale 12, u_bar 2 x 12, back-EMF k 0.1)
  highway_v2               TEST-C3 highway_traffic + box u in [-0.25, 0.2] (u_scale 0.25, u_bar 2 x 0.25)
Variants: white (exact / partial levels), realistic (30 % mismatch + AR(1) Student-t(3) noise), mismatch30.
"""
from __future__ import annotations

import copy
import json

import env  # noqa: F401
from env import LIVE

FAMILY_A = ["vdp", "fhn", "rossler", "lorenz", "lorenz96", "network", "lorenz96_n10"]
FAMILY_B = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp", "cartpole_b", "fhn", "highway_v2",
            "network", "rossler", "two_link_arm_v2", "lorenz96"]
A_NOMINAL_ONLY = {"lorenz", "rossler"}   # lorenz collapses at +0.3; rossler escapes from the IC shell (DESIGN.md)
DEV = {"battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp", "lorenz96", "lorenz96_n10"}
SYSTEM = {"highway_v2": "highway_traffic", "two_link_arm_v2": "two_link_arm_b"}
WHITE = {"noise": 0.03}
REAL = {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}


def _tree_cfg(system):
    if system in ("cartpole_b", "two_link_arm_b"):
        node = "TEST-C3-B"
    elif system in DEV:
        node = "C3-realistic"
    else:
        node = "TEST-C3"
    f = LIVE / "experiments" / "tree" / node / "runs" / system / "seed0" / "results.json"
    return json.load(open(f))["config"], str(f)


def config(family: str, plant: str, variant: str = "realistic", seed: int = 0) -> dict:
    assert family in ("autonomous", "driven"), family
    system = SYSTEM.get(plant, plant)
    if system.startswith("lorenz96"):
        from hyperkkl.config import load
        cfg = load(system)
        src = f"configs/systems/{system}.yaml (l96_default)"
    else:
        cfg, src = _tree_cfg(system)
        cfg = copy.deepcopy(cfg)
    s = cfg["system"]
    if plant == "two_link_arm_v2":
        s.update(name="two_link_arm_v2", u_scale=12.0, actuator={"kind": "saturating", "u_bar": 2.0, "j": 2, "k": 0.1})
    if plant == "highway_v2":
        s.update(name="highway_v2", u_scale=0.25, u_bounds=[-0.25, 0.2], actuator={"kind": "saturating", "u_bar": 2.0})
    s["knowledge"] = "partial"
    if variant == "white":
        s["mismatch"] = 0.0
        cfg["data"]["sensor"] = dict(WHITE)
    elif variant == "realistic":
        s["mismatch"] = 0.3
        cfg["data"]["sensor"] = dict(REAL)
    elif variant == "mismatch30":
        s["mismatch"] = 0.3
        cfg["data"]["sensor"] = dict(WHITE)
    else:
        raise KeyError(variant)
    if family == "autonomous":
        assert plant in FAMILY_A, plant
        if plant in A_NOMINAL_ONLY and s["mismatch"] != 0.0:
            raise ValueError(f"{plant}: no autonomous cell at +0.3 (collapses to a fixed point)")
    cfg["seed"] = int(seed)
    cfg["eval"] = dict(cfg["eval"], conditions=["clean"], save_traj=False, memory_probe=False)
    cfg["baselines"] = []
    cfg["v2"] = {"family": family, "plant": plant, "variant": variant, "base_config": src}
    return cfg


def level_of(variant):
    return {"white": "partial", "realistic": "realistic", "mismatch30": "mismatch"}[variant]
