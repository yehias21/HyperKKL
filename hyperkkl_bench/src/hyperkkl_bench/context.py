"""What a method may know about a (plant, knowledge level): the harness builds this dict and passes
it to ``Estimator.fit``. It deliberately leaves out the REAL plant's parameters (metadata key
``mismatch30_params``) and the hidden-test seeds."""

from __future__ import annotations

import numpy as np

from .data import metadata
from .registry import KNOWLEDGE

ALLOWED_META = ("sensor_noise_std_physical", "sensor_hold_samples", "plant", "role", "n_x", "n_y", "n_u", "state_names", "sensors", "sensor_names",
                "unmeasured", "unmodelled_term", "nominal_params", "x_std", "y_std", "x_mean",
                "operating_domain", "dt", "N", "T", "t_ref", "alpha", "nz", "settle_frac",
                "settle_index", "input_box", "actuator", "ic_box", "process_noise_rel",
                "system_config")


def modeller_model(plant: str, knowledge: str, root=None):
    """The model a model-based method may use: NOMINAL parameters; f_known at the partial and
    realistic levels. torch-batched: model.f(x (B, n_x), u (B, 1)), model.h(x)."""
    from .plants import build_plant
    meta = metadata(plant, root)
    return build_plant(meta["system_config"], knowledge=KNOWLEDGE[knowledge]["model_knowledge"])


def build_context(plant: str, knowledge: str, root=None, with_model: bool = True) -> dict:
    meta = metadata(plant, root)
    m = {k: meta[k] for k in ALLOWED_META if k in meta}
    box = np.asarray(meta["ic_box"], float)
    sensor = {"noise": 0.03}
    if meta.get("sensor_noise_std_physical"):
        sensor = {"noise_std_physical": meta["sensor_noise_std_physical"], "hold_samples": meta.get("sensor_hold_samples")}
    if knowledge == "realistic":
        sensor = dict(sensor, kind="colored_t", rho=0.9, nu=3.0)
    return {
        "plant": plant, "knowledge": knowledge, "meta": m, "dt": float(meta["dt"]),
        "sensors": list(meta["sensors"]), "unmeasured": list(meta["unmeasured"]),
        "x_std": np.asarray(meta["x_std"], float), "y_std": np.asarray(meta["y_std"], float),
        "x_mean": np.asarray(meta["x_mean"], float),
        # nominal noise description a modeller is given (PROTOCOL.md section 3)
        "q_std": float(meta["process_noise_rel"]) * np.asarray(meta["x_std"], float),   # per sqrt(s)
        "r_std": (np.asarray(meta["sensor_noise_std_physical"], float) if meta.get("sensor_noise_std_physical")
                  else 0.03 * np.asarray(meta["y_std"], float)),
        "sensor_model": sensor,
        "prior_mean": box.mean(1), "prior_std": (box[:, 1] - box[:, 0]) / np.sqrt(12.0),
        "model": modeller_model(plant, knowledge, root) if with_model else None,
        "model_knowledge": KNOWLEDGE[knowledge]["model_knowledge"],
    }
