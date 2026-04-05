"""
Configuration system using dataclasses + YAML.

Provides typed configs for systems, training, and experiments.
Handles loading from YAML files with CLI overrides.

Usage:
    python -m src.config --system duffing
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    name: str = "duffing"
    class_name: str = "RevDuff"
    init_args: dict = field(default_factory=dict)
    x_size: int = 2
    y_size: int = 1
    z_size: int = 5
    num_hidden: int = 3
    hidden_size: int = 150
    M: list = field(default_factory=list)
    K: list = field(default_factory=list)
    limits: list = field(default_factory=list)
    time_start: float = 0.0
    time_end: float = 50.0
    n_steps: int = 1000
    natural_inputs: List[str] = field(
        default_factory=lambda: ["zero", "sinusoid", "square", "constant"]
    )

    @property
    def M_np(self) -> np.ndarray:
        return np.array(self.M, dtype=np.float64)

    @property
    def K_np(self) -> np.ndarray:
        return np.array(self.K, dtype=np.float64)

    @property
    def limits_np(self) -> np.ndarray:
        return np.array(self.limits, dtype=np.float64)

    @property
    def dt(self) -> float:
        return (self.time_end - self.time_start) / self.n_steps


@dataclass
class Phase1Config:
    epochs: int = 20
    batch_size: int = 2049
    lr: float = 1e-3
    num_ic: int = 200
    use_pde: bool = True
    lambda_pde: float = 1.0


@dataclass
class Phase2Config:
    epochs: int = 30
    batch_size: int = 4098
    lr: float = 1e-3
    encoder_type: str = "lstm"  # "lstm" or "gru" — model choice, not method
    window_size: int = 100
    latent_dim: int = 32
    n_train_traj: int = 400
    rnn_hidden: int = 64
    hypernet_hidden: int = 128
    lora_rank: int = 4


@dataclass
class CurriculumConfig:
    lr: float = 1e-3
    batch_size: int = 4098
    window_size: int = 100
    latent_dim: int = 32
    n_traj_per_stage: int = 50
    stage1_epochs: int = 10
    stage2_epochs: int = 20
    stage3_epochs: int = 30
    stage4_epochs: int = 40


@dataclass
class EvalConfig:
    n_trials: int = 10
    settle_time: float = 5.0


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    system: SystemConfig = field(default_factory=SystemConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    methods: List[str] = field(
        default_factory=lambda: [
            "autonomous", "augmented", "full", "lora"
        ]
    )
    output_dir: str = "./results"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update base dict with overrides."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = v
    return result


_NESTED_TYPES = {
    "system": SystemConfig,
    "phase1": Phase1Config,
    "phase2": Phase2Config,
    "curriculum": CurriculumConfig,
    "evaluation": EvalConfig,
}


def _dict_to_dataclass(cls, data: dict):
    """Convert a dict to a nested dataclass, ignoring unknown keys."""
    if not isinstance(data, dict):
        return data
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {}
    for k, v in data.items():
        if k in field_names:
            # Check if it's a known nested dataclass field
            if k in _NESTED_TYPES and isinstance(v, dict):
                filtered[k] = _dict_to_dataclass(_NESTED_TYPES[k], v)
            else:
                filtered[k] = v
    return cls(**filtered)


def load_config(
    system_name: str,
    config_dir: Optional[str] = None,
    overrides: Optional[dict] = None,
) -> ExperimentConfig:
    """Load experiment config from YAML files.

    Merges: default.yaml -> systems/{system_name}.yaml -> overrides dict.
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    # Load default config
    default_path = config_dir / "default.yaml"
    data = {}
    if default_path.exists():
        with open(default_path) as f:
            data = yaml.safe_load(f) or {}

    # Load system-specific config
    sys_path = config_dir / "systems" / f"{system_name}.yaml"
    if sys_path.exists():
        with open(sys_path) as f:
            sys_data = yaml.safe_load(f) or {}
        # System config goes under the 'system' key
        data = _deep_update(data, {"system": sys_data})

    # Apply overrides
    if overrides:
        data = _deep_update(data, overrides)

    return _dict_to_dataclass(ExperimentConfig, data)


def save_config(config: ExperimentConfig, path: Path):
    """Save experiment config as YAML alongside results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def config_to_flat_dict(config: ExperimentConfig) -> dict:
    """Flatten nested config to dot-separated keys for logging."""
    flat = {}

    def _flatten(d, prefix=""):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, key)
            elif isinstance(v, list) and len(v) < 20:
                flat[key] = str(v)
            else:
                flat[key] = v

    _flatten(asdict(config))
    return flat


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show experiment config")
    parser.add_argument("--system", default="duffing")
    parser.add_argument("--config_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.system, args.config_dir)
    print(yaml.dump(asdict(cfg), default_flow_style=False))
