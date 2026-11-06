"""Standalone plant, sensor and metric definitions of the HyperKKL observer benchmark.

    from hyperkkl_bench import build_plant, load_meta
    meta = load_meta("vdp")                                    # metadata/vdp.json of the dataset
    true_plant  = build_plant(meta["system_config"], perturb=0.0)      # C1 / C3-partial data
    real_plant  = build_plant(meta["system_config"], perturb=0.3)      # C3-mismatch30 / C3-realistic data
    model_exact = build_plant(meta["system_config"])                   # modeller, knowledge = full
    model_part  = build_plant(meta["system_config"], knowledge="partial")  # f_known (UNMODELLED term missing)

Requires torch, numpy (scipy for the colored_t sensor model).
"""
import json
from pathlib import Path

from . import systems_extra, systems_bounded, systems_casestudy, systems_scaling, systems_real   # noqa: F401  (register battery, network, cartpole_b, quad_planar, cstr_exo)
from .systems import REGISTRY, build_plant, Plant   # noqa: F401
from .sensors import measure                  # noqa: F401
from .metrics import settled_nrmse, episode_metrics   # noqa: F401
from .simulate import rk4_zoh_step            # noqa: F401


def load_meta(plant: str, root=None) -> dict:
    """Read metadata/<plant>.json from a local copy of the dataset (default: ../../metadata)."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    return json.loads((root / "metadata" / f"{plant}.json").read_text())
