"""hyperkkl-bench: a benchmark for state estimation of autonomous and input-driven nonlinear systems.

    import hyperkkl_bench as hkb
    tr = hkb.load("driven/battery/realistic", "train", recipe="mixed")   # v2: family/plant/level
    te = hkb.load("autonomous/vdp/partial", "test_ic_ood")
    rep = hkb.evaluate_v2(est, "driven/battery/realistic")            # v2 tiers x conditions
    hkb.list_plants()                                   # v1: 12 frozen plants (legacy tiers)
    train = hkb.load("vdp", "realistic", "train")       # EpisodeSet (u, y, x, dt, ...)
    est = hkb.baselines.EKF(gate=True)                  # any object with fit / reset / step
    rep = hkb.evaluate(est, "vdp", "realistic")         # standard report (table + JSON)
    print(rep.table())
"""
from .version import BENCH_VERSION, DATA_VERSION, SCORER_VERSION  # noqa: F401
from .registry import KNOWLEDGE, list_plants  # noqa: F401
from . import registry, tracks, hidden, conditions  # noqa: F401
from .data import load, metadata, EpisodeSet, Episode, PUBLIC_SPLITS, HIDDEN_SPLITS  # noqa: F401
from .estimator import Estimator, BaseEstimator  # noqa: F401
from .context import build_context, modeller_model  # noqa: F401
from .evaluate import evaluate, data_efficiency, Report  # noqa: F401
from .cards import plant_card  # noqa: F401
from . import baselines  # noqa: F401
from .families import evaluate_v2, context_v2, metadata_v2, FAMILIES, AUTONOMOUS, DRIVEN  # noqa: F401
from .families import TIERS as TIERS_V2, CONDITIONS as CONDITIONS_V2  # noqa: F401

__version__ = BENCH_VERSION
