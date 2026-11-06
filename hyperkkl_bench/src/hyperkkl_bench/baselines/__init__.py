"""Reference estimators bundled with the benchmark (standalone) + adapters for the stored
research models (``stored``, needs the research repository; see baselines/README in docs)."""
from .filters import EKF, UKF, TunedKF  # noqa: F401
from .simple import PriorMean, GRUObserver  # noqa: F401
from .ude import UDEFilter  # noqa: F401

REGISTRY = {
    "prior_mean": lambda **kw: PriorMean(**kw),
    "ekf": lambda **kw: EKF(gate=False, **kw),
    "ekf_gated": lambda **kw: EKF(gate=True, **kw),
    "ekf_huber": lambda **kw: EKF(gate=False, huber=True, **kw),
    "ukf": lambda **kw: UKF(gate=False, **kw),
    "ukf_gated": lambda **kw: UKF(gate=True, **kw),
    "ekf_gated_tuned": lambda **kw: TunedKF("ekf", gate=True, **kw),
    "ukf_gated_tuned": lambda **kw: TunedKF("ukf", gate=True, **kw),
    "ude_ekf_gated": lambda **kw: UDEFilter(hybrid=True, kind="ekf", gate=True, **kw),
    "ude_ukf_gated": lambda **kw: UDEFilter(hybrid=True, kind="ukf", gate=True, **kw),
    "node_ekf_gated": lambda **kw: UDEFilter(hybrid=False, kind="ekf", gate=True, **kw),
    "gru": lambda **kw: GRUObserver(**kw),
}


def make(name: str, **kw):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
