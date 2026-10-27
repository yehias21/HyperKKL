"""HyperKKL: where the input should enter a learned KKL observer.

Modules, in dependency order:
    systems       plants f(x, u), h(x)  (systems_extra: battery, network)
    signals       input families and evaluation regimes
    design        latent filter (A, B, dt), horizons, exact latent step
    config        YAML configs and dotted overrides
    data          episode generation, sensor models, caching
    nets          base maps and context encoders
    conditioning  the five places the input can enter (C1)
    observers     the observer contract and every observer
    metrics       NRMSE and hierarchical paired bootstrap
    train         phase 1, phase 2, NLOX
    evaluate      tiers x conditions
    job           one (node, system, seed)
    tree, sched   experiment tree and GPU worker pool
"""

from . import systems as _systems


class _Registry(dict):
    """systems.REGISTRY, completed on demand: a name systems.py does not define imports
    systems_extra (battery, network), whose @register calls land in this dict. Lazy, so
    importing hyperkkl.job loads exactly job.RESULT_MODULES (tests/test_verify_pipeline)
    and runs on the existing plants never touch the new module."""

    def __missing__(self, key):
        from . import systems_extra  # noqa: F401  (registers its plants)
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        # bounded test-plant variants (2026-10-26, experiments/cartpole_bounded): own module so
        # no cache-hashed module changes; loaded only on a lookup of one of its names
        from . import systems_bounded  # noqa: F401
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        # deployment case-study plants (2026-10-27, experiments/casestudies): own module, same reason
        from . import systems_casestudy  # noqa: F401
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        raise KeyError(key)


_systems.REGISTRY = _Registry(_systems.REGISTRY)
