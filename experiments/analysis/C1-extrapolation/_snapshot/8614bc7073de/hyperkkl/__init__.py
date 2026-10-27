"""HyperKKL: where the input should enter a learned KKL observer.

Modules, in dependency order:
    systems       plants f(x, u), h(x)
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
