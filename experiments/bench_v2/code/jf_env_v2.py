"""Small helpers shared by the bench_v2 runners (copies of jitter_fix/code/jf_env.py pieces)."""
import torch


def cat(a, b):
    from hyperkkl import data as Dm
    kw = {k: torch.cat([getattr(a, k), getattr(b, k)]) for k in ("x", "u", "y_clean", "y", "z", "z_obs")}
    assert a.burn == b.burn and abs(a.dt - b.dt) < 1e-12, (a.burn, b.burn, a.dt, b.dt)
    return Dm.EpisodeSet(**kw, families=list(a.families) + list(b.families), burn=a.burn, dt=a.dt,
                         meta=list(a.meta) + list(b.meta), n_rejected=a.n_rejected + b.n_rejected)
