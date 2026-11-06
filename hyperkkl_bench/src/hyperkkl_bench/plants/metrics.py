r"""Benchmark metric (identical to the definitions used in the HyperKKL experiments).

    e_j(t)   = xhat_j(t) - x_j(t)
    sigma_j  = reference std of state j (metadata ``x_std``): forced trajectories of the NOMINAL
               plant at in-distribution inputs after the washout, measured once per plant. Fixed per
               plant, so every method, tier and setting shares one yardstick.
    S        = settled window: sample indices k >= settle = int(settle_frac * N), settle_frac = 0.5
               (metadata ``settle_index``), i.e. the second half of every episode
    settled NRMSE       = sqrt( mean_{k in S} mean_j (e_j(k) / sigma_j)^2 )       per episode
    NRMSE_j             = sqrt( mean_{k in S} (e_j(k) / sigma_j)^2 )
    unmeasured NRMSE    = mean of NRMSE_j over the unmeasured states (metadata ``unmeasured``)
    measured NRMSE      = mean of NRMSE_j over the measured states (metadata ``sensors``)
    Every normalised error is clipped to [-100, 100] (non-finite -> 100) before averaging;
    an episode is flagged diverged if the estimate is non-finite or |e_j / sigma_j| > 100.
Report the mean over the episodes of a tier (64 per tier).
"""
import numpy as np

DIVERGE = 100.0


def episode_metrics(x, xh, sigma, settle: int, sensors=None) -> dict:
    """x, xh: arrays (E, N+1, nx). Returns per-episode arrays (length E)."""
    x, xh = np.asarray(x, np.float64), np.asarray(xh, np.float64)
    s = np.asarray(sigma, np.float64)
    en = (xh - x) / s
    fin = np.isfinite(xh).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        big = (np.abs(en) > DIVERGE).any(axis=(1, 2))
    en = np.clip(np.nan_to_num(en, nan=DIVERGE, posinf=DIVERGE, neginf=-DIVERGE), -DIVERGE, DIVERGE)
    st = en[:, settle:]
    out = {"nrmse": np.sqrt((st ** 2).mean(axis=(1, 2)))}
    per = np.sqrt((st ** 2).mean(axis=1))                       # (E, nx)
    for j in range(x.shape[-1]):
        out[f"nrmse_x{j + 1}"] = per[:, j]
    if sensors is not None:
        meas = sorted({int(j) for j in sensors})
        unm = [j for j in range(x.shape[-1]) if j not in meas]
        nan = np.full(len(x), np.nan)
        out["nrmse_measured"] = per[:, meas].mean(-1) if meas else nan
        out["nrmse_unmeasured"] = per[:, unm].mean(-1) if unm else nan
    out["diverged"] = ~fin | big
    return out


def settled_nrmse(x, xh, meta: dict) -> dict:
    """Tier-level summary with the plant's metadata: mean settled NRMSE and unmeasured NRMSE."""
    m = episode_metrics(x, xh, meta["x_std"], int(meta["settle_index"]), meta["sensors"])
    return {"nrmse": float(np.mean(m["nrmse"])),
            "nrmse_unmeasured": float(np.nanmean(m["nrmse_unmeasured"])),
            "nrmse_measured": float(np.nanmean(m["nrmse_measured"])),
            "div_rate": float(np.mean(m["diverged"]))}
