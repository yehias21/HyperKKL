r"""Scoring.

Accuracy (identical to the paper's definitions; plants/metrics.py is the reference copy):

    e_j(k)   = xhat_j(k) - x_j(k);   en_j(k) = e_j(k) / x_std_j  (x_std fixed per plant, metadata)
    clipped to [-100, 100], non-finite -> 100
    S        = settled window k >= settle_index (= int(0.5 N): the second half of the episode)
    nrmse              = sqrt(mean_{k in S} mean_j en_j(k)^2)               all-state settled NRMSE
    nrmse_x{j}         = sqrt(mean_{k in S} en_j(k)^2)
    nrmse_unmeasured   = mean_j over UNMEASURED states of nrmse_x{j}          (HEADLINE metric)
    nrmse_measured     = mean_j over measured states of nrmse_x{j}
    diverged           = any non-finite estimate or |en| > 100
    Tier value = mean over episodes (64 per public tier).

Calibration (only if the estimator returns a covariance P(k); settled window):
    coverage68 / coverage95   fraction of (k, j) with |e_j(k)| <= z sqrt(P_jj(k)), z = 1 / 1.96
                              (ideal 0.683 / 0.95)
    anees                     mean_k e(k)' P(k)^-1 e(k) / n_x  (ideal 1; > 1 overconfident)
    nees_in_95                fraction of k with NEES inside the two-sided 95% chi2_{n_x} interval
    Diverged episodes are excluded from calibration and counted in ``calib_excluded``.

Transient (robustness conditions):
    t_conv   per episode: first time after which the 1-t_ref moving RMS of ne(k) = sqrt(mean_j en_j^2)
             stays below max(2 x the clean settled median, 0.05); censored at T (conv_censored).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

DIVERGE = 100.0
CONV_BAND = 0.05


def _norm_err(x, xh, sigma):
    x, xh = np.asarray(x, np.float64), np.asarray(xh, np.float64)
    s = np.asarray(sigma, np.float64)
    en = (xh - x) / s
    fin = np.isfinite(xh).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        big = (np.abs(en) > DIVERGE).any(axis=(1, 2))
    en = np.clip(np.nan_to_num(en, nan=DIVERGE, posinf=DIVERGE, neginf=-DIVERGE), -DIVERGE, DIVERGE)
    return en, ~fin | big


def episode_metrics(x, xh, sigma, settle: int, sensors) -> dict:
    """x, xh (E, N+1, n_x) -> per-episode arrays (length E)."""
    en, div = _norm_err(x, xh, sigma)
    st = en[:, settle:]
    per = np.sqrt((st ** 2).mean(axis=1))                         # (E, n_x)
    meas = sorted({int(j) for j in sensors})
    unm = [j for j in range(x.shape[-1]) if j not in meas]
    nan = np.full(len(en), np.nan)
    out = {"nrmse": np.sqrt((st ** 2).mean(axis=(1, 2))),
           "nrmse_unmeasured": per[:, unm].mean(-1) if unm else nan,
           "nrmse_measured": per[:, meas].mean(-1) if meas else nan,
           "nrmse_transient": np.sqrt((en[:, :settle] ** 2).mean(axis=(1, 2))) if settle > 0 else nan,
           "diverged": div}
    for j in range(x.shape[-1]):
        out[f"nrmse_x{j + 1}"] = per[:, j]
    return out


def calibration(x, xh, P, settle: int, diverged=None) -> dict:
    """P: (E, N+1, n_x, n_x) covariances or (E, N+1, n_x) variances."""
    x, xh, P = (np.asarray(a, np.float64) for a in (x, xh, P))
    keep = np.ones(len(x), bool) if diverged is None else ~np.asarray(diverged, bool)
    if not keep.any():
        return {"coverage68": None, "coverage95": None, "anees": None, "nees_in_95": None,
                "calib_excluded": int((~keep).sum())}
    e = (xh - x)[keep, settle:]
    P = P[keep, settle:]
    nx = x.shape[-1]
    var = np.einsum("...ii->...i", P) if P.ndim == 4 else P
    sd = np.sqrt(np.clip(var, 1e-300, None))
    a = np.abs(e) / sd
    if P.ndim == 4:
        Pr = P + 1e-12 * np.eye(nx) * np.maximum(np.einsum("...ii->...", P)[..., None, None] / nx, 1e-300)
        nees = np.einsum("...i,...i->...", e, np.linalg.solve(Pr, e[..., None])[..., 0])
    else:
        nees = (a ** 2).sum(-1)
    lo, hi = chi2.ppf(0.025, nx), chi2.ppf(0.975, nx)
    finite = np.isfinite(nees)
    return {"coverage68": float(np.mean(a <= 1.0)), "coverage95": float(np.mean(a <= 1.959964)),
            "anees": float(np.mean(nees[finite]) / nx) if finite.any() else None,
            "nees_in_95": float(np.mean((nees >= lo) & (nees <= hi))),
            "calib_excluded": int((~keep).sum())}


def conv_time(x, xh, sigma, dt: float, t_ref: float, thr: float) -> np.ndarray:
    """Per-episode convergence time (s); NaN = never converged (censored)."""
    en, _ = _norm_err(x, xh, sigma)
    ne = np.sqrt((en ** 2).mean(-1))                              # (E, N+1)
    w = max(1, int(round(t_ref / dt)))
    c = np.cumsum(np.pad(ne ** 2, ((0, 0), (w, 0))), axis=1)
    mrms = np.sqrt(np.clip((c[:, w:] - c[:, :-w]) / w, 0, None))  # trailing RMS
    above = mrms > thr
    out = np.full(len(ne), np.nan)
    for i in range(len(ne)):
        idx = np.nonzero(above[i])[0]
        if len(idx) == 0:
            out[i] = 0.0
        elif idx[-1] < ne.shape[1] - 1:
            out[i] = (idx[-1] + 1) * dt
    return out


def summarize(m: dict) -> dict:
    """Tier summary of per-episode metrics (means; diverged episodes included with clipped error)."""
    out = {}
    for k, v in m.items():
        v = np.asarray(v, float)
        if k == "diverged":
            out["div_rate"] = float(v.mean())
        elif np.isfinite(v).any():
            out[k] = float(np.nanmean(v))
    nr = np.asarray(m["nrmse_unmeasured"], float)
    if np.isfinite(nr).any():
        out["nrmse_unmeasured_median"] = float(np.nanmedian(nr))
        out["nrmse_unmeasured_p95"] = float(np.nanpercentile(nr, 95))
    return out


def bootstrap_ci(v, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Episode-bootstrap CI of the mean (episodes are independent units in the test tiers). This
    measures episode variability only; training-seed variability needs several fitted seeds
    (PROTOCOL.md section 5)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return [None, None]
    rng = np.random.RandomState(seed)
    bs = v[rng.randint(0, len(v), (n_boot, len(v)))].mean(1)
    a = (1 - level) / 2
    return [float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))]
