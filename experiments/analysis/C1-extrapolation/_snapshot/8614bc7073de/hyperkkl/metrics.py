r"""Metrics. SMAPE is not used anywhere (CDC R10: it is undefined near zero crossings).

Definitions (stated once, used everywhere; the paper quotes these verbatim):

    e_j(t)   = xhat_j(t) - x_j(t)
    sigma_j  = reference std of state j: ID-amplitude forced trajectories of the
               NOMINAL plant after washout, measured once per system (data.system_scales).
               Fixed per system, so every method, tier and condition shares one yardstick.
    NRMSE    = sqrt( mean_{t in S} mean_j (e_j(t) / sigma_j)^2 )
    RMSE_j   = sqrt( mean_{t in S} e_j(t)^2 )        (plant units, per state)
    MAE_j    = mean_{t in S} |e_j(t)|
    S        = steady window, t >= settle_frac * T (default: second half of the episode)
    transient NRMSE: same formula on t < settle_frac * T
    max_nerr = max_{t in S} sqrt( mean_j (e_j(t) / sigma_j)^2 )
    t_conv   = first time after which the normalised error stays below 0.05
    diverged = non-finite estimate or |e_j / sigma_j| > 100 anywhere
    Every normalised error is clipped to [-100, 100] (non-finite -> 100) before any
    average, and RMSE_j / MAE_j use the same clipped error times sigma_j, so a diverged
    episode contributes a bounded, finite value to every metric (and is flagged).

Uncertainty: the unit of replication is the training SEED. Means and paired comparisons
use a Student-t interval over seeds (per-seed means, per-seed relative changes); a node
with a single seed falls back to an episode bootstrap, which measures episode variability
only. Method comparisons are PAIRED: same seeds, same episodes.
"""

from __future__ import annotations

import numpy as np
import torch

DIVERGE = 100.0
CONV_BAND = 0.05


@torch.no_grad()
def episode_metrics(x: torch.Tensor, xh: torch.Tensor, sigma, settle: int) -> dict:
    """x, xh: (E, N+1, nx). Returns per-episode arrays (numpy, length E)."""
    s = torch.as_tensor(np.asarray(sigma), dtype=x.dtype, device=x.device)
    e = xh - x
    en = e / s
    fin = torch.isfinite(xh).all(-1).all(-1)
    big = (en.abs() > DIVERGE).any(-1).any(-1)
    en = torch.nan_to_num(en, nan=DIVERGE, posinf=DIVERGE, neginf=-DIVERGE).clamp(-DIVERGE, DIVERGE)
    st, tr = en[:, settle:], en[:, :settle]
    out = {
        "nrmse": st.pow(2).mean((1, 2)).sqrt(),
        "nrmse_transient": tr.pow(2).mean((1, 2)).sqrt() if settle > 0 else st[:, :0].sum((1, 2)),
        "max_nerr": st.pow(2).mean(-1).sqrt().amax(1),
    }
    ee = st * s                                   # clipped error in plant units (NaN is not 0)
    for j in range(x.shape[-1]):
        out[f"rmse_x{j + 1}"] = ee[..., j].pow(2).mean(1).sqrt()
        out[f"mae_x{j + 1}"] = ee[..., j].abs().mean(1)
        out[f"nrmse_x{j + 1}"] = st[..., j].pow(2).mean(1).sqrt()
    ne = en.pow(2).mean(-1).sqrt()                                   # (E, N+1)
    above = ne > CONV_BAND
    last = torch.where(above.any(1), (above.float() * torch.arange(ne.shape[1], device=ne.device)).amax(1),
                       torch.full_like(ne[:, 0], -1))
    out["t_conv_steps"] = last + 1
    out = {k: v.float().cpu().numpy() for k, v in out.items()}
    out["diverged"] = (~fin | big).cpu().numpy()
    return out


def bootstrap(values, n_boot=2000, seed=0, alpha=0.05) -> tuple:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]),) * 3
    rng = np.random.RandomState(seed)
    b = v[rng.randint(0, len(v), (n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def _t_interval(v: np.ndarray, alpha=0.05) -> tuple:
    from scipy.stats import t as student_t
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = float(student_t.ppf(1 - alpha / 2, len(v) - 1)) * sd / np.sqrt(len(v))
    return m, m - h, m + h


def hierarchical(per_seed: list, n_boot=2000, seed=0, alpha=0.05) -> tuple:
    """per_seed: list of 1-D arrays (episodes of one training seed) -> (mean, lo, hi).

    The point estimate is the mean of the seed means. With >= 2 seeds the CI is a
    Student-t interval on the seed means: the seed is the unit of replication, and a
    two-level bootstrap over 3-5 seeds under-covers badly (measured 80% / 86% nominal-95%
    coverage at S = 3 / 5; the t-interval gives 95% / 96%, tests/test_verify_pipeline.py).
    With one seed only, an episode bootstrap is the only option, and it measures episode
    variability, not training variability -- reports flag single-seed nodes."""
    groups = [np.asarray(g, float)[np.isfinite(g)] for g in per_seed]
    groups = [g for g in groups if len(g)]
    if not groups:
        return (np.nan, np.nan, np.nan)
    if len(groups) == 1:
        return bootstrap(groups[0], n_boot, seed, alpha)
    return _t_interval(np.array([g.mean() for g in groups]), alpha)


def paired_relative(a_per_seed: list, b_per_seed: list, n_boot=2000, seed=0, alpha=0.05) -> dict:
    """Relative change of method a against b on the SAME episodes.

    Per seed s: r_s = (mean_a_s - mean_b_s) / mean_b_s over that seed's paired episodes.
    Point estimate mean_s r_s; with >= 2 seeds a Student-t CI over seeds (see
    `hierarchical` for why), with 1 seed a paired episode bootstrap. A seed or episode count
    that differs between a and b raises: silently dropping a seed would misreport."""
    if len(a_per_seed) != len(b_per_seed):
        raise ValueError(f"paired_relative: {len(a_per_seed)} vs {len(b_per_seed)} seeds")
    pairs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in zip(a_per_seed, b_per_seed)]
    if any(len(a) != len(b) for a, b in pairs):
        raise ValueError(f"paired_relative: episode counts differ {[(len(a), len(b)) for a, b in pairs]}")
    pairs = [(a, b) for a, b in pairs if len(a)]
    if not pairs:
        return {}
    rel = lambda ma, mb: (ma - mb) / np.maximum(mb, 1e-12)
    S = len(pairs)
    if S >= 2:
        r = np.array([rel(a.mean(), b.mean()) for a, b in pairs])
        point, lo, hi = _t_interval(r, alpha)
        method = "t-over-seeds"
    else:
        a, b = pairs[0]
        rng = np.random.RandomState(seed)
        j = rng.randint(0, len(a), (n_boot, len(a)))          # SAME resampled episodes
        est = rel(a[j].mean(1), b[j].mean(1))
        point = float(rel(a.mean(), b.mean()))
        lo, hi = np.percentile(est, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        method = "bootstrap-episodes-1-seed"
    return {"rel": float(point), "ci": [float(lo), float(hi)], "n_seeds": S, "ci_method": method,
            "better": bool(hi < 0), "worse": bool(lo > 0)}


def bound_terms(eps_dec, eps_res, l_dec, kappa, alpha, l_phi=0.0, l_c=0.0, delta_c=0.0) -> dict:
    """Terms of Proposition 1 (THEORY.md):
    limsup ||xi|| <= eps_dec + l_dec kappa eps_res / (alpha - kappa l_phi) + l_c delta_c."""
    margin = alpha - kappa * l_phi
    drive = kappa * eps_res / margin if margin > 0 else float("inf")
    return {"eps_dec": eps_dec, "eps_res": eps_res, "l_dec": l_dec, "margin": margin,
            "drive": drive, "context_term": l_c * delta_c,
            "bound": eps_dec + l_dec * drive + l_c * delta_c}
