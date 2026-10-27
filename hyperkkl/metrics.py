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
    nrmse_measured / nrmse_unmeasured = mean of NRMSE_j over the measured (plant.sensors) /
               unmeasured states (episode_metrics with ``sensors``; NaN if the set is empty)
    Every normalised error is clipped to [-100, 100] (non-finite -> 100) before any
    average, and RMSE_j / MAE_j use the same clipped error times sigma_j, so a diverged
    episode contributes a bounded, finite value to every metric (and is flagged).

Deployment scorecard (``block_scorecard``, one value per method x tier x condition block,
ne(t) = sqrt(mean_j (e_j(t) / sigma_j)^2), the clipped normalised error above):
    nrmse_p95 / nrmse_p99   95th / 99th percentile of the episode NRMSE; div_rate = mean diverged
    t_conv                  per episode: first time after which ne_s, the centred moving
                            RMS of ne over 1 t_ref, stays below
                            thr = max(2 median_{episodes, t in S} ne_s, CONV_BAND) to the end;
                            median / p90 in s and in t_ref; never-converged episodes are
                            censored at T (conv_censored = their fraction)
    recovery (burst blocks) pre = RMS ne over the burst-length window before the outage;
                            recovery = first time after the first fresh sample from which the
                            RMS of ne over the next 0.5 t_ref is <= 1.5 pre; median over
                            episodes (censored at the episode end, rec_not_recovered = fraction)
    t_forget (init blocks)  paired: first time after which the moving RMS (1 t_ref) of the
                            normalised difference to the SAME observer's estimate from the
                            nominal prior (same episode) stays <= CONV_BAND (metrics.forget_time)
    drift (long blocks)     RMS ne over the last 20% of the episode / RMS ne over the 20%
                            starting at the nominal settle time (1 = no drift), pooled over episodes
    alarm_auroc             r_k = EMA over t_ref of ||(y_k - h(xhat_k)) / y_std||, scored as a
                            detector of steps with ne > 3 x the median settled ne (Mann-Whitney
                            AUROC, ties averaged; subsampled steps, EMA warm-up of 1 t_ref skipped)

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
def episode_metrics(x: torch.Tensor, xh: torch.Tensor, sigma, settle: int, sensors=None) -> dict:
    """x, xh: (E, N+1, nx). Returns per-episode arrays (numpy, length E). With ``sensors``
    (the measured state indices) also nrmse_measured / nrmse_unmeasured."""
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
    if sensors is not None:
        nx = x.shape[-1]
        meas = sorted({int(j) for j in sensors})
        unm = [j for j in range(nx) if j not in meas]
        per = torch.stack([out[f"nrmse_x{j + 1}"] for j in range(nx)], -1)
        nan = torch.full_like(per[:, 0], float("nan"))
        out["nrmse_measured"] = per[:, meas].mean(-1) if meas else nan
        out["nrmse_unmeasured"] = per[:, unm].mean(-1) if unm else nan
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


@torch.no_grad()
def state_coverage(x_train: torch.Tensor, x_test: torch.Tensor, x_std, n_ref: int = 20000,
                   n_query: int = 4000, seed: int = 0, chunk: int = 1024) -> dict:
    """How far evaluation states lie from the states seen in training.

    x_train: (E_tr, N_tr, nx) training trajectories; x_test: (E, N, nx) evaluation states
    (callers pass them after burn-in). Two measures, both on states normalised by x_std:

      outside_box   fraction of test states with ANY coordinate outside the per-state
                    [min, max] of all training states (and per state: outside_box_j)
      outside_knn   fraction of test states whose nearest-training-state distance exceeds
                    q99 of the same distance for HELD-OUT training states. The training
                    episodes are split in two halves: references come from one half,
                    held-out queries from the other, so neighbouring time steps of one
                    episode never match each other and test and held-out training
                    queries are judged against the same reference set (a test set drawn
                    like the training set gives ~0.01).

    Subsampled (n_ref references, n_query queries per set) to stay cheap."""
    s = torch.as_tensor(np.asarray(x_std, dtype=np.float64)).double()
    tr = (x_train.double().cpu() / s)
    te = (x_test.double().cpu() / s).reshape(-1, tr.shape[-1])
    tf = tr.reshape(-1, tr.shape[-1])
    lo, hi = tf.min(0).values, tf.max(0).values
    out_j = (te < lo) | (te > hi)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(tr.shape[0])
    half = max(1, tr.shape[0] // 2)
    ref = tr[perm[:half]].reshape(-1, tr.shape[-1])
    hold = tr[perm[half:]].reshape(-1, tr.shape[-1]) if tr.shape[0] > 1 else ref
    pick = lambda a, n: a[torch.as_tensor(rng.choice(len(a), min(n, len(a)), replace=False))]
    ref, hold, q = pick(ref, n_ref), pick(hold, n_query), pick(te, n_query)

    def nn(a):
        return torch.cat([torch.cdist(a[i:i + chunk], ref).min(1).values for i in range(0, len(a), chunk)])
    d_hold, d_te = nn(hold), nn(q)
    q99 = float(torch.quantile(d_hold, 0.99))
    return {"outside_box": float(out_j.any(-1).double().mean()),
            "outside_box_j": out_j.double().mean(0).tolist(),
            "outside_knn": float((d_te > q99).double().mean()),
            "knn_q99_train": q99, "knn_median_test": float(d_te.median()),
            "knn_median_train": float(d_hold.median()), "n_test": int(len(te)), "n_query": int(len(q))}


def bound_terms(eps_dec, eps_res, l_dec, kappa, alpha, l_phi=0.0, l_c=0.0, delta_c=0.0) -> dict:
    """Terms of Proposition 1 (THEORY.md):
    limsup ||xi|| <= eps_dec + l_dec kappa eps_res / (alpha - kappa l_phi) + l_c delta_c."""
    margin = alpha - kappa * l_phi
    drive = kappa * eps_res / margin if margin > 0 else float("inf")
    return {"eps_dec": eps_dec, "eps_res": eps_res, "l_dec": l_dec, "margin": margin,
            "drive": drive, "context_term": l_c * delta_c,
            "bound": eps_dec + l_dec * drive + l_c * delta_c}


# ---------------------------------------------------------------------------
# Deployment scorecard (block level; definitions in the module docstring)
# ---------------------------------------------------------------------------

def auroc(score, label) -> float:
    """Area under the ROC curve of ``score`` for the binary ``label`` (Mann-Whitney U with
    average ranks for ties). NaN when a class is empty."""
    from scipy.stats import rankdata
    s = np.asarray(score, float).ravel()
    l = np.asarray(label, bool).ravel()
    ok = np.isfinite(s)
    s, l = s[ok], l[ok]
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[l].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _nerr(x, xh, sigma):
    """Clipped normalised error per step, ne (E, N+1), as in episode_metrics."""
    s = torch.as_tensor(np.asarray(sigma), dtype=x.dtype, device=x.device)
    en = (xh - x) / s
    en = torch.nan_to_num(en, nan=DIVERGE, posinf=DIVERGE, neginf=-DIVERGE).clamp(-DIVERGE, DIVERGE)
    return en.pow(2).mean(-1).sqrt()


def _q(v, p):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, p)) if len(v) else float("nan")


def moving_rms(ne: np.ndarray, W: int) -> np.ndarray:
    """Centred moving RMS over W samples along axis 1 (window clipped at the ends)."""
    E, N1 = ne.shape
    c = np.concatenate([np.zeros((E, 1)), np.cumsum(ne.astype(float) ** 2, 1)], 1)
    k = np.arange(N1)
    lo, hi = np.clip(k - W // 2, 0, N1), np.clip(k - W // 2 + W, 0, N1)
    return np.sqrt((c[:, hi] - c[:, lo]) / (hi - lo))


def convergence_time(ne: np.ndarray, thr: float) -> tuple:
    """Per episode: index of the first sample after which ne stays <= thr (ne (E, N+1)),
    and a censored flag (ne > thr at the last sample: never converged)."""
    above = ne > thr
    N1 = ne.shape[1]
    idx = np.where(above.any(1), N1 - np.argmax(above[:, ::-1], 1), 0)
    return idx, above[:, -1]


def recovery_time(ne: np.ndarray, k0: int, L: int, W: int, factor: float = 1.5) -> dict:
    """Burst outage on samples k0+1..k0+L (first fresh sample k1 = k0+L+1). Per episode:
    pre = RMS ne over [max(0, k0-L), k0]; recovery index j - k1 for the first j >= k1 with
    RMS(ne[j:j+W]) <= factor * pre; censored (-1) when no such j exists."""
    E, N1 = ne.shape
    k1 = k0 + L + 1
    lo = max(0, k0 - max(L, 1))
    pre = np.sqrt((ne[:, lo:k0 + 1] ** 2).mean(1))
    W = max(1, int(W))
    if k1 + W > N1:
        return {"pre": pre, "steps": np.full(E, -1), "during": np.full(E, np.nan), "avail": 0}
    c = np.concatenate([np.zeros((E, 1)), np.cumsum(ne[:, k1:] ** 2, 1)], 1)
    fwd = np.sqrt((c[:, W:] - c[:, :-W]) / W)                 # window starting at k1 + i
    ok = fwd <= factor * pre[:, None]
    steps = np.where(ok.any(1), np.argmax(ok, 1), -1)
    during = np.sqrt((ne[:, k0 + 1:k1] ** 2).mean(1)) / np.maximum(pre, 1e-12) if L > 0 else np.ones(E)
    return {"pre": pre, "steps": steps, "during": during, "avail": fwd.shape[1]}


@torch.no_grad()
def block_scorecard(x, xh, y, sigma, y_std, sensors, settle: int, dt: float, t_ref: float = 1.0,
                    burst=None, drift_start: int = None, tol: float = CONV_BAND,
                    alarm_max: int = 20000, seed: int = 0) -> dict:
    """Scorecard scalars for one (method, tier, condition) block.

    x, xh (E, N+1, nx); y (E, N+1, ny) the measurements the observer received; sensors the
    measured state indices (h = selection); burst = (k0, L) of a burst outage
    (data.burst_window) or None; drift_start = the nominal settle index for long episodes
    (drift ratio), or None."""
    E, N1, nx = x.shape
    em = episode_metrics(x, xh, sigma, settle, sensors)
    ne_t = _nerr(x, xh, sigma)
    ne = ne_t.double().cpu().numpy()
    T, tr = (N1 - 1) * dt, float(t_ref)
    out = {"n_episodes": int(E), "nrmse_mean": float(np.mean(em["nrmse"])),
           "nrmse_median": _q(em["nrmse"], 50), "nrmse_p95": _q(em["nrmse"], 95),
           "nrmse_p99": _q(em["nrmse"], 99), "div_rate": float(np.mean(em["diverged"])),
           "nrmse_state": [float(np.mean(em[f"nrmse_x{j + 1}"])) for j in range(nx)],
           "nrmse_measured": float(np.mean(em["nrmse_measured"])),
           "nrmse_unmeasured": float(np.mean(em["nrmse_unmeasured"]))}
    med = float(np.median(ne[:, settle:])) if settle < N1 else float(np.median(ne))
    # convergence time, on the centred moving RMS of ne over 1 t_ref: the instantaneous error
    # of a noisy estimator exceeds 2x its median every few samples, so "stays below" on raw ne
    # measured the last noise excursion of the episode, not convergence
    ns = moving_rms(ne, max(1, int(round(tr / dt))))
    med_s = float(np.median(ns[:, settle:])) if settle < N1 else float(np.median(ns))
    thr = max(2.0 * med_s, float(tol))
    idx, cens = convergence_time(ns, thr)
    tc = np.where(cens, T, idx * dt)
    out.update({"settled_median_nerr": med, "conv_thr": thr,
                "t_conv_median_s": float(np.median(tc)), "t_conv_p90_s": float(np.percentile(tc, 90)),
                "t_conv_median_tref": float(np.median(tc)) / tr, "t_conv_p90_tref": float(np.percentile(tc, 90)) / tr,
                "conv_censored": float(np.mean(cens))})
    # recovery after a burst outage
    if burst is not None:
        k0, L = burst
        W = max(1, int(round(0.5 * tr / dt)))
        r = recovery_time(ne, int(k0), int(L), W)
        rec = r["steps"] >= 0
        rs = np.where(rec, r["steps"] * dt, max(r["avail"] - 1, 0) * dt)
        out.update({"rec_median_s": float(np.median(rs)), "rec_median_tref": float(np.median(rs)) / tr,
                    "rec_p90_s": float(np.percentile(rs, 90)), "rec_not_recovered": float(1 - rec.mean()),
                    "rec_pre_nerr": float(np.median(r["pre"])),
                    "burst_err_ratio": float(np.nanmedian(r["during"])) if np.isfinite(r["during"]).any() else float("nan"),
                    "burst_k0": int(k0), "burst_len": int(L)})
    # long-horizon drift
    if drift_start is not None:
        w = max(1, int(round(0.2 * (N1 - 1))))
        a0 = int(drift_start)
        A = ne[:, a0:a0 + w]
        B = ne[:, N1 - w:]
        ra, rb = np.sqrt((A ** 2).mean()), np.sqrt((B ** 2).mean())
        per = np.sqrt((B ** 2).mean(1)) / np.maximum(np.sqrt((A ** 2).mean(1)), 1e-12)
        out.update({"drift_ratio": float(rb / max(ra, 1e-12)), "drift_ratio_median": float(np.median(per)),
                    "drift_nrmse_early": float(ra), "drift_nrmse_late": float(rb)})
    # self-diagnosis alarm: output residual EMA vs large state error
    from scipy.signal import lfilter
    sens = list(sensors)
    ys = torch.as_tensor(np.asarray(y_std), dtype=xh.dtype, device=xh.device)
    res = torch.nan_to_num((y.to(xh.dtype) - xh[..., sens]) / ys, nan=1e6, posinf=1e6, neginf=-1e6)
    rn = res.double().norm(dim=-1).clamp(max=1e12).cpu().numpy()            # (E, N+1)
    a = 1.0 - np.exp(-dt / tr)
    rk, _ = lfilter([a], [1.0, -(1.0 - a)], rn, axis=1, zi=((1.0 - a) * rn[:, :1]))
    k_w = min(N1 - 1, int(round(tr / dt)))
    lab = ne > 3.0 * med
    sc, lb = rk[:, k_w:].ravel(), lab[:, k_w:].ravel()
    if len(sc) > alarm_max:
        pick = np.random.RandomState(seed).choice(len(sc), alarm_max, replace=False)
        sc, lb = sc[pick], lb[pick]
    out.update({"alarm_auroc": auroc(sc, lb), "alarm_pos_frac": float(lb.mean()) if len(lb) else float("nan"),
                "alarm_n": int(len(lb))})
    return out


@torch.no_grad()
def forget_time(xh, xh_ref, sigma, dt: float, t_ref: float = 1.0, tol: float = CONV_BAND) -> dict:
    """Paired convergence from a poor prior: per episode, the first time after which the
    centred moving RMS (1 t_ref) of the normalised DIFFERENCE between this estimate and the
    same observer's estimate from the nominal prior (same episode, same measurements) stays
    <= tol. Unlike t_conv it does not depend on how the steady error varies with the input."""
    s = torch.as_tensor(np.asarray(sigma), dtype=xh.dtype, device=xh.device)
    d = torch.nan_to_num((xh - xh_ref) / s, nan=DIVERGE, posinf=DIVERGE, neginf=-DIVERGE).clamp(-DIVERGE, DIVERGE)
    dn = d.pow(2).mean(-1).sqrt().double().cpu().numpy()
    ds = moving_rms(dn, max(1, int(round(float(t_ref) / dt))))
    idx, cens = convergence_time(ds, float(tol))
    T = (dn.shape[1] - 1) * dt
    tf = np.where(cens, T, idx * dt)
    return {"t_forget_median_s": float(np.median(tf)), "t_forget_p90_s": float(np.percentile(tf, 90)),
            "t_forget_median_tref": float(np.median(tf)) / float(t_ref), "forget_censored": float(np.mean(cens))}


def fit_decay(L, d, dt: float, floor: float = 1e-9, tail: float = 0.25) -> dict:
    """Exponential decay rate (1/s) of a memory-probe curve d(L) (L in samples): a
    least-squares line through log d against L dt on the points above ``floor`` with
    L >= tail * (largest such L) -- the tail, where the slowest mode dominates. Also the
    first L (s) at which d falls to the floor."""
    L, d = np.asarray(L, float), np.asarray(d, float)
    ok = np.isfinite(d) & (d > floor)
    below = np.where(~ok & np.isfinite(d))[0]
    L_floor = float(L[below[0]] * dt) if len(below) else float("nan")
    if ok.sum() < 2:
        return {"rate": float("nan"), "n_fit": int(ok.sum()), "L_floor_s": L_floor}
    Lv, dv = L[ok], d[ok]
    sel = Lv >= tail * Lv.max()
    if sel.sum() < 2:
        sel = np.zeros_like(sel); sel[-2:] = True
    slope = np.polyfit(Lv[sel] * dt, np.log(dv[sel]), 1)[0]
    return {"rate": float(-slope), "n_fit": int(sel.sum()), "L_fit_s": [float(Lv[sel].min() * dt), float(Lv[sel].max() * dt)],
            "L_floor_s": L_floor}
