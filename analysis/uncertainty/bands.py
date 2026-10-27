"""Stage 2 of the uncertainty study (experiments/uncertainty/PREREG.md): from the stage-1 arrays of
the 3 seeds of one (node, system), build the bands U1, U2, U2-mf, U3 (deployable ACI), U3-oracle,
U4, U4c, EKF / UKF / ekf_oracle covariance bands and EKF-c, and score them on every test block.
Pure numpy + a small torch MLP on CPU.

usage: bands.py NODE SYSTEM NPZDIR OUTDIR
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ALPHA, Z90, GAMMA, CAP = 0.10, 1.6448536269514722, 0.005, 3.0
ETA = 0.05            # exploratory (A3): step of the log-scale quantile tracker


def conf_level(n):
    return min(1.0, (1 - ALPHA) * (1 + 1.0 / n))


def qfun(sorted_scores, n_ep):
    """Quantile function of pooled calibration scores (equal episode weights, equal lengths):
    Q(p) = the smallest q with (1/(n+1)) sum_e F_e(q) >= p, i.e. the pooled quantile at p (1 + 1/n).
    p > n/(n+1) (pooled level > 1) -> inf."""
    M = len(sorted_scores)

    def Q(p):
        p = np.asarray(p, float)
        lev = p * (1 + 1.0 / n_ep)
        idx = np.ceil(lev * M).astype(int) - 1
        out = sorted_scores[np.clip(idx, 0, M - 1)]
        out = np.where(lev > 1.0, np.inf, out)
        return np.where(p <= 0, 0.0, out)
    return Q


def pooled_q(scores, n_ep):                          # scores (E, T) -> q at the U1 level
    s = np.sort(scores.reshape(-1))
    return float(qfun(s, n_ep)(1 - ALPHA))


# ------------------------------------------------------------------ scale model (U2)
def feats(D, pre, use_gram=True):
    f = [np.log(D[f"{pre}/lora_rema"] + 1e-6), np.log(D[f"{pre}/lora_ctxd"][..., None] + 1e-6),
         np.log(D[f"{pre}/lora_jac"] + 1e-6)]
    if use_gram:
        f.append(np.log(D[f"{pre}/lora_gram"] + 1e-6))
    return np.concatenate(f, -1)


def fit_scale(F, E, seed):
    """MLP log|e| ~ features (PREREG: 2 x 64 SiLU, Adam 1e-3, 3000 steps, wd 1e-4)."""
    torch.manual_seed(seed)
    dv = "cuda" if torch.cuda.is_available() else "cpu"
    F = torch.as_tensor(F.reshape(-1, F.shape[-1]), dtype=torch.float32, device=dv)
    T = torch.as_tensor(np.log(np.abs(E).reshape(-1, E.shape[-1]) + 1e-3), dtype=torch.float32, device=dv)
    ok = torch.isfinite(F).all(1) & torch.isfinite(T).all(1)
    F, T = F[ok], T[ok]
    mu, sd = F.mean(0), F.std(0).clamp(min=1e-6)
    net = nn.Sequential(nn.Linear(F.shape[1], 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                        nn.Linear(64, T.shape[1])).to(dv)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(3000):
        i = torch.randint(0, len(F), (4096,), device=dv)
        loss = (net((F[i] - mu) / sd) - T[i]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def predict(G):
        sh = G.shape
        G = torch.as_tensor(G.reshape(-1, sh[-1]), dtype=torch.float32, device=dv)
        G = torch.nan_to_num((G - mu) / sd, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        return np.exp(net(G).cpu().numpy().astype(np.float64)).reshape(*sh[:-1], -1)
    return predict


# ------------------------------------------------------------------ ACI
def aci(scale, Qj, miss_fn, n_states):
    """Generic ACI over stored samples of each episode. scale (E, T, nx); Qj: list of per-state
    quantile functions; miss_fn(t, w) -> err (E,) or (E, nx). Returns w (E, T, nx), alpha path."""
    E, T, nx = scale.shape
    shared = None
    W = np.zeros((E, T, nx))
    a = np.full((E, nx), ALPHA)
    A = np.zeros((E, T, nx))
    for t in range(T):
        lev = 1 - a
        w = np.stack([Qj[j](lev[:, j]) for j in range(nx)], -1) * scale[:, t]
        W[:, t] = w
        A[:, t] = a
        err = miss_fn(t, w, a)
        if err.ndim == 1:
            err = err[:, None].repeat(nx, 1)
        a = a + GAMMA * (ALPHA - err)
    return W, A


def qtrack(w0, miss_fn):
    """Exploratory (PREREG A3): multiplicative quantile tracking (Angelopoulos, Candes & Tibshirani 2023,
    conformal PID 'quantile tracking' on log scale): w(t) = exp(theta_t) w0(t),
    theta_{t+1} = theta_t + ETA (err_t - alpha). Always finite. miss_fn(t, w, theta) -> (E,) or (E, nx)."""
    E, T, nx = w0.shape
    th = np.zeros((E, nx))
    W = np.zeros((E, T, nx))
    for t in range(T):
        w = np.exp(th) * w0[:, t]
        W[:, t] = w
        err = miss_fn(t, w, th)
        if err.ndim == 1:
            err = err[:, None].repeat(nx, 1)
        th = th + ETA * (err - ALPHA)
    return W


# ------------------------------------------------------------------ metrics
def spearman(a, b):
    from scipy.stats import spearmanr
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def score(err, w, cal_max, rng, hi_ep):
    """err, w: (E, T) normalised error and half-width (inf allowed). Returns metrics dict."""
    cov = np.abs(err) <= w
    capw = np.minimum(w, CAP * cal_max)
    width = 2 * capw
    dist = np.maximum(np.abs(err) - capw, 0) * (~np.isinf(w))
    IS = width + (2 / ALPHA) * dist
    sub = rng.choice(err.size, min(err.size, 20000), replace=False)
    wf, ef = w.reshape(-1)[sub], np.abs(err).reshape(-1)[sub]
    fin = np.isfinite(wf)
    return {"coverage": float(cov.mean()), "width": float(width.mean()), "width_med": float(np.median(width)),
            "rmse": float(np.sqrt(np.mean(np.minimum(err ** 2, 1e6)))), "IS": float(IS.mean()),
            "cov_hi": float(cov[hi_ep].mean()), "rho_w_e": spearman(wf[fin], ef[fin]) if fin.sum() > 10 else float("nan"),
            "cap_frac": float(np.isinf(w).mean()), "ep_cov_min": float(cov.mean(1).min()),
            "ep_cov_sd": float(cov.mean(1).std())}


def hi_episodes(errE):                              # (E, T, nx) -> top-20% episodes by NRMSE
    nr = np.sqrt(np.nanmean(errE ** 2, (1, 2)))
    return nr >= np.quantile(nr, 0.8)


def main(node, system, npzdir, outdir):
    files = sorted(Path(npzdir).glob(f"{node}_{system}_s*.npz"))
    runs = {}
    for f in files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        runs[meta["seed"]] = ({k: z[k] for k in z.files if k != "meta"}, meta)
    seeds = sorted(runs)
    meta0 = runs[seeds[0]][1]
    sig_x = np.asarray(meta0["x_std"])
    nx, sensors = meta0["nx"], meta0["sensors"]
    blocks = [s for s in meta0["sets"] if s not in ("fit", "cal")]
    rows, extra = [], {"node": node, "system": system, "seeds": seeds, "checks": {}}
    rng = np.random.default_rng(0)

    # ensemble (U4) availability: identical episodes across seeds
    ens_ok = len(seeds) >= 2 and all(
        np.allclose(runs[s][0][f"{b}/x"], runs[seeds[0]][0][f"{b}/x"], atol=1e-3, rtol=1e-3)
        for s in seeds[1:] for b in ["cal"] + blocks)
    extra["checks"]["ensemble_same_episodes"] = bool(ens_ok)

    def add(seed, blk, method, err, w, cal_max, hi, extra_m=None):
        tier, cond = blk.split("|")
        for j in range(nx):
            m = score(err[..., j], w[..., j], cal_max[j], rng, hi)
            m.update({"node": node, "system": system, "seed": seed, "tier": tier, "cond": cond,
                      "method": method, "state": j, "measured": j in sensors})
            if extra_m:
                m.update(extra_m)
            rows.append(m)

    for seed in seeds:
        D, meta = runs[seed]
        n_cal = D["cal/x"].shape[0]
        E = {b: (D[f"{b}/x"] - D[f"{b}/lora_xh"]) / sig_x for b in ["fit", "cal"] + blocks}
        # reproduction check: settled NRMSE of the stored clean blocks vs the run's results.json
        extra["checks"][f"s{seed}_nrmse_lora"] = {
            t: [float(np.mean(np.sqrt(np.mean(E[f"{t}|clean"] ** 2, (1, 2))))), meta["stored_nrmse_lora"][t]]
            for t in ("id", "shape_ood", "sequence_ood")}
        # ---- U1
        q1 = np.array([pooled_q(np.abs(E["cal"][..., j]), n_cal) for j in range(nx)])
        cmax1 = np.abs(E["cal"]).max((0, 1))
        # ---- U2 / U2-mf
        sc = {}
        for nm, ug in (("U2", True), ("U2mf", False)):
            pred = fit_scale(feats(D, "fit", ug), E["fit"], seed)
            s_cal = pred(feats(D, "cal", ug))
            S = np.abs(E["cal"]) / s_cal
            qs = np.array([pooled_q(S[..., j], n_cal) for j in range(nx)])
            Qj = [qfun(np.sort(S[..., j].reshape(-1)), n_cal) for j in range(nx)]
            sc[nm] = (pred, qs, Qj, S.max((0, 1)), ug)
        # ---- U3 proxy: constant-scale conformal band on the output residual (max over outputs)
        r_cal = D["cal/lora_res"]
        r_sd = r_cal.reshape(-1, r_cal.shape[-1]).std(0) + 1e-9
        rho_cal = np.sort((np.abs(r_cal) / r_sd).max(-1).reshape(-1))
        Qr = qfun(rho_cal, n_cal)
        # ---- EKF-c (conformalised filter std), per filter
        fcal = {}
        for fn in ("ekf", "ukf", "ekf_oracle"):
            if f"cal/{fn}_xh" in D:
                ef = (D["cal/x"] - D[f"cal/{fn}_xh"]) / sig_x
                sf = D[f"cal/{fn}_pd"] / sig_x + 1e-12
                fcal[fn] = (np.array([pooled_q(np.abs(ef[..., j]) / sf[..., j], n_cal) for j in range(nx)]),
                            (np.abs(ef) / sf).max((0, 1)))
        extra["checks"][f"s{seed}_q"] = {"U1": q1.tolist(), "U2": sc["U2"][1].tolist()}

        for b in blocks:
            err = E[b]
            hi = hi_episodes(err)
            add(seed, b, "U1", err, np.broadcast_to(q1, err.shape), cmax1, hi)
            for nm, (pred, qs, Qj, smax, ug) in sc.items():
                s_t = pred(feats(D, b, ug))
                add(seed, b, nm, err, qs * s_t, qs * np.nanmax(s_t, (0, 1)), hi)
                if nm != "U2":
                    continue
                # U3 deployable: shared alpha_t, feedback = output-residual miscoverage
                rho = (np.abs(D[f"{b}/lora_res"]) / r_sd).max(-1)        # (E, T)
                W3, A3 = aci(s_t, Qj, lambda t, w, a: (rho[:, t] > Qr(1 - a[:, 0])).astype(float), nx)
                capref = np.array([Qj[j](np.array([1 - ALPHA]))[0] for j in range(nx)]) * np.nanmax(s_t, (0, 1))
                add(seed, b, "U3", err, W3, capref, hi, {"alpha_mean": float(A3.mean())})
                # U3-oracle: per-state alpha_t, true miss feedback
                W3o, A3o = aci(s_t, Qj, lambda t, w, a: (np.abs(err[:, t]) > w).astype(float), nx)
                add(seed, b, "U3oracle", err, W3o, capref, hi, {"alpha_mean": float(A3o.mean())})
                # exploratory (A3): log-scale quantile tracking, deployable proxy and oracle feedback
                w0 = qs * s_t
                qr0 = float(Qr(np.array([1 - ALPHA]))[0])
                Wq = qtrack(w0, lambda t, w, th: (rho[:, t] > np.exp(th[:, 0]) * qr0).astype(float))
                add(seed, b, "U3qt", err, Wq, np.full(nx, np.inf), hi)
                Wqo = qtrack(w0, lambda t, w, th: (np.abs(err[:, t]) > w).astype(float))
                add(seed, b, "U3qt_oracle", err, Wqo, np.full(nx, np.inf), hi)
            for fn in ("ekf", "ukf", "ekf_oracle"):
                if f"{b}/{fn}_xh" not in D:
                    continue
                ef = (D[f"{b}/x"] - D[f"{b}/{fn}_xh"]) / sig_x
                sf = D[f"{b}/{fn}_pd"] / sig_x
                hif = hi_episodes(np.clip(ef, -1e3, 1e3))
                nees = D[f"{b}/{fn}_nees"] / nx
                ex = {"anees": float(np.mean(np.minimum(nees, 1e6))), "mnees": float(np.median(nees)),
                      "nrmse_centre": float(np.mean(np.sqrt(np.mean(np.clip(ef, -1e3, 1e3) ** 2, (1, 2)))))}
                add(seed, b, fn, ef, Z90 * sf, Z90 * np.nanmax(np.minimum(sf, 1e3), (0, 1)), hif, ex)
                if fn in fcal:
                    qf, _ = fcal[fn]
                    add(seed, b, fn + "-c", ef, qf * sf, qf * np.nanmax(np.minimum(sf, 1e3), (0, 1)), hif, ex)

    # ---- U4 / U4c: ensemble over seeds (centre = mean)
    if ens_ok:
        Dm = {b: np.stack([runs[s][0][f"{b}/lora_xh"] for s in seeds]) for b in ["cal"] + blocks}
        X = {b: runs[seeds[0]][0][f"{b}/x"] for b in ["cal"] + blocks}
        n_cal = X["cal"].shape[0]
        ec = (X["cal"] - Dm["cal"].mean(0)) / sig_x
        sdc = Dm["cal"].std(0) / sig_x + 1e-9
        q4 = np.array([pooled_q(np.abs(ec[..., j]) / sdc[..., j], n_cal) for j in range(nx)])
        for b in blocks:
            e4 = (X[b] - Dm[b].mean(0)) / sig_x
            sd4 = Dm[b].std(0) / sig_x + 1e-9
            hi = hi_episodes(e4)
            ex = {"nrmse_centre": float(np.mean(np.sqrt(np.mean(e4 ** 2, (1, 2)))))}
            add("ens", b, "U4", e4, Z90 * sd4, Z90 * sd4.max((0, 1)), hi, ex)
            add("ens", b, "U4c", e4, q4 * sd4, q4 * sd4.max((0, 1)), hi, ex)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{node}_{system}.json").write_text(json.dumps({"rows": rows, "extra": extra}))
    print(node, system, "rows", len(rows), json.dumps(extra["checks"])[:600])


if __name__ == "__main__":
    torch.set_num_threads(4)
    main(*sys.argv[1:5])
