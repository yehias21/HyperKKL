"""Stage 1 follow-up (CPU): feature-space excitation of eval relative to training (generalised
eigenvalues, Mahalanobis coverage), where x3/x4 information sits in z, Hankel PE with the
episode mean removed. Appends to s1b.json."""
import json
import math

from common_qt import *  # noqa
from diag_s1 import lp_bank, load, TIERS, TAU_SLOW
from scipy.linalg import eigh

torch.set_default_dtype(torch.float64)
OUT = HERE / "s1b.json"
R = {}


def feats(es, k0, dt, alpha, stride=25):
    taus = [(1 / alpha) * 1.5 ** j for j in range(6)]
    z = es.z_obs.double().numpy()[:, k0::stride]
    c = lp_bank(es.u[..., 0].double().numpy(), dt, taus)[:, k0::stride]
    x = es.x.double().numpy()[:, k0::stride]
    nz = z.shape[-1]
    return z.reshape(-1, nz), c.reshape(-1, c.shape[-1]), x.reshape(-1, x.shape[-1])


for node, system in (("C3-partial", "quad_tank"), ("C3-mismatch30", "quad_tank"), ("C3-partial", "vdp"),
                     ("C3-partial", "battery")):
    cfg, sc, d, ev = load(node, system)
    dt, alpha = d["train"].dt, float(cfg["system"]["alpha"])
    xs = np.asarray(sc["x_std"])
    Z, C, X = feats(d["train"], d["train"].burn, dt, alpha)
    out = {}
    for tag, F in (("z", Z), ("zc", np.concatenate([Z, C], 1))):
        mu, Ct = F.mean(0), np.cov(F.T)
        L = np.linalg.cholesky(Ct + 1e-12 * np.trace(Ct) * np.eye(len(Ct)))
        Li = np.linalg.inv(L)
        mt = ((F - mu) @ Li.T)
        r_tr = np.sqrt((mt ** 2).sum(1))
        p99 = float(np.percentile(r_tr, 99))
        row = {"train_maha_p99": p99}
        for t, es in ev.items():
            Ze, Ce, _ = feats(es, int(0.5 * (es.x.shape[1] - 1)), dt, alpha)
            Fe = Ze if tag == "z" else np.concatenate([Ze, Ce], 1)
            g = eigh(np.cov(Fe.T) + np.outer(Fe.mean(0) - mu, Fe.mean(0) - mu), Ct, eigvals_only=True)
            me = ((Fe - mu) @ Li.T)
            r = np.sqrt((me ** 2).sum(1))
            # per train-eigendirection: eval second moment / train variance
            lam, V = np.linalg.eigh(Ct)
            pe_ = ((Fe - mu) @ V) ** 2
            ratio = pe_.mean(0) / lam
            row[t] = {"gen_eig_eval_over_train_desc": [float(v) for v in g[::-1]],
                      "frac_maha_beyond_train_p99": float((r > p99).mean()),
                      "eval_over_train_by_train_eigdir_smallest_first": [float(v) for v in ratio]}
        out[tag] = row
    # where does the x3/x4 information sit in z? R^2 of x_j on the train-eigencomponents of z
    if system == "quad_tank":
        mu, Ct = Z.mean(0), np.cov(Z.T)
        lam, V = np.linalg.eigh(Ct)
        P = (Z - mu) @ V
        info = {"z_eig_rel": [float(v / lam[-1]) for v in lam]}
        for j in range(4):
            y = (X[:, j] - X[:, j].mean()) / xs[j]
            corr2 = [float(np.corrcoef(P[:, i], y)[0, 1] ** 2) for i in range(P.shape[1])]
            # R^2 of a linear fit using only the 3 weakest z directions vs only the 3 strongest
            A_w = np.c_[P[:, :3], np.ones(len(P))]
            A_s = np.c_[P[:, 3:], np.ones(len(P))]
            r2 = lambda A: 1 - np.var(y - A @ np.linalg.lstsq(A, y, rcond=None)[0]) / np.var(y)
            info[f"x{j+1}"] = {"corr2_by_dir_smallest_first": corr2, "R2_weak3": float(r2(A_w)),
                               "R2_strong3": float(r2(A_s)), "R2_all": float(r2(np.c_[P, np.ones(len(P))]))}
        # local slope: how many x_std does x3 move per unit (train-std) motion along the weakest dirs
        out["z_info"] = info
    # Hankel PE of the input with the per-episode mean removed (span 3 tau_slow)
    Lh = 15
    s = max(1, int(round(3 * TAU_SLOW[system] / dt / Lh)))
    hk = {"L": Lh, "spacing": s, "span_s": Lh * s * dt}
    for t, es in {"train": d["train"], **ev}.items():
        u = es.u[..., 0].double().numpy()
        u = u - u.mean(1, keepdims=True)
        k0 = es.burn if t == "train" else 0
        rows = [u[:, k - np.arange(Lh) * s] for k in range(k0 + (Lh - 1) * s, u.shape[1], max(1, s // 2))]
        if not rows:
            continue
        Hm = np.concatenate(rows, 0)
        lamh = np.linalg.eigvalsh(Hm.T @ Hm / len(Hm))
        hk[t] = {"lam_min_over_max": float(lamh[0] / lamh[-1]), "lam_min_over_trace": float(lamh[0] / lamh.sum()),
                 "eig_desc_over_trace": [float(v) for v in (lamh[::-1] / lamh.sum())]}
    out["hankel_demeaned_episode"] = hk
    R[f"{node}/{system}"] = out
    print(node, system, "done", flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
