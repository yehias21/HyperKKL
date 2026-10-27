"""Stage 1 diagnostics (CPU only): coverage, input spectra, empirical observability Gramian,
persistent excitation, error-vs-coverage. Writes s1.json next to this file.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python diag_s1.py
"""
import json
import math
import time

from common_qt import *  # noqa
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

torch.set_num_threads(8)
torch.set_default_dtype(torch.float64)
OUT = HERE / "s1.json"
TIERS = ("id", "shape_ood", "sequence_ood")
NODES = ("C3-partial", "C3-mismatch30")
BG = HERE.parent / "beat_greybox" / "runs"
TAU_SLOW = {"quad_tank": 49.0, "vdp": 3.0, "battery": 1.0}   # config-stated median 1/|Re eig| (s / s / min)
R = {}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load(node, system, seed=0):
    cfg = run_cfg(node, system, seed)
    sc = Dm.system_scales(cfg, DEV)
    d = Dm.training_data(cfg, seed, DEV)
    ev = {t: Dm.eval_data(cfg, t, device=DEV) for t in TIERS}
    return cfg, sc, d, ev


# ------------------------------------------------------------------ (a) coverage
def coverage(cfg, sc, d, ev):
    box = np.asarray(cfg["system"]["ic_box"], float)
    xs = np.asarray(sc["x_std"])
    sets = {"train": d["train"], **ev}
    out = {}
    for name, es in sets.items():
        x = es.x.double().numpy()
        k0 = es.burn if name == "train" else int(0.5 * (x.shape[1] - 1))
        xa = x[:, ::10]                                   # whole episode
        xw = x[:, k0::10]                                 # labelled (train) / scored (eval) window
        row = {}
        for j in range(x.shape[-1]):
            a, w = xa[..., j], xw[..., j]
            within = w.std(1).mean()                      # mean within-episode std in the window
            row[f"x{j+1}"] = {
                "std_all": float(a.std()), "std_window": float(w.std()),
                "within_ep_std_window": float(within), "within_over_xstd": float(within / xs[j]),
                "range_window": [float(w.min()), float(w.max())],
                "p1_p99_window": [float(np.percentile(w, 1)), float(np.percentile(w, 99))],
                "box_coverage_window": float((min(w.max(), box[j, 1]) - max(w.min(), box[j, 0]))
                                             / (box[j, 1] - box[j, 0])),
                "box_coverage_p1p99": float(max(0.0, min(np.percentile(w, 99), box[j, 1]) - max(np.percentile(w, 1), box[j, 0]))
                                            / (box[j, 1] - box[j, 0])),
                "ep_range_window_median": float(np.median(w.max(1) - w.min(1))),
            }
        out[name] = row
    return out


def knn_coverage(sc, d, ev, results_bg=None):
    """Distance of eval (scored) states to the nearest TRAINING (labelled) state, standardised x.
    Reference: held-out validation episodes (same distribution as training)."""
    xs = np.asarray(sc["x_std"])
    tr = d["train"].x.double().numpy()[:, d["train"].burn::20].reshape(-1, len(xs)) / xs
    tree = cKDTree(tr)
    va = d["val"].x.double().numpy()
    k0v = int(0.5 * (va.shape[1] - 1))
    ref = tree.query(va[:, k0v::20].reshape(-1, len(xs)) / xs)[0]
    thr = float(np.percentile(ref, 99))
    out = {"val_nn_median": float(np.median(ref)), "val_nn_p99": thr}
    per_ep = {}
    for t, es in ev.items():
        x = es.x.double().numpy()
        k0 = int(0.5 * (x.shape[1] - 1))
        q = x[:, k0::20] / xs
        dist = tree.query(q.reshape(-1, len(xs)))[0].reshape(q.shape[0], q.shape[1])
        out[t] = {"nn_median": float(np.median(dist)), "nn_p99": float(np.percentile(dist, 99)),
                  "frac_beyond_val_p99": float((dist > thr).mean())}
        per_ep[t] = dist.mean(1)
    # same for the unmeasured pair only (x3, x4)
    tr34 = tr[:, 2:4]
    tree34 = cKDTree(tr34)
    ref34 = tree34.query(va[:, k0v::20].reshape(-1, len(xs))[:, 2:4] / xs[2:4])[0]
    thr34 = float(np.percentile(ref34, 99))
    for t, es in ev.items():
        x = es.x.double().numpy()
        k0 = int(0.5 * (x.shape[1] - 1))
        q = (x[:, k0::20].reshape(-1, len(xs)) / xs)[:, 2:4]
        dd = tree34.query(q)[0]
        out[t]["x34_frac_beyond_val_p99"] = float((dd > thr34).mean())
    return out, per_ep


# ------------------------------------------------------------------ (b) spectra
def spectra(cfg, d, ev, tau_slow):
    dt = float(d["train"].dt)
    alpha = float(cfg["system"]["alpha"])
    wc = 1.0 / tau_slow
    sets = {"train": d["train"], **ev}
    out = {"w_slow_corner_rad_s": wc, "alpha_rad_s": alpha}
    for name, es in sets.items():
        u = es.u[..., 0].double().numpy()
        E, N1 = u.shape
        out["episode_T_s"] = (N1 - 1) * dt
        out["fft_resolution_rad_s"] = 2 * math.pi / ((N1 - 1) * dt)
        nfft = 8 * N1
        U = np.fft.rfft(u, n=nfft, axis=1)
        w = 2 * math.pi * np.fft.rfftfreq(nfft, dt)
        P = np.abs(U) ** 2
        tot = P.sum(1)
        uc = u - u.mean(1, keepdims=True)
        Pc = np.abs(np.fft.rfft(uc, n=nfft, axis=1)) ** 2
        totc = Pc.sum(1)
        fam = np.asarray(es.families)
        row = {"dc_share_of_energy": float((u.mean(1) ** 2 * N1).sum() / (u ** 2).sum()),
               "frac_energy_below_wc": float(P[:, w <= wc].sum() / tot.sum()),
               "frac_energy_below_alpha": float(P[:, w <= alpha].sum() / tot.sum()),
               "frac_fluct_energy_below_wc": float(Pc[:, w <= wc].sum() / totc.sum()),
               "frac_fluct_energy_below_alpha": float(Pc[:, w <= alpha].sum() / totc.sum()),
               "u_rms": float(np.sqrt((u ** 2).mean())), "u_fluct_rms": float(np.sqrt((uc ** 2).mean()))}
        # time-domain: variance of u passed through the slow plant-mode low-pass 1/(tau s + 1),
        # mean removed per episode, relative to the fluctuation variance
        a = math.exp(-dt / tau_slow)
        from scipy.signal import lfilter
        ul = lfilter([1 - a], [1, -a], uc, axis=1)
        row["lp_tau_slow_var_over_fluct_var"] = float(ul[:, N1 // 2:].var() / max(uc.var(), 1e-12))
        per_fam = {}
        for f in sorted(set(fam)):
            m = fam == f
            per_fam[f] = {"frac_fluct_below_wc": float(Pc[m][:, w <= wc].sum() / max(totc[m].sum(), 1e-12)),
                          "frac_fluct_below_alpha": float(Pc[m][:, w <= alpha].sum() / max(totc[m].sum(), 1e-12)),
                          "fluct_rms": float(np.sqrt((uc[m] ** 2).mean())), "n": int(m.sum())}
        row["per_family"] = per_fam
        out[name] = row
    return out


# ------------------------------------------------------------------ (c) empirical observability Gramian
def rk4(f, x, u, h):
    k1 = f(x, u); k2 = f(x + h / 2 * k1, u); k3 = f(x + h / 2 * k2, u); k4 = f(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def gramian(cfg, sc, es, H_list, n_ep=48, starts=(0.35, 0.6), eps=1e-3, seed=0):
    """Local empirical observability Gramian along training trajectories (true plant of the run).
    Coordinates: x in units of x_std, y in units of the sensor-noise std (noise * y_std), so
    I0 = sum_k S_k^T S_k is the Fisher information of x(t0) from the samples y_{t0..t0+H}
    (no process noise, no prior). I_end = Phi^-T I0 Phi^-1 is the information on x(t0+H)
    (the filtering end of the window). CRB std in x_std units = sqrt(diag(I^-1))."""
    scfg = cfg["system"]
    plant = build_plant(scfg, perturb=float(scfg.get("mismatch", 0.0)))
    dt = es.dt
    xs = torch.as_tensor(sc["x_std"])
    sy = float(cfg["data"]["sensor"]["noise"]) * torch.as_tensor(sc["y_std"])
    rng = np.random.RandomState(seed)
    eps_ = rng.choice(len(es), n_ep, replace=False)
    N1 = es.x.shape[1]
    Hmax = max(H_list)
    Kmax = int(round(Hmax / dt))
    base, ublk = [], []
    for e in eps_:
        for s in starts:
            k0 = int(s * (N1 - 1))
            if k0 + Kmax >= N1:
                k0 = N1 - 1 - Kmax
            if k0 < 0:
                continue
            base.append(es.x[e, k0].double())
            ublk.append(es.u[e, k0:k0 + Kmax + 1, 0].double())
    x0 = torch.stack(base)
    U = torch.stack(ublk)                                # (n, Kmax+1)
    n, nx = x0.shape
    P = [x0]
    for i in range(nx):
        for sg in (1, -1):
            dx = torch.zeros(nx); dx[i] = sg * eps * xs[i]
            P.append(x0 + dx)
    X = torch.cat(P)
    Ufull = U.repeat(2 * nx + 1, 1)
    Ks = sorted(int(round(Hh / dt)) for Hh in H_list)
    I = torch.zeros(n, nx, nx)
    res = {}

    def sens(X):
        Y = plant.h(X) / sy
        Yp = Y[n:].view(nx, 2, n, -1)
        return ((Yp[:, 0] - Yp[:, 1]) / (2 * eps)).permute(1, 2, 0)    # (n, ny, nx) dY/dx0 (x_std units)

    S = sens(X)
    I += S.transpose(1, 2) @ S
    for k in range(1, Kmax + 1):
        X = rk4(plant.f, X, Ufull[:, k - 1:k], dt)
        S = sens(X)
        I += S.transpose(1, 2) @ S
        if k in Ks:
            Xp = X[n:].view(nx, 2, n, nx)
            Phi = ((Xp[:, 0] - Xp[:, 1]) / (2 * eps)).permute(1, 2, 0) / xs[None, :, None]   # (n, nx, nx)
            Pinv = torch.linalg.pinv(Phi)
            Iend = Pinv.transpose(1, 2) @ I @ Pinv
            row = {}
            for nm, M in (("x0", I), ("end", Iend)):
                ev_, V = torch.linalg.eigh(0.5 * (M + M.transpose(1, 2)))
                lam_min, lam_max = ev_[:, 0].clamp(min=1e-300), ev_[:, -1]
                crb = torch.sqrt(torch.diagonal(torch.linalg.pinv(M), dim1=1, dim2=2).clamp(min=0))
                vmin = V[:, :, 0].abs()
                row[nm] = {"lam_min_median": float(lam_min.median()), "lam_max_median": float(lam_max.median()),
                           "cond_median": float((lam_max / lam_min).median()),
                           "cond_q90": float(torch.quantile(lam_max / lam_min, 0.9)),
                           "crb_std_xstd_median": [float(v) for v in crb.median(0).values],
                           "crb_rms_xstd_median": float(crb.pow(2).mean(1).sqrt().median()),
                           "weakest_dir_abs_loading_mean": [float(v) for v in vmin.mean(0)]}
            res[f"H={k * dt:g}"] = row
    return res


# ------------------------------------------------------------------ (d) persistent excitation
def lp_bank(u, dt, taus):
    from scipy.signal import lfilter
    outs = []
    for tau in taus:
        a = math.exp(-dt / tau)
        outs.append(lfilter([1 - a], [1, -a], u, axis=1))
    return np.stack(outs, -1)


def pe(cfg, sc, d, ev, tau_slow):
    dt = d["train"].dt
    alpha = float(cfg["system"]["alpha"])
    taus = [(1 / alpha) * 1.5 ** j for j in range(6)]     # SFM-like bank (init poles alpha .. alpha/1.5^5)
    xs = np.asarray(sc["x_std"])
    sets = {"train": d["train"], **ev}
    feats, xsamp = {}, {}
    for name, es in sets.items():
        k0 = es.burn if name == "train" else int(0.5 * (es.x.shape[1] - 1))
        sl = slice(k0, None, 25)
        z = es.z_obs.double().numpy()[:, sl]
        c = lp_bank(es.u[..., 0].double().numpy(), dt, taus)[:, sl]
        feats[name] = np.concatenate([z, c], -1).reshape(-1, z.shape[-1] + c.shape[-1])
        xsamp[name] = es.x.double().numpy()[:, sl].reshape(-1, len(xs)) / xs
    mu, sd = feats["train"].mean(0), feats["train"].std(0) + 1e-12
    out = {"feature_note": f"[z_obs ({d['train'].z.shape[-1]}), u low-passed at tau = {[round(t, 1) for t in taus]} s], "
                           "standardised with training stats; train samples k >= burn, eval k >= settle"}
    nzf = d["train"].z.shape[-1]
    for name, F in feats.items():
        G = (F - mu) / sd
        C = np.cov(G.T)
        lam = np.linalg.eigvalsh(C)
        Cz = np.cov(G[:, :nzf].T)
        lz = np.linalg.eigvalsh(Cz)
        out[name] = {"eig_zc": [float(v) for v in lam], "cond_zc": float(lam[-1] / max(lam[0], 1e-300)),
                     "eig_z": [float(v) for v in lz], "cond_z": float(lz[-1] / max(lz[0], 1e-300))}
    # unmeasured-state excitation: how much of (x3, x4) variance is NOT linearly explained by the
    # measured (x1, x2) + input filters (train) -- the "independent" excitation the map has to learn
    for name in sets:
        X = xsamp[name]
        F = (feats[name] - mu) / sd
        cpart = F[:, nzf:]
        A = np.concatenate([X[:, :2], cpart, np.ones((len(X), 1))], 1)
        res = {}
        for j in range(2, len(xs)):
            beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
            r = X[:, j] - A @ beta
            res[f"x{j+1}"] = {"var_xstd2": float(X[:, j].var()), "resid_var_given_x12_c": float(r.var()),
                              "frac_unexplained": float(r.var() / max(X[:, j].var(), 1e-12))}
        out[name]["unmeasured_independent_var"] = res
    # input Hankel PE over a window of 3 tau_slow
    L = 15
    s = max(1, int(round(3 * tau_slow / dt / L)))
    out["hankel"] = {"L": L, "spacing_samples": s, "span_s": L * s * dt}
    for name, es in sets.items():
        u = es.u[..., 0].double().numpy()
        k0 = es.burn if name == "train" else 0
        rows = []
        for k in range(k0 + (L - 1) * s, u.shape[1], max(1, s // 2)):
            rows.append(u[:, k - np.arange(L) * s])
        Hk = np.concatenate(rows, 0)
        for tag, M in (("raw", Hk), ("demeaned", Hk - Hk.mean(1, keepdims=True))):
            Rm = M.T @ M / len(M)
            lam = np.linalg.eigvalsh(Rm)
            out["hankel"][f"{name}_{tag}"] = {"lam_min_over_max": float(lam[0] / lam[-1]),
                                              "lam_min_over_trace": float(lam[0] / lam.sum()),
                                              "eig_desc_over_trace": [float(v) for v in (lam[::-1] / lam.sum())[:6]]}
    return out


# ------------------------------------------------------------------ errors per state / vs coverage
def errors(node, system, per_ep_cov):
    out = {}
    for t in TIERS:
        rows = {}
        for m in ("lora", "greybox_ekf", "greybox_ukf", "greybox_ekf_oracle", "ekf"):
            per = {f"x{j}": [] for j in range(1, 5)}
            tot = []
            for s in range(3):
                r = c3_results(node, system, s)[t]["clean"]["methods"].get(m)
                if r is None:
                    continue
                for j in range(1, 5):
                    per[f"x{j}"].append(np.mean(r[f"nrmse_x{j}"]))
                tot.append(np.mean(r["nrmse"]))
            if tot:
                rows[m] = {"nrmse": float(np.mean(tot)), **{k: float(np.mean(v)) for k, v in per.items()}}
        # per-episode: campaign A2 (seed 0) vs best grey-box (seed 0) against eval-episode coverage
        a2 = BG / node / system / "seed0" / "A2.json"
        if a2.exists():
            ea = np.asarray(json.load(open(a2))["results"][t]["nrmse"])
            gb = np.asarray(c3_results(node, system, 0)[t]["clean"]["methods"]["greybox_ekf"]["nrmse"])
            lo = np.asarray(c3_results(node, system, 0)[t]["clean"]["methods"]["lora"]["nrmse"])
            cov = per_ep_cov[t]
            rows["A2_seed0_nrmse"] = float(ea.mean())
            rows["spearman_A2err_vs_coverage_dist"] = float(spearmanr(ea, cov)[0])
            rows["spearman_A2_over_gb_vs_coverage_dist"] = float(spearmanr(ea / gb, cov)[0])
            rows["spearman_lora_vs_coverage_dist"] = float(spearmanr(lo, cov)[0])
            rows["spearman_gb_vs_coverage_dist"] = float(spearmanr(gb, cov)[0])
        out[t] = rows
    return out


def main():
    R["meta"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tau_slow": TAU_SLOW}
    # linearisation of the true quad_tank at its nominal zero-input equilibrium
    cfg = run_cfg("C3-partial", "quad_tank", 0)
    pl = build_plant(cfg["system"])
    from torch.func import jacrev
    xe = torch.tensor((11.66, 10.9, 5.36, 3.3), dtype=torch.float64)[None]
    for _ in range(20000):                                 # settle onto the true equilibrium
        xe = rk4(pl.f, xe, torch.zeros(1, 1), 0.5)
    J = jacrev(lambda x: pl.f(x[None], torch.zeros(1, 1))[0])(xe[0])
    lam = np.linalg.eigvals(J.numpy())
    C = np.zeros((2, 4)); C[0, 0] = C[1, 1] = 1
    Ob = np.concatenate([C @ np.linalg.matrix_power(J.numpy(), k) for k in range(4)])
    so = np.linalg.svd(Ob, compute_uv=False)
    R["quad_tank_linear"] = {"x_eq": xe[0].tolist(), "eig": sorted(lam.real.tolist()),
                             "time_constants_s": sorted((-1 / lam.real).tolist()),
                             "obs_matrix_sv": so.tolist()}
    log("linear", R["quad_tank_linear"])
    for node in NODES:
        cfg, sc, d, ev = load(node, "quad_tank")
        key = f"{node}/quad_tank"
        R[key] = {"x_std": sc["x_std"], "burn": d["train"].burn, "families_train": sorted(set(d["train"].families))}
        R[key]["coverage"] = coverage(cfg, sc, d, ev); log(key, "coverage done")
        kc, per_ep = knn_coverage(sc, d, ev)
        R[key]["knn"] = kc; log(key, "knn", kc)
        R[key]["spectra"] = spectra(cfg, d, ev, TAU_SLOW["quad_tank"]); log(key, "spectra done")
        R[key]["pe"] = pe(cfg, sc, d, ev, TAU_SLOW["quad_tank"]); log(key, "pe done")
        R[key]["errors"] = errors(node, "quad_tank", per_ep); log(key, "errors done")
        if node == "C3-partial" or True:
            a = float(cfg["system"]["alpha"])
            H = [1 / a, 3 / a, TAU_SLOW["quad_tank"], 2 * TAU_SLOW["quad_tank"], 3 * TAU_SLOW["quad_tank"]]
            R[key]["gramian"] = gramian(cfg, sc, d["train"], H); log(key, "gramian done")
        json.dump(R, open(OUT, "w"), indent=1)
    for system in ("vdp", "battery"):
        cfg, sc, d, ev = load("C3-partial", system)
        key = f"C3-partial/{system}"
        a = float(cfg["system"]["alpha"])
        ts = TAU_SLOW[system]
        H = [1 / a, 3 / a, ts, 2 * ts, 3 * ts]
        R[key] = {"x_std": sc["x_std"], "dt": d["train"].dt, "T": d["train"].dt * (d["train"].x.shape[1] - 1)}
        R[key]["gramian"] = gramian(cfg, sc, d["train"], H); log(key, "gramian done")
        R[key]["spectra"] = {k: v for k, v in spectra(cfg, d, ev, ts).items() if k != "per_family"}
        R[key]["pe"] = pe(cfg, sc, d, ev, ts); log(key, "pe done")
        json.dump(R, open(OUT, "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
