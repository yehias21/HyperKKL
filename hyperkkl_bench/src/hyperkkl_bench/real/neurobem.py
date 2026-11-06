r"""Real-data track ``neurobem``: body velocity and tilt of an agile quadrotor from its own IMU and rotor speeds (Bauersfeld
et al., RSS 2021). No licence text on the source: fetched from the original host, never redistributed (card, RELEASE.md).

The measured channels are NOT in the providers' processed CSVs (a smoothed, fused state): they come from the raw Betaflight
logs, decoded by real/blackbox.py: gyroADC, accSmooth, and the bidirectional-DShot rotor eRPM telemetry (debug[0..3]).
Vicon gives the labels: body velocity and attitude. docs/cards/neurobem_datacheck.md has the evidence.

    u = rotor speeds - 1350 rad/s (4)      y = gyro (3), accelerometer (3)      hidden, scored: v_b (3), g_b (3)
    x = (v_b, g_b, w, a_m); w, a_m are the measured channels themselves (x[6:12] = y)

Every (t_log) sample is mapped to the merged clock by a per-flight (offset, drift) fit of the raw gyro to the published
smoothed gyro. On a 100 Hz grid: labels are interpolated from the 400 Hz CSV; u and y are CAUSAL boxcar means over
(t_k - 10 ms, t_k], so they never see the future (the CSV's smoothed `mot` and `ang vel` do, and are not used).

Episodes: 10-s windows (N = 1000) inside a segment (a Vicon-valid flying part), every 2.5 s (train) or 5 s (val / test), started
from a blind prior: v_b = 0, level. Score: settled (t >= 5 s) errors in m/s and deg; headline: hidden NRMSE, mean of test_id and
test_ood. Splits by flight, resources/real_neurobem.json. Model: plants/systems_real.NeuroBEMNominal, identified on train only.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .battery_pf import RealEpisodeSet
from .blackbox import read_bfl
from .registry import processed_dir, raw_dir, spec

NAME = "neurobem"
SCORER = "settled-quad-hidden-1"
HEADLINE = "nrmse_hidden"
METRIC_ABOUT = ("settled hidden NRMSE (t >= 5 s of 10-s windows; v_b and g_b normalised by their training std); headline = "
                "mean of test_id and test_ood; RMSEs in m/s and deg in each split's summary")
PREP_VERSION = "neurobem-prep-1"
HZ = 100
DT = 1.0 / HZ
N = 1000
STRIDE = {"train": 250, "val": 500, "test_id": 500, "test_ood": 500}
SPLITS = ["train", "val", "test_id", "test_ood"]
ERPM = 14 * math.pi / 30                      # debug value -> rad/s, the providers' constant (log_smoother.m)
G0 = 9.80665
STATE = ["vx", "vy", "vz", "gx", "gy", "gz", "wx", "wy", "wz", "ax", "ay", "az"]
HIDDEN = list(range(6))
SENSORS = list(range(6, 12))
HKKL = {"nz": 13, "alpha": 1.2, "replay": "real_windows"}
CSV_COLS = ["t", "ang vel x", "ang vel y", "ang vel z", "quat x", "quat y", "quat z", "quat w", "acc x", "acc y", "acc z",
            "vel x", "vel y", "vel z", "ang acc x", "ang acc y", "ang acc z"]


def splits():
    return list(SPLITS)


def condition_of(split):
    return "default"


# ------------------------------------------------------------------------------------- raw -> arrays
def _logs(root=None):
    return raw_dir(NAME, root) / "rawlogs" / "raw_data"


def _csvs(flight, root=None):
    d = raw_dir(NAME, root) / "processed" / "processed_data"
    return sorted(d.glob(f"merged_{flight}_seg_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def decode_log(path):
    """.BFL -> t [s], gyro [rad/s], acc [m/s^2], rotor speed [rad/s] at the log rate (about 1051 Hz)."""
    L = read_bfl(path)
    t = (L["time"] - L["time"][0]) / 1e6
    gy = np.deg2rad(np.stack([L[f"gyroADC[{i}]"] for i in range(3)], 1).astype(float))
    ac = np.stack([L[f"accSmooth[{i}]"] for i in range(3)], 1) * G0 / int(L["_header"]["acc_1G"])
    om = np.stack([L[f"debug[{i}]"] for i in range(4)], 1) * ERPM
    return {"t": t, "gy": gy, "ac": ac, "om": om}


def read_csv(path):
    import pandas as pd
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    return d[CSV_COLS].to_numpy(float)


def _smooth_gyro(gy):
    from scipy.signal import savgol_filter
    return savgol_filter(gy, 121, 6, axis=0)


def fit_clock(lg, segs):
    """(offset, slope) of t_log = (1 + slope) t_csv + offset, maximising the match of the raw gyro (smoothed as the providers
    smooth it) to the CSV's ang vel over the WHOLE flight. A global search: drift is 3-4 ms/s, and the periodic flights give
    false peaks to any local correlation. Uses the gyro only, no label: nothing is learned from the split."""
    from scipy.optimize import minimize
    from scipy.signal import correlate
    t, gs = lg["t"], _smooth_gyro(lg["gy"])
    h = 0.0025
    tg = np.arange(0, t[-1], h)
    G = np.stack([np.interp(tg, t, gs[:, i]) for i in range(3)], 1)
    tp = np.concatenate([s[:, 0] for s in segs])
    P = np.concatenate([s[:, 1:4] for s in segs])
    best, s0, o0 = -np.inf, 0.0, 0.0
    for s in np.arange(-0.0005, 0.0066, 0.0001):
        k = np.round((tp * (1 + s) - tp[0]) / h).astype(int)
        L = k[-1] + 1
        W = np.zeros((L, 3))
        W[k] = P
        mode = "valid" if len(G) >= L else "full"
        c = sum(correlate(G[:, i], W[:, i], mode, method="fft") for i in range(3))
        j = int(c.argmax())
        if c.max() > best:
            best, s0, o0 = float(c.max()), float(s), (j - (0 if mode == "valid" else L - 1)) * h - tp[0]

    def J(th):
        x = np.stack([np.interp(tp * (1 + th[0] * 1e-3) + th[1], t, gs[:, i]) for i in range(3)], 1)
        return float(np.mean((x - P) ** 2))
    a, b = s0 * 1e3, o0
    r = minimize(J, [a, b], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7,
                                                           "initial_simplex": [[a, b], [a + 0.2, b], [a, b + 0.004]]})
    return {"offset": float(r.x[1]), "slope": float(r.x[0] * 1e-3), "gyro_rms": float(math.sqrt(r.fun)), "gyro_std": float(P.std())}


def quat_rot(q):
    """R (body -> world), q = (x, y, z, w), Hamilton (measured: docs/cards/neurobem_datacheck.md)."""
    x, y, z, w = q.T
    R = np.empty((len(q), 3, 3))
    R[:, 0] = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1)
    R[:, 1] = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1)
    R[:, 2] = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    return R


def boxcar(t, X, tq, h=DT):
    """Causal mean of the samples of X in (tq - h, tq] (t sorted). Sample-and-hold if a bin is empty."""
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    i1 = np.searchsorted(t, tq, side="right")
    i0 = np.searchsorted(t, tq - h, side="right")
    n = i1 - i0
    return np.where((n > 0)[:, None], (c[i1] - c[i0]) / np.maximum(n, 1)[:, None], X[np.clip(i1 - 1, 0, len(X) - 1)])


def grid_segment(lg, csv, clock, om0):
    """One segment on the 100-Hz grid: dict of t, u (4), y (6), x (12), acc_true (3), dw_true (3), gy_smooth (3)."""
    tc = csv[:, 0]
    t = np.arange(tc[0], tc[-1] + 1e-9, DT)
    cols = np.stack([np.interp(t, tc, csv[:, j]) for j in range(1, csv.shape[1])], 1)
    wsm, q, acc, v, dw = cols[:, :3], cols[:, 3:7], cols[:, 7:10], cols[:, 10:13], cols[:, 13:16]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    g = quat_rot(q)[:, 2, :]                       # R^T e_z: world up in the body frame
    tl = t * (1 + clock["slope"]) + clock["offset"]
    gy, ac = boxcar(lg["t"], lg["gy"], tl), boxcar(lg["t"], lg["ac"], tl)
    om = boxcar(lg["t"], lg["om"], tl)
    x = np.concatenate([v, g, gy, ac], 1)
    return {"t": t, "u": om - om0, "y": np.concatenate([gy, ac], 1), "x": x, "acc_true": acc, "dw_true": dw, "gy_smooth": wsm}


def _decode_one(a):
    fl, root, cache = a
    p = Path(cache) / f"{fl}.npz"
    if not p.exists():
        np.savez(p, **decode_log(_logs(root) / f"{fl}.BFL"))
    return fl


def _align_one(a):
    fl, root, cache = a
    z = np.load(Path(cache) / f"{fl}.npz")
    return fl, fit_clock(z, [read_csv(p) for p in _csvs(fl, root)])


def _workers():
    return int(os.environ.get("HKB_WORKERS", min(12, os.cpu_count() or 1)))


def _map(fn, items):
    with ProcessPoolExecutor(_workers()) as ex:
        return list(ex.map(fn, items, chunksize=1))


def _cache(root):
    c = processed_dir(NAME, root) / "decoded"
    c.mkdir(parents=True, exist_ok=True)
    return c


def flights():
    return spec(NAME)["flights"]


def flight_segments(fl, root=None, om0=None):
    from ..plants.systems_real import NB_OM0
    z = np.load(_cache(root) / f"{fl}.npz")
    ck = spec(NAME)["frozen"]["alignment"][fl]
    return [dict(grid_segment(z, read_csv(p), ck, NB_OM0 if om0 is None else om0), seg=int(p.stem.rsplit("_", 1)[1]))
            for p in _csvs(fl, root)]


def _seg_one(a):
    fl, root = a
    return fl, flight_segments(fl, root)


# ------------------------------------------------------------------------------------- identification (train flights only)
def _identify(segs, log=print):
    from ..plants.systems_real import NB_OM0, NeuroBEMNominal as M
    from scipy.signal import butter, lfilter, sosfiltfilt
    edges = np.cumsum([0] + [len(s["u"]) for s in segs])
    u = np.concatenate([s["u"] for s in segs])
    x = np.concatenate([s["x"] for s in segs])
    a_true = np.concatenate([s["acc_true"] for s in segs])
    dw = np.concatenate([s["dw_true"] for s in segs])
    wsm = np.concatenate([s["gy_smooth"] for s in segs])
    y = x[:, 6:12]
    w2 = ((u + NB_OM0) / NB_OM0) ** 2
    sc = np.sqrt(w2.mean(1))
    p = dict(M.PARAMS)
    Jv = np.array([p["Jx"], p["Jy"], p["Jz"]])
    v = x[:, :3]
    # drag and thrust: least squares on the mocap-derived specific force
    p["dx"] = float(-np.sum(a_true[:, 0] * sc * v[:, 0]) / np.sum((sc * v[:, 0]) ** 2))
    p["dy"] = float(-np.sum(a_true[:, 1] * sc * v[:, 1]) / np.sum((sc * v[:, 1]) ** 2))
    A = np.hstack([w2, -(sc * v[:, 2])[:, None]])
    th = np.linalg.lstsq(A, a_true[:, 2], rcond=None)[0]
    for i in range(4):
        p[f"bT{i}"] = float(th[i])
    p["dz"] = float(th[4])
    # control effectiveness: Euler's equation with the Readme's J. B comes from the 2-Hz high-passed signals (the incremental
    # form of Smeur, Chu, de Croon, JGCD 2016): at low frequency the rotors balance the steady aerodynamic moment, so a plain
    # regression gives B too small by a factor ~8 (rotor arm 1 cm instead of 8.5 cm). The moment is then fitted on what remains.
    rhs = dw + np.cross(wsm, Jv * wsm) / Jv
    so = butter(3, 2.0 / (HZ / 2), "high", output="sos")
    hp = lambda a: np.concatenate([sosfiltfilt(so, a[i:j], axis=0) for i, j in zip(edges[:-1], edges[1:])])
    Bm = np.linalg.lstsq(hp(w2), hp(rhs), rcond=None)[0].T          # (3, 4)
    Bm[2] = np.linalg.lstsq(w2 - w2.mean(0), rhs[:, 2] - rhs[:, 2].mean(), rcond=None)[0]     # yaw: no such confound, plain fit
    for r, a in enumerate("xyz"):
        for i in range(4):
            p[f"B{a}{i}"] = float(Bm[r, i])
    rem = rhs - w2 @ Bm.T
    one = np.ones_like(sc)
    mx = np.linalg.lstsq(np.stack([sc * v[:, 1], one], 1), rem[:, 0], rcond=None)[0]
    my = np.linalg.lstsq(np.stack([sc * v[:, 0], one], 1), rem[:, 1], rcond=None)[0]
    p["Mx"], p["c0x"], p["My"], p["c0y"], p["c0z"] = float(mx[0]), float(mx[1]), float(my[0]), float(my[1]), float(rem[:, 2].mean())
    res_w = rem - np.stack([p["Mx"] * sc * v[:, 1] + p["c0x"], p["My"] * sc * v[:, 0] + p["c0y"], np.full_like(sc, p["c0z"])], 1)
    # accelerometer: affine calibration + first-order lag (grid on tau), on the mocap-derived specific force
    best = None
    for tau in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        al = 1 - math.exp(-DT / tau)
        f = np.concatenate([lfilter([al], [1, al - 1], a_true[a:b] - a_true[a], axis=0) + a_true[a]
                            for a, b in zip(edges[:-1], edges[1:])])
        X = np.hstack([f, np.ones((len(f), 1))])
        Bc = np.linalg.lstsq(X, y[:, 3:], rcond=None)[0]
        res = y[:, 3:] - X @ Bc
        if best is None or res.var() < best[0]:
            best = (float(res.var()), tau, Bc, res)
    _, tau, Bc, res_a = best
    p["tau_a"] = tau
    for c, col in enumerate("xyz"):
        for r, row in enumerate("xyz"):
            p[f"C{row}{col}"] = float(Bc[c, r])
    for r, row in enumerate("xyz"):
        p[f"b{row}"] = float(Bc[3, r])
    r_w = (y[:, :3] - wsm).std(0)
    r_a = res_a.std(0)
    res_f = np.stack([a_true[:, 0] + p["dx"] * sc * v[:, 0], a_true[:, 1] + p["dy"] * sc * v[:, 1], a_true[:, 2] - A @ th], 1)
    xs, xm = x.std(0), x.mean(0)
    info = {"x_std": xs.tolist(), "x_mean": xm.tolist(), "y_mean": y.mean(0).tolist(), "r_y": np.r_[r_w, r_a].tolist(),
            "u_std": u.std(0).tolist(), "u_mean": u.mean(0).tolist(), "u_scale": float(3 * u.std()),
            "specific_force_residual_std": res_f.std(0).tolist(), "specific_force_std": a_true.std(0).tolist(),
            "angular_acc_residual_std": res_w.std(0).tolist(), "angular_acc_std": dw.std(0).tolist(),
            "accelerometer_residual_std": r_a.tolist(),
            "n_train_samples": int(len(u)), "n_train_segments": len(segs),
            "arm_check": {"B_note": "thrust/torque consistency: rotor arm implied by B and bT with the Readme J, m",
                          "ly_m": float(np.mean(np.abs(Bm[0])) * Jv[0] / max(np.mean(np.abs(list(th[:4]))) * 0.772, 1e-9)),
                          "lx_m": float(np.mean(np.abs(Bm[1])) * Jv[1] / max(np.mean(np.abs(list(th[:4]))) * 0.772, 1e-9))}}
    log(f"[freeze] neurobem: tau_a {tau}, dx {p['dx']:.3f} dy {p['dy']:.3f} dz {p['dz']:.3f}, r_y {np.round(info['r_y'], 3)}")
    return p, info


def tilt_deg(gh, g):
    gh = gh / np.maximum(np.linalg.norm(gh, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(gh, g), axis=-1), (gh * g).sum(-1)))


def roll_pitch(g):
    return np.degrees(np.arctan2(g[..., 1], g[..., 2])), np.degrees(np.arctan2(-g[..., 0], np.hypot(g[..., 1], g[..., 2])))


def _model_fit(params, by_split, log=print):
    """One-step error (all samples) and open-loop simulation error from the true state at horizons 0.1, 1, 10 s."""
    import torch
    from ..plants.simulate import rk4_zoh_step
    from ..plants.systems_real import NeuroBEMNominal as M
    m = M(sensors=tuple(SENSORS), **params)
    out = {}
    for split, segs in by_split.items():
        ws = [(s["x"][k:k + N + 1], s["u"][k:k + N + 1]) for s in segs for k in range(0, len(s["u"]) - N, 500)]
        X = np.stack([w[0] for w in ws])
        U = np.stack([w[1] for w in ws])
        x = torch.as_tensor(X[:, 0], dtype=torch.float64)
        Ut = torch.as_tensor(U, dtype=torch.float64)
        sim = [x.numpy()]
        with torch.no_grad():
            for k in range(N):
                x = torch.nan_to_num(rk4_zoh_step(m, x, Ut[:, k], DT), nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
                sim.append(x.numpy())
            xs = torch.as_tensor(X[:, :-1].reshape(-1, 12), dtype=torch.float64)
            one = rk4_zoh_step(m, xs, Ut[:, :-1].reshape(-1, 4), DT).numpy().reshape(X[:, 1:].shape)
        sim = np.stack(sim, 1)
        e = {"onestep": {"v_ms": float(np.sqrt(np.mean((one[..., :3] - X[:, 1:, :3]) ** 2))),
                         "tilt_deg": float(np.sqrt(np.mean(tilt_deg(one[..., 3:6], X[:, 1:, 3:6]) ** 2))),
                         "w_rads": float(np.sqrt(np.mean((one[..., 6:9] - X[:, 1:, 6:9]) ** 2))),
                         "am_ms2": float(np.sqrt(np.mean((one[..., 9:12] - X[:, 1:, 9:12]) ** 2)))}, "n_windows": len(X)}
        for H, nm in ((10, "0.1s"), (50, "0.5s"), (100, "1s"), (N, "10s")):      # the open loop tumbles: finite windows only
            ok = np.isfinite(sim[:, H]).all(1) & (np.abs(sim[:, H, 6:9]).max(1) < 100)
            e[f"open_loop_{nm}"] = {"windows_not_tumbled": float(ok.mean())}
            if ok.any():
                e[f"open_loop_{nm}"].update({"v_ms": float(np.sqrt(np.mean((sim[ok, H, :3] - X[ok, H, :3]) ** 2))),
                                             "tilt_deg": float(np.sqrt(np.mean(tilt_deg(sim[ok, H, 3:6], X[ok, H, 3:6]) ** 2))),
                                             "w_rads": float(np.sqrt(np.mean((sim[ok, H, 6:9] - X[ok, H, 6:9]) ** 2)))})
        out[split] = e
    log(f"[freeze] neurobem model fit {json.dumps(out)}")
    return out


def _write_spec(fn):
    p = Path(__file__).resolve().parents[1] / "resources" / "real_neurobem.json"
    s = json.loads(p.read_text())
    fn(s)
    p.write_text(json.dumps(s, indent=1))
    spec.cache_clear()


def freeze(root=None, log=print):
    fl = sorted(flights())
    cache = str(_cache(root))
    _map(_decode_one, [(f, root, cache) for f in fl])
    log(f"[freeze] neurobem: {len(fl)} logs decoded")
    al = dict(_map(_align_one, [(f, root, cache) for f in fl]))
    _write_spec(lambda s: s.__setitem__("frozen", {"alignment": al}))
    log(f"[freeze] neurobem: clocks fitted, gyro residual max {max(a['gyro_rms'] for a in al.values()):.4f} rad/s")
    segs = dict(_map(_seg_one, [(f, root) for f in fl]))
    by = {k: [s for f in fl if flights()[f]["split"] == k for s in segs[f] if len(s["u"]) > N] for k in ("train", "val")}
    params, info = _identify(by["train"], log)
    info["model_fit"] = _model_fit(params, by, log)
    _write_spec(lambda s: s["frozen"].update({"params": params, "identification": info}))
    return spec(NAME)["frozen"]


def _windows(split, segs):
    X, U, Y, P, meta = [], [], [], [], []
    for fl, ss in segs.items():
        if flights()[fl]["split"] != split:
            continue
        for s in ss:
            for k in range(0, len(s["u"]) - N, STRIDE[split]):
                sl = slice(k, k + N + 1)
                X.append(s["x"][sl]); U.append(s["u"][sl]); Y.append(s["y"][sl])
                P.append(np.r_[0, 0, 0, 0, 0, 1.0, s["y"][k]])
                meta.append({"rec": fl, "family": flights()[fl]["family"], "seg": s["seg"], "start": int(k)})
    f32 = lambda a: np.asarray(a, np.float32)
    return {"x": f32(X), "u": f32(U), "y": f32(Y), "prior": f32(P), "meta": meta}


def prepare(root=None, log=print):
    if not spec(NAME).get("frozen", {}).get("params"):
        freeze(root, log)
    fl = sorted(flights())
    cache = str(_cache(root))
    _map(_decode_one, [(f, root, cache) for f in fl])
    segs = dict(_map(_seg_one, [(f, root) for f in fl]))
    out = processed_dir(NAME, root)
    files = {}
    for split in SPLITS:
        w = _windows(split, segs)
        np.savez(out / f"{split}.npz", **{k: v for k, v in w.items() if k != "meta"}, meta=np.asarray(json.dumps(w["meta"])))
        files[f"{split}.npz"] = int(len(w["u"]))
    (out / "MANIFEST.json").write_text(json.dumps({"prep_version": PREP_VERSION, "files": files}, indent=1))
    log(f"[prepare] neurobem: {files}")
    if "filter_tuning" not in spec(NAME)["frozen"]:
        tune_filter(log=log)
    return {"files": files}


def tune_filter(log=print, q_grid=(0.3, 1, 3, 10), r_grid=(1, 3, 10), n_val=48):
    """EKF process and sensor noise inflation chosen on VAL flights (PROTOCOL 3): model error dominates the nominal noise."""
    from ..baselines import EKF
    es = load("val")
    sub = es.subset(np.linspace(0, len(es) - 1, min(n_val, len(es))).astype(int))
    scores = {}
    ctx = context()
    for q in q_grid:
        for r in r_grid:
            f = EKF(gate=False, q_inflate=q, r_inflate=r)
            f.fit(None, None, ctx)
            xh, _ = f.run_batch(sub.u.astype(float), sub.y.astype(float), sub.prior.astype(float), np.tile(es.prior_std, (len(sub), 1)))
            scores[f"q{q}_r{r}"] = score(xh, sub)["summary"][HEADLINE]
            log(f"[tune] q {q} r {r}: {scores[f'q{q}_r{r}']:.4f}")
    k = min(scores, key=scores.get)
    q, r = (float(t[1:]) for t in k.split("_"))
    _write_spec(lambda s: s["frozen"].update({"filter_tuning": {"q_inflate": q, "r_inflate": r, "val_scores": scores, "n_val": len(sub)}}))
    return scores


# ------------------------------------------------------------------------------------- load / context / score
def frozen():
    fr = spec(NAME).get("frozen")
    if not fr or "params" not in fr:
        raise RuntimeError("neurobem: not frozen; run hyperkkl-bench prepare neurobem")
    return fr


def model(params=None):
    from ..plants.systems_real import NeuroBEMNominal
    return NeuroBEMNominal(sensors=tuple(SENSORS), **(params or frozen()["params"]))


def x_std():
    return np.asarray(frozen()["identification"]["x_std"], float)


def prior_std():
    i = frozen()["identification"]
    return np.r_[np.asarray(i["x_std"][:6], float), 3 * np.asarray(i["r_y"], float)]


def load(split, root=None) -> RealEpisodeSet:
    if split not in SPLITS:
        raise ValueError(f"{NAME}: unknown split {split!r}; available: {SPLITS}")
    p = processed_dir(NAME, root) / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `hyperkkl-bench fetch {NAME}` then `hyperkkl-bench prepare {NAME}`")
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    x = z["x"]
    return RealEpisodeSet(u=z["u"], y=z["y"], x=x, dt=DT, plant=NAME, knowledge="real", split=split,
                          families=[m["family"] for m in meta], signals=[f"{m['rec']}_seg{m['seg']}" for m in meta], x0=x[:, 0].copy(),
                          episode_ids=np.arange(len(x)), source=str(p), prior=z["prior"], prior_std=prior_std(),
                          label_mask=np.ones(12, bool), meta=meta, settle_index=N // 2)


def q_std():
    """sqrt(one-step model residual variance / dt) per state (train windows)."""
    fr = frozen()
    if "q_std" in fr:
        return np.asarray(fr["q_std"], float)
    i = fr["identification"]
    sf = np.asarray(i["specific_force_residual_std"], float)
    aw = np.asarray(i["angular_acc_residual_std"], float)
    s = math.sqrt(DT)
    return np.r_[sf * s, np.full(3, 0.02), aw * s, 1.0 * np.asarray(i["accelerometer_residual_std"], float) * s / 0.04]


def context(condition="default", **kw):
    fr = frozen()
    i = fr["identification"]
    xs = x_std()
    xm = np.asarray(i["x_mean"], float)
    ic = np.stack([xm - 4 * xs, xm + 4 * xs], 1)
    ic[3:6] = [[-1.2, 1.2]] * 3
    meta = {"plant": NAME, "n_x": 12, "n_y": 6, "n_u": 4, "state_names": STATE, "sensors": SENSORS, "unmeasured": HIDDEN,
            "labelled": list(range(12)), "x_std": xs.tolist(), "dt": DT, "dt_seconds": DT, "time_unit_s": 1.0, "N": N,
            "settle_index": N // 2, "t_ref": 1.0, "input_box": {"u_scale": float(i["u_scale"])}}
    t = fr.get("filter_tuning", {})
    return {"plant": NAME, "knowledge": "real", "meta": meta, "dt": DT, "sensors": SENSORS, "unmeasured": HIDDEN,
            "labelled": list(range(12)), "x_std": xs, "y_std": xs[6:12], "x_mean": xm,
            "q_std": q_std() * math.sqrt(t.get("q_inflate", 1.0)), "r_std": np.asarray(i["r_y"], float) * math.sqrt(t.get("r_inflate", 1.0)),
            "prior_mean": None, "prior_std": prior_std(), "model": model(), "model_knowledge": "partial", "condition": condition,
            "hold_missing": False, "hyperkkl": dict(HKKL, ic_box=ic.tolist())}


def score(X_hat, es) -> dict:
    """Settled errors (k >= settle_index). Per state and aggregated, in physical units; non-finite estimates count as 100 std."""
    Xh = np.asarray(X_hat, float)
    X = es.x.astype(float)
    s = es.settle_index
    xs = x_std()
    bad = ~np.isfinite(Xh).all((1, 2))
    Xc = np.where(np.isfinite(Xh), Xh, X + 100 * xs)
    Xc = np.clip(Xc, X - 100 * xs, X + 100 * xs)
    a, b = Xc[:, s:], X[:, s:]
    rms = lambda e: np.sqrt(np.mean(e ** 2, 1))
    out = {"nrmse_hidden": np.sqrt(np.mean(((a - b)[..., HIDDEN] / xs[HIDDEN]) ** 2, (1, 2)))}
    for j, nm in zip(range(3), "xyz"):
        out[f"vel_{nm}_ms"] = rms(a[..., j] - b[..., j])
    out["vel_rmse_ms"] = np.sqrt(np.mean(np.sum((a[..., :3] - b[..., :3]) ** 2, -1), 1))
    t = tilt_deg(a[..., 3:6], b[..., 3:6])
    out["tilt_rmse_deg"] = rms(t)
    ra, pa = roll_pitch(a[..., 3:6])
    rb, pb = roll_pitch(b[..., 3:6])
    out["roll_rmse_deg"] = rms((ra - rb + 180) % 360 - 180)
    out["pitch_rmse_deg"] = rms(pa - pb)
    out["nrmse_measured"] = np.sqrt(np.mean(((a - b)[..., SENSORS] / xs[SENSORS]) ** 2, (1, 2)))
    dev = np.abs(np.nan_to_num((Xh[..., HIDDEN] - X[..., HIDDEN]) / xs[HIDDEN], nan=np.inf)).max(-1)
    out["diverged"] = bad | (dev[:, s:] > 100).any(1)          # settled window: a start-up transient from a blind prior is not divergence
    out["transient_peak_std"] = np.minimum(dev[:, :s].max(1), 1e6)      # reported: largest hidden error before settling, in stds
    summ = {k: float(np.mean(v)) for k, v in out.items()}
    summ["n_episodes"] = int(len(Xh))
    fam = np.array(es.families)
    summ["by_family"] = {f: {"vel_rmse_ms": float(out["vel_rmse_ms"][fam == f].mean()), "tilt_rmse_deg": float(out["tilt_rmse_deg"][fam == f].mean()),
                             "n": int((fam == f).sum())} for f in sorted(set(fam))}
    return {"summary": summ, "per_episode": {k: np.asarray(v).tolist() for k, v in out.items()}}


def headline(results):
    return float(np.mean([results[s]["summary"][HEADLINE] for s in ("test_id", "test_ood")]))
