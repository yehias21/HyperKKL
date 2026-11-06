r"""Real-data track ``battery_pf``: SOC estimation of a Panasonic 18650PF Li-ion cell on held-out
drive cycles (Kollmeyer, Mendeley Data, doi:10.17632/wykht8y7tg.1, CC BY 4.0).

The preprocessing below is the research pipeline (hyperkkl/realdata.py + realdata_run.py,
experiments/real/DESIGN.md) vendored verbatim; tests/test_real_battery.py checks that the windows
and labels it produces equal the research pipeline's, value for value.

Conventions
    u      current [A], POSITIVE = discharge, zero-order hold: u_k = mean current over [t_k, t_k + 1 s)
           (charge-preserving resampling of the 0.1 s log)
    y      (dT, Vm): dT = case temperature - (ambient + thermocouple rest offset) [K];
           Vm = bin-mean terminal voltage - 3.7 V behind a first-order acquisition filter (tau 2 s)
    x      (SOC, V1, V2, dT, Vm). LABELLED: SOC (the lab's coulomb counter, reset at full charge,
           Q = 2.995 Ah from the C/20 test), dT and Vm (measured). V1, V2 (RC polarisation voltages)
           have NO label on real data: they are NaN in every split (label_mask).
    dt     1 s = 1/60 model time unit (the ECM's time unit is 1 min, as the simulated battery plant)

Splits (by drive cycle; pre-registered in the research, DESIGN.md)
    train          Cycle 1, 2, 3, UDDS, LA92 at 25 degC: 30-min windows every 2 min (labels SOC, dT, Vm)
    val            Cycle 4 at 25 degC: windows every 5 min, x 2 priors (SOC0 +- 0.2 kept inside [0, 1])
    test_US06, test_HWFET (HWFTa + HWFTb pooled), test_NN       25 degC, as val
    test0_<cycle>  every 0 degC drive cycle (US06, HWFET, NN, UDDS, LA92, Cycle_1..4): a temperature
                   shift (models fitted at 25 degC); the model-based context at 0 degC is the 25 degC
                   ECM with Arrhenius-scaled resistances (``context(condition="0C")``)
    train_recordings / val_recordings: whole recordings (``recordings()``), for methods that want them

Task and score
    Every val / test episode starts from a prior x0_hat = (SOC0 + e, 0, 0, dT_0, Vm_0), e = +-0.2
    (in ``prior``). Score: settled SOC RMSE in % SOC over k >= 900 (t >= 15 min of each 30-min
    window); mean over the split's episodes. Headline: the mean of the three 25 degC test families.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..data import EpisodeSet
from ..plants.systems_real import TAU_F, TIME_UNIT, V_REF, cold_params, ocv_eval, ocv_basis  # noqa: F401
from .registry import processed_dir, raw_dir, spec

NAME = "battery_pf"
DATA_ROOT_REL = "Panasonic_18650PF_Data"
C20_REL = "25degC/C20_OCV_and_1C_discharge_tests_start_of_tests/05-08-17_13.26_C20_OCV_Test_C20_25dC.mat"
FIELDS = ("Time", "Voltage", "Current", "Ah", "Battery_Temp_degC", "Chamber_Temp_degC")
LABELLED = (0, 3, 4)
STATE_NAMES = ("SOC", "V1", "V2", "dT", "Vm")
SPLIT = {"train": ("Cycle_1", "Cycle_2", "Cycle_3", "UDDS", "LA92"),
         "val": ("Cycle_4",),
         "test": ("US06", "HWFTa", "HWFTb", "NN")}
FAMILY = {"HWFTa": "HWFET", "HWFTb": "HWFET", "HWFET": "HWFET"}
TEST_FAMILIES = ("US06", "HWFET", "NN")
COLD_CYCLES = ("Cycle_1", "Cycle_2", "LA92", "NN", "Cycle_3", "Cycle_4", "US06", "HWFET", "UDDS")
N_STEPS, DT_S = 1800, 1.0
STRIDE_TRAIN, STRIDE_EVAL = 120, 300
INIT_ERR = 0.2
SETTLE = N_STEPS // 2
PREP_VERSION = "battery_pf-prep-1"


def splits() -> list:
    return (["train", "val"] + [f"test_{f}" for f in TEST_FAMILIES] + [f"test0_{c}" for c in COLD_CYCLES])


# ---------------------------------------------------------------------------
# Loading and resampling (vendored from hyperkkl/realdata.py, unchanged)
# ---------------------------------------------------------------------------

def load_mat(path) -> dict:
    """The ``meas`` struct of one file as float64 arrays (+ 'TimeStamp' strings). scipy.io.loadmat
    (MAT v5 parser; no code execution, no pickles)."""
    import scipy.io as sio
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "meas" not in m:
        raise ValueError(f"{path}: no 'meas' struct")
    s = m["meas"]
    out = {}
    for f in FIELDS:
        if f == "Chamber_Temp_degC" and f not in s._fieldnames:      # absent in the LG HG2 files (unused)
            continue
        if f not in s._fieldnames:
            raise ValueError(f"{path}: field {f} missing")
        out[f] = np.atleast_1d(np.asarray(getattr(s, f), dtype=np.float64))
    ts = getattr(s, "TimeStamp", None)
    out["TimeStamp"] = [str(v) for v in np.atleast_1d(ts)] if ts is not None else []
    n = len(out["Time"])
    for f in out:
        if f != "TimeStamp" and len(out[f]) != n:
            raise ValueError(f"{path}: field {f} has {len(out[f])} samples, Time has {n}")
    return out


def parse_name(path) -> tuple:
    m = re.search(r"(n?)(\d+)degC_(.+?)_Pan18650PF", Path(path).name)
    if not m:
        raise ValueError(f"unrecognised file name {Path(path).name}")
    temp = float(m.group(2)) * (-1.0 if m.group(1) == "n" else 1.0)
    return temp, m.group(3)


def _parse_ts(s: str):
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S"):
        try:
            return _dt.datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def _cumtrapz(y, t):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))])


@dataclass
class Recording:
    name: str
    temp: float
    cycle: str
    t: np.ndarray
    u: np.ndarray
    vbar: np.ndarray
    vm: np.ndarray
    temp_case: np.ndarray
    ah: np.ndarray
    soc: np.ndarray
    Q: float
    dt: float
    t_amb_eff: float
    abs_start: object = None
    abs_end: object = None
    path: str = ""
    sha256: str = ""
    cc_check: float = 0.0

    @property
    def family(self) -> str:
        return FAMILY.get(self.cycle, self.cycle)

    @property
    def dT(self) -> np.ndarray:
        return self.temp_case - self.t_amb_eff

    @property
    def y(self) -> np.ndarray:
        return np.stack([self.dT, self.vm], 1)

    def __len__(self):
        return len(self.t)


def resample(raw: dict, dt: float = 1.0, Q: float = 2.9, t_amb_eff: float = 25.0, tau_f: float = TAU_F) -> dict:
    t = raw["Time"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    I = -raw["Current"][keep]
    V = raw["Voltage"][keep]
    n = int(math.floor((t[-1] - t[0]) / dt))
    if n < 2:
        raise ValueError("recording shorter than two samples")
    g = t[0] + dt * np.arange(n + 1)
    qI = np.interp(g, t, _cumtrapz(I, t))
    qV = np.interp(g, t, _cumtrapz(V, t))
    u = np.diff(qI) / dt
    vbar = np.diff(qV) / dt
    tk = g[:-1]
    ah = np.interp(tk, t, raw["Ah"][keep])
    a = math.exp(-dt / tau_f)
    vm = np.empty(n)
    vm[0] = vbar[0] - V_REF
    for k in range(n - 1):
        vm[k + 1] = a * vm[k] + (1 - a) * (vbar[k] - V_REF)
    own = -(qI[:-1] - qI[0]) / 3600.0
    return {"t": tk - t[0], "u": u, "vbar": vbar, "vm": vm,
            "temp_case": np.interp(tk, t, raw["Battery_Temp_degC"][keep]),
            "ah": ah, "soc": 1.0 + ah / Q, "cc_check": float(np.abs(own - (ah - ah[0])).max())}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recording(path, Q: float, t_amb_eff: float = None, dt: float = 1.0, offset: float = 0.0) -> Recording:
    raw = load_mat(path)
    temp, cyc = parse_name(path)
    tae = temp + offset if t_amb_eff is None else t_amb_eff
    r = resample(raw, dt, Q, tae)
    ts = [_parse_ts(raw["TimeStamp"][0]) if raw["TimeStamp"] else None]
    end = ts[0] + _dt.timedelta(seconds=float(raw["Time"][-1] - raw["Time"][0])) if ts[0] else None
    return Recording(name=Path(path).stem, temp=temp, cycle=cyc, Q=Q, dt=dt, t_amb_eff=tae,
                     abs_start=ts[0], abs_end=end, path=str(path), sha256=_sha256(path),
                     **{k: r[k] for k in ("t", "u", "vbar", "vm", "temp_case", "ah", "soc")},
                     cc_check=r["cc_check"])


def drive_cycle_files(temp: float, root: Path) -> dict:
    tag = f"{'n' if temp < 0 else ''}{abs(int(temp))}degC"
    out = {}
    for p in sorted((root / tag / "Drive_cycles").glob("*.mat")):
        t, c = parse_name(p)
        if t != temp or "_" in c.replace("Cycle_", "Cycle"):
            continue
        out[c] = p
    return out


def c20_capacity_and_ocv(path, n_grid: int = 201) -> dict:
    raw = load_mat(path)
    I, ah, V = raw["Current"], raw["Ah"], raw["Voltage"]
    dis = I < -0.05
    chg = I > 0.05
    if not dis.any() or not chg.any():
        raise ValueError("C/20 test needs a discharge and a charge segment")
    idis, ichg = np.flatnonzero(dis), np.flatnonzero(chg)
    a0, a1 = ah[idis[0]], ah[idis[-1]]
    Q = float(a0 - a1)
    soc_d = 1.0 - (a0 - ah[idis]) / Q
    soc_c = (ah[ichg] - ah[ichg[0]]) / Q
    g = np.linspace(0.0, 1.0, n_grid)
    od = np.interp(g, soc_d[::-1], V[idis][::-1])
    top = float(soc_c.max())
    oc = np.interp(g, soc_c, V[ichg], right=np.nan)
    ov = ~np.isnan(oc)
    sel = ov & (g > top - 0.2 * top)
    half_gap = float(np.mean(oc[sel] - od[sel]) / 2)
    ocv = np.where(ov, 0.5 * (od + np.nan_to_num(oc)), od + half_gap)
    return {"Q": Q, "soc": g, "ocv": ocv, "dis": od, "chg": oc, "top_charge_soc": top, "half_gap": half_gap}


def fit_ocv(soc, ocv) -> dict:
    best = None
    for w0 in (0.01, 0.02, 0.04, 0.08):
        for w1 in (0.02, 0.05, 0.1, 1e9):
            X = np.stack(ocv_basis(soc, w0, w1), 1)
            c, *_ = np.linalg.lstsq(X, ocv, rcond=None)
            r = X @ c - ocv
            rms = float(np.sqrt(np.mean(r ** 2)))
            if best is None or rms < best["rms"]:
                best = {"coef": c.tolist(), "w0": w0, "w1": w1, "rms": rms, "max": float(np.abs(r).max())}
    return best


def _rc_avg(u, R, tau, dt):
    from scipy.signal import lfilter
    e = math.exp(-dt / tau)
    Vk = lfilter([0.0, R * (1 - e)], [1.0, -e], u)
    g = tau / dt * (1 - e)
    return g * Vk + R * (1 - g) * u, Vk


def ecm_voltage(theta, u, soc_mid, ocv_fn, dt):
    R0, R1, t1, R2, t2 = np.exp(theta)
    a1, _ = _rc_avg(u, R1, t1, dt)
    a2, _ = _rc_avg(u, R2, t2, dt)
    return ocv_fn(soc_mid) - a1 - a2 - R0 * u


def fit_ecm(recs, ocv_fn, dt=1.0, x0=(0.02, 0.01, 10.0, 0.01, 200.0)) -> dict:
    from scipy.optimize import least_squares
    data = []
    for r in recs:
        soc_mid = r.soc - 0.5 * r.u * dt / (3600.0 * r.Q)
        data.append((r.u, soc_mid, r.vbar))

    def res(th):
        return np.concatenate([ecm_voltage(th, u, s, ocv_fn, dt) - v for u, s, v in data])

    sol = least_squares(res, np.log(np.asarray(x0)), x_scale=1.0, method="trf")
    R0, R1, t1, R2, t2 = np.exp(sol.x)
    if t1 > t2:
        R1, t1, R2, t2 = R2, t2, R1, t1
    rr = res(np.log([R0, R1, t1, R2, t2]))
    return {"R0": R0, "R1": R1, "C1": t1 / R1, "R2": R2, "C2": t2 / R2, "tau1_s": t1, "tau2_s": t2,
            "v_rmse_mV": 1e3 * float(np.sqrt(np.mean(rr ** 2))), "v_max_mV": 1e3 * float(np.abs(rr).max()),
            "nfev": int(sol.nfev)}


def fit_thermal(recs, ecm, dUdT=-1.0e-4, dt=1.0) -> dict:
    from scipy.optimize import least_squares
    from scipy.signal import lfilter
    data = []
    for r in recs:
        _, V1 = _rc_avg(r.u, ecm["R1"], ecm["tau1_s"], dt)
        _, V2 = _rc_avg(r.u, ecm["R2"], ecm["tau2_s"], dt)
        T = r.t_amb_eff + 273.15 + r.dT
        P = r.u ** 2 * ecm["R0"] + V1 ** 2 / ecm["R1"] + V2 ** 2 / ecm["R2"] - r.u * T * dUdT
        data.append((P, r.dT))

    def sim(th, P, d0):
        C, h = np.exp(th)
        e = 1 - dt * h / C
        zi = np.array([e * d0])
        out, _ = lfilter([0.0, dt / C], [1.0, -e], P, zi=zi)
        return out + 0.0

    def res(th):
        out = []
        for P, d in data:
            s = sim(th, P, d[0])
            s[0] = d[0]
            out.append(s - d)
        return np.concatenate(out)

    sol = least_squares(res, np.log([60.0, 0.1]))
    C, h = np.exp(sol.x)
    rr = res(sol.x)
    return {"C_th": C, "hA": h, "tau_th_s": C / h, "dT_rmse_K": float(np.sqrt(np.mean(rr ** 2)))}


def rest_offset(recs, n_s: int = 30) -> float:
    return float(np.median(np.concatenate([r.temp_case[:n_s] - r.temp for r in recs])))


def identify(train_recs, c20_path) -> dict:
    """The partial model a BMS engineer would fit, from the training recordings + the C/20 test."""
    oc = c20_capacity_and_ocv(c20_path)
    fit = fit_ocv(oc["soc"], oc["ocv"])
    ocv_fn = lambda s: ocv_eval(s, fit["coef"], fit["w0"], fit["w1"])
    ecm = fit_ecm(train_recs, ocv_fn)
    th = fit_thermal(train_recs, ecm)
    params = {"Q": oc["Q"], "R0": ecm["R0"], "R1": ecm["R1"], "C1": ecm["C1"], "R2": ecm["R2"],
              "C2": ecm["C2"], "C_th": th["C_th"], "hA": th["hA"], "Ea_R": 3000.0, "dUdT": -1.0e-4,
              "T_amb": 273.15 + float(np.mean([r.t_amb_eff for r in train_recs])), "tau_f": TAU_F,
              "ow0": fit["w0"], "ow1": fit["w1"], **{f"o{i}": float(c) for i, c in enumerate(fit["coef"])}}
    return {"params": {k: float(v) for k, v in params.items()}, "ocv_fit": fit, "ecm": ecm, "thermal": th}


def check_split(split: dict, recs: dict) -> None:
    """No leakage: each recording in at most one role, and no two roles share an absolute time interval."""
    roles = {}
    for role, names in split.items():
        for n in names:
            if n in roles:
                raise ValueError(f"recording {n} in both {roles[n]} and {role}")
            roles[n] = role
    items = [(roles[n], recs[n]) for n in roles if n in recs]
    for i, (ra, a) in enumerate(items):
        for rb, b in items[i + 1:]:
            if ra == rb or a.abs_start is None or b.abs_start is None:
                continue
            if a.abs_start < b.abs_end and b.abs_start < a.abs_end:
                raise ValueError(f"{a.name} ({ra}) and {b.name} ({rb}) overlap in time")


def window_starts(n: int, N: int, stride: int, offset: int = 0) -> list:
    return list(range(offset, n - N - 1, stride))


def replay(plant, rec: "Recording") -> np.ndarray:
    """The partial model driven by a recording's current from its measured start (SOC = reference,
    V1 = V2 = 0 at rest after the CC-CV charge, dT and Vm measured): RK4, zero-order hold, float64.
    (realdata.replay_many.) Its V1, V2 are the PLACEHOLDERS of the unlabelled states (``x_fill``):
    never a loss target, never scored."""
    import torch
    from ..plants.simulate import rk4_zoh_step
    dt_m = rec.dt / TIME_UNIT
    x = torch.tensor([[rec.soc[0], 0.0, 0.0, rec.dT[0], rec.vm[0]]], dtype=torch.float64)
    U = torch.as_tensor(rec.u, dtype=torch.float64)
    out = np.empty((len(rec), 5))
    out[0] = x[0].numpy()
    with torch.no_grad():
        for k in range(len(rec) - 1):
            x = rk4_zoh_step(plant, x, U[k].reshape(1, 1), dt_m)
            out[k + 1] = x[0].numpy()
    return out


# ---------------------------------------------------------------------------
# Windows -> arrays (the research's realdata.episodes + realdata_run.priors/expand, labelled columns)
# ---------------------------------------------------------------------------

def windows(recs, stride: int, replays=None) -> dict:
    """Raw windows (float32, as the research EpisodeSet): u (E, N+1), y (E, N+1, 2), x (E, N+1, 5)
    with V1, V2 = NaN (no label), and per-window meta."""
    X, F, U, Y, meta = [], [], [], [], []
    for i, r in enumerate(recs):
        for s in window_starts(len(r), N_STEPS, stride):
            sl = slice(s, s + N_STEPS + 1)
            nan = np.full(N_STEPS + 1, np.nan)
            X.append(np.stack([r.soc[sl], nan, nan, r.dT[sl], r.vm[sl]], 1))
            if replays is not None:
                F.append(np.stack([nan, replays[i][sl, 1], replays[i][sl, 2], nan, nan], 1))
            U.append(r.u[sl]); Y.append(r.y[sl])
            meta.append({"rec": r.name, "cycle": r.cycle, "family": r.family, "temp": r.temp,
                         "start": int(s), "soc0": float(r.soc[s])})
    if not X:
        raise ValueError("no windows")
    f32 = lambda a: np.stack(a).astype(np.float32)
    out = {"u": f32(U), "y": f32(Y), "x": f32(X), "meta": meta}
    if replays is not None:
        out["x_fill"] = f32(F)
    return out


def expand(w: dict, err: float = INIT_ERR) -> dict:
    """Two episodes per window, prior SOC0 + e and SOC0 - e, each kept when inside [0, 1]
    (realdata_run.priors). x0_hat = (SOC0 + e, 0, 0, dT_0, Vm_0), float32."""
    idx, sgn, X0 = [], [], []
    for i in range(len(w["u"])):
        s0 = float(w["x"][i, 0, 0])
        for e in (+err, -err):
            if 0.0 <= s0 + e <= 1.0:
                idx.append(i); sgn.append(e)
                X0.append([s0 + e, 0.0, 0.0, float(w["y"][i, 0, 0]), float(w["y"][i, 0, 1])])
    idx = np.asarray(idx)
    meta = [dict(w["meta"][i], window=int(i), prior_sign=float(e)) for i, e in zip(idx, sgn)]
    out = {"u": w["u"][idx], "y": w["y"][idx], "x": w["x"][idx], "prior": np.asarray(X0, np.float32), "meta": meta}
    if "x_fill" in w:
        out["x_fill"] = w["x_fill"][idx]
    return out


# ---------------------------------------------------------------------------
# prepare: raw files -> frozen episodes
# ---------------------------------------------------------------------------

def _raw_root(root=None) -> Path:
    return raw_dir(NAME, root) / DATA_ROOT_REL


def load_all(root=None, raw=None) -> dict:
    """Every recording, with the research's thermocouple offset and Q. ``raw`` overrides the folder
    that holds 25degC/ and 0degC/ (e.g. the research copy)."""
    base = Path(raw) if raw else _raw_root(root)
    oc = c20_capacity_and_ocv(base / C20_REL)
    Q = oc["Q"]
    f25 = drive_cycle_files(25.0, base)
    recs = {c: load_recording(p, Q) for c, p in f25.items()}
    off = rest_offset([recs[c] for c in SPLIT["train"]])
    for r in recs.values():
        r.t_amb_eff = r.temp + off
    check_split(SPLIT, recs)
    f0 = drive_cycle_files(0.0, base)
    recs0 = {c: load_recording(p, Q, t_amb_eff=0.0 + off) for c, p in f0.items()}
    return {"Q": Q, "offset_K": off, "recs25": recs, "recs0": recs0, "base": base}


def build_sets(d: dict) -> dict:
    recs, recs0 = d["recs25"], d["recs0"]
    m25 = model("25C")
    rp = {c: replay(m25, recs[c]) for c in SPLIT["train"] + SPLIT["val"]}
    sets = {"train": windows([recs[c] for c in SPLIT["train"]], STRIDE_TRAIN, [rp[c] for c in SPLIT["train"]]),
            "val": expand(windows([recs[c] for c in SPLIT["val"]], STRIDE_EVAL, [rp[c] for c in SPLIT["val"]]))}
    for fam in TEST_FAMILIES:
        parts = [expand(windows([recs[c]], STRIDE_EVAL)) for c in SPLIT["test"] if FAMILY.get(c, c) == fam]
        sets[f"test_{fam}"] = {k: (np.concatenate([p[k] for p in parts]) if k != "meta" else sum((p[k] for p in parts), []))
                               for k in parts[0]}
    for c in COLD_CYCLES:
        sets[f"test0_{c}"] = expand(windows([recs0[c]], STRIDE_EVAL))
    return sets


def prepare(root=None, raw=None, log=print, check_identification: bool = True) -> dict:
    """Raw files -> processed/<split>.npz + recordings.npz + MANIFEST.json. Deterministic."""
    sp = spec(NAME)
    d = load_all(root, raw)
    fr = sp["frozen"]
    if abs(d["Q"] - fr["Q_Ah"]) > 1e-9 or abs(d["offset_K"] - fr["thermocouple_offset_K"]) > 1e-9:
        raise RuntimeError(f"Q / offset differ from the frozen values: {d['Q']}, {d['offset_K']}")
    ident_diff = None
    if check_identification:
        idn = identify([d["recs25"][c] for c in SPLIT["train"]], d["base"] / C20_REL)
        ident_diff = max(abs(idn["params"][k] - v) / max(abs(v), 1e-12) for k, v in fr["params_25C"].items())
        log(f"[prepare] re-identified ECM vs frozen parameters: max relative difference {ident_diff:.2e}")
    sets = build_sets(d)
    out = processed_dir(NAME, root)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, s in sets.items():
        p = out / f"{name}.npz"
        arrs = {k: v for k, v in s.items() if k != "meta"}
        np.savez(p, **arrs, meta=np.asarray(json.dumps(s["meta"])))
        files[p.name] = {"n_episodes": int(len(s["u"])), "sha256_arrays": _arrays_hash(arrs)}
    rec_arrays = {}
    for tag, rr in (("25", d["recs25"]), ("0", d["recs0"])):
        for c, r in rr.items():
            for k in ("t", "u", "soc", "vbar"):
                rec_arrays[f"{tag}/{c}/{k}"] = getattr(r, k)
            rec_arrays[f"{tag}/{c}/y"] = r.y
    np.savez(out / "recordings.npz", **rec_arrays)
    man = {"prep_version": PREP_VERSION, "dataset": NAME, "files": files, "Q_Ah": d["Q"], "offset_K": d["offset_K"],
           "identification_max_rel_diff": ident_diff,
           "raw_sha256": {r.name: r.sha256 for r in list(d["recs25"].values()) + list(d["recs0"].values())}}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    log(f"[prepare] {NAME}: " + ", ".join(f"{k} {v['n_episodes']}" for k, v in files.items()) + f" -> {out}")
    return man


def _arrays_hash(arrs: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading the frozen episodes
# ---------------------------------------------------------------------------

@dataclass
class RealEpisodeSet(EpisodeSet):
    """EpisodeSet + what a real-data track needs: ``prior`` (E, n_x) and ``prior_std`` (n_x,) for
    val / test episodes, ``label_mask`` (n_x,) bool (False = never labelled: NaN in x), and per-episode
    ``meta`` (recording, cycle, start index, prior sign)."""
    prior: np.ndarray | None = None
    x_fill: np.ndarray | None = None
    prior_std: np.ndarray | None = None
    label_mask: np.ndarray | None = None
    meta: list = field(default_factory=list)
    settle_index: int = SETTLE
    score_mask: np.ndarray | None = None      # (E, N+1) bool: samples scored instead of k >= settle_index (outage splits)

    def subset(self, idx) -> "RealEpisodeSet":
        idx = np.asarray(idx, dtype=int)
        base = EpisodeSet.subset(self, idx)
        return RealEpisodeSet(**{k: getattr(base, k) for k in EpisodeSet.__dataclass_fields__},
                              prior=None if self.prior is None else self.prior[idx], prior_std=self.prior_std,
                              x_fill=None if self.x_fill is None else self.x_fill[idx],
                              label_mask=self.label_mask, meta=[self.meta[i] for i in idx],
                              settle_index=self.settle_index,
                              score_mask=None if self.score_mask is None else self.score_mask[idx])


def prior_std() -> np.ndarray:
    """The research filters' P0: (0.2, std V1, std V2, 0.5 K, 10 mV)."""
    xs = np.asarray(spec(NAME)["frozen"]["x_std"], float)
    return np.array([INIT_ERR, xs[1], xs[2], 0.5, 0.01])


def load(split: str, root=None) -> RealEpisodeSet:
    if split not in splits():
        raise ValueError(f"{NAME}: unknown split {split!r}; available: {splits()}")
    p = processed_dir(NAME, root) / f"{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `hyperkkl-bench fetch {NAME}` then `hyperkkl-bench prepare {NAME}`")
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    u, y, x = z["u"], z["y"], z["x"]
    E = len(u)
    return RealEpisodeSet(u=u, y=y, x=x, dt=DT_S / TIME_UNIT, plant=NAME, knowledge="real", split=split,
                          families=[m["family"] for m in meta], signals=[m["rec"] for m in meta],
                          x0=x[:, 0].copy(), episode_ids=np.arange(E), source=str(p),
                          prior=z["prior"] if "prior" in z.files else None,
                          x_fill=z["x_fill"] if "x_fill" in z.files else None,
                          prior_std=prior_std(), label_mask=np.isin(np.arange(5), LABELLED), meta=meta)


def recordings(split: str = "train", temp: str = "25", root=None) -> dict:
    """Whole recordings {cycle: {t [s], u, y, soc, vbar}} of a split's drive cycles (train / val / test)."""
    z = np.load(processed_dir(NAME, root) / "recordings.npz", allow_pickle=False)
    names = SPLIT[split] if temp == "25" else COLD_CYCLES
    return {c: {k: z[f"{temp}/{c}/{k}"] for k in ("t", "u", "y", "soc", "vbar")} for c in names}


# ---------------------------------------------------------------------------
# What a method may know (the partial model and the nominal noise)
# ---------------------------------------------------------------------------

def model(condition: str = "25C"):
    """The partial model: the identified 2-RC ECM (frozen parameters), moved to 0 degC by the
    Arrhenius factor for condition '0C'. torch-batched like every benchmark plant."""
    from ..plants.systems_real import Pan18650PF
    fr = spec(NAME)["frozen"]
    p = dict(fr["params_25C"])
    if condition == "0C":
        p, _ = cold_params(p, 0.0 + fr["thermocouple_offset_K"])
    elif condition != "25C":
        raise ValueError("condition must be '25C' or '0C'")
    return Pan18650PF(sensors=(3, 4), **p)


def context(condition: str = "25C", filter_kind: str = "ekf") -> dict:
    """The fit() context of the track. ``q_std`` / ``r_std`` are the research filter noise chosen
    on the validation cycle (72-point grid; PROTOCOL section 3: tuning on validation is allowed)."""
    fr = spec(NAME)["frozen"]
    xs = np.asarray(fr["x_std"], float)
    g = fr["filter_tuning"][filter_kind]
    q = g["c"] * xs
    q[0] = g["q_soc"]
    meta = {"plant": NAME, "n_x": 5, "n_y": 2, "n_u": 1, "state_names": list(STATE_NAMES), "sensors": [3, 4],
            "sensor_names": ["dT", "Vm"], "unmeasured": [0, 1, 2], "labelled": list(LABELLED),
            "x_std": xs.tolist(), "dt": DT_S / TIME_UNIT, "dt_seconds": DT_S, "N": N_STEPS,
            "settle_index": SETTLE, "t_ref": 1.0, "time_unit_s": TIME_UNIT, "condition": condition,
            "ambient_C": 25.0 if condition == "25C" else 0.0,
            "input_box": {"u_scale": float(fr["Q_Ah"]), "about": "current [A], discharge positive; 1 C = Q A"}}
    return {"plant": NAME, "knowledge": "real", "meta": meta, "dt": DT_S / TIME_UNIT, "sensors": [3, 4],
            "unmeasured": [0, 1, 2], "labelled": list(LABELLED), "x_std": xs, "y_std": np.asarray(fr["y_std"], float),
            "x_mean": np.asarray(fr["x_mean"], float), "q_std": q, "r_std": np.array([0.3, g["r_vm"]]),
            "prior_mean": None, "prior_std": prior_std(), "model": model(condition), "model_knowledge": "partial",
            "condition": condition, "hold_missing": False,
            # HyperKKL latent design of the research run (DESIGN.md): n_z = n_x + 2, Butterworth alpha 1.3 /min
            # (the largest value <= the simulated battery's 1.5 that passes max|lambda| dt <= 0.1 at dt = 1 s)
            "hyperkkl": {"nz": 7, "alpha": 1.3, "ic_box": ic_box()}}


def ic_box() -> list:
    """Operating box from the TRAINING windows (labelled states + the placeholders of V1, V2), SOC [0, 1]."""
    tr = load("train")
    x = np.where(np.isfinite(tr.x), tr.x, tr.x_fill).reshape(-1, 5).astype(np.float64)
    b = [[float(x[:, j].min()), float(x[:, j].max())] for j in range(5)]
    b[0] = [0.0, 1.0]
    return b


def condition_of(split: str) -> str:
    return "0C" if split.startswith("test0_") else "25C"


# ---------------------------------------------------------------------------
# Scoring (research realdata_run.soc_metrics)
# ---------------------------------------------------------------------------

def soc_metrics(soc_hat, soc, settle=SETTLE, dt_min=DT_S / 60.0, band=0.05) -> dict:
    e = (np.asarray(soc_hat, float) - np.asarray(soc, float)) * 100.0
    e = np.nan_to_num(e, nan=1e4, posinf=1e4, neginf=-1e4)
    rs = np.sqrt(np.mean(e[:, settle:] ** 2, 1))
    ra = np.sqrt(np.mean(e ** 2, 1))
    mx = np.abs(e[:, settle:]).max(1)
    bad = np.abs(e) > 100 * band
    T = e.shape[1]
    last = np.where(bad.any(1), T - 1 - np.argmax(bad[:, ::-1], 1), -1)
    tc = (last + 1) * dt_min
    return {"rmse_settled": rs, "rmse_all": ra, "max_settled": mx, "t_conv_min": tc, "conv_censored": (last == T - 1)}


HEADLINE = "rmse_settled"
METRIC_ABOUT = ("settled SOC RMSE [% SOC] over t >= 15 min of each 30-min window, mean over episodes; "
                "headline = mean of the US06 / HWFET / NN test families at 25 degC")
SCORER = "settled-soc-rmse-1"


def score(X_hat, es: RealEpisodeSet) -> dict:
    """X_hat (E, N+1, 5) or (E, N+1) SOC estimates -> per-episode metrics and the split summary."""
    Xh = np.asarray(X_hat, float)
    soc_hat = Xh[..., 0] if Xh.ndim == 3 else Xh
    m = soc_metrics(soc_hat, es.x[..., 0].astype(np.float64), es.settle_index)
    summ = {k: float(np.mean(v)) for k, v in m.items() if k != "conv_censored"}
    summ["conv_censored_frac"] = float(np.mean(m["conv_censored"]))
    summ["n_episodes"] = int(len(soc_hat))
    return {"summary": summ, "per_episode": {k: np.asarray(v).tolist() for k, v in m.items()}}


def headline(results: dict) -> float:
    """Mean over the three 25 degC test families of the split means of settled SOC RMSE [% SOC]."""
    return float(np.mean([results[f"test_{f}"]["summary"][HEADLINE] for f in TEST_FAMILIES]))


def coulomb_counting(es: RealEpisodeSet) -> np.ndarray:
    """The no-training reference: integrate the current from the prior SOC (the research 'cc' row)."""
    Q = spec(NAME)["frozen"]["Q_Ah"]
    U = es.u.astype(np.float64)
    soc0 = es.prior[:, 0].astype(np.float64)
    return soc0[:, None] - np.concatenate([np.zeros((len(U), 1)), np.cumsum(U[:, :-1], 1)], 1) * DT_S / (3600 * Q)
