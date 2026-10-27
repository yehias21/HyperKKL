r"""Real-data battery SOC experiment: loader, drive-cycle split, ECM identification, episodes.

Data: Panasonic 18650PF Li-ion cell (P. Kollmeyer, Univ. Wisconsin-Madison), Mendeley Data
DOI 10.17632/wykht8y7tg.1, CC BY 4.0 (experiments/real/DATA.md: URLs, checksums). Each
drive-cycle file is one discharge from full charge (1 C CC-CV to 4.2 V, 50 mA cut-off) to
2.5 V (25 degC) or to 80 % DOD (0 degC), logged at 0.1 s: Time [s], Voltage [V], Current [A,
negative = discharge], Ah (counter reset at the start of the file), Battery_Temp_degC (case
thermocouple), Chamber_Temp_degC.

Conventions (as hyperkkl/systems_extra.LiIonCell):
    u      current, POSITIVE = discharge [A]; zero-order hold on [t_k, t_k+1): u_k is the
           mean current of that bin (charge-preserving resampling of the 0.1 s log)
    SOC    reference = 1 + Ah_lab / Q (the lab's coulomb counter, Q = measured C/20 capacity)
    y      (dT, Vm):  dT = T_case - T_amb_eff [K]; Vm = BMS voltage channel, the bin-mean
           terminal voltage minus V_REF behind a first-order acquisition filter (tau_f), applied
           in software with its exact ZOH discretisation, Vm_{k+1} = a Vm_k + (1 - a)(Vbar_k - V_REF),
           a = exp(-dt / tau_f). This is the measured quantity the plant's Vm state models,
           so the filters' y = x[:, sensors] contract holds on real data.
    x      (SOC, V1, V2, dT, Vm). Labelled on real data: SOC (reference), dT and Vm (measured).
           V1, V2 (RC polarisation voltages) are NOT measurable; episodes carry the identified
           model's replay of them as placeholders, masked out of every loss (``LABELLED``).

The identified model ``Pan18650PF`` is the standard 2-RC Thevenin ECM a BMS engineer would
fit: OCV(SOC) from the C/20 test (mean of the discharge and charge branches), constant R0,
R1, C1, R2, C2 fitted by least squares to the bin-mean terminal voltage of the TRAINING
cycles, a lumped thermal model (C_th, hA fitted the same way), Arrhenius R(T) with a
literature activation energy, no hysteresis, no SOC- or rate-dependence of R or capacity.
It is therefore a realistic, imperfect model.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .config import ROOT
from .systems import Plant, _pos, register

REAL = ROOT / "experiments" / "real"
RAW = REAL / "raw" / "panasonic18650pf" / "Panasonic_18650PF_Data"
FIELDS = ("Time", "Voltage", "Current", "Ah", "Battery_Temp_degC", "Chamber_Temp_degC")
V_REF = 3.7
TAU_F = 2.0            # s, software BMS acquisition filter (as LiIonCell.tau_f)
TIME_UNIT = 60.0       # s per model time unit (minutes, as LiIonCell)
LABELLED = (0, 3, 4)   # SOC, dT, Vm carry labels on real data; V1, V2 do not

# Drive-cycle split (pre-registered in experiments/real/DESIGN.md). Cycles 1-4 are random
# mixes of US06, HWFET, UDDS, LA92 and NN segments and NN combines US06 and LA92 portions
# (dataset readme), so the test is SEQUENCE OOD (event order, dwell, pure repetition of one
# profile), not strict shape OOD; the strict variant (no recording containing any US06 /
# HWFET material in training) is SPLIT_STRICT.
SPLIT = {"train": ("Cycle_1", "Cycle_2", "Cycle_3", "UDDS", "LA92"),
         "val": ("Cycle_4",),
         "test": ("US06", "HWFTa", "HWFTb", "NN")}
SPLIT_STRICT = {"train": ("UDDS", "LA92"), "val": ("Cycle_4",), "test": ("US06", "HWFTa", "HWFTb", "NN")}
FAMILY = {"HWFTa": "HWFET", "HWFTb": "HWFET", "HWFET": "HWFET"}
TEST_FAMILIES = ("US06", "HWFET", "NN")


# ---------------------------------------------------------------------------
# Loading and resampling
# ---------------------------------------------------------------------------

def load_mat(path) -> dict:
    """The ``meas`` struct of one file as float64 arrays (+ 'TimeStamp' strings). Uses
    scipy.io.loadmat (MAT v5 parser; no code execution, no pickles)."""
    import scipy.io as sio
    m = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if "meas" not in m:
        raise ValueError(f"{path}: no 'meas' struct")
    s = m["meas"]
    out = {}
    for f in FIELDS:
        if f not in s._fieldnames:
            raise ValueError(f"{path}: field {f} missing")
        out[f] = np.atleast_1d(np.asarray(getattr(s, f), dtype=np.float64))
    ts = getattr(s, "TimeStamp", None)
    out["TimeStamp"] = [str(v) for v in np.atleast_1d(ts)] if ts is not None else []
    n = len(out["Time"])
    for f in FIELDS:
        if len(out[f]) != n:
            raise ValueError(f"{path}: field {f} has {len(out[f])} samples, Time has {n}")
    return out


def parse_name(path) -> tuple:
    """'03-20-17_01.43_25degC_US06_Pan18650PF.mat' -> (25.0, 'US06')."""
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
    temp: float                # nominal ambient [degC]
    cycle: str                 # file descriptor (US06, HWFTa, Cycle_1, ...)
    t: np.ndarray              # (n,) s, grid k dt
    u: np.ndarray              # (n,) A, discharge +, mean over [t_k, t_k + dt)
    vbar: np.ndarray           # (n,) V, mean terminal voltage over [t_k, t_k + dt)
    vm: np.ndarray             # (n,) V, filtered channel at t_k (minus V_REF)
    temp_case: np.ndarray      # (n,) degC at t_k
    ah: np.ndarray             # (n,) lab Ah counter at t_k
    soc: np.ndarray            # (n,) reference SOC at t_k
    Q: float
    dt: float
    t_amb_eff: float           # degC, ambient incl. the thermocouple's rest offset
    abs_start: object = None
    abs_end: object = None
    path: str = ""
    sha256: str = ""
    cc_check: float = 0.0      # max |own current integral - lab Ah| [Ah]

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


def resample(raw: dict, dt: float = 1.0, Q: float = 2.9, t_amb_eff: float = 25.0,
             tau_f: float = TAU_F) -> dict:
    """0.1 s log -> fixed grid t_k = k dt. Bin means of current and voltage come from the
    cumulative trapezoid integral (charge-preserving ZOH), samples of T and Ah by linear
    interpolation. Duplicate / non-increasing time stamps are dropped."""
    t = raw["Time"]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    I = -raw["Current"][keep]                           # discharge positive
    V = raw["Voltage"][keep]
    n = int(math.floor((t[-1] - t[0]) / dt))            # bins [t_k, t_k+1) inside the log
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
    own = -(qI[:-1] - qI[0]) / 3600.0                    # lab sign: Ah < 0 on discharge
    return {"t": tk - t[0], "u": u, "vbar": vbar, "vm": vm,
            "temp_case": np.interp(tk, t, raw["Battery_Temp_degC"][keep]),
            "ah": ah, "soc": 1.0 + ah / Q, "cc_check": float(np.abs(own - (ah - ah[0])).max())}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recording(path, Q: float, t_amb_eff: float = None, dt: float = 1.0,
                   offset: float = 0.0) -> Recording:
    """One drive-cycle file. ``t_amb_eff`` defaults to the nominal ambient + ``offset``
    (the case thermocouple's reading at rest minus the chamber, from the training files)."""
    raw = load_mat(path)
    temp, cyc = parse_name(path)
    tae = temp + offset if t_amb_eff is None else t_amb_eff
    r = resample(raw, dt, Q, tae)
    ts = [_parse_ts(raw["TimeStamp"][0]) if raw["TimeStamp"] else None]
    # end = start + logged duration (some files' last TimeStamp is not the end of the log)
    end = ts[0] + _dt.timedelta(seconds=float(raw["Time"][-1] - raw["Time"][0])) if ts[0] else None
    return Recording(name=Path(path).stem, temp=temp, cycle=cyc, Q=Q, dt=dt, t_amb_eff=tae,
                     abs_start=ts[0], abs_end=end, path=str(path), sha256=sha256(path),
                     **{k: r[k] for k in ("t", "u", "vbar", "vm", "temp_case", "ah", "soc")},
                     cc_check=r["cc_check"])


def drive_cycle_files(temp: float, root: Path = RAW) -> dict:
    """{cycle descriptor: path} for the single-cycle files at one ambient temperature
    (the long concatenated files, e.g. 'LA92_NN', are excluded: they duplicate data)."""
    tag = f"{'n' if temp < 0 else ''}{abs(int(temp))}degC"
    out = {}
    for p in sorted((root / tag / "Drive_cycles").glob("*.mat")):
        t, c = parse_name(p)
        if t != temp or "_" in c.replace("Cycle_", "Cycle"):
            continue
        out[c] = p
    return out


def c20_capacity_and_ocv(path, n_grid: int = 201) -> dict:
    """C/20 discharge then charge (25 degC). Q = Ah removed between full charge and 2.5 V.
    OCV(SOC) = mean of the discharge and charge branches where both exist (the R I drop and
    hysteresis cancel to first order); above the charge branch's top SOC the discharge
    branch is shifted up by the mean half-gap of the top 20 % of the overlap."""
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
    return {"Q": Q, "soc": g, "ocv": ocv, "dis": od, "chg": oc, "top_charge_soc": top,
            "half_gap": half_gap}


# ---------------------------------------------------------------------------
# OCV curve: smooth, C^1 beyond [0, 1] (linear extension), torch and numpy
# ---------------------------------------------------------------------------

OCV_DEG = 7
OCV_W = (0.02, 0.98)   # widths of the two exponential edge terms are fitted on a grid


def ocv_basis(s, w0, w1, xp=np):
    """[1, s, ..., s^DEG, exp(-s / w0), exp(-(1 - s) / w1)] on s in [0, 1]."""
    cols = [s ** i for i in range(OCV_DEG + 1)]
    cols += [xp.exp(-s / w0), xp.exp(-(1.0 - s) / w1)]
    return cols


def fit_ocv(soc, ocv) -> dict:
    """Least squares in the coefficients, grid over the two edge widths."""
    best = None
    for w0 in (0.01, 0.02, 0.04, 0.08):
        for w1 in (0.02, 0.05, 0.1, 1e9):
            X = np.stack(ocv_basis(soc, w0, w1), 1)
            c, *_ = np.linalg.lstsq(X, ocv, rcond=None)
            r = X @ c - ocv
            rms = float(np.sqrt(np.mean(r ** 2)))
            if best is None or rms < best["rms"]:
                best = {"coef": c.tolist(), "w0": w0, "w1": w1, "rms": rms,
                        "max": float(np.abs(r).max())}
    return best


def ocv_eval(s, coef, w0, w1, xp=np):
    """OCV with the linear extension beyond [0, 1] (slope of the edge), C^1."""
    if xp is np:
        sc = np.clip(s, 0.0, 1.0)
    else:
        sc = s.clamp(0.0, 1.0)
    v = sum(c * b for c, b in zip(coef, ocv_basis(sc, w0, w1, xp)))
    h = 1e-4
    lo = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + h, w0, w1, xp)))
    v0 = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc, w0, w1, xp)))
    hi = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + 1.0, w0, w1, xp)))
    hm = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + 1.0 - h, w0, w1, xp)))
    s0, s1 = (lo - v0) / h, (hi - hm) / h
    return v + s0 * xp.minimum(s - sc, 0.0 * s) + s1 * xp.maximum(s - sc, 0.0 * s) if xp is np else \
        v + s0 * torch.clamp(s - sc, max=0.0) + s1 * torch.clamp(s - sc, min=0.0)


# ---------------------------------------------------------------------------
# The identified plant
# ---------------------------------------------------------------------------

N_OCV = OCV_DEG + 3


@register("battery_pf")
class Pan18650PF(Plant):
    r"""Identified 2-RC ECM of the Panasonic 18650PF cell, same state, input and output
    convention as LiIonCell (time unit 1 min, SI parameters):

        SOC' = -I / (3600 Q)
        Vi'  = I / Ci - Vi / (Ri(T) Ci),  i = 1, 2
        C_th dT' = I^2 R0(T) + V1^2 / R1(T) + V2^2 / R2(T) - I T dUdT - hA dT
        tau_f Vm' = OCV(SOC) + dUdT dT - V1 - V2 - R0(T) I - V_REF - Vm
        y = (dT, Vm)

    I is the measured cell current (no BMS derating: the lab current is what the cell saw).
    OCV coefficients o0..o9 and edge widths ow0, ow1 come from fit_ocv; R(T) Arrhenius
    relative to T_amb (Ea_R = 3000 K, as LiIonCell; not identifiable from 25 degC data)."""
    nx = 5
    PARAMS = {"Q": 2.9, "R0": 0.03, "R1": 0.01, "C1": 2000.0, "R2": 0.01, "C2": 50000.0,
              "C_th": 60.0, "hA": 0.1, "Ea_R": 3000.0, "dUdT": -1.0e-4, "T_amb": 298.15,
              "tau_f": TAU_F, "ow0": 0.02, "ow1": 0.05,
              **{f"o{i}": 0.0 for i in range(N_OCV)}}
    NOT_PERTURBED = ("T_amb", "tau_f", "ow0", "ow1") + tuple(f"o{i}" for i in range(N_OCV))
    DEFAULT_SENSORS = (3, 4)
    TIME_UNIT = TIME_UNIT
    V_REF = V_REF
    UNMODELLED = "identified ECM: no hysteresis, no SOC/rate dependence of R, Q"

    def ocv(self, soc):
        return ocv_eval(soc, [getattr(self, f"o{i}") for i in range(N_OCV)], self.ow0, self.ow1,
                        xp=torch)

    def f(self, x, u):
        soc, v1, v2, dT, vm = x.unbind(1)
        I = u[:, 0]
        T = self.T_amb + dT
        arr = torch.exp(self.Ea_R * (1.0 / T - 1.0 / self.T_amb))
        r0, r1, r2 = self.R0 * arr, self.R1 * arr, self.R2 * arr
        heat = I ** 2 * r0 + v1 ** 2 / r1 + v2 ** 2 / r2 - I * T * self.dUdT - self.hA * dT
        v_t = self.ocv(soc) + self.dUdT * dT - v1 - v2 - r0 * I
        s = self.TIME_UNIT
        return torch.stack([-s * I / (3600.0 * self.Q),
                            s * (I - v1 / r1) / self.C1,
                            s * (I - v2 / r2) / self.C2,
                            s * heat / self.C_th,
                            s * (v_t - self.V_REF - vm) / self.tau_f], 1)

    def f_known(self, x, u):
        return self.f(x, u)


# ---------------------------------------------------------------------------
# ECM identification (training recordings only)
# ---------------------------------------------------------------------------

def _rc_avg(u, R, tau, dt):
    """Bin-average of the RC voltage under ZOH current, from rest:
    V_{k+1} = e V_k + R (1 - e) u_k; mean over the bin = g V_k + R (1 - g) u_k,
    g = (tau / dt)(1 - e)."""
    from scipy.signal import lfilter
    e = math.exp(-dt / tau)
    Vk = lfilter([0.0, R * (1 - e)], [1.0, -e], u)       # V_k (state at the bin start)
    g = tau / dt * (1 - e)
    return g * Vk + R * (1 - g) * u, Vk


def ecm_voltage(theta, u, soc_mid, ocv_fn, dt):
    R0, R1, t1, R2, t2 = np.exp(theta)
    a1, _ = _rc_avg(u, R1, t1, dt)
    a2, _ = _rc_avg(u, R2, t2, dt)
    return ocv_fn(soc_mid) - a1 - a2 - R0 * u


def fit_ecm(recs, ocv_fn, dt=1.0, x0=(0.02, 0.01, 10.0, 0.01, 200.0)) -> dict:
    """Least squares of the bin-mean terminal voltage over the TRAINING recordings
    (each starts from rest after the CC-CV charge: V1 = V2 = 0)."""
    from scipy.optimize import least_squares
    data = []
    for r in recs:
        soc_mid = r.soc - 0.5 * r.u * dt / (3600.0 * r.Q)
        data.append((r.u, soc_mid, r.vbar))

    def res(th):
        return np.concatenate([ecm_voltage(th, u, s, ocv_fn, dt) - v for u, s, v in data])

    sol = least_squares(res, np.log(np.asarray(x0)), x_scale=1.0, method="trf")
    R0, R1, t1, R2, t2 = np.exp(sol.x)
    if t1 > t2:                                   # order the branches: fast first
        R1, t1, R2, t2 = R2, t2, R1, t1
    rr = res(np.log([R0, R1, t1, R2, t2]))
    return {"R0": R0, "R1": R1, "C1": t1 / R1, "R2": R2, "C2": t2 / R2, "tau1_s": t1, "tau2_s": t2,
            "v_rmse_mV": 1e3 * float(np.sqrt(np.mean(rr ** 2))), "v_max_mV": 1e3 * float(np.abs(rr).max()),
            "nfev": int(sol.nfev)}


def fit_thermal(recs, ecm, dUdT=-1.0e-4, dt=1.0) -> dict:
    """C_th, hA by least squares of dT under the identified heat source (Euler, 1 s)."""
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
    """Thermocouple reading at rest minus the nominal ambient: median over the first n_s
    seconds of the given recordings (the cell rests in the chamber before each test)."""
    return float(np.median(np.concatenate([r.temp_case[:n_s] - r.temp for r in recs])))


def identify(train_recs, c20_path) -> dict:
    """Everything the modeller identifies, from the training recordings + C/20 test."""
    oc = c20_capacity_and_ocv(c20_path)
    fit = fit_ocv(oc["soc"], oc["ocv"])
    ocv_fn = lambda s: ocv_eval(s, fit["coef"], fit["w0"], fit["w1"])
    ecm = fit_ecm(train_recs, ocv_fn)
    th = fit_thermal(train_recs, ecm)
    params = {"Q": oc["Q"], "R0": ecm["R0"], "R1": ecm["R1"], "C1": ecm["C1"], "R2": ecm["R2"],
              "C2": ecm["C2"], "C_th": th["C_th"], "hA": th["hA"], "Ea_R": 3000.0, "dUdT": -1.0e-4,
              "T_amb": 273.15 + float(np.mean([r.t_amb_eff for r in train_recs])), "tau_f": TAU_F,
              "ow0": fit["w0"], "ow1": fit["w1"], **{f"o{i}": float(c) for i, c in enumerate(fit["coef"])}}
    return {"params": {k: float(v) for k, v in params.items()}, "ocv_fit": fit, "c20": {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in oc.items()}, "ecm": ecm,
        "thermal": th}


# ---------------------------------------------------------------------------
# Splits and episodes
# ---------------------------------------------------------------------------

def check_split(split: dict, recs: dict) -> None:
    """No leakage: each recording in at most one role, and no two roles share any
    absolute time interval (catches a file that duplicates another's data)."""
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
    """Starts s with s + N + 1 <= n - 1 (u_{s+N} must exist)."""
    return list(range(offset, n - N - 1, stride))


def replay_many(plant, recs, device="cpu") -> list:
    """The identified model driven by each recording's current from its measured start
    (SOC = reference, V1 = V2 = 0 at rest after the charge, dT and Vm measured), one padded
    batch (RK4, ZOH input, float64). Returns a list of (n_i, 5) arrays."""
    from .data import Design, integrate
    from . import design as D
    recs = list(recs)
    n = max(len(r) for r in recs)
    dt_m = recs[0].dt / TIME_UNIT
    A, B = D.butterworth_AB(1, 1.0, 2)
    des = Design(A, B, dt_m, n - 1, 1.0, {})
    x0 = np.array([[r.soc[0], 0.0, 0.0, r.dT[0], r.vm[0]] for r in recs])
    sigs = []
    for r in recs:
        u = np.zeros(n); u[:len(r)] = r.u
        sigs.append(lambda t, u=u: u[np.clip(np.round(t / dt_m).astype(int), 0, n - 1)])
    x, _, _ = integrate(plant, des, x0, sigs, device=device)
    x = x.cpu().numpy()
    return [x[i, :len(r)] for i, r in enumerate(recs)]


def replay(plant, rec: Recording, device="cpu") -> np.ndarray:
    return replay_many(plant, [rec], device)[0]


def episodes(recs, replays, N: int, stride: int, des, burn: int, offset: int = 0):
    """EpisodeSet of windows: x = (SOC_ref, V1_model, V2_model, dT_meas, Vm_meas), u, y =
    (dT, Vm), z = z_obs = the deployed latent filter run on y from z0 = 0. The x columns
    V1, V2 are the identified model's replay (``replays``, from replay_many; placeholders,
    masked from every loss)."""
    from .data import EpisodeSet, run_latent
    X, U, Y, meta = [], [], [], []
    for r, rp in zip(recs, replays):
        for s in window_starts(len(r), N, stride, offset):
            sl = slice(s, s + N + 1)
            x = np.stack([r.soc[sl], rp[sl, 1], rp[sl, 2], r.dT[sl], r.vm[sl]], 1)
            X.append(x); U.append(r.u[sl, None]); Y.append(r.y[sl])
            meta.append({"rec": r.name, "cycle": r.cycle, "family": r.family, "temp": r.temp,
                         "start": int(s), "soc0": float(r.soc[s])})
    if not X:
        raise ValueError("no windows")
    t = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32)
    x, u, y = t(X), t(U), t(Y)
    zo = run_latent(des, y.double()).float()
    return EpisodeSet(x, u, y.clone(), y, zo.clone(), zo, [m["family"] for m in meta], burn,
                      des.dt, meta, 0)


def masked_scales(scales: dict, labelled=LABELLED) -> dict:
    """scales whose x_std makes train.phase2's loss the MASKED mean over the labelled states:
    unlabelled states get sigma = inf (their error contributes exactly 0, and so does its
    gradient), labelled ones sigma * sqrt(n_lab / nx), so mean_j (e_j / s_j)^2 over all nx
    states equals the mean over the labelled ones with their original sigma."""
    xs = np.asarray(scales["x_std"], float)
    nx, lab = len(xs), sorted(set(labelled))
    f = math.sqrt(len(lab) / nx)
    out = np.full(nx, np.inf)
    out[lab] = xs[lab] * f
    return dict(scales, x_std=out.tolist())


# ---------------------------------------------------------------------------
# EKF with the analytic Jacobian of Pan18650PF (same algorithm as observers.EKF; vmap(jacrev)
# costs most of an EKF step on a loaded CPU)
# ---------------------------------------------------------------------------

def jacobian(plant: Pan18650PF, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """d f / d x, (B, 5, 5), analytic. ocv'(soc) by central difference (h = 1e-6)."""
    p = plant
    soc, v1, v2, dT, vm = x.unbind(1)
    I = u[:, 0]
    T = p.T_amb + dT
    arr = torch.exp(p.Ea_R * (1.0 / T - 1.0 / p.T_amb))
    r0, r1, r2 = p.R0 * arr, p.R1 * arr, p.R2 * arr
    dlog = -p.Ea_R / T ** 2                                   # d r_i / d dT = r_i dlog
    h = 1e-6
    docv = (p.ocv(soc + h) - p.ocv(soc - h)) / (2 * h)
    s = p.TIME_UNIT
    J = torch.zeros(x.shape[0], 5, 5, dtype=x.dtype, device=x.device)
    J[:, 1, 1] = -s / (r1 * p.C1)
    J[:, 1, 3] = s * v1 / (p.C1 * r1) * dlog
    J[:, 2, 2] = -s / (r2 * p.C2)
    J[:, 2, 3] = s * v2 / (p.C2 * r2) * dlog
    J[:, 3, 1] = s * 2 * v1 / r1 / p.C_th
    J[:, 3, 2] = s * 2 * v2 / r2 / p.C_th
    J[:, 3, 3] = s * (I ** 2 * r0 * dlog - v1 ** 2 / r1 * dlog - v2 ** 2 / r2 * dlog
                      - I * p.dUdT - p.hA) / p.C_th
    J[:, 4, 0] = s * docv / p.tau_f
    J[:, 4, 1] = -s / p.tau_f
    J[:, 4, 2] = -s / p.tau_f
    J[:, 4, 3] = s * (p.dUdT - I * r0 * dlog) / p.tau_f
    J[:, 4, 4] = -s / p.tau_f
    return J


def make_ekf(plant, dt, q, r, p0):
    """observers.EKF whose Jacobian is ``jacobian`` instead of vmap(jacrev(f))."""
    from .observers import EKF

    class AnalyticEKF(EKF):
        def reset(self, B, x0_hat, u0):
            super().reset(B, x0_hat, u0)
            self.jac = lambda x, u: jacobian(self.m, x, u)
    f = AnalyticEKF(plant, dt, q, r, p0)
    f.name = "ekf"
    return f
