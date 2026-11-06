r"""Real-data track ``calce``: SOC of an A123 APR18650M1A LiFePO4 cell (1.1 Ah) on held-out dynamic profiles and at
other temperatures (CALCE battery group, University of Maryland, https://calce.umd.edu/battery-data). Licence: checked by
the author (2026-11-02); the page states only 'Use of this data for publication purposes should include references to
the CALCE article(s)'. Fetched from the source, never redistributed.

Why this cell: a flat LiFePO4 voltage plateau, so SOC is much less observable from voltage than on the NCA / NMC cells of
battery_pf and battery_lg. The source's other dynamic set (INR 18650-20R, NMC) would repeat those tracks and has no
temperature channel; not used.

Each file holds three discharge profiles from a full CC-CV charge to 2.0 V: step 8 = DST, step 16 = US06, step 24 = FUDS
(order on the source page, confirmed by the logged rest intervals 361 s, 601 s, 1373 s = the profile periods).

Splits (fixed in this file BEFORE any estimator was run):
    train            DST of cell A1-007 at 20, 25, 30 degC and of cell A1-008 at 20, 30 degC
    val              DST of cell A1-008 at 25 degC (the only input family available for selection: test inputs held out)
    test_US06, test_FUDS            both cells, 25 degC (input shift, no temperature shift)
    test{T}_{US06, FUDS}            both cells, T = 0, 10, 40, 50 degC (plant shift too: PITFALLS P17)
    -10 degC not used: cell A1-008's records are truncated. Sample 2 of the 20 degC file ('newprofile') not used.
Windows: 30 min every 2 min (train), every 5 min x 2 priors (val / test), as battery_pf.

Label: the lab's coulomb counter from the end of the CC-CV charge, SOC = 1 + Ah / Q, Q = the C/20 discharge capacity of the
same cell at 25 degC (cell Q also used at other temperatures, as battery_pf). The label is an integral of the input, with
the capacity measured once: PITFALLS P15. The case temperature is measured; dT = T - T(first 30 s of the profile).

Partial model: 2-RC ECM (plants/systems_real.Pan18650PF), OCV = mean of the C/20 discharge and charge of cell A1-007 at
25 degC (averages the LFP hysteresis, which the model does not have), R, C on the training DST of A1-007 at 25 degC.
"""

from __future__ import annotations

import re

import numpy as np

from . import battery_lg as BL
from . import battery_pf as BP
from . import registry as R
from .battery_field import FieldBattery
from .battery_pf import Recording, resample

NAME = "calce"
STEP = {"DST": 8, "US06": 16, "FUDS": 24}
FILES = {0: "20120813", 10: "20120815", 20: "20120817", 25: "20120827", 30: "20120820", 40: "20120822", 50: "20120824"}
TEMPS_SHIFT = (0, 10, 40, 50)
# OCV branch of the partial model. 'mean' (first version) averages the C/20 charge and discharge curves; every profile here is a
# discharge from full, and the LFP half-hysteresis (25.8 mV) is worth ~29 % SOC on the 0.9 mV / % plateau. Chosen on VAL (DIAG_REAL.md).
OCV_BRANCH = "discharge"
# Fit range of the ECM identification: the voltage collapse below ~10 % SOC (outside a 2-RC model) dominated the first fit
# (train max residual 805 mV, R0 155 mOhm). Candidate: identify on SOC >= ECM_MIN_SOC only. Chosen on VAL (DIAG_REAL.md).
ECM_MIN_SOC = 0.1


def _parsed(root=None):
    return R.real_root(root) / NAME / "parsed"


def parse_raw(root=None, log=print):
    """Workbooks -> one npz per file under <real root>/calce/parsed. Needs openpyxl; cell values only (no macros, no formulas
    run). The '~$' lock files and the 'newprofile' test inside the zips are skipped."""
    import io
    import zipfile

    import pandas as pd
    out = _parsed(root)
    out.mkdir(parents=True, exist_ok=True)
    for zp in sorted(R.raw_dir(NAME, root).glob("A123_*.zip")):
        z = zipfile.ZipFile(zp)
        for m in z.namelist():
            b = m.rsplit("/", 1)[-1]
            o = out / (b[:-5] + ".npz")
            if not b or b.startswith("~$") or not b.endswith(".xlsx") or "newprofile" in b or o.exists():
                continue
            x = pd.ExcelFile(io.BytesIO(z.read(m)))
            d = pd.concat([x.parse(s) for s in x.sheet_names if s.startswith("Channel")], ignore_index=True)
            tc = [c for c in d.columns if c.lower().startswith("temperature")]
            if not tc and "Sheet1" in x.sheet_names:
                s1 = x.parse("Sheet1")
                t1 = [c for c in s1.columns if c.lower().startswith("temperature")]
                if t1 and len(s1) == len(d):
                    d["Temperature"], tc = s1[t1[0]].to_numpy(), ["Temperature"]
            np.savez_compressed(o, t=d["Test_Time(s)"].to_numpy(float), step=d["Step_Index"].to_numpy(int), cyc=d["Cycle_Index"].to_numpy(int),
                                I=d["Current(A)"].to_numpy(float), V=d["Voltage(V)"].to_numpy(float), qc=d["Charge_Capacity(Ah)"].to_numpy(float),
                                qd=d["Discharge_Capacity(Ah)"].to_numpy(float), T=d[tc[0]].to_numpy(float) if tc else np.full(len(d), np.nan))
            log(f"[parse] {b}")


def _ocv_file(cell, root=None):
    return _parsed(root) / f"A1-00{cell}-OCV-25-20120905.npz"


def cell_q_ocv(cell, root=None):
    z = np.load(_ocv_file(cell, root))
    t, I, V, s = z["t"], z["I"], z["V"], z["step"]
    def seg(k):
        m = s == k
        ah = np.concatenate([[0.0], np.cumsum(0.5 * (I[m][1:] + I[m][:-1]) * np.diff(t[m]))]) / 3600
        return ah, V[m]
    a_d, v_d = seg(5)
    a_c, v_c = seg(7)
    Q = float(-a_d[-1])
    g = np.linspace(0.0, 1.0, 201)
    od = np.interp(g, (1 + a_d / Q)[::-1], v_d[::-1])
    oc = np.interp(g, a_c / a_c[-1], v_c)
    return {"Q": Q, "soc": g, "ocv": od if OCV_BRANCH == "discharge" else 0.5 * (od + oc), "ocv_mean": 0.5 * (od + oc), "ocv_dis": od,
            "hyst_V_mid": float(np.median((oc - od)[50:150]))}


def recording(cell, temp, cyc, root=None) -> Recording:
    p = _parsed(root) / f"A1-00{cell}-DST-US06-FUDS-{temp}-{FILES[temp]}.npz"
    z = np.load(p)
    m = z["step"] == STEP[cyc]
    t, I, V, T = z["t"][m], z["I"][m], z["V"][m], z["T"][m]
    Q = cell_q_ocv(cell, root)["Q"]
    ah = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t))]) / 3600.0
    raw = {"Time": t, "Current": I, "Voltage": V, "Ah": ah, "Battery_Temp_degC": T}
    T0 = float(np.median(T[:30]))
    r = resample(raw, 1.0, Q, T0)
    return Recording(name=f"A1-00{cell}_{cyc}_{temp}C", temp=float(temp), cycle=cyc, Q=Q, dt=1.0, t_amb_eff=T0, path=str(p),
                     cc_check=r["cc_check"], **{k: r[k] for k in ("t", "u", "vbar", "vm", "temp_case", "ah", "soc")})


def load_sets(root=None) -> dict:
    if not any(_parsed(root).glob("*.npz")):
        parse_raw(root)
    d = {"train": [recording(7, t, "DST", root) for t in (20, 25, 30)] + [recording(8, t, "DST", root) for t in (20, 30)],
         "val": [recording(8, 25, "DST", root)]}
    for c in ("US06", "FUDS"):
        d[f"test_{c}"] = [recording(k, 25, c, root) for k in (7, 8)]
        for T in TEMPS_SHIFT:
            d[f"test{T}_{c}"] = [recording(k, T, c, root) for k in (7, 8)]
    d["_Q"] = cell_q_ocv(7, root)["Q"]
    d["_offset"] = 0.0
    return d


def identify(d) -> dict:
    oc = cell_q_ocv(7)
    fit = BP.fit_ocv(oc["soc"], oc["ocv"])
    ocv_fn = lambda s: BP.ocv_eval(s, fit["coef"], fit["w0"], fit["w1"])
    import dataclasses
    recs = []
    for r in d["train"]:
        if r.name != "A1-007_DST_25C":
            continue
        n = int(np.argmax(r.soc < ECM_MIN_SOC)) if (r.soc < ECM_MIN_SOC).any() else len(r)
        recs.append(dataclasses.replace(r, **{k: getattr(r, k)[:n] for k in ("t", "u", "vbar", "vm", "temp_case", "ah", "soc")}))
    ecm = BL.fit_ecm_bounded(recs, ocv_fn)
    th = BP.fit_thermal(recs, ecm)
    p = {"Q": oc["Q"], "R0": ecm["R0"], "R1": ecm["R1"], "C1": ecm["C1"], "R2": ecm["R2"], "C2": ecm["C2"], "C_th": th["C_th"], "hA": th["hA"],
         "Ea_R": 3000.0, "dUdT": -1.0e-4, "T_amb": 273.15 + 25.0, "tau_f": BP.TAU_F, "ow0": fit["w0"], "ow1": fit["w1"],
         **{f"o{i}": float(c) for i, c in enumerate(fit["coef"])}}
    return {"params": {k: float(v) for k, v in p.items()}, "ecm": ecm, "thermal": th, "ocv_rms": fit["rms"],
            "note": f"LFP hysteresis (charge - discharge C/20 voltage, mid SOC) {oc['hyst_V_mid'] * 1e3:.1f} mV; OCV = DISCHARGE branch, ECM on SOC >= {ECM_MIN_SOC} (chosen on val)"}


def _cond(split):
    m = re.match(r"test(\d+)_", split)
    return f"{m.group(1)}C" if m else "25C"


T = FieldBattery(name=NAME, N=1800, load_sets=load_sets, identify=identify, condition_of=_cond,
                 test_splits=["test_US06", "test_FUDS"] + [f"test{t}_{c}" for t in TEMPS_SHIFT for c in ("US06", "FUDS")],
                 headline_splits=["test_US06", "test_FUDS"], prep_version="calce-prep-1",
                 metric_about="settled SOC RMSE [% SOC] over t >= 15 min of each 30-min window, mean over episodes; headline = mean of US06 / FUDS at 25 degC (LiFePO4 plateau)")
T.export(globals())
