r"""Real-data track ``battery_lg``: SOC estimation of an LG 18650HG2 Li-ion cell (3 Ah) on held-out drive
cycles and at other ambient temperatures (Kollmeyer, Naguib, Skells, McMaster University; Mendeley Data,
doi:10.17632/b5mj79w5w9.2, CC BY 4.0).

The same task, preprocessing, model class and scorer as ``battery_pf`` (whose code it reuses), on a second
cell chemistry, a second lab and a second set of drive cycles, with no information from battery_pf. The
split and the latent design were fixed in this file BEFORE any estimator was run on these data:

    train          UDDS, LA92, Mixed1-6 at 25 degC (windows of 30 min every 2 min)
    val            Mixed7 at 25 degC (every 5 min, x 2 priors SOC0 +- 0.2)
    test_US06, test_Mixed8            25 degC
    test{T}_{US06, HWFET, Mixed8}     T = 0, 10, 40 degC (temperature shift; the 25 degC model moved by its
                                      Arrhenius factor, ``context(condition="0C")`` etc.)
    The 25 degC HWFET file is a 600 s rest with no current, and the 40 degC Mixed8 file is truncated (1438 s,
    aborted at SOC ~0.93): both verified, neither used.

Differences to battery_pf, all stated: no chamber temperature channel (dT = case temperature minus the
nominal ambient plus the thermocouple rest offset of the training files, as there); the case sensor is an
AD592 (+-1 degC); the drive-cycle power profiles are repeated until 95 % of the 1C capacity is discharged.

Partial model: the 2-RC ECM of plants/systems_real.Pan18650PF (same equations; the class name is historical),
identified on the training cycles of THIS cell + its C/20 test (``identify_lg``: battery_pf's procedure with
physical bounds on the RC branches, see fit_ecm_bounded; frozen in
resources/real_battery_lg.json by ``python -m hyperkkl_bench.real.battery_lg freeze``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from . import battery_pf as BP
from .battery_pf import (DT_S, INIT_ERR, LABELLED, N_STEPS, SETTLE, STATE_NAMES, STRIDE_EVAL, STRIDE_TRAIN,  # noqa: F401
                         TIME_UNIT, RealEpisodeSet, Recording, check_split, expand, load_mat, replay,
                         resample, rest_offset, soc_metrics, windows)
from .registry import processed_dir, raw_dir, spec

NAME = "battery_lg"
C20_GLOB = "25degC/*C20DisCh_25degC_LGHG2.mat"
SPLIT = {"train": ("UDDS", "LA92", "Mixed1", "Mixed2", "Mixed3", "Mixed4", "Mixed5", "Mixed6"),
         "val": ("Mixed7",), "test": ("US06", "Mixed8")}
TEST_FAMILIES = ("US06", "Mixed8")
SHIFT_TEMPS = (0, 10, 40)
SHIFT_CYCLES = ("US06", "HWFET", "Mixed8")
SHIFT_SKIP = {(40, "Mixed8")}          # the 40 degC Mixed8 file is truncated (1438 s, aborted at SOC ~0.93): not used
PREP_VERSION = "battery_lg-prep-1"
HEADLINE = "rmse_settled"
SCORER = BP.SCORER
METRIC_ABOUT = ("settled SOC RMSE [% SOC] over t >= 15 min of each 30-min window, mean over episodes; "
                "headline = mean of the US06 / Mixed8 test families at 25 degC")
score = BP.score


def splits() -> list:
    return ["train", "val"] + [f"test_{f}" for f in TEST_FAMILIES] + \
        [f"test{T}_{c}" for T in SHIFT_TEMPS for c in SHIFT_CYCLES if (T, c) not in SHIFT_SKIP]


def parse_name(path) -> tuple:
    m = re.search(r"_(\w+?)_(n?)(\d+)degC_LGHG2", Path(path).name)
    if not m:
        raise ValueError(f"unrecognised file name {Path(path).name}")
    return float(m.group(3)) * (-1.0 if m.group(2) else 1.0), m.group(1)


def load_recording(path, Q: float, t_amb_eff: float = None, offset: float = 0.0) -> Recording:
    import datetime as _dt
    raw = load_mat(path)
    temp, cyc = parse_name(path)
    tae = temp + offset if t_amb_eff is None else t_amb_eff
    r = resample(raw, DT_S, Q, tae)
    ts = BP._parse_ts(raw["TimeStamp"][0]) if raw["TimeStamp"] else None
    end = ts + _dt.timedelta(seconds=float(raw["Time"][-1] - raw["Time"][0])) if ts else None
    return Recording(name=Path(path).stem, temp=temp, cycle=cyc, Q=Q, dt=DT_S, t_amb_eff=tae, abs_start=ts,
                     abs_end=end, path=str(path), sha256=BP._sha256(path),
                     **{k: r[k] for k in ("t", "u", "vbar", "vm", "temp_case", "ah", "soc")}, cc_check=r["cc_check"])


def _files(base: Path, temp: int) -> dict:
    out = {}
    for p in sorted((base / f"{temp}degC").glob("*_LGHG2.mat")):
        t, c = parse_name(p)
        if c != "C20DisCh":
            out[c] = p
    return out


def _c20(base: Path) -> Path:
    hits = sorted(base.glob(C20_GLOB))
    if len(hits) != 1:
        raise FileNotFoundError(f"C/20 test not found under {base}")
    return hits[0]


def load_all(root=None) -> dict:
    base = raw_dir(NAME, root)
    oc = BP.c20_capacity_and_ocv(_c20(base))
    Q = oc["Q"]
    f25 = _files(base, 25)
    recs = {c: load_recording(f25[c], Q) for c in SPLIT["train"] + SPLIT["val"] + SPLIT["test"]}
    off = rest_offset([recs[c] for c in SPLIT["train"]])
    for r in recs.values():
        r.t_amb_eff = r.temp + off
    check_split(SPLIT, recs)
    shift = {}
    for T in SHIFT_TEMPS:
        fT = _files(base, T)
        shift[T] = {c: load_recording(fT[c], Q, t_amb_eff=T + off) for c in SHIFT_CYCLES if (T, c) not in SHIFT_SKIP}
    return {"Q": Q, "offset_K": off, "recs25": recs, "shift": shift, "base": base}


def model(condition: str = "25C", frozen: dict | None = None):
    from ..plants.systems_real import Pan18650PF, cold_params
    fr = frozen or spec(NAME)["frozen"]
    if not fr:
        raise RuntimeError(f"{NAME}: the identified model is not frozen yet (python -m hyperkkl_bench.real.battery_lg freeze)")
    p = dict(fr["params_25C"])
    T = float(condition.rstrip("C"))
    if T != 25.0:
        p, _ = cold_params(p, T + fr["thermocouple_offset_K"])
    return Pan18650PF(sensors=(3, 4), **p)


def condition_of(split: str) -> str:
    m = re.match(r"test(\d+)_", split)
    return f"{m.group(1)}C" if m else "25C"


def build_sets(d: dict, frozen=None) -> dict:
    recs = d["recs25"]
    m25 = model("25C", frozen)
    rp = {c: replay(m25, recs[c]) for c in SPLIT["train"] + SPLIT["val"]}
    sets = {"train": windows([recs[c] for c in SPLIT["train"]], STRIDE_TRAIN, [rp[c] for c in SPLIT["train"]]),
            "val": expand(windows([recs[c] for c in SPLIT["val"]], STRIDE_EVAL, [rp[c] for c in SPLIT["val"]]))}
    for f in TEST_FAMILIES:
        sets[f"test_{f}"] = expand(windows([recs[f]], STRIDE_EVAL))
    for T in SHIFT_TEMPS:
        for c in d["shift"][T]:
            sets[f"test{T}_{c}"] = expand(windows([d["shift"][T][c]], STRIDE_EVAL))
    return sets


def fit_ecm_bounded(recs, ocv_fn, dt=1.0, x0=(0.02, 0.01, 10.0, 0.01, 200.0)) -> dict:
    """battery_pf's fit_ecm with physical bounds (R in [0.1 mOhm, 0.5 Ohm], tau1 in [1, 100] s, tau2 in
    [100, 3600] s). Unbounded, this cell's fit put the second branch at tau2 = 30 days and R2 = 15 Ohm (an
    integrator absorbing a capacity / OCV mismatch, iteration limit reached): not a polarisation voltage."""
    from scipy.optimize import least_squares
    data = [(r.u, r.soc - 0.5 * r.u * dt / (3600.0 * r.Q), r.vbar) for r in recs]

    def res(th):
        return np.concatenate([BP.ecm_voltage(th, u, s, ocv_fn, dt) - v for u, s, v in data])

    lo = np.log([1e-4, 1e-4, 1.0, 1e-4, 100.0])
    hi = np.log([0.5, 0.5, 100.0, 0.5, 3600.0])
    th0 = np.clip(np.log(np.asarray(x0)), lo + 1e-9, hi - 1e-9)
    sol = least_squares(res, th0, bounds=(lo, hi), method="trf")
    R0, R1, t1, R2, t2 = np.exp(sol.x)
    rr = res(sol.x)
    return {"R0": R0, "R1": R1, "C1": t1 / R1, "R2": R2, "C2": t2 / R2, "tau1_s": t1, "tau2_s": t2,
            "v_rmse_mV": 1e3 * float(np.sqrt(np.mean(rr ** 2))), "v_max_mV": 1e3 * float(np.abs(rr).max()),
            "nfev": int(sol.nfev), "active_bounds": sol.active_mask.tolist()}


def identify_lg(train_recs, c20_path) -> dict:
    oc = BP.c20_capacity_and_ocv(c20_path)
    fit = BP.fit_ocv(oc["soc"], oc["ocv"])
    ocv_fn = lambda s: BP.ocv_eval(s, fit["coef"], fit["w0"], fit["w1"])
    ecm = fit_ecm_bounded(train_recs, ocv_fn)
    th = BP.fit_thermal(train_recs, ecm)
    params = {"Q": oc["Q"], "R0": ecm["R0"], "R1": ecm["R1"], "C1": ecm["C1"], "R2": ecm["R2"], "C2": ecm["C2"],
              "C_th": th["C_th"], "hA": th["hA"], "Ea_R": 3000.0, "dUdT": -1.0e-4,
              "T_amb": 273.15 + float(np.mean([r.t_amb_eff for r in train_recs])), "tau_f": BP.TAU_F,
              "ow0": fit["w0"], "ow1": fit["w1"], **{f"o{i}": float(c) for i, c in enumerate(fit["coef"])}}
    return {"params": {k: float(v) for k, v in params.items()}, "ocv_fit": fit, "ecm": ecm, "thermal": th}


def freeze(root=None, log=print) -> dict:
    """Identify the partial model on the TRAINING cycles (+ C/20 test) and write the frozen constants."""
    d = load_all(root)
    idn = identify_lg([d["recs25"][c] for c in SPLIT["train"]], _c20(d["base"]))
    params = idn["params"]
    m = model("25C", {"params_25C": params, "thermocouple_offset_K": d["offset_K"]})
    # scales: std of the labelled states over the training windows; V1, V2 from the model replay
    rp = [replay(m, d["recs25"][c]) for c in SPLIT["train"]]
    w = windows([d["recs25"][c] for c in SPLIT["train"]], STRIDE_TRAIN, rp)
    xf = np.where(np.isfinite(w["x"]), w["x"], w["x_fill"]).reshape(-1, 5).astype(np.float64)
    fit = {}
    for c in SPLIT["train"] + SPLIT["val"]:
        r = d["recs25"][c]
        e = replay(m, r)[:, 4] - r.vm
        fit[f"25/{c}"] = {"vm_rmse_mV": 1e3 * float(np.sqrt(np.mean(e ** 2)))}
    fr = {"Q_Ah": d["Q"], "thermocouple_offset_K": d["offset_K"], "params_25C": params, "ecm_fit": idn["ecm"],
          "thermal_fit": idn["thermal"], "ocv_fit_rms_V": idn["ocv_fit"]["rms"], "x_std": xf.std(0).tolist(),
          "x_mean": xf.mean(0).tolist(), "y_std": xf[:, 3:].std(0).tolist(),
          "filter_tuning": {"ekf": {"q_soc": 3e-4, "c": 0.1, "r_vm": 0.1}},
          "filter_tuning_note": "battery_pf's validation-tuned EKF noise reused a priori (no tuning on this cell)",
          "open_loop_model_fit_train_val": fit}
    p = Path(__file__).resolve().parents[1] / "resources" / f"real_{NAME}.json"
    s = json.loads(p.read_text())
    s["frozen"] = fr
    p.write_text(json.dumps(s, indent=1))
    spec.cache_clear()
    log(f"[freeze] {NAME}: Q {d['Q']:.4f} Ah, offset {d['offset_K']:.3f} K, ECM {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in idn['ecm'].items()})}")
    return fr


def prepare(root=None, log=print) -> dict:
    sp = spec(NAME)
    if not sp.get("frozen"):
        freeze(root, log)
        sp = spec(NAME)
    d = load_all(root)
    fr = sp["frozen"]
    if abs(d["Q"] - fr["Q_Ah"]) > 1e-9 or abs(d["offset_K"] - fr["thermocouple_offset_K"]) > 1e-9:
        raise RuntimeError("Q / offset differ from the frozen values")
    sets = build_sets(d)
    out = processed_dir(NAME, root)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, s in sets.items():
        arrs = {k: v for k, v in s.items() if k != "meta"}
        np.savez(out / f"{name}.npz", **arrs, meta=np.asarray(json.dumps(s["meta"])))
        files[f"{name}.npz"] = {"n_episodes": int(len(s["u"])), "sha256_arrays": BP._arrays_hash(arrs)}
    man = {"prep_version": PREP_VERSION, "dataset": NAME, "files": files, "Q_Ah": d["Q"], "offset_K": d["offset_K"]}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    log(f"[prepare] {NAME}: " + ", ".join(f"{k} {v['n_episodes']}" for k, v in files.items()))
    return man


def prior_std() -> np.ndarray:
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
    x = z["x"]
    return RealEpisodeSet(u=z["u"], y=z["y"], x=x, dt=DT_S / TIME_UNIT, plant=NAME, knowledge="real", split=split,
                          families=[m["family"] for m in meta], signals=[m["rec"] for m in meta], x0=x[:, 0].copy(),
                          episode_ids=np.arange(len(x)), source=str(p),
                          prior=z["prior"] if "prior" in z.files else None,
                          x_fill=z["x_fill"] if "x_fill" in z.files else None, prior_std=prior_std(),
                          label_mask=np.isin(np.arange(5), LABELLED), meta=meta)


def ic_box() -> list:
    tr = load("train")
    x = np.where(np.isfinite(tr.x), tr.x, tr.x_fill).reshape(-1, 5).astype(np.float64)
    b = [[float(x[:, j].min()), float(x[:, j].max())] for j in range(5)]
    b[0] = [0.0, 1.0]
    return b


def context(condition: str = "25C", filter_kind: str = "ekf") -> dict:
    fr = spec(NAME)["frozen"]
    xs = np.asarray(fr["x_std"], float)
    g = fr["filter_tuning"][filter_kind]
    q = g["c"] * xs
    q[0] = g["q_soc"]
    T = float(condition.rstrip("C"))
    meta = {"plant": NAME, "n_x": 5, "n_y": 2, "n_u": 1, "state_names": list(STATE_NAMES), "sensors": [3, 4],
            "sensor_names": ["dT", "Vm"], "unmeasured": [0, 1, 2], "labelled": list(LABELLED), "x_std": xs.tolist(),
            "dt": DT_S / TIME_UNIT, "dt_seconds": DT_S, "N": N_STEPS, "settle_index": SETTLE, "t_ref": 1.0,
            "time_unit_s": TIME_UNIT, "condition": condition, "ambient_C": T,
            "input_box": {"u_scale": float(fr["Q_Ah"]), "about": "current [A], discharge positive; 1 C = Q A"}}
    return {"plant": NAME, "knowledge": "real", "meta": meta, "dt": DT_S / TIME_UNIT, "sensors": [3, 4],
            "unmeasured": [0, 1, 2], "labelled": list(LABELLED), "x_std": xs, "y_std": np.asarray(fr["y_std"], float),
            "x_mean": np.asarray(fr["x_mean"], float), "q_std": q, "r_std": np.array([0.3, g["r_vm"]]),
            "prior_mean": None, "prior_std": prior_std(), "model": model(condition), "model_knowledge": "partial",
            "condition": condition, "hold_missing": False,
            "hyperkkl": {"nz": 7, "alpha": 1.3, "ic_box": ic_box()}}


def headline(results: dict) -> float:
    return float(np.mean([results[f"test_{f}"]["summary"][HEADLINE] for f in TEST_FAMILIES]))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
