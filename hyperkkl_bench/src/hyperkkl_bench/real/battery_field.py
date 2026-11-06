r"""Shared machinery of the battery tracks whose window length, recordings or identification differ from
battery_pf / battery_lg (``calce``). Same state, sensors, task and scorer as battery_pf: x = (SOC, V1,
V2, dT, Vm), y = (dT, Vm), V1, V2 unlabelled (NaN), prior SOC0 +- 0.2, settled SOC RMSE [% SOC]."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import battery_pf as BP
from . import registry as R
from .battery_pf import INIT_ERR, LABELLED, STATE_NAMES, TIME_UNIT, RealEpisodeSet, expand, replay

DT_S = 1.0


def windows_n(recs, N, stride, replays=None) -> dict:
    """battery_pf.windows with a track-specific window length."""
    X, F, U, Y, meta = [], [], [], [], []
    for i, r in enumerate(recs):
        for s in range(0, len(r) - N - 1, stride):
            sl = slice(s, s + N + 1)
            nan = np.full(N + 1, np.nan)
            X.append(np.stack([r.soc[sl], nan, nan, r.dT[sl], r.vm[sl]], 1))
            if replays is not None:
                F.append(np.stack([nan, replays[i][sl, 1], replays[i][sl, 2], nan, nan], 1))
            U.append(r.u[sl]); Y.append(r.y[sl])
            meta.append({"rec": r.name, "cycle": r.cycle, "family": r.family, "temp": r.temp, "start": int(s), "soc0": float(r.soc[s])})
    if not X:
        raise ValueError("no windows")
    f32 = lambda a: np.stack(a).astype(np.float32)
    out = {"u": f32(U), "y": f32(Y), "x": f32(X), "meta": meta}
    if replays is not None:
        out["x_fill"] = f32(F)
    return out


@dataclass
class FieldBattery:
    name: str
    N: int
    load_sets: object                  # root -> {"train": [Recording], "val": [...], "test_*": [...], "_Q": Q, ...}
    identify: object                   # (sets) -> {"params": ..., "ecm": ..., "thermal": ..., "ocv_rms": ...}
    test_splits: list
    headline_splits: list
    prep_version: str
    metric_about: str
    stride_train: int = 120
    stride_eval: int = 300
    condition_of: object = lambda s: "25C"
    r_dT: float = 0.3
    hyperkkl: dict = field(default_factory=lambda: {"nz": 7, "alpha": 1.3})
    filter_tuning: dict = field(default_factory=lambda: {"q_soc": 3e-4, "c": 0.1, "r_vm": 0.1})

    def splits(self):
        return ["train", "val"] + list(self.test_splits)

    def frozen(self):
        fr = R.spec(self.name).get("frozen")
        if not fr:
            raise RuntimeError(f"{self.name}: not frozen; prepare it first")
        return fr

    def model(self, condition="25C", fr=None):
        from ..plants.systems_real import Pan18650PF, cold_params
        fr = fr or self.frozen()
        p = dict(fr["params_25C"])
        if fr.get("arrhenius_condition", True):
            T = float(condition.rstrip("C"))
            if T != 25.0:
                p, _ = cold_params(p, T + fr["thermocouple_offset_K"])
        return Pan18650PF(sensors=(3, 4), **p)

    def freeze(self, sets, log=print):
        idn = self.identify(sets)
        params = idn["params"]
        fr0 = {"params_25C": params, "thermocouple_offset_K": sets.get("_offset", 0.0), "arrhenius_condition": sets.get("_arrhenius", True)}
        m = self.model("25C", fr0)
        rp = [replay(m, r) for r in sets["train"]]
        w = windows_n(sets["train"], self.N, self.stride_train, rp)
        xf = np.where(np.isfinite(w["x"]), w["x"], w["x_fill"]).reshape(-1, 5).astype(np.float64)
        fit = {}
        for sp in ("train", "val"):
            for r in sets[sp]:
                e = replay(m, r)
                fit[f"{sp}/{r.name}"] = {"vm_rmse_mV": 1e3 * float(np.sqrt(np.mean((e[:, 4] - r.vm) ** 2))),
                                         "soc_end_err_pct": 100 * float(e[-1, 0] - r.soc[-1])}
        fr = {"Q_Ah": sets["_Q"], **fr0, "ecm_fit": idn["ecm"], "thermal_fit": idn["thermal"], "ocv_fit_rms_V": idn["ocv_rms"],
              "x_std": xf.std(0).tolist(), "x_mean": xf.mean(0).tolist(), "y_std": xf[:, 3:].std(0).tolist(),
              "filter_tuning": {"ekf": self.filter_tuning}, "filter_tuning_note": "battery_pf's validation-tuned EKF noise reused a priori",
              "open_loop_model_fit_train_val": fit, "identification_note": idn.get("note", "")}
        p = Path(R.RESOURCES) / f"real_{self.name}.json"
        s = json.loads(p.read_text())
        s["frozen"] = fr
        p.write_text(json.dumps(s, indent=1))
        R.spec.cache_clear()
        log(f"[freeze] {self.name}: Q {sets['_Q']:.3f}, ECM {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in idn['ecm'].items() if k != 'active_bounds'})}, "
            f"thermal {json.dumps(idn['thermal'])}")
        return fr

    def prepare(self, root=None, log=print):
        sets = self.load_sets(root)
        if not R.spec(self.name).get("frozen"):
            self.freeze(sets, log)
        m25 = self.model("25C")
        out = R.processed_dir(self.name, root)
        out.mkdir(parents=True, exist_ok=True)
        files, seen = {}, {}
        for sp in self.splits():
            recs = sets[sp]
            for r in recs:
                seen.setdefault(r.name, set()).add(sp if sp in ("train", "val") else "test")
            if sp == "train":
                s = windows_n(recs, self.N, self.stride_train, [replay(m25, r) for r in recs])
            elif sp == "val":
                s = expand(windows_n(recs, self.N, self.stride_eval, [replay(m25, r) for r in recs]))
            else:
                s = expand(windows_n(recs, self.N, self.stride_eval))
            arrs = {k: v for k, v in s.items() if k != "meta"}
            np.savez(out / f"{sp}.npz", **arrs, meta=np.asarray(json.dumps(s["meta"])))
            files[f"{sp}.npz"] = {"n_episodes": int(len(s["u"])), "recordings": sorted({r.name for r in recs}), "sha256_arrays": BP._arrays_hash(arrs)}
        bad = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
        if bad:
            raise ValueError(f"{self.name}: recording in several roles: {bad}")
        man = {"prep_version": self.prep_version, "dataset": self.name, "files": files, "excluded": sets.get("_excluded", {})}
        (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
        log(f"[prepare] {self.name}: " + ", ".join(f"{k} {v['n_episodes']}" for k, v in files.items()))
        return man

    def prior_std(self):
        xs = np.asarray(self.frozen()["x_std"], float)
        return np.array([INIT_ERR, xs[1], xs[2], 0.5, 0.01])

    def load(self, split, root=None):
        if split not in self.splits():
            raise ValueError(f"{self.name}: unknown split {split!r}; available: {self.splits()}")
        p = R.processed_dir(self.name, root) / f"{split}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: fetch and prepare {self.name}")
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        x = z["x"]
        return RealEpisodeSet(u=z["u"], y=z["y"], x=x, dt=DT_S / TIME_UNIT, plant=self.name, knowledge="real", split=split,
                              families=[m["family"] for m in meta], signals=[m["rec"] for m in meta], x0=x[:, 0].copy(),
                              episode_ids=np.arange(len(x)), source=str(p), prior=z["prior"] if "prior" in z.files else None,
                              x_fill=z["x_fill"] if "x_fill" in z.files else None, prior_std=self.prior_std(),
                              label_mask=np.isin(np.arange(5), LABELLED), meta=meta, settle_index=self.N // 2)

    def ic_box(self):
        tr = self.load("train")
        x = np.where(np.isfinite(tr.x), tr.x, tr.x_fill).reshape(-1, 5).astype(np.float64)
        b = [[float(x[:, j].min()), float(x[:, j].max())] for j in range(5)]
        b[0] = [0.0, 1.0]
        return b

    def context(self, condition="25C", filter_kind="ekf"):
        fr = self.frozen()
        xs = np.asarray(fr["x_std"], float)
        g = fr["filter_tuning"]["ekf"]
        q = g["c"] * xs
        q[0] = g["q_soc"]
        meta = {"plant": self.name, "n_x": 5, "n_y": 2, "n_u": 1, "state_names": list(STATE_NAMES), "sensors": [3, 4],
                "sensor_names": ["dT", "Vm"], "unmeasured": [0, 1, 2], "labelled": list(LABELLED), "x_std": xs.tolist(),
                "dt": DT_S / TIME_UNIT, "dt_seconds": DT_S, "N": self.N, "settle_index": self.N // 2, "t_ref": 1.0,
                "time_unit_s": TIME_UNIT, "condition": condition,
                "input_box": {"u_scale": float(fr["Q_Ah"]), "about": "current [A], discharge positive; 1 C = Q A"}}
        return {"plant": self.name, "knowledge": "real", "meta": meta, "dt": DT_S / TIME_UNIT, "sensors": [3, 4],
                "unmeasured": [0, 1, 2], "labelled": list(LABELLED), "x_std": xs, "y_std": np.asarray(fr["y_std"], float),
                "x_mean": np.asarray(fr["x_mean"], float), "q_std": q, "r_std": np.array([self.r_dT, g["r_vm"]]),
                "prior_mean": None, "prior_std": self.prior_std(), "model": self.model(condition), "model_knowledge": "partial",
                "condition": condition, "hold_missing": False, "hyperkkl": dict(self.hyperkkl, ic_box=self.ic_box())}

    def headline(self, results):
        return float(np.mean([results[s]["summary"]["rmse_settled"] for s in self.headline_splits]))

    def export(self, g):
        g.update(splits=self.splits, condition_of=self.condition_of, prepare=self.prepare, load=self.load, context=self.context,
                 score=BP.score, headline=self.headline, model=self.model, HEADLINE="rmse_settled", SCORER=BP.SCORER,
                 METRIC_ABOUT=self.metric_about, PREP_VERSION=self.prep_version, NAME=self.name)
