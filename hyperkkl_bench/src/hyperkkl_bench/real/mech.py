r"""Shared machinery of the 2-state mechanical real-data tracks (``emps``, ``silverbox``).

Task (both): from the measured input u (force / input voltage) and the measured position y, estimate the
VELOCITY, a hidden state. No velocity sensor exists in these data sets, so the velocity LABEL is derived
offline and non-causally, exactly as the data providers' own identification pipelines do: a zero-phase
Butterworth low-pass (filtfilt) of the position followed by a central difference (EMPS: Janot et al., the
'bandpass filtering process' of IDIM-LS; Silverbox: the same process with a cut-off above the excited band).
An estimator only ever sees (u_0..u_k, y_0..y_k); the label is never an input. The label is itself an
estimate (PITFALLS P15): its error is reported in the card (noise floor: the difference between two cut-offs).

    x = (position, velocity)      sensors = [0] (position), labelled = [0, 1] (velocity: derived label)
    episode prior: (y_0, 0), prior_std = (3 r_y, std of the training velocity)
    score: settled velocity NRMSE = sqrt(mean_{k >= N/2} (v_hat - v_ref)^2) / std(v_ref over train),
           clipped at 100 (diverged); headline = mean over the test splits
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .battery_pf import RealEpisodeSet
from .registry import processed_dir, raw_dir, spec

SCORER = "settled-velocity-nrmse-1"
HEADLINE = "nrmse_velocity"


def zero_phase_velocity(y, dt_s, cutoff_hz, order=4):
    """filtfilt Butterworth low-pass of y, then central difference (diffcent); per second."""
    from scipy.signal import butter, filtfilt
    fs = 1.0 / dt_s
    b, a = butter(order, cutoff_hz / (fs / 2))
    yf = filtfilt(b, a, y)
    v = np.empty_like(yf)
    v[1:-1] = (yf[2:] - yf[:-2]) / (2 * dt_s)
    v[0], v[-1] = (yf[1] - yf[0]) / dt_s, (yf[-1] - yf[-2]) / dt_s
    return yf, v


@dataclass
class Segment:
    split: str
    record: str
    start: int
    stop: int
    stride: int


@dataclass
class MechTrack:
    name: str
    N: int
    dt_s: float
    time_unit_s: float
    cutoff_hz: float
    segments: list
    test_splits: list
    plant_class: str
    hyperkkl: dict
    edge: int = 200                    # samples cut at each end of a record (filtfilt edge effects)
    loader: object = None              # raw_dir -> {record: (u [model input units], y)}
    identify_fn: object = None         # (u, y, v_ref, a_ref, dt_model) -> (params, info)
    metric_about: str = ""
    prep_version: str = ""
    extra: dict = field(default_factory=dict)
    outage: dict = field(default_factory=dict)   # split -> list of (start, length): position sensor frozen there
    hold_missing: bool = False                   # filters' stale-sample rule (True when outages are part of a split)

    # ---------------------------------------------------------------- splits
    def splits(self):
        return ["train", "val"] + list(self.test_splits)

    def condition_of(self, split):
        return "default"

    @property
    def dt(self):
        return self.dt_s / self.time_unit_s

    # ---------------------------------------------------------------- prepare
    def records(self, root=None):
        recs = self.loader(raw_dir(self.name, root))
        out = {}
        for k, (u, y) in recs.items():
            yf, v = zero_phase_velocity(y, self.dt_s, self.cutoff_hz)
            _, v2 = zero_phase_velocity(y, self.dt_s, 0.5 * self.cutoff_hz)
            a = np.gradient(v, self.dt_s)
            out[k] = {"u": np.asarray(u, float), "y": np.asarray(y, float), "v": v * self.time_unit_s,
                      "a": a * self.time_unit_s ** 2, "v_half_cutoff": v2 * self.time_unit_s, "yf": yf}
        return out

    def _windows(self, recs, seg: Segment):
        r = recs[seg.record]
        lo = max(seg.start, self.edge)
        hi = min(seg.stop, len(r["u"]) - self.edge)
        X, U, Y, P, M, meta = [], [], [], [], [], []
        bursts = self.outage.get(seg.split, [])
        for s in range(lo, hi - self.N, seg.stride):
            sl = slice(s, s + self.N + 1)
            X.append(np.stack([r["y"][sl], r["v"][sl]], 1)); U.append(r["u"][sl])
            y = r["y"][sl].copy()
            m = np.zeros(self.N + 1, bool)
            for b0, L in bursts:                      # frozen sensor: the last sample before the burst is held
                y[b0:b0 + L] = y[b0 - 1]
                m[b0:b0 + L] = True
            Y.append(y[:, None]); M.append(m)
            P.append([r["y"][s], 0.0])
            meta.append({"rec": seg.record, "family": seg.record, "start": int(s)})
        f32 = lambda a: np.asarray(a, np.float32)
        out = {"x": f32(X), "u": f32(U), "y": f32(Y), "prior": f32(P), "meta": meta}
        if bursts:
            out["score_mask"] = np.asarray(M, bool)
        return out

    def _cat(self, parts):
        return {k: (np.concatenate([p[k] for p in parts]) if k != "meta" else sum((p[k] for p in parts), [])) for k in parts[0]}

    def freeze(self, root=None, log=print):
        recs = self.records(root)
        tr = [s for s in self.segments if s.split == "train"]
        sl = lambda r, s: slice(max(s.start, self.edge), min(s.stop, len(r["u"]) - self.edge))
        cat = lambda key: np.concatenate([recs[s.record][key][sl(recs[s.record], s)] for s in tr])
        u, y, v, a = cat("u"), cat("y"), cat("v"), cat("a")
        params, info = self.identify_fn(u, y, v, a, self.dt)
        r_y = float(np.std(cat("y") - cat("yf")))
        info.update({"r_y": max(r_y, 1e-9), "v_std": float(np.std(v)), "y_std": float(np.std(y)),
                     "y_mean": float(np.mean(y)), "v_mean": float(np.mean(v)),
                     "label_noise_floor_nrmse": float(np.std(cat("v") - cat("v_half_cutoff")) / np.std(v))})
        info["model_fit"] = self._model_fit(params, recs, log)
        p = Path(__file__).resolve().parents[1] / "resources" / f"real_{self.name}.json"
        s = json.loads(p.read_text())
        s["frozen"] = {"params": params, "identification": info}
        p.write_text(json.dumps(s, indent=1))
        spec.cache_clear()
        log(f"[freeze] {self.name}: params {json.dumps({k: round(v, 6) for k, v in params.items()})}")
        return s["frozen"]

    def _model_fit(self, params, recs, log):
        """Open-loop simulation error on val windows (from the true (y, v_ref) start) and one-step error."""
        import torch
        from ..plants.simulate import rk4_zoh_step
        m = self.model(params)
        out = {}
        for split in ("train", "val"):
            w = self._cat([self._windows(recs, s) for s in self.segments if s.split == split])
            x = torch.as_tensor(w["x"][:, 0], dtype=torch.float64)
            X = np.asarray(w["x"], np.float64)
            U = torch.as_tensor(w["u"], dtype=torch.float64)
            sim = [x.numpy()]
            with torch.no_grad():
                for k in range(self.N):
                    x = rk4_zoh_step(m, x, U[:, k:k + 1], self.dt)
                    sim.append(x.numpy())
                xs = torch.as_tensor(X[:, :-1].reshape(-1, 2))
                one = rk4_zoh_step(m, xs, U[:, :-1].reshape(-1, 1), self.dt).numpy().reshape(X[:, 1:].shape)
            sim = np.stack(sim, 1)
            sd = X.reshape(-1, 2).std(0)
            out[split] = {"sim_nrmse_pos": float(np.sqrt(np.nanmean((sim[..., 0] - X[..., 0]) ** 2)) / sd[0]),
                          "sim_nrmse_vel": float(np.sqrt(np.nanmean((sim[..., 1] - X[..., 1]) ** 2)) / sd[1]),
                          "onestep_nrmse_vel": float(np.sqrt(np.mean((one[..., 1] - X[:, 1:, 1]) ** 2)) / sd[1])}
        log(f"[freeze] {self.name}: model fit {json.dumps(out)}")
        return out

    def prepare(self, root=None, log=print):
        if not spec(self.name).get("frozen"):
            self.freeze(root, log)
        recs = self.records(root)
        out = processed_dir(self.name, root)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for split in self.splits():
            w = self._cat([self._windows(recs, s) for s in self.segments if s.split == split])
            arrs = {k: v for k, v in w.items() if k != "meta"}
            np.savez(out / f"{split}.npz", **arrs, meta=np.asarray(json.dumps(w["meta"])))
            files[f"{split}.npz"] = int(len(w["u"]))
        self._check_disjoint()
        (out / "MANIFEST.json").write_text(json.dumps({"prep_version": self.prep_version, "files": files}, indent=1))
        log(f"[prepare] {self.name}: {files}")
        return {"files": files}

    def _check_disjoint(self):
        by = {}
        for s in self.segments:
            by.setdefault(s.record, []).append(s)
        for rec, ss in by.items():
            for i, a in enumerate(ss):
                for b in ss[i + 1:]:
                    if a.split != b.split and a.start < b.stop and b.start < a.stop and not (
                            a.split.startswith("test") and b.split.startswith("test")):
                        raise ValueError(f"{self.name}: {a.split} and {b.split} overlap in {rec}")

    # ---------------------------------------------------------------- load / context / score
    def frozen(self):
        fr = spec(self.name).get("frozen")
        if not fr:
            raise RuntimeError(f"{self.name}: not frozen; run hyperkkl-bench prepare {self.name}")
        return fr

    def model(self, params=None):
        from ..plants import systems_real as SR
        cls = getattr(SR, self.plant_class)
        return cls(sensors=(0,), **(params or self.frozen()["params"]))

    def x_std(self):
        i = self.frozen()["identification"]
        return np.array([i["y_std"], i["v_std"]])

    def prior_std(self):
        i = self.frozen()["identification"]
        return np.array([3 * i["r_y"], i["v_std"]])

    def load(self, split, root=None) -> RealEpisodeSet:
        if split not in self.splits():
            raise ValueError(f"{self.name}: unknown split {split!r}; available: {self.splits()}")
        p = processed_dir(self.name, root) / f"{split}.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `hyperkkl-bench fetch {self.name}` then `hyperkkl-bench prepare {self.name}`")
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        x = z["x"]
        return RealEpisodeSet(u=z["u"], y=z["y"], x=x, dt=self.dt, plant=self.name, knowledge="real", split=split,
                              families=[m["family"] for m in meta], signals=[m["rec"] for m in meta], x0=x[:, 0].copy(),
                              episode_ids=np.arange(len(x)), source=str(p), prior=z["prior"], prior_std=self.prior_std(),
                              label_mask=np.array([True, True]), meta=meta, settle_index=self.N // 2,
                              score_mask=z["score_mask"] if "score_mask" in z.files else None)

    def context(self, condition="default", **kw):
        i = self.frozen()["identification"]
        xs = self.x_std()
        q = np.array([1e-3 * i["q_v"], i["q_v"]])
        u_scale = float(i.get("u_scale", 1.0))
        meta = {"plant": self.name, "n_x": 2, "n_y": 1, "n_u": 1, "state_names": ["position", "velocity"],
                "sensors": [0], "unmeasured": [1], "labelled": [0, 1], "x_std": xs.tolist(), "dt": self.dt,
                "dt_seconds": self.dt_s, "time_unit_s": self.time_unit_s, "N": self.N, "settle_index": self.N // 2,
                "t_ref": 1.0, "input_box": {"u_scale": u_scale}}
        b = np.array([[i["y_mean"] - 4 * i["y_std"], i["y_mean"] + 4 * i["y_std"]],
                      [i["v_mean"] - 4 * i["v_std"], i["v_mean"] + 4 * i["v_std"]]])
        return {"plant": self.name, "knowledge": "real", "meta": meta, "dt": self.dt, "sensors": [0], "unmeasured": [1],
                "labelled": [0, 1], "x_std": xs, "y_std": np.array([i["y_std"]]),
                "x_mean": np.array([i["y_mean"], i["v_mean"]]), "q_std": q, "r_std": np.array([i["r_y"]]),
                "prior_mean": None, "prior_std": self.prior_std(), "model": self.model(), "model_knowledge": "partial",
                "condition": condition, "hold_missing": self.hold_missing,
                "hyperkkl": dict(self.hyperkkl, ic_box=b.tolist())}

    def score(self, X_hat, es) -> dict:
        Xh = np.asarray(X_hat, float)
        s = es.settle_index
        xs = self.x_std()
        mask = np.zeros(es.u.shape, bool)
        mask[:, s:] = True
        if es.score_mask is not None:                 # outage splits: scored during the outages only
            mask = es.score_mask.astype(bool)
        out = {}
        for j, nm in ((1, "nrmse_velocity"), (0, "nrmse_position")):
            e = (Xh[..., j] - es.x[..., j].astype(float)) / xs[j]
            e = np.clip(np.nan_to_num(e, nan=100.0, posinf=100.0, neginf=-100.0), -100, 100)
            out[nm] = np.sqrt((np.where(mask, e, 0.0) ** 2).sum(1) / mask.sum(1))
        out["diverged"] = (~np.isfinite(Xh).all((1, 2))) | (np.abs((Xh[..., 1] - es.x[..., 1]) / xs[1]) > 100).any(1)
        summ = {k: float(np.mean(v)) for k, v in out.items()}
        summ["n_episodes"] = int(len(Xh))
        return {"summary": summ, "per_episode": {k: np.asarray(v).tolist() for k, v in out.items()}}

    def headline(self, results):
        return float(np.mean([results[s]["summary"][HEADLINE] for s in self.test_splits]))

    def export(self, g: dict):
        """Bind the module-level API of a track module to this track."""
        g.update(splits=self.splits, condition_of=self.condition_of, prepare=self.prepare, freeze=self.freeze,
                 load=self.load, context=self.context, score=self.score, headline=self.headline, model=self.model,
                 records=self.records, HEADLINE=HEADLINE, SCORER=SCORER, METRIC_ABOUT=self.metric_about,
                 PREP_VERSION=self.prep_version, NAME=self.name)
