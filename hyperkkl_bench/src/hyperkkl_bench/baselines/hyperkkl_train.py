"""HyperKKL (the paper's method) TRAINED on a benchmark dataset with the research training code.

    est = HyperKKLTrain(seed=0, device="cuda:0", gpu="5", steps_scale=0.2)   # 0.2 = smoke run
    rep = hyperkkl_bench.real.evaluate(est, "battery_pf")

Recipe: A3 (beat_greybox: SOAP optimiser, phase-1 base decoder trainable in phase 2 at 0.1 lr, strided
latent window as extra hypernetwork context) on the low-rank (lora, r = 4) weight-space decoder, with the
latent design (n_z, alpha) and the partial model given by the dataset's context (``context['hyperkkl']``,
``context['model']``). The 'mixed data' (D1 event-structured simulated episodes) and C5 (jitter fix) parts of
the full default recipe need a simulator of the TRUE plant, which real data do not have: they are not used
(stated in every report's fit_info).

It needs the research repository next to the package (the beat_greybox package copy in
archive/beat_greybox_scratch/pkg); training runs in a subprocess (_hkkl_train_runner), on the GPU named by
``gpu`` (CUDA_VISIBLE_DEVICES) or on the CPU. Batch-only (run_batch): latency is not measured.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from ..estimator import BaseEstimator
from ..paths import CHECKOUT


class HyperKKLTrain(BaseEstimator):
    def __init__(self, seed: int = 0, device: str = "cpu", gpu: str | None = None, steps_scale: float = 1.0,
                 arm: str = "A3", work=None, name: str | None = None, n_sim: int = 600):
        self.seed, self.device, self.gpu, self.steps_scale, self.arm = int(seed), device, gpu, float(steps_scale), arm
        self.n_sim = int(n_sim)
        self.work = Path(work) if work else None
        self.name = name or f"hyperkkl_{arm}" + (f"_x{steps_scale:g}" if steps_scale != 1 else "")

    def _env(self):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.gpu if (self.gpu is not None and self.device != "cpu") else ""
        env["PYTHONPATH"] = str(CHECKOUT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def fit(self, train, val, context):
        hk = context.get("hyperkkl")
        if not hk:
            raise ValueError(f"{context.get('plant')}: the dataset context declares no HyperKKL latent design")
        m = context["model"]
        self.context = context
        self.work = self.work or (CHECKOUT / "logs" / "hkkl_train" / f"{context['plant']}_{self.arm}_s{self.seed}_x{self.steps_scale:g}")
        self.work.mkdir(parents=True, exist_ok=True)

        def filled(es):
            x = es.x.astype(np.float32)
            if getattr(es, "x_fill", None) is not None:
                x = np.where(np.isfinite(x), x, es.x_fill.astype(np.float32))
            return x
        npz = self.work / "data.npz"
        np.savez(npz, train_x=filled(train), train_u=train.u, train_y=train.y,
                 val_x=filled(val), val_u=val.u, val_y=val.y)
        lab = context.get("labelled") or list(range(len(context["x_std"])))
        spec = {"plant_module": type(m).__module__, "plant_class": type(m).__name__, "params": m.params(),
                "sensors": list(m.sensors), "labelled": list(lab), "nz": hk["nz"], "alpha": hk["alpha"],
                "dt": float(context["dt"]), "N": int(train.u.shape[1] - 1),
                "u_scale": float(context["meta"].get("input_box", {}).get("u_scale", 1.0)),
                "ic_box": hk["ic_box"], "x_std": [float(v) for v in context["x_std"]],
                "nu": int(getattr(m, "nu", 1)), "replay": hk.get("replay", "model"),
                "arm": self.arm, "seed": self.seed, "steps_scale": self.steps_scale, "n_sim": self.n_sim,
                "device": "cuda:0" if self.device != "cpu" else "cpu", "train_npz": str(npz), "work": str(self.work)}
        self.spec_path = self.work / "spec.json"
        self.spec_path.write_text(json.dumps(spec, indent=1))
        t0 = time.time()
        r = subprocess.run([sys.executable, "-m", "hyperkkl_bench.baselines._hkkl_train_runner", "train",
                            str(self.spec_path)], capture_output=True, text=True, env=self._env())
        (self.work / "train.stdout").write_text(r.stdout + "\n" + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"HyperKKL training failed:\n{r.stderr[-4000:]}")
        info = json.loads((self.work / "train_info.json").read_text())
        self.fit_info = dict(info, fit_s=time.time() - t0, recipe=f"{self.arm} (S+P+H), lora r=4; mixed data / C5 "
                             "not applicable on real data (need a simulator of the true plant)", work=str(self.work))

    def run_batch(self, U, Y, X0, P0=None):
        with tempfile.TemporaryDirectory() as td:
            fi, fo = f"{td}/in.npz", f"{td}/out.npz"
            np.savez(fi, Y=np.asarray(Y, np.float32), U=np.asarray(U, np.float32), X0=np.asarray(X0, np.float32))
            r = subprocess.run([sys.executable, "-m", "hyperkkl_bench.baselines._hkkl_train_runner", "predict",
                                str(self.spec_path), fi, fo], capture_output=True, text=True, env=self._env())
            if r.returncode != 0:
                raise RuntimeError(f"HyperKKL predict failed:\n{r.stderr[-4000:]}")
            d = np.load(fo)
            self._np = int(d["n_params"])
            return d["X_hat"]

    def reset(self, x0_prior, P0_prior=None):
        raise NotImplementedError("HyperKKLTrain is batch-only (run_batch); it runs in a subprocess")

    def step(self, u, y):
        raise NotImplementedError("HyperKKLTrain is batch-only (run_batch); it runs in a subprocess")

    def n_params(self):
        return getattr(self, "_np", None)
