"""End-to-end smoke runs of the bundled baselines and of HyperKKL on a real-data track, through the
benchmark API (fit on train/val -> run every test split from the track's prior -> score -> report).

    python tools/real_smoke.py battery_pf ekf
    python tools/real_smoke.py battery_pf hyperkkl --gpu 5 [--steps-scale 0.2]
    methods: prior_mean, coulomb (battery tracks), finite_diff (mechanical tracks), imu_static (neurobem), ekf, ekf_gated,
             ude_ekf_gated, gru, hyperkkl

Reports: reference/real/smoke/<dataset>__<method>.json (+ .md table). Smoke = short training, 1 seed:
the numbers show the path works and are NOT benchmark results (PROTOCOL: >= 3 seeds, full budgets).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "reference" / "real" / "smoke"


class Coulomb:
    """Coulomb counting from the prior (battery tracks): the no-training floor."""
    name = "coulomb"

    def fit(self, train, val, context):
        self.mod = context["plant"]

    def run_batch(self, U, Y, X0, P0=None):
        from hyperkkl_bench.real import registry
        Q = registry.spec(self.mod)["frozen"]["Q_Ah"]
        dt_s = 1.0
        soc = X0[:, 0][:, None] - np.concatenate([np.zeros((len(U), 1)), np.cumsum(U[:, :-1], 1)], 1) * dt_s / (3600 * Q)
        out = np.repeat(X0[:, None, :], U.shape[1], 1).astype(float)
        out[..., 0] = soc
        return out


class FiniteDiff:
    """Mechanical tracks: causal second-order backward difference of the measured position (no model, no
    training): the floor a velocity estimator must beat to be meaningful."""
    name = "finite_diff"

    def fit(self, train, val, context):
        self.dt = float(context["dt"])

    def run_batch(self, U, Y, X0, P0=None):
        y = np.asarray(Y, float)[..., 0]
        v = np.zeros_like(y)
        v[:, 1] = (y[:, 1] - y[:, 0]) / self.dt
        v[:, 2:] = (3 * y[:, 2:] - 4 * y[:, 1:-1] + y[:, :-2]) / (2 * self.dt)
        return np.stack([y, v], -1)


class ImuStatic:
    """neurobem: the no-dynamics floor. Low-pass the accelerometer (causal, 0.5 s), undo the identified calibration, read the
    body velocity off the drag term of the identified model and the gravity direction off the remaining specific force.
    Uses the frozen identification, no filter, no training."""
    name = "imu_static"

    def fit(self, train, val, context):
        self.m = context["model"]

    def run_batch(self, U, Y, X0, P0=None):
        from scipy.signal import lfilter
        from hyperkkl_bench.plants.systems_real import NB_OM0
        m = self.m
        C = m._C.numpy()
        b = m._b.numpy()
        al = 1 - np.exp(-0.01 / 0.5)
        Y = np.asarray(Y, float)
        y0 = Y[:, :1]
        yl = lfilter([al], [1, al - 1], Y - y0, axis=1) + y0
        a = (yl[..., 3:] - b) @ np.linalg.inv(C).T
        s = np.sqrt((((np.asarray(U, float) + NB_OM0) / NB_OM0) ** 2).mean(-1))
        s = lfilter([al], [1, al - 1], s - s[:, :1], axis=1) + s[:, :1]
        v = np.stack([-a[..., 0] / (m.dx * s), -a[..., 1] / (m.dy * s), 0 * s], -1)
        g = a - np.cross(yl[..., :3], v)
        g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
        return np.concatenate([v, g, Y], -1)


def make(method, a):
    from hyperkkl_bench import baselines as B
    if method == "prior_mean":
        return B.PriorMean()
    if method == "coulomb":
        return Coulomb()
    if method == "finite_diff":
        return FiniteDiff()
    if method == "imu_static":
        return ImuStatic()
    if method == "ekf":
        e = B.EKF(gate=False); e.refit_for_condition = True; return e
    if method == "ekf_gated":
        e = B.EKF(gate=True); e.refit_for_condition = True; return e
    if method == "ude_ekf_gated":
        return B.UDEFilter(hybrid=True, kind="ekf", gate=True, steps=a.ude_steps, lrs=tuple(a.ude_lrs), threads=a.threads)
    if method == "gru":
        return B.GRUObserver(hidden=64, layers=1, steps=a.gru_steps, window=512, warmup=64, threads=a.threads)
    if method == "hyperkkl":
        from hyperkkl_bench.baselines.hyperkkl_train import HyperKKLTrain
        return HyperKKLTrain(seed=a.seed, device="cuda" if a.gpu else "cpu", gpu=a.gpu, steps_scale=a.steps_scale)
    raise KeyError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("method")
    ap.add_argument("--gpu", default=None); ap.add_argument("--steps-scale", type=float, default=0.2)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--splits", default=None)
    ap.add_argument("--ude-steps", type=int, default=1000); ap.add_argument("--ude-lrs", type=float, nargs="+", default=[1e-3, 3e-3])
    ap.add_argument("--gru-steps", type=int, default=1500); ap.add_argument("--n-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="HyperKKL seed; seed != 0 reports go to smoke/seeds/, outside the tables")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import hyperkkl_bench.real as hr
    est = make(a.method, a)
    t0 = time.time()
    rep = hr.evaluate(est, a.dataset, splits_=a.splits.split(",") if a.splits else None, verbose=True,
                      threads=a.threads, n_episodes=a.n_episodes)
    rep.meta["smoke"] = {"method": a.method, "steps_scale": a.steps_scale, "ude_steps": a.ude_steps,
                         "ude_lrs": a.ude_lrs, "gru_steps": a.gru_steps, "seed": a.seed, "wall_s": time.time() - t0}
    out = OUT if a.seed == 0 else OUT / "seeds"
    tag = a.method if a.seed == 0 else f"{a.method}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    rep.save(out / f"{a.dataset}__{tag}.json")
    (out / f"{a.dataset}__{tag}.md").write_text(rep.table() + "\n")
    print(rep.table())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
