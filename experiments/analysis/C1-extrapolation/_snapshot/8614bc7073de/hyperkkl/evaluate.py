"""Evaluation: every observer x every tier x every stress condition, on paired episodes.

Tiers (signals.py): id, ood, waveform_ood. Conditions (configs/base.yaml ``conditions``)
change only what the observers receive or the plant they face:

    sensor      sensor model override (noise level, outliers, dropout, colour, ...)
    input       corruption of the u given to observers (noise, bias, delay)
    init        prior x0_hat: "center" of the IC box, or "poor" (a far corner)
    plant_shift the plant drifted after training; baselines keep the nominal model

Sensor/input/init conditions reuse the SAME simulated episodes as `clean`, so condition
effects are paired too.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from . import data as Dm
from .conditioning import ConditionedDecoder
from .config import deep_merge
from .metrics import episode_metrics
from .nets import ContextEncoder
from .observers import (EKF, UKF, KKLObserver, LatentStep, ModelBasedKKL, ParticleFilter,
                        run, time_per_step)
from .systems import build_plant


class RawInput(ContextEncoder):
    """c_k = u_k (used by NLOX, whose injection is omega(z) u)."""
    dim = 1

    def seq(self, u):
        return u

    def init_state(self, B, device):
        return None

    def step(self, u_k, state):
        return u_k, None


def build_observers(cfg, des, scales, base, trained: dict, nlox_res, device) -> dict:
    lat = LatentStep(des, device)
    T0, Ti = (base["T0"], base["T0inv"]) if base else (None, None)   # None: filters only
    obs = {}
    for name, r in trained.items():
        if name.split("@")[0] == "latent":
            obs[name] = KKLObserver(name, lat, ConditionedDecoder(Ti, "none", 0, 0),
                                    ctx=r["ctx"], injection=r["cond"], z0_map=T0)
        else:
            obs[name] = KKLObserver(name, lat, r["cond"], ctx=r["ctx"], z0_map=T0)
    if nlox_res is not None:
        om = nlox_res["omega"]
        obs["nlox"] = KKLObserver("nlox", lat, nlox_res["dec"], ctx=RawInput(),
                                  injection=lambda z, c: om(z, c), z0_map=None)
    model = build_plant(cfg["system"])                       # NOMINAL model
    sc, dcfg = cfg["system"], cfg["data"]
    q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
    r = max(float(dcfg["sensor"].get("noise", 0.0)), 1e-3) * np.asarray(scales["y_std"])
    box = np.asarray(sc["ic_box"], float)
    p0 = (box[:, 1] - box[:, 0]) / np.sqrt(12.0)
    for b in cfg["baselines"]:
        if b == "ekf":
            obs[b] = EKF(model, des.dt, q, r, p0)
        elif b == "ukf":
            obs[b] = UKF(model, des.dt, q, r, p0)
        elif b == "pf":
            pf = cfg["eval"].get("pf_schedule") or {"inflate": 3.0, "anneal_tau": 0.0}
            obs[b] = ParticleFilter(model, des.dt, q, r, p0,
                                    n_particles=int(cfg["eval"].get("pf_particles", 512)),
                                    inflate=float(pf["inflate"]), anneal_tau=float(pf["anneal_tau"]))
        elif b == "kkl_mb":
            obs[b] = ModelBasedKKL(lat, T0, Ti, model)
    return obs


def initial_guess(cfg, n, kind, seed, device) -> torch.Tensor:
    box = np.asarray(cfg["system"]["ic_box"], float)
    ctr, half = box.mean(1), (box[:, 1] - box[:, 0]) / 2
    if kind == "center":
        x0 = np.tile(ctr, (n, 1))
    elif kind == "poor":
        rng = np.random.RandomState(seed)
        x0 = ctr + 0.9 * half * rng.choice([-1.0, 1.0], size=(n, len(ctr)))
    else:
        raise ValueError(kind)
    return torch.as_tensor(x0, dtype=torch.float32, device=device)


def evaluate(cfg, des, scales, observers: dict, device, log=print) -> dict:
    """All (tier, condition) blocks are concatenated into ONE batch and every observer
    runs over it once: the per-step cost is launch-bound, so this is ~blocks x faster
    than looping, and it changes nothing per episode."""
    ecfg = cfg["eval"]
    sigma = scales["x_std"]
    base_sets, blocks = {}, []
    for tier in ecfg["tiers"]:
        for cname in ecfg["conditions"]:
            cond = cfg["conditions"][cname] or {}
            shift = float(cond.get("plant_shift", 0.0))
            if (tier, shift) not in base_sets:
                base_sets[(tier, shift)] = Dm.eval_data(cfg, tier, plant_shift=shift, device=device)
            es = base_sets[(tier, shift)]
            seed = Dm.split_seed(cfg["system"]["name"], tier, ecfg.get("split", "dev"),
                                 int(ecfg.get("rep", 0)))          # same stream as eval_data
            if cond.get("sensor"):
                es = es.with_sensor(des, deep_merge(cfg["data"]["sensor"], cond["sensor"]),
                                    scales["y_std"], seed + 1)
            u_obs = es.u
            if cond.get("input"):
                u_obs = Dm.corrupt_input(es.u, cond["input"], float(cfg["system"].get("u_scale", 1.0)), seed + 2)
            x0 = initial_guess(cfg, len(es), cond.get("init", "center"), seed + 3, "cpu")
            blocks.append((tier, cname, es, u_obs, x0))
    X = torch.cat([b[2].x for b in blocks]).to(device)
    Y = torch.cat([b[2].y for b in blocks]).to(device)
    U = torch.cat([b[3] for b in blocks]).to(device)
    X0 = torch.cat([b[4] for b in blocks]).to(device)
    settle = int(float(ecfg.get("settle_frac", 0.5)) * (X.shape[1] - 1))
    chunk = int(ecfg.get("chunk", 1024))
    out = {t: {} for t in ecfg["tiers"]}
    for tier, cname, es, _, _ in blocks:
        out[tier][cname] = {"families": es.families, "methods": {}}
    for name, ob in observers.items():
        t0 = time.time()
        c = chunk if name != "pf" else max(1, chunk // 8)
        xh = torch.cat([run(ob, Y[i:i + c], U[i:i + c], X0[i:i + c]) for i in range(0, len(Y), c)])
        m = episode_metrics(X, xh, sigma, settle)
        o = 0
        for tier, cname, es, _, _ in blocks:
            n = len(es)
            r = {k: v[o:o + n].tolist() for k, v in m.items()}
            out[tier][cname]["methods"][name] = r
            if ecfg.get("save_traj") and tier == "ood" and cname == "clean":
                out[tier][cname].setdefault("_traj", {})[name] = xh[o:o + 4].cpu().numpy()
                out[tier][cname]["_traj"]["true"] = X[o:o + 4].cpu().numpy()
            o += n
        summ = "  ".join(f"{t[:3]}/{c_}={np.mean(out[t][c_]['methods'][name]['nrmse']):.4f}"
                         for t in ecfg["tiers"] for c_ in list(ecfg["conditions"])[:3])
        log(f"    eval {name:14s} {summ}  ({time.time() - t0:.0f}s, {len(Y)} episodes)")
    return out


def costs(observers: dict, nx, ny, device) -> dict:
    """Online cost per step: microseconds at batch 1 and batch 64 (CUDA-synchronised)."""
    out = {}
    for name, ob in observers.items():
        try:
            out[name] = {"us_b1": time_per_step(ob, ny, 1, nx, device, B=1),
                         "us_b64": time_per_step(ob, ny, 1, nx, device, B=64)}
        except Exception as e:                                  # cost is diagnostic only
            out[name] = {"error": repr(e)}
    return out
