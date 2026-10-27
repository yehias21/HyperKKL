"""Sanity / leakage audit of the UDE / NODE pipeline (AUDIT_UDE.md). CPU only (cache keys use the
'cuda' device type, under which the stored data were built; tensors are loaded to CPU).

    CUDA_VISIBLE_DEVICES= python audit_ude.py C3-partial/bioreactor ... > ../logs/audit_ude.txt
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import torch

import b2  # noqa: F401
from b2 import HERE
from b2 import models as M
from b2 import train as T
from b2.job import load_cfg
from hyperkkl import baselines_data as BD
from hyperkkl import data as Dm
from hyperkkl import signals as S
from hyperkkl.observers import _rk4
from hyperkkl.systems import build_plant
from hyperkkl import systems_extra  # noqa: F401
from hyperkkl import systems_bounded  # noqa: F401

KEY = "cuda"            # cache-key device type of the stored data
torch.set_num_threads(6)


def ep_hash(x):
    return {hashlib.sha1(np.ascontiguousarray(e[:4].numpy()).tobytes()).hexdigest() for e in x}


@torch.no_grad()
def rollout_err(f_model, X, U, e, s0, K, dt, sig):
    """Normalised MSE after k = 1..K open-loop RK4/ZOH steps from the true state (float64)."""
    x = X[e, s0].double()
    out = []
    for j in range(K):
        u = U[e, s0 + j].double()
        x = _rk4(f_model, x, u, u, u, dt)
        err = ((x - X[e, s0 + j + 1].double()) / sig).pow(2).mean(-1)
        out.append(float(torch.nan_to_num(err, nan=1e6, posinf=1e6).clamp(max=1e6).mean()))
    return out


class _F:
    def __init__(self, f, base):
        self.f, self.nx, self.nu, self.sensors, self.ny, self.h = f, base.nx, base.nu, base.sensors, base.ny, base.h


def audit(cell, seed=0):
    node, system = cell.split("/")
    cfg, src = load_cfg(node, system, seed)
    sc = cfg["system"]
    scales = Dm.system_scales(cfg, KEY)
    des = Dm.make_design(sc)
    data = Dm.training_data(cfg, seed, KEY)
    R = {"cell": cell, "seed": seed, "cfg_src": src, "mismatch": sc.get("mismatch", 0.0),
         "knowledge": sc.get("knowledge", "full")}
    model = Dm.model_plant(sc)                                   # what job.py uses
    real = build_plant(sc, perturb=float(sc.get("mismatch", 0.0)))   # data-generating plant
    nominal_full = build_plant(sc)
    R["model_params_equal_nominal"] = json.dumps(model.params(), sort_keys=True, default=str) == \
        json.dumps(nominal_full.params(), sort_keys=True, default=str)
    R["model_params_equal_real"] = json.dumps(model.params(), sort_keys=True, default=str) == \
        json.dumps(real.params(), sort_keys=True, default=str)
    tr, p1, va = data["train"], data["p1"], data["val"]
    Xall = torch.cat([tr.x, p1.x]).reshape(-1, tr.x.shape[-1])
    idx = torch.randperm(len(Xall), generator=torch.Generator().manual_seed(0))[:20000]
    xs_, us_ = Xall[idx].double(), torch.cat([tr.u, p1.u]).reshape(-1, tr.u.shape[-1])[idx].double()
    fr, fm = real.f(xs_, us_), model.f(xs_, us_)
    R["rel_model_error_median"] = float(((fr - fm).norm(dim=1) / fr.norm(dim=1).clamp(min=1e-9)).median())
    # (2) data counts and overlap
    N1 = tr.x.shape[1]
    R["episodes"] = {"train": len(tr), "p1": len(p1), "val": len(va), "N+1": N1}
    R["transitions_train_p1"] = int((len(tr) + len(p1)) * (N1 - 1))
    R["greybox_n_fit_cap"] = int(BD.DEFAULTS["gb_n_fit"])
    hs_tr = ep_hash(tr.x) | ep_hash(p1.x)
    ov = {}
    for tier in ("id", "shape_ood", "sequence_ood"):
        es = Dm.eval_data(cfg, tier, device=KEY)
        ov[tier] = {"n": len(es), "identical_to_train": len(ep_hash(es.x) & hs_tr),
                    "families": sorted(set(es.families))}
    ov["val_identical_to_train"] = len(ep_hash(va.x) & hs_tr)
    R["overlap"] = ov
    R["train_families"] = sorted(set(tr.families))
    # (6) one-step grey-box vs UDE: open-loop rollout errors on validation windows
    sig = torch.as_tensor(scales["x_std"], dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    K = 128 if N1 - 1 > 200 else 32
    e = torch.randint(0, len(va), (2048,), generator=g)
    s0 = torch.randint(0, N1 - 1 - K, (2048,), generator=g)
    # refit the grey-box residual exactly as the job (same seed, same data)
    trn = BD.transitions([tr, p1], int(BD.DEFAULTS["n_fit"]), seed + 1, "cpu")
    van = BD.transitions([va], int(BD.DEFAULTS["n_val"]), seed + 2, "cpu")
    gb = BD.fit_greybox(model, tuple(a[:int(BD.DEFAULTS["gb_n_fit"])] for a in trn), van, des.dt, scales,
                        float(sc.get("u_scale", 1.0)), seed, "cpu", BD.options(cfg), log=lambda m: None)
    R["greybox_fit"] = {k: gb["info"][k] for k in ("n_params", "n_fit", "steps", "val_loss", "val_r2")}
    models = {"f_known only": model, "grey-box (one-step r)": gb["model"]}
    for name in ("ude5k_ekf", "ude140k_ekf", "node5k_ekf"):
        p = HERE / "models" / node / system / f"seed{seed}" / f"{name}.pt"
        if p.exists():
            d = torch.load(p, weights_only=False)
            net = BD.ResidualMLP(model.nx, model.nu, d["width"], scales.get("x_mean"), scales["x_std"],
                                 float(sc.get("u_scale", 1.0)), d["t_mean"], d["t_std"])
            net.load_state_dict(d["state"])
            models[name.replace("_ekf", "")] = M.deployed_dynamics(model, net, d["hybrid"])
            R.setdefault("ude_lr", {})[name] = d["lr"]
    models["TRUE plant (reference only)"] = real
    roll = {}
    for name, fmod in models.items():
        r = rollout_err(_F(fmod.f, model), va.x, va.u, e, s0, K, des.dt, sig)
        roll[name] = {k: r[k - 1] for k in (1, 8, 32, K) if k <= K}
    R["val_rollout_nmse"] = roll
    # per-episode evaluation results (stored run files)
    ev = {}
    for b, ms in (("greybox", ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")), ("ude5k", ("ude5k_ekf", "ude5k_ukf"))):
        f = HERE / "runs" / node / system / f"seed{seed}" / f"{b}.json"
        if f.exists():
            r = json.load(open(f))["results"]["sequence_ood"]
            for m in ms:
                a = np.asarray(r[m]["nrmse"])
                ev[m] = {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.quantile(a, 0.9)),
                         "diverged": float(np.mean(r[m]["diverged"]))}
    R["seq_ood_eval_seed"] = ev
    return R


if __name__ == "__main__":
    out = [audit(c) for c in sys.argv[1:]]
    print(json.dumps(out, indent=1, default=str))
