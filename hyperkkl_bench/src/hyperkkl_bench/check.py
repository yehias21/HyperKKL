"""Validation of a contributed plant (``hyperkkl-bench check-plant``). The same inclusion rules
the frozen plants passed (experiments/cartpole_bounded/INCLUSION.md), CPU-only, ~1-5 min:

  interface   f(x, u) -> (B, n_x) and h(x) -> (B, n_y) on the IC box, finite; f_known present
              (needed for the partial / realistic knowledge levels) and different from f
  observable  linearised observability Gramian over a 3/alpha window (states normalised by
              x_std, outputs by the sensor noise), at 12 settled times x 16 episodes:
              lambda_min / lambda_max > 1e-10 at >= 90 % of the points (rule 1b);
              WEAK if the median ratio < 1e-6. (Rule 1a, the posterior CRB < 1, needs the
              information library of the research repo and is reported only as the Gramian
              proxy crb_proxy_unmeasured_max = sqrt(diag((W/W_scale + P0^-1)^-1)) < 1.)
  bounded     5T rollouts under train / id / shape_ood / sequence_ood inputs, process noise
              0.005 x_std: S_j(5T) <= 2 S_j(T) and mean spread over [4T, 5T] <= 2 x over [T, 2T],
              all finite (rule 2)
  design      the KKL latent design gate (plants/design.validate) for (nz, alpha, n_y, dt):
              controllable, kappa = 1, max|lambda| dt <= 0.1, Gramian cond < 1e12. Only
              meaningful for KKL-type methods; reported, required for inclusion because the
              reference HyperKKL baseline is built on it.

A plant spec is a dict like metadata['system_config']:
  {"plant": "<registered name>" or "module.path:ClassName", "params": {...}, "sensors": [...],
   "ic_box": [[lo, hi], ...], "T": 50, "alpha": 1.5, "nz": 5, "u_scale": 1.0, "t_ref": 1.0,
   "actuator": {"kind": "saturating", "u_bar": 2.0}, "dt": (optional), "u_bounds": (optional)}
"""

from __future__ import annotations

import importlib
import json
import math
import time

import numpy as np
import torch

from .plants import design as D
from .plants import signals as S
from .plants.systems import REGISTRY, build_plant


def _build(spec: dict, knowledge="full"):
    name = spec["plant"]
    if ":" in name:                                       # module.path:ClassName -> register
        mod, cls = name.split(":")
        c = getattr(importlib.import_module(mod), cls)
        REGISTRY[cls] = c
        spec = dict(spec, plant=cls)
    return build_plant(spec, knowledge=knowledge), spec


def _design(spec, ny):
    nz, alpha = int(spec["nz"]), float(spec["alpha"])
    A, B = D.butterworth_AB(nz, alpha, ny)
    dt = float(spec.get("dt") or D.choose_dt(A, 0.1))
    return A, B, dt


def _simulate(plant, x0, sigs, dt, N, sigma_p=None, seed=0):
    t = np.arange(N + 1) * dt
    ug = torch.as_tensor(np.stack([s(t) for s in sigs], 1), dtype=torch.float64)    # (N+1, E)
    x = torch.as_tensor(x0, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    sp = None if sigma_p is None else torch.as_tensor(sigma_p, dtype=torch.float64) * math.sqrt(dt)
    xs = [x]
    with torch.no_grad():
        for k in range(N):
            u = ug[k][:, None]
            a1 = plant.f(x, u); a2 = plant.f(x + dt / 2 * a1, u)
            a3 = plant.f(x + dt / 2 * a2, u); a4 = plant.f(x + dt * a3, u)
            x = x + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            if sp is not None:
                x = x + sp * torch.randn(x.shape, generator=g, dtype=x.dtype)
            xs.append(x)
    return torch.stack(xs, 1), ug.T[..., None]


def _signals(fams, n, regime, rng, dt, T, spec):
    out = []
    for r in range(n):
        s = S.sample_signal(fams[r % len(fams)], regime, rng, dt, T, stratum=r // len(fams), n_strata=4,
                            scale=float(spec.get("u_scale", 1.0)), t_ref=float(spec.get("t_ref", 1.0)))
        out.append(S.enforce_bounds(s, dt, T, spec.get("u_bounds")))
    return out


def _ics(box, n, seed):
    from scipy.stats import qmc
    m = int(math.ceil(math.log2(max(n, 2))))
    X = qmc.Sobol(d=len(box), scramble=True, seed=seed).random_base2(m)[:n]
    return box[:, 0] + X * (box[:, 1] - box[:, 0])


def check_interface(plant, spec):
    box = np.asarray(spec["ic_box"], float)
    x = torch.as_tensor(_ics(box, 64, 0), dtype=torch.float64)
    u = torch.zeros(len(x), 1, dtype=torch.float64)
    out = {"f_shape_ok": False, "h_shape_ok": False, "finite": False, "has_f_known": False}
    fx = plant.f(x, u)
    hx = plant.h(x)
    out["f_shape_ok"] = tuple(fx.shape) == tuple(x.shape)
    out["h_shape_ok"] = hx.shape[0] == len(x) and hx.shape[1] == len(plant.sensors)
    out["finite"] = bool(torch.isfinite(fx).all() and torch.isfinite(hx).all())
    try:
        pk, _ = _build(spec, "partial")
        d = (pk.f(x, u) - fx).abs().max().item()
        out["has_f_known"] = True
        out["f_known_differs"] = bool(d > 1e-9)
    except NotImplementedError:
        out["has_f_known"] = False
    out["pass"] = out["f_shape_ok"] and out["h_shape_ok"] and out["finite"]
    return out


def check_observable(plant, spec, x, u, dt, x_std, y_noise, n_ep=16, n_t=12):
    from torch.func import jacrev, vmap
    alpha = float(spec["alpha"])
    x, u = x[:n_ep], u[:n_ep]
    E, N1, nx = x.shape
    settle = (N1 - 1) // 2
    W = min(int(round(3 / alpha / dt)), settle)

    def step1(xi, ui):
        a1 = plant.f(xi[None], ui[None]); a2 = plant.f(xi[None] + dt / 2 * a1, ui[None])
        a3 = plant.f(xi[None] + dt / 2 * a2, ui[None]); a4 = plant.f(xi[None] + dt * a3, ui[None])
        return (xi[None] + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4))[0]
    jf = vmap(jacrev(step1, argnums=0))
    jh = vmap(jacrev(lambda xi: plant.h(xi[None])[0]))
    Dg, Di = torch.diag(torch.as_tensor(x_std)), torch.diag(1 / torch.as_tensor(x_std))
    Rinv = torch.diag(1 / torch.as_tensor(y_noise) ** 2)
    ks = np.linspace(settle, N1 - 1, n_t).round().astype(int)
    ratios, crbs = [], []
    unm = [j for j in range(nx) if j not in plant.sensors]
    p0 = torch.as_tensor((np.asarray(spec["ic_box"], float)[:, 1] - np.asarray(spec["ic_box"], float)[:, 0])
                         / np.sqrt(12) / x_std, dtype=torch.float64)
    for k in ks:
        k0 = k - W
        Phi = torch.eye(nx, dtype=torch.float64).expand(E, nx, nx).clone()
        Wo = torch.zeros_like(Phi)
        for i in range(W + 1):
            H = jh(x[:, k0 + i]) @ Dg
            Pn = Di @ Phi @ Dg
            Wo = Wo + Pn.transpose(-1, -2) @ H.transpose(-1, -2) @ Rinv @ H @ Pn
            if i < W:
                Phi = jf(x[:, k0 + i], u[:, k0 + i]) @ Phi
        Ws = 0.5 * (Wo + Wo.transpose(-1, -2))
        ev = torch.linalg.eigvalsh(Ws)
        ratios.append((ev[:, 0].clamp(min=0) / ev[:, -1]).numpy())
        J = Ws + torch.diag(1 / p0 ** 2)
        crbs.append(torch.sqrt(torch.diagonal(torch.linalg.inv(J), dim1=-2, dim2=-1)).numpy())
    r = np.concatenate(ratios)
    crb = np.concatenate(crbs)
    cu = float(np.median(crb[:, unm], 0).max()) if unm else 0.0
    out = {"window_samples": W, "ratio_median": float(np.median(r)), "ratio_p10": float(np.percentile(r, 10)),
           "frac_full_rank": float((r > 1e-10).mean()), "cond_median": float(1 / max(np.median(r), 1e-300)),
           "crb_proxy_unmeasured_max": cu}
    out["pass"] = out["frac_full_rank"] >= 0.9 and cu < 1.0
    out["weak"] = out["ratio_median"] < 1e-6 or cu >= 0.1
    return out


def check_bounded(plant, spec, dt, N, x_std, n=32, mult=5):
    box = np.asarray(spec["ic_box"], float)
    sp = 0.005 * np.asarray(x_std)
    win = max(1, N // 10)
    out = {}
    for si, (name, regime) in enumerate((("train", "train"), ("id", "id"), ("shape_ood", "shape_ood"),
                                          ("sequence_ood", "sequence_ood"))):
        fams = S.TRAIN_FAMILIES if name == "train" else S.families_for(regime)
        rng = np.random.RandomState(1000 + si)
        sigs = _signals(fams, n, regime, rng, dt, mult * N * dt, spec)
        x, _ = _simulate(plant, _ics(box, n, 77 + si), sigs, dt, mult * N, sp, seed=5 + si)
        fin = bool(torch.isfinite(x).all())
        xs = torch.nan_to_num(x, nan=1e30, posinf=1e30, neginf=-1e30)
        Sd = xs.std(0)
        Sw = lambda k: Sd[max(0, k - win):k + 1].mean(0)
        seg = lambda a, b: Sd[a * N:b * N + 1].mean(0)
        g_end = (Sw(mult * N) / Sw(N).clamp(min=1e-12)).numpy()
        g_sec = (seg(mult - 1, mult) / seg(1, 2).clamp(min=1e-12)).numpy()
        out[name] = {"finite": fin, "ratio_end_max": float(g_end.max()), "ratio_secular_max": float(g_sec.max()),
                     "pass": fin and bool((g_end <= 2).all()) and bool((g_sec <= 2).all())}
    out["pass"] = all(v["pass"] for v in out.values() if isinstance(v, dict))
    return out


def check_plant(spec: dict, fast: bool = False, verbose: bool = True) -> dict:
    t0 = time.time()
    torch.set_num_threads(1)
    plant, spec = _build(spec)
    log = print if verbose else (lambda *a: None)
    ny = len(plant.sensors)
    A, B, dt = _design(spec, ny)
    N = int(math.ceil(float(spec["T"]) / dt))
    res = {"spec": spec, "dt": dt, "N": N}
    res["interface"] = check_interface(plant, spec)
    log("interface", res["interface"])
    if not res["interface"]["pass"]:
        res["pass"] = False
        return res
    res["design"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                     for k, v in D.validate(A, B, dt, float(spec["alpha"])).items()}
    log("design gate", res["design"]["ok"], res["design"]["checks"])
    n = 8 if fast else 16
    rng = np.random.RandomState(1)
    box = np.asarray(spec["ic_box"], float)
    sigs = _signals(S.families_for("id"), n, "id", rng, dt, N * dt, spec)
    x, u = _simulate(plant, _ics(box, n, 3), sigs, dt, N)
    if not torch.isfinite(x).all():
        res["observable"] = {"pass": False, "error": "non-finite id trajectories"}
        res["pass"] = False
        return res
    xs = x[:, N // 2:].reshape(-1, x.shape[-1]).std(0).clamp(min=1e-9).numpy()
    ys = plant.h(x[:, N // 2:].reshape(-1, x.shape[-1])).std(0).clamp(min=1e-9).numpy()
    res["x_std_estimate"] = xs.tolist()
    res["observable"] = check_observable(plant, spec, x, u, dt, xs, 0.03 * ys, n_ep=n)
    log("observable", res["observable"])
    res["bounded"] = check_bounded(plant, spec, dt, N, xs, n=8 if fast else 32)
    log("bounded", res["bounded"]["pass"], {k: v for k, v in res["bounded"].items() if k != "pass"})
    res["pass"] = bool(res["interface"]["pass"] and res["design"]["ok"] and res["observable"]["pass"]
                       and res["bounded"]["pass"])
    res["wall_s"] = time.time() - t0
    log("INCLUDE" if res["pass"] else "REJECT", f"({res['wall_s']:.0f}s)")
    return res


def spec_from_file(path: str) -> dict:
    return json.loads(open(path).read())
