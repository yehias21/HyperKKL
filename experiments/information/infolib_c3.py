"""POST HOC (2026-10-26): Bayesian PCRB under the TEST-C3 sensor model (budget fix, BUDGET_FIX.md).

The original budgets (infolib.py / budgets.py) assumed white Gaussian sensor noise and the exact
model.  The TEST-C3 comparison generates its data with (hyperkkl.data.measure, kind colored_t)

    y_k = h(x_k) + v_k,      v_k = rho v_{k-1} + sqrt(1 - rho^2) s e_k,   v_0 = s e_0,
    e_k iid unit-variance Student-t(nu),   s = noise * y_std  (per output),

on the TRUE plant build_plant(scfg, perturb=mismatch) (hyperkkl.data.eval_data).

Whitening.  Conditioning on y_k, the pre-whitened measurement
    y_{k+1} - rho y_k = h(x_{k+1}) - rho h(x_k) + sqrt(1 - rho^2) s e_{k+1}
has independent noise, so p(x_{0:K}, y_{0:K}) factorises as
    p(x_0) p(y_0 | x_0) prod_k p(x_{k+1} | x_k) p(y_{k+1} | y_k, x_k, x_{k+1}),
a chain whose k-th factor couples only (x_k, x_{k+1}).  The Tichavsky-Muravchik-Nehorai (1998)
recursion applies with the measurement factor included in the transition terms:
    D11 = E[F^T Q^-1 F] + rho^2 E[H_k^T J_e H_k]
    D12 = -E[F^T] Q^-1   - rho   E[H_k^T J_e H_{k+1}]
    D22 = Q^-1           +       E[H_{k+1}^T J_e H_{k+1}]
    J_{k+1} = D22 - D12^T (J_k + D11)^-1 D12,      J_0 = P0^-1 + E[H_0^T J_v H_0].

Student-t.  For additive noise the expected negative Hessian of log p is the noise's Fisher
information (location family).  For a unit-variance Student-t(nu) (scale^2 = (nu-2)/nu):
    J_t = (nu + 1) / (nu + 3) * nu / (nu - 2)          (= 2 for nu = 3; = 1 for nu -> inf),
so J_e = diag(J_t / ((1 - rho^2) s^2)) and J_v = diag(J_t / s^2).  (At a window start the
marginal of v_k0 is a weighted sum of t's; by Stam's inequality its Fisher information is
<= J_t / s^2, so using J_t / s^2 there can only lower the bound: still a valid lower bound.)
rho = 0, J_t = 1 recovers infolib's white-Gaussian recursion exactly.
"""
from __future__ import annotations

import math

import numpy as np
import torch

import infolib as L


def t_fisher_factor(nu: float) -> float:
    """Fisher information of a unit-variance Student-t(nu) location family (Gaussian = 1)."""
    if nu is None or not math.isfinite(nu):
        return 1.0
    return (nu + 1.0) / (nu + 3.0) * nu / (nu - 2.0)


def sensor_fisher(cfg: dict, scales: dict) -> tuple:
    """(rho, J_v (ny,ny), J_e (ny,ny), J_t) for the run's sensor model."""
    sen = cfg["data"]["sensor"]
    kind = sen.get("kind", "gaussian")
    sd = float(sen.get("noise", 0.0))
    s = np.maximum(sd * np.asarray(scales["y_std"]), 1e-6 * np.asarray(scales["y_std"]))
    if kind == "gaussian":
        rho, Jt = 0.0, 1.0
    elif kind == "colored":
        rho, Jt = float(sen.get("rho", 0.95)), 1.0
    elif kind == "colored_t":
        rho, Jt = float(sen.get("rho", 0.9)), t_fisher_factor(float(sen.get("nu", 3.0)))
    else:
        raise ValueError(kind)
    Jv = torch.diag(torch.as_tensor(Jt / s ** 2, dtype=torch.float64))
    Je = torch.diag(torch.as_tensor(Jt / ((1 - rho ** 2) * s ** 2), dtype=torch.float64))
    return rho, Jv, Je, Jt


def _step(J, F, Hk, Hk1, Qinv, Je, rho):
    Ft = F.transpose(-1, -2)
    D11 = (Ft @ Qinv @ F).mean(0)
    D12 = -(Ft.mean(0) @ Qinv)
    D22 = Qinv + (Hk1.transpose(-1, -2) @ Je @ Hk1).mean(0)
    if rho != 0.0:
        D11 = D11 + rho ** 2 * (Hk.transpose(-1, -2) @ Je @ Hk).mean(0)
        D12 = D12 - rho * (Hk.transpose(-1, -2) @ Je @ Hk1).mean(0)
    return D22 - D12.transpose(-1, -2) @ torch.linalg.solve(J + D11, D12)


def bayes_full(F, H, Q, rho, Jv, Je, P0, k_eval):
    Qinv = torch.linalg.inv(Q)
    J = torch.linalg.inv(P0) + (H[:, 0].transpose(-1, -2) @ Jv @ H[:, 0]).mean(0)
    out, ke = [], set(int(k) for k in k_eval)
    if 0 in ke:
        out.append(torch.linalg.inv(J))
    for k in range(F.shape[1]):
        J = _step(J, F[:, k], H[:, k], H[:, k + 1], Qinv, Je, rho)
        if (k + 1) in ke:
            out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def bayes_window(F, H, Q, rho, Jv, Je, P0, k_eval, W):
    Qinv = torch.linalg.inv(Q)
    out = []
    for k in k_eval.tolist():
        k0 = max(k, W) - W
        J = torch.linalg.inv(P0) + (H[:, k0].transpose(-1, -2) @ Jv @ H[:, k0]).mean(0)
        for i in range(W):
            J = _step(J, F[:, k0 + i], H[:, k0 + i], H[:, k0 + i + 1], Qinv, Je, rho)
        out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def budget_c3(plant, x, u, cfg, scales, sensors, windows_s, Sigma, n_ep=64, n_t=12, settle_frac=0.5,
              sensor_override: dict = None) -> dict:
    """Bayesian PCRB (full past and windows) under the run's sensor model; sensor_override
    replaces cfg['data']['sensor'] (used for the decomposition white/colored/colored_t)."""
    if sensor_override is not None:
        cfg = dict(cfg, data=dict(cfg["data"], sensor=sensor_override))
    E = min(n_ep, x.shape[0])
    x, u = x[:E].double(), u[:E].double()
    N1 = x.shape[1]
    dt = float(scales["dt"])
    settle = int(settle_frac * (N1 - 1))
    k_eval = torch.linspace(settle, N1 - 1, n_t).round().long()
    Q, _ = L.noise_matrices(cfg, scales)
    rho, Jv, Je, Jt = sensor_fisher(cfg, scales)
    F, H = L.jacobians(plant, x, u, dt)
    sig = np.asarray(scales["x_std"])
    out = {"n_episodes": E, "n_times": n_t, "settle": settle, "rho": rho, "J_t": Jt,
           "sensor": cfg["data"]["sensor"], "bayes": {"windows": {}}}
    Pb = bayes_full(F, H, Q, rho, Jv, Je, Sigma, k_eval)[None]
    out["bayes"]["full"] = L.summarise(Pb, Sigma, sig, sensors)
    for Ws in windows_s:
        W = min(max(1, int(round(Ws / dt))), settle)
        Pw = bayes_window(F, H, Q, rho, Jv, Je, Sigma, k_eval, W)[None]
        out["bayes"]["windows"][f"{Ws:g}"] = {"W_samples": W, "W_s": W * dt, **L.summarise(Pw, Sigma, sig, sensors)}
    return out
