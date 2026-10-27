"""Information budget (Bayesian PCRB, full past and 3/alpha window) for the case studies.

Extends experiments/information/infolib_c3.py (the BUDGET_FIX method: whitened AR(1) sensor noise,
Student-t Fisher factor, true plant, the run's own sensor model) to what the case-study sensors add:

* per-output noise level (data.sensor.noise_mult): s_j = noise * mult_j * y_std_j;
* multi-rate outputs (data.sensor.hold = P_j): output j carries a FRESH sample only at k = 0, P_j,
  2 P_j, ...; held samples carry no information (they are exact repeats). A held output is treated
  as white at its own rate (Fisher J_t / s_j^2 at each fresh sample): its AR(1) correlation at the
  fresh-sample spacing is rho^P_j (0.35 for rho = 0.9, P = 10; 0.66 for P = 4), which this ignores
  -- an approximation, disclosed; the per-step channels are whitened exactly as infolib_c3.
* fresh outputs at every step use the whitened recursion of infolib_c3 with their own rho.

Recursion (Tichavsky, Muravchik & Nehorai 1998; whitened chain factor couples x_k, x_{k+1}):
    D11 = E[F'Q^-1 F] + sum_{j fast} rho^2 E[h_kj' Je_j h_kj]
    D12 = -E[F'] Q^-1 - sum_{j fast} rho E[h_kj' Je_j h_k+1,j]
    D22 = Q^-1 + sum_{j fast} E[h_k+1,j' Je_j h_k+1,j] + sum_{j held, fresh at k+1} E[h_k+1,j' Jv_j h_k+1,j]
with h_kj the j-th row of H_k (here a selection row: y = x[sensors]), Je_j = J_t / ((1 - rho^2) s_j^2),
Jv_j = J_t / s_j^2. With hold = 1 everywhere it is exactly infolib_c3.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

INFO = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl/experiments/information")
if str(INFO) not in sys.path:
    sys.path.insert(0, str(INFO))
import infolib as L  # noqa: E402
import infolib_c3 as C3  # noqa: E402


def sensor_terms(cfg, scales):
    """(rho, Jt, s (ny,), hold (ny,)) of the run's sensor model."""
    sen = cfg["data"]["sensor"]
    kind = sen.get("kind", "gaussian")
    ny = len(scales["y_std"])
    mult = np.asarray(sen.get("noise_mult", [1.0] * ny), float)
    s = float(sen.get("noise", 0.0)) * mult * np.asarray(scales["y_std"], float)
    s = np.maximum(s, 1e-6 * np.asarray(scales["y_std"], float))
    if kind == "gaussian":
        rho, Jt = 0.0, 1.0
    elif kind == "colored":
        rho, Jt = float(sen.get("rho", 0.95)), 1.0
    elif kind == "colored_t":
        rho, Jt = float(sen.get("rho", 0.9)), C3.t_fisher_factor(float(sen.get("nu", 3.0)))
    else:
        raise ValueError(kind)
    hold = np.asarray(sen.get("hold") or [1] * ny, int)
    return rho, Jt, s, hold


def _info_mats(H, rho, Jt, s, hold, k):
    """Measurement information terms of the step k -> k+1, averaged over episodes.
    H (E, N+1, ny, nx). Returns (A11, A12, A22) to add to D11, D12, D22."""
    nx = H.shape[-1]
    A11 = torch.zeros(nx, nx, dtype=H.dtype)
    A12 = torch.zeros(nx, nx, dtype=H.dtype)
    A22 = torch.zeros(nx, nx, dtype=H.dtype)
    for j in range(H.shape[2]):
        hk, hk1 = H[:, k, j, :], H[:, k + 1, j, :]                       # (E, nx)
        if hold[j] <= 1:
            je = Jt / ((1 - rho ** 2) * s[j] ** 2)
            A22 = A22 + je * (hk1[:, :, None] * hk1[:, None, :]).mean(0)
            if rho != 0.0:
                A11 = A11 + rho ** 2 * je * (hk[:, :, None] * hk[:, None, :]).mean(0)
                A12 = A12 - rho * je * (hk[:, :, None] * hk1[:, None, :]).mean(0)
        elif (k + 1) % int(hold[j]) == 0:                                  # fresh sample of a held output
            jv = Jt / s[j] ** 2
            A22 = A22 + jv * (hk1[:, :, None] * hk1[:, None, :]).mean(0)
    return A11, A12, A22


def _J0(P0, H, Jt, s, k0):
    J = torch.linalg.inv(P0)
    for j in range(H.shape[2]):
        h = H[:, k0, j, :]
        J = J + Jt / s[j] ** 2 * (h[:, :, None] * h[:, None, :]).mean(0)
    return J


def _step(J, F, Qinv, A):
    Ft = F.transpose(-1, -2)
    A11, A12, A22 = A
    D11 = (Ft @ Qinv @ F).mean(0) + A11
    D12 = -(Ft.mean(0) @ Qinv) + A12
    D22 = Qinv + A22
    return D22 - D12.transpose(-1, -2) @ torch.linalg.solve(J + D11, D12)


def bayes_full(F, H, Q, terms, P0, k_eval):
    rho, Jt, s, hold = terms
    Qinv = torch.linalg.inv(Q)
    J = _J0(P0, H, Jt, s, 0)
    out, ke = [], set(int(k) for k in k_eval)
    for k in range(F.shape[1]):
        J = _step(J, F[:, k], Qinv, _info_mats(H, rho, Jt, s, hold, k))
        if (k + 1) in ke:
            out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def bayes_window(F, H, Q, terms, P0, k_eval, W):
    rho, Jt, s, hold = terms
    Qinv = torch.linalg.inv(Q)
    out = []
    for k in k_eval.tolist():
        k0 = max(k, W) - W
        J = _J0(P0, H, Jt, s, k0)
        for i in range(W):
            J = _step(J, F[:, k0 + i], Qinv, _info_mats(H, rho, Jt, s, hold, k0 + i))
        out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def budget(plant, x, u, cfg, scales, sensors, windows_s, Sigma, n_ep=48, n_t=10, settle_frac=0.5) -> dict:
    E = min(n_ep, x.shape[0])
    x, u = x[:E].double(), u[:E].double()
    N1 = x.shape[1]
    dt = float(scales["dt"])
    settle = int(settle_frac * (N1 - 1))
    k_eval = torch.linspace(settle, N1 - 1, n_t).round().long()
    Q, _ = L.noise_matrices(cfg, scales)
    terms = sensor_terms(cfg, scales)
    F, H = L.jacobians(plant, x, u, dt)
    sig = np.asarray(scales["x_std"])
    out = {"n_episodes": E, "n_times": n_t, "settle": settle, "rho": terms[0], "J_t": terms[1],
           "s": terms[2].tolist(), "hold": terms[3].tolist(), "sensor": cfg["data"]["sensor"],
           "bayes": {"windows": {}}}
    kmax = int(k_eval.max())
    Pb = bayes_full(F[:, :kmax], H[:, :kmax + 1], Q, terms, Sigma, k_eval)[None]
    out["bayes"]["full"] = L.summarise(Pb, Sigma, sig, sensors)
    for Ws in windows_s:
        W = min(max(1, int(round(Ws / dt))), settle)
        Pw = bayes_window(F, H, Q, terms, Sigma, k_eval, W)[None]
        out["bayes"]["windows"][f"{Ws:g}"] = {"W_samples": W, "W_s": W * dt, **L.summarise(Pw, Sigma, sig, sensors)}
    return out


def selftest():
    """hold = 1 everywhere must reproduce infolib_c3 (white and coloured)."""
    torch.manual_seed(0)
    E, N, nx, ny = 6, 40, 3, 2
    F = torch.eye(nx, dtype=torch.float64) + 0.05 * torch.randn(E, N, nx, nx, dtype=torch.float64)
    H = torch.zeros(E, N + 1, ny, nx, dtype=torch.float64)
    H[..., 0, 0] = 1.0; H[..., 1, 2] = 1.0
    Q = 0.01 * torch.eye(nx, dtype=torch.float64)
    P0 = torch.eye(nx, dtype=torch.float64)
    k = torch.tensor([20, 39])
    for rho, Jt in ((0.0, 1.0), (0.9, 2.0)):
        s = np.array([0.1, 0.1])
        Jv = torch.diag(torch.as_tensor(Jt / s ** 2))
        Je = torch.diag(torch.as_tensor(Jt / ((1 - rho ** 2) * s ** 2)))
        a = C3.bayes_full(F, H, Q, rho, Jv, Je, P0, k)
        b = bayes_full(F, H, Q, (rho, Jt, s, np.array([1, 1])), P0, k)
        err = float((a - b).abs().max() / a.abs().max())
        a = C3.bayes_window(F, H, Q, rho, Jv, Je, P0, k, 10)
        b = bayes_window(F, H, Q, (rho, Jt, s, np.array([1, 1])), P0, k, 10)
        err = max(err, float((a - b).abs().max() / a.abs().max()))
        assert err < 1e-10, err
    # a held output carries less information than a fresh one
    b1 = bayes_full(F, H, Q, (0.0, 1.0, np.array([0.1, 0.1]), np.array([1, 1])), P0, k)
    b5 = bayes_full(F, H, Q, (0.0, 1.0, np.array([0.1, 0.1]), np.array([5, 1])), P0, k)
    assert (b5.diagonal(dim1=-2, dim2=-1) >= b1.diagonal(dim1=-2, dim2=-1) - 1e-12).all()
    return True


if __name__ == "__main__":
    print("selftest", selftest())
