"""Information budget of a sampled plant along forced trajectories (theory/m4_information.tex).

Model (exactly the data generator of hyperkkl.data):
    x_{k+1} = Phi(x_k, u_k) + w_k,   w_k ~ N(0, Q),  Q = diag((noise_proc x_std)^2 dt)   (Euler-Maruyama)
    y_k     = h(x_k) + v_k,          v_k ~ N(0, R),  R = diag((sensor.noise y_std)^2)
with Phi the RK4 step under a zero-order-hold input. The Bayesian (posterior) Cramer-Rao
recursion [Tichavsky, Muravchik, Nehorai 1998], linearised along the realised trajectory,
    P_{k+1}^- = F_k P_k F_k^T + Q,     P_{k+1} = (P_{k+1}^{-1,-} + H_{k+1}^T R^{-1} H_{k+1})^{-1},
F_k = d Phi / d x (autograd through the RK4 step), H = dh/dx, is the Kalman-filter covariance when the
plant is linear (exact: the KF attains it) and the standard local approximation otherwise.

Quantities (all in the sigma-normalised units of hyperkkl.metrics, sigma_j = x_std_j):
    bits(W)      = 1/2 log2 det(Sigma_x) - 1/2 log2 det(P_W)  : information about x(t) carried by the last
                   W samples of (y, u) beyond climatology (prior Sigma_x = covariance of settled states)
    crb_j(W)     = sqrt(mean_{episodes, t} P_W[j, j]) / sigma_j : lower bound on NRMSE_j of any estimator
    nrmse_lb(W)  = sqrt(mean_j crb_j^2),  unmeasured_lb = mean_{j unmeasured} crb_j
    rd_lb(W)     = (det Sigma~_x)^{1/(2n)} 2^{-bits/n}        : Shannon lower bound (rate-distortion) on the
                   per-state RMS error from the bit count alone (weaker than the CRB by AM-GM; Gaussian-tight)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/raid0_gpu4/demo/data/papers/TMLR/hyperkkl")
import importlib.util as _ilu  # noqa: E402
if _ilu.find_spec("hyperkkl") is None:            # a caller may have put a code snapshot on sys.path first
    sys.path.insert(0, str(ROOT))

from hyperkkl import data as Dm                       # noqa: E402
from hyperkkl import systems_extra                    # noqa: E402,F401  (registers battery, network)
from hyperkkl.cache import DATA_MODULES, module_hash  # noqa: E402
from hyperkkl.config import digest                    # noqa: E402
from hyperkkl.systems import build_plant              # noqa: E402

TREE = ROOT / "experiments" / "tree"
CACHE = ROOT / "experiments" / "cache"
DEV = ["battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp"]
TEST = ["cartpole", "fhn", "highway_traffic", "network", "rossler", "two_link_arm"]
ALL = DEV + TEST
RUN_FULL = {s: "C1-where" for s in DEV} | {s: "TEST-C1" for s in TEST}      # knowledge: full
RUN_PARTIAL = {s: "C3-regimes" for s in DEV} | {s: "TEST-C3" for s in TEST}  # knowledge: partial


# ---------------------------------------------------------------------------
# Read-only access to finished runs and their cached episodes
# ---------------------------------------------------------------------------

def run_payload(system: str, node: str = None, seed: int = 0) -> dict:
    node = node or RUN_FULL[system]
    return json.loads((TREE / node / "runs" / system / f"seed{seed}" / "results.json").read_text())


def load_cached_eval(cfg: dict, tier: str, device_key: str = "cuda"):
    """The evaluation episodes of a run (cached by hyperkkl.data.eval_data; the runs generated them
    on CUDA, so the key carries 'cuda'). Loaded on CPU, never rebuilt."""
    scfg, ecfg = cfg["system"], cfg["eval"]
    kc = dict(cfg, system={k: v for k, v in scfg.items() if k != "knowledge"})
    extra = [tier, ecfg.get("split", "dev"), int(ecfg.get("rep", 0)), 0.0, module_hash(DATA_MODULES), device_key]
    key = digest(kc, "system", "data", "eval.n_episodes", extra=extra)
    path = CACHE / "data" / f"{scfg['name']}_eval_{tier}_{key}.pt"
    if not path.exists():
        raise FileNotFoundError(path.name)
    return torch.load(path, weights_only=False, map_location="cpu")


def load_cached_train(cfg: dict, seed: int = 0, device_key: str = "cuda") -> dict:
    scfg = cfg["system"]
    key = digest(cfg, "system", "data", extra=[seed, module_hash(DATA_MODULES), device_key])
    path = CACHE / "data" / f"{scfg['name']}_train_{key}.pt"
    if not path.exists():
        raise FileNotFoundError(path.name)
    return torch.load(path, weights_only=False, map_location="cpu")


# ---------------------------------------------------------------------------
# Linearisation along trajectories
# ---------------------------------------------------------------------------

def rk4_step(plant, x, u, dt):
    """One RK4 step of xdot = f(x, u) with u held (exactly hyperkkl.data.integrate's x-part)."""
    a1 = plant.f(x, u)
    a2 = plant.f(x + dt / 2 * a1, u)
    a3 = plant.f(x + dt / 2 * a2, u)
    a4 = plant.f(x + dt * a3, u)
    return x + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)


def jacobians(plant, x: torch.Tensor, u: torch.Tensor, dt: float, chunk: int = 4096) -> tuple:
    """F[e, k] = d rk4_step / dx at (x[e, k], u[e, k]) and H[e, k] = dh/dx, float64.
    x (E, N+1, nx), u (E, N+1, nu). Returns F (E, N, nx, nx) (steps k = 0..N-1), H (E, N+1, ny, nx)."""
    E, N1, nx = x.shape
    xf = x[:, :-1].reshape(-1, nx).double()
    uf = u[:, :-1].reshape(-1, u.shape[-1]).double()

    def step1(xi, ui):
        return rk4_step(plant, xi[None], ui[None], dt)[0]

    def h1(xi):
        return plant.h(xi[None])[0]
    jf = torch.func.vmap(torch.func.jacrev(step1, argnums=0))
    jh = torch.func.vmap(torch.func.jacrev(h1))
    F = torch.cat([jf(xf[i:i + chunk], uf[i:i + chunk]) for i in range(0, len(xf), chunk)])
    xa = x.reshape(-1, nx).double()
    H = torch.cat([jh(xa[i:i + chunk]) for i in range(0, len(xa), chunk)])
    return F.reshape(E, N1 - 1, nx, nx), H.reshape(E, N1, -1, nx)


# ---------------------------------------------------------------------------
# Posterior Cramer-Rao recursion (information form of the Riccati recursion)
# ---------------------------------------------------------------------------

def _update(Pm, H, Rinv):
    """P = (Pm^{-1} + H^T R^{-1} H)^{-1}, computed in covariance form (Joseph-free, nx small)."""
    n = Pm.shape[-1]
    I = torch.eye(n, dtype=Pm.dtype, device=Pm.device)
    Jm = torch.linalg.solve(Pm, I.expand_as(Pm))
    J = Jm + H.transpose(-1, -2) @ Rinv @ H
    return torch.linalg.solve(J, I.expand_as(J))


def pcrb_full(F, H, Q, R, P0, k_eval: torch.Tensor) -> torch.Tensor:
    """Run the recursion from k = 0 with prior P0 (nx, nx) along every episode; return P at the
    sample indices k_eval (shape (E, len(k_eval), nx, nx)). 'Full past' bound (any observer)."""
    E, N, n, _ = F.shape
    Rinv = torch.linalg.inv(R)
    P = P0.expand(E, n, n).clone()
    P = _update(P, H[:, 0], Rinv)                       # y_0 is measured
    out, ke = [], set(int(k) for k in k_eval)
    if 0 in ke:
        out.append(P)
    for k in range(N):
        P = F[:, k] @ P @ F[:, k].transpose(-1, -2) + Q
        P = _update(P, H[:, k + 1], Rinv)
        if (k + 1) in ke:
            out.append(P)
    return torch.stack(out, 1)


def pcrb_window(F, H, Q, R, P0, k_eval: torch.Tensor, W: int) -> torch.Tensor:
    """Information from the last W samples only: prior P0 (climatology) at k - W, recursion to k.
    Batched over (episode, evaluation time). Returns (E, len(k_eval), nx, nx)."""
    E, N, n, _ = F.shape
    T = len(k_eval)
    Rinv = torch.linalg.inv(R)
    ks = k_eval.clamp(min=W)                             # need W steps of history
    e_idx = torch.arange(E)[:, None].expand(E, T).reshape(-1)
    k0 = (ks - W)[None, :].expand(E, T).reshape(-1)
    P = P0.expand(E * T, n, n).clone()
    P = _update(P, H[e_idx, k0], Rinv)
    for i in range(W):
        Fi = F[e_idx, k0 + i]
        P = Fi @ P @ Fi.transpose(-1, -2) + Q
        P = _update(P, H[e_idx, k0 + i + 1], Rinv)
    return P.reshape(E, T, n, n)


def _bayes_step(J, F, H1, Qinv, Rinv):
    """One step of the Bayesian PCRB recursion [Tichavsky et al. 1998, eq. (18)-(21)] with the
    D-terms averaged over the batch dimension of F (E, n, n), H1 (E, ny, n):
        D11 = E[F^T Q^-1 F],  D12 = -E[F^T] Q^-1,  D22 = Q^-1 + E[H^T R^-1 H],
        J+  = D22 - D12^T (J + D11)^-1 D12."""
    Ft = F.transpose(-1, -2)
    D11 = (Ft @ Qinv @ F).mean(0)
    D12 = -(Ft.mean(0) @ Qinv)
    D22 = Qinv + (H1.transpose(-1, -2) @ Rinv @ H1).mean(0)
    return D22 - D12.transpose(-1, -2) @ torch.linalg.solve(J + D11, D12)


def pcrb_bayes_full(F, H, Q, R, P0, k_eval: torch.Tensor) -> torch.Tensor:
    """The Bayesian posterior CRB (a theorem for Gaussian w, v: a lower bound on the MSE matrix of
    ANY estimator of x_k from (y_0..k, u_0..k), averaged over the episode distribution); the
    expectations are Monte Carlo averages over the episodes at each time. Returns P_k = J_k^{-1} at
    k_eval, shape (len(k_eval), n, n)."""
    E, N, n, _ = F.shape
    Qinv, Rinv = torch.linalg.inv(Q), torch.linalg.inv(R)
    J = torch.linalg.inv(P0) + (H[:, 0].transpose(-1, -2) @ Rinv @ H[:, 0]).mean(0)
    out, ke = [], set(int(k) for k in k_eval)
    if 0 in ke:
        out.append(torch.linalg.inv(J))
    for k in range(N):
        J = _bayes_step(J, F[:, k], H[:, k + 1], Qinv, Rinv)
        if (k + 1) in ke:
            out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def pcrb_bayes_window(F, H, Q, R, P0, k_eval: torch.Tensor, W: int) -> torch.Tensor:
    """Bayesian PCRB from the last W samples only (prior P0 at k - W). Returns (len(k_eval), n, n)."""
    E, N, n, _ = F.shape
    Qinv, Rinv = torch.linalg.inv(Q), torch.linalg.inv(R)
    out = []
    for k in k_eval.tolist():
        k0 = max(k, W) - W
        J = torch.linalg.inv(P0) + (H[:, k0].transpose(-1, -2) @ Rinv @ H[:, k0]).mean(0)
        for i in range(W):
            J = _bayes_step(J, F[:, k0 + i], H[:, k0 + i + 1], Qinv, Rinv)
        out.append(torch.linalg.inv(J))
    return torch.stack(out, 0)


def summarise(P: torch.Tensor, Sigma: torch.Tensor, sigma: np.ndarray, sensors: list) -> dict:
    """P (E, T, n, n) posterior covariances -> bits, per-state CRB bounds (sigma-normalised)."""
    n = P.shape[-1]
    s = torch.as_tensor(sigma, dtype=P.dtype)
    Sn = Sigma / (s[:, None] * s[None, :])               # climatology covariance in normalised units
    Pn = P / (s[:, None] * s[None, :])
    ld_S = torch.logdet(Sn)
    ld_P = torch.logdet(Pn)
    bits = float(((ld_S - ld_P) / (2 * math.log(2))).mean())
    bits_t = ((ld_S - ld_P) / (2 * math.log(2)))          # (E, T)
    crb = Pn.diagonal(dim1=-2, dim2=-1).mean((0, 1)).sqrt()   # per-state RMS bound (NRMSE_j >= .)
    unm = [j for j in range(n) if j not in sensors]
    out = {
        "bits": bits, "bits_std_over_episodes": float(bits_t.mean(1).std()) if P.shape[0] > 1 else 0.0,
        "bits_per_state": bits / n,
        "crb_state": crb.tolist(),
        "nrmse_lb": float(crb.pow(2).mean().sqrt()),
        "unmeasured_lb": float(crb[unm].mean()) if unm else float("nan"),
        "measured_lb": float(crb[sensors].mean()) if sensors else float("nan"),
        # Shannon lower bound from the bit count alone: per-state RMS >= det(Sn)^{1/2n} 2^{-bits/n}
        "rd_lb": float(torch.exp(ld_S / (2 * n)) * 2 ** (-bits / n)),
        "logdet_climatology_bits": float(ld_S / (2 * math.log(2))),
    }
    return out


def climatology(x_settled: torch.Tensor) -> torch.Tensor:
    """Full covariance of the settled states (E*T, n) -> (n, n); the NRMSE's reference prior."""
    xf = x_settled.reshape(-1, x_settled.shape[-1]).double()
    return torch.cov(xf.T)


def noise_matrices(cfg: dict, scales: dict, sensor_noise: float = None) -> tuple:
    dcfg = cfg["data"]
    dt = float(scales["dt"])
    q = float(dcfg.get("noise_proc", 0.0)) * np.asarray(scales["x_std"])
    r = (float(dcfg["sensor"].get("noise", 0.0)) if sensor_noise is None else sensor_noise) * np.asarray(scales["y_std"])
    Q = torch.diag(torch.as_tensor(q ** 2 * dt, dtype=torch.float64))
    R = torch.diag(torch.as_tensor(np.maximum(r, 1e-6 * np.asarray(scales["y_std"])) ** 2, dtype=torch.float64))
    return Q, R


def slow_mode_rate(plant, x_mean: np.ndarray, u0: float = 0.0) -> float:
    """|Re lambda|_min of df/dx at the mean state (1/s): the plant's slowest local mode."""
    x = torch.as_tensor(x_mean, dtype=torch.float64)[None]
    u = torch.full((1, 1), float(u0), dtype=torch.float64)
    J = torch.func.jacrev(lambda xi: plant.f(xi[None], u)[0])(x[0])
    ev = torch.linalg.eigvals(J)
    return float(ev.real.abs().min())


def budget_along(plant, x, u, cfg, scales, sensors, windows_s: list, n_ep: int = 32, n_t: int = 12,
                 Sigma: torch.Tensor = None, settle_frac: float = 0.5, prior_scale: float = 1.0,
                 sensor_noise: float = None) -> dict:
    """Budgets along episodes x (E, N+1, nx), u (E, N+1, nu): full-past and windowed."""
    E = min(n_ep, x.shape[0])
    x, u = x[:E].double(), u[:E].double()
    N1 = x.shape[1]
    dt = float(scales["dt"])
    settle = int(settle_frac * (N1 - 1))
    k_eval = torch.linspace(settle, N1 - 1, n_t).round().long()
    if Sigma is None:
        Sigma = climatology(x[:, settle:])
    Sigma = Sigma * prior_scale
    Q, R = noise_matrices(cfg, scales, sensor_noise)
    F, H = jacobians(plant, x, u, dt)
    sig = np.asarray(scales["x_std"])
    out = {"n_episodes": E, "n_times": n_t, "dt": dt, "settle": settle, "k_eval": k_eval.tolist(),
           "bayes": {"windows": {}}, "local": {"windows": {}}}
    # (i) the theorem: Bayesian PCRB, information averaged over the episode distribution
    Pb = pcrb_bayes_full(F, H, Q, R, Sigma, k_eval)[None]
    out["bayes"]["full"] = summarise(Pb, Sigma, sig, sensors)
    # (ii) the local (conditional) recursion along each realised trajectory: what an EKF computes
    Pf = pcrb_full(F, H, Q, R, Sigma, k_eval)
    out["local"]["full"] = summarise(Pf, Sigma, sig, sensors)
    out["local"]["full"]["per_episode_total"] = _per_episode_total(Pf, sig).tolist()
    for Ws in windows_s:
        W = max(1, int(round(Ws / dt)))
        W = min(W, settle)                                 # history available before the first eval time
        tag = f"{Ws:g}"
        Pw = pcrb_bayes_window(F, H, Q, R, Sigma, k_eval, W)[None]
        out["bayes"]["windows"][tag] = {"W_samples": W, "W_s": W * dt, **summarise(Pw, Sigma, sig, sensors)}
        Pw = pcrb_window(F, H, Q, R, Sigma, k_eval, W)
        out["local"]["windows"][tag] = {"W_samples": W, "W_s": W * dt, **summarise(Pw, Sigma, sig, sensors)}
    # no-information reference: climatology itself (bits 0, crb = 1 per state up to correlations)
    out["climatology"] = summarise(Sigma.expand(1, 1, *Sigma.shape), Sigma, sig, sensors)
    return out


def _per_episode_total(P: torch.Tensor, sigma) -> torch.Tensor:
    """Per-episode RMS bound on the total NRMSE: sqrt(mean_t mean_j P_jj / sigma_j^2), (E,)."""
    s = torch.as_tensor(sigma, dtype=P.dtype)
    Pn = P / (s[:, None] * s[None, :])
    return Pn.diagonal(dim1=-2, dim2=-1).mean(1).mean(-1).sqrt()


def input_descriptors(u: torch.Tensor, dt: float, f_slow: float) -> dict:
    """Excitation descriptors of the input set: RMS, fraction of power below the plant's slowest
    mode (|Re lambda|_min / 2pi Hz), mean dwell between level changes."""
    U = u[..., 0].double()
    U0 = U - U.mean(1, keepdim=True)
    spec = torch.fft.rfft(U0, dim=1).abs().pow(2).mean(0)
    freqs = torch.fft.rfftfreq(U.shape[1], d=dt)
    tot = float(spec[1:].sum())
    slow = float(spec[1:][freqs[1:] <= f_slow].sum())
    dU = (U[:, 1:] - U[:, :-1]).abs()
    changes = (dU > 1e-9).double().sum(1).clamp(min=1)
    return {"rms": float(U.pow(2).mean().sqrt()), "rms_ac": float(U0.pow(2).mean().sqrt()),
            "slow_band_fraction": slow / max(tot, 1e-30), "f_slow_hz": f_slow,
            "mean_switch_interval_s": float((U.shape[1] * dt / changes).mean())}
