"""KKL observer design: the latent filter  zdot = A z + B y  and the quantities it fixes.

    butterworth_AB(nz, alpha, ny)   normal real-Jordan A with spectral abscissa alpha
                                    (so ||e^{At}|| = e^{-alpha t} exactly, kappa = 1)
    choose_dt(A)                    largest dt with max|lambda| dt <= 0.1
    washout_time / memory_length    horizons from the exponential tail bound
    latent_step(A, B, dt)           exact one-step map of the latent filter
    sylvester_reference             linear-plant ground truth P with P F = A P + B C
    validate(A, B, dt)              design gate, refuses ill-posed designs
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_lyapunov, solve_sylvester


def butterworth_poles(nz: int, alpha: float) -> np.ndarray:
    """Poles on a circle in the open left half-plane with real part >= alpha."""
    r = alpha / np.sin(np.pi / (2 * nz))
    th = np.pi / 2 + (2 * np.arange(nz) + 1) * np.pi / (2 * nz)
    return r * np.exp(1j * th)


def real_jordan(poles: np.ndarray) -> np.ndarray:
    """Block-diagonal real matrix with the given conjugate-closed spectrum."""
    poles = list(np.asarray(poles))
    blocks = []
    while poles:
        lam = poles.pop(0)
        if abs(lam.imag) < 1e-12:
            blocks.append(np.array([[lam.real]]))
            continue
        j = int(np.argmin([abs(p - np.conj(lam)) for p in poles]))
        poles.pop(j)
        s, w = lam.real, abs(lam.imag)
        blocks.append(np.array([[s, w], [-w, s]]))
    n = sum(b.shape[0] for b in blocks)
    A, o = np.zeros((n, n)), 0
    for b in blocks:
        A[o:o + len(b), o:o + len(b)] = b
        o += len(b)
    return A


def input_matrix(nz: int, ny: int) -> np.ndarray:
    """All-ones for ny = 1; distinct well-spread columns for ny > 1 (identical columns
    would collapse the measurements into their sum)."""
    if ny == 1:
        return np.ones((nz, 1))
    i, c = np.arange(nz)[:, None], np.arange(ny)[None, :]
    return 1.0 + 0.5 * np.cos(np.pi * i * (c + 1) / nz)


def butterworth_AB(nz: int, alpha: float, ny: int = 1) -> tuple:
    return real_jordan(butterworth_poles(nz, alpha)), input_matrix(nz, ny)


def spectral_bounds(A: np.ndarray, n_grid: int = 400) -> tuple:
    """(kappa, alpha) in ||e^{At}|| <= kappa e^{-alpha t}; kappa is measured."""
    alpha = float(-np.linalg.eigvals(A).real.max())
    if alpha <= 0:
        raise ValueError("A is not Hurwitz")
    ts = np.linspace(0.0, 10.0 / alpha, n_grid)
    kappa = max(np.linalg.norm(expm(A * t), 2) * np.exp(alpha * t) for t in ts)
    return float(kappa), alpha


def choose_dt(A: np.ndarray, target: float = 0.1) -> float:
    return float(target / np.abs(np.linalg.eigvals(A)).max())


def washout_time(kappa, alpha, B_norm, y_bar, z_scale, eps_rel=1e-3, z0_norm=0.0):
    """t such that  kappa e^{-alpha t}(||z0|| + kappa ||B|| ybar / alpha) <= eps ||z||."""
    c0 = kappa * (z0_norm + kappa * B_norm * y_bar / alpha)
    tgt = eps_rel * max(z_scale, 1e-12)
    return 0.0 if c0 <= tgt else float(np.log(c0 / tgt) / alpha)


def memory_length(kappa, alpha, B_norm, y_bar, z_scale, eps_rel=1e-3):
    """L past which output history contributes below eps to the latent state
    (the same tail bound; Lemma 1 in THEORY.md)."""
    c0 = kappa * B_norm * y_bar / alpha
    tgt = eps_rel * max(z_scale, 1e-12)
    return 0.0 if c0 <= tgt else float(np.log(c0 / tgt) / alpha)


def latent_step(A: np.ndarray, B: np.ndarray, dt: float, n_quad: int = 16) -> dict:
    r"""Exact one-step map of  zdot = A z + B y(t) + g  with y linear on [t_k, t_{k+1}]:

        z_{k+1} = Ad z_k + W0 y_k + W1 y_{k+1} + Wg g

    Ad = e^{A dt}; W0, W1 integrate e^{A(dt-s)} B against the two hat functions; Wg is
    A^{-1}(Ad - I). The linear part is exact; the only error is the interpolation of y.
    """
    nz = A.shape[0]
    xg, wg = np.polynomial.legendre.leggauss(n_quad)
    s = 0.5 * dt * (xg + 1.0)
    w = 0.5 * dt * wg
    W0 = np.zeros((nz, nz)); W1 = np.zeros((nz, nz)); Wg = np.zeros((nz, nz))
    for sj, wj in zip(s, w):
        E = expm(A * (dt - sj))
        W0 += wj * (1 - sj / dt) * E
        W1 += wj * (sj / dt) * E
        Wg += wj * E
    return {"Ad": expm(A * dt), "W0": W0 @ B, "W1": W1 @ B, "Wg": Wg}


def sylvester_reference(F, C, A, B) -> np.ndarray:
    """Exact KKL map z = P x of a linear plant: P F = A P + B C."""
    gap = np.abs(np.linalg.eigvals(A)[:, None] - np.linalg.eigvals(F)[None, :]).min()
    if gap < 1e-8:
        raise ValueError("spec(A) and spec(F) intersect")
    return solve_sylvester(-A, F, B @ C)


def validate(A: np.ndarray, B: np.ndarray, dt: float, alpha: float) -> dict:
    """Design gate: every check must hold before anything is trained."""
    nz, ny = B.shape[0], B.shape[1]
    kappa, a = spectral_bounds(A)
    Km = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(nz)])
    Kn = Km / np.linalg.norm(Km, axis=0, keepdims=True)
    W = solve_lyapunov(A, -B @ B.T)
    sw = np.linalg.svd(W, compute_uv=False)
    lam_dt = float(np.abs(np.linalg.eigvals(A)).max() * dt)
    checks = {
        "B_shape": B.shape == (nz, ny),
        "alpha_matches": abs(a - alpha) < 1e-6 * max(1.0, alpha),
        "controllable": int(np.linalg.matrix_rank(Kn)) == nz,
        "normal_kappa_1": abs(kappa - 1.0) < 1e-6,
        "resolution": lam_dt <= 0.1 + 1e-9,
        "gramian_ok": bool(sw[0] / max(sw[-1], 1e-300) < 1e12),
    }
    return {"ok": all(checks.values()), "checks": checks, "kappa": kappa, "alpha": a,
            "max_lam_dt": lam_dt, "cond_gramian": float(sw[0] / max(sw[-1], 1e-300))}
