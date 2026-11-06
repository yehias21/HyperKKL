"""Sensor model used to generate the measurements y (copied from the benchmark generator).

Settings of the benchmark:
    C1, C3-partial, C3-mismatch30:  {"noise": 0.03}                                  white Gaussian
    C3-realistic:                   {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}
Magnitudes are relative to each output's reference std y_std (metadata).
"""
import math

import numpy as np
import torch


def burst_window(burst: dict, N1: int, dt: float, t_ref: float = 1.0) -> tuple:
    """(k0, L) of a burst sensor outage on an episode of N1 samples: the sensor freezes at
    its sample k0 = round(start_frac (N1 - 1)) and repeats it for the next L samples
    (k0 + 1 .. k0 + L, L = round(len_tref t_ref / dt), clipped to the episode); the first
    fresh sample is k0 + L + 1. Shared by measure() and the recovery metric."""
    k0 = int(round(float(burst.get("start_frac", 0.6)) * (N1 - 1)))
    k0 = min(max(k0, 0), N1 - 1)
    L = int(round(float(burst.get("len_tref", 2.0)) * float(t_ref) / float(dt)))
    return k0, max(0, min(L, N1 - 1 - k0))


def _student_t_unit(u: torch.Tensor, nu: float) -> torch.Tensor:
    """Unit-variance Student-t(nu) draws from uniforms u (inverse CDF; nu > 2)."""
    from scipy.stats import t as student_t
    uu = u.detach().cpu().double().numpy().clip(1e-15, 1 - 1e-15)
    w = student_t.ppf(uu, nu) * math.sqrt((nu - 2.0) / nu)
    return torch.as_tensor(w, dtype=u.dtype, device=u.device)


@torch.no_grad()
def measure(y_clean: torch.Tensor, sensor: dict, y_std, seed: int, dt: float = None,
            t_ref: float = 1.0) -> torch.Tensor:
    """Apply the sensor model. Every magnitude is relative to the output's reference std.

    sensor keys: noise (rel. std), kind (gaussian|colored|colored_t), rho, nu,
    outlier_rate, outlier_scale, quant (step, rel.), drift (amplitude, rel.), dropout (hold
    prob.), burst ({start_frac, len_tref}: deterministic sample-and-hold outage, needs dt)

    kind colored:   e_k = rho e_{k-1} + sqrt(1 - rho^2) w_k, w Gaussian
    kind colored_t: the same AR(1) driven by Student-t(nu) w scaled to unit variance
                    (nu > 2): unit stationary variance, heavy tails (non-Gaussian, coloured)
    Both are deterministic in ``seed`` (one generator for every random draw).
    """
    dev = y_clean.device
    g = torch.Generator(device=dev).manual_seed(int(seed) + 7)
    ys = torch.as_tensor(np.asarray(y_std), dtype=y_clean.dtype, device=dev)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=y_clean.dtype, device=dev)
    ru = lambda *s: torch.rand(*s, generator=g, dtype=y_clean.dtype, device=dev)
    E, N1, ny = y_clean.shape
    y = y_clean.clone()
    sd = float(sensor.get("noise", 0.0))
    kind = sensor.get("kind", "gaussian")
    if kind not in ("gaussian", "colored", "colored_t"):
        raise ValueError(f"unknown sensor kind {kind!r}")
    if kind == "colored_t":
        nu, rho = float(sensor.get("nu", 3.0)), float(sensor.get("rho", 0.9))
        if not nu > 2.0:
            raise ValueError(f"colored_t needs nu > 2 (finite variance), got {nu}")
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"colored_t needs 0 <= rho < 1, got {rho}")
    if sd > 0:
        if kind == "colored_t":
            from scipy.signal import lfilter
            w = _student_t_unit(ru(E, N1, ny), nu).cpu().numpy()
            b0 = math.sqrt(1 - rho ** 2)
            zi = ((1.0 - b0) * w[:, :1])                      # e_0 = w_0 (stationary start)
            v, _ = lfilter([b0], [1.0, -rho], w, axis=1, zi=zi)
            w = torch.as_tensor(v, dtype=y_clean.dtype, device=dev)
        else:
            w = rn(E, N1, ny)
            if kind == "colored":
                rho = float(sensor.get("rho", 0.95))
                v = torch.empty_like(w)
                v[:, 0] = w[:, 0]
                for k in range(1, N1):
                    v[:, k] = rho * v[:, k - 1] + math.sqrt(1 - rho ** 2) * w[:, k]
                w = v
        y = y + sd * ys * w
    if float(sensor.get("drift", 0.0)) > 0:
        b = torch.cumsum(rn(E, N1, ny), 1)
        b = b / b.abs().amax(1, keepdim=True).clamp(min=1e-12)
        y = y + float(sensor["drift"]) * ys * b
    if float(sensor.get("outlier_rate", 0.0)) > 0:
        m = ru(E, N1, ny) < float(sensor["outlier_rate"])
        sgn = torch.where(ru(E, N1, ny) < 0.5, -1.0, 1.0).to(y.dtype)
        y = torch.where(m, y + float(sensor.get("outlier_scale", 12.0)) * ys * sgn, y)
    if float(sensor.get("quant", 0.0)) > 0:
        q = float(sensor["quant"]) * ys
        y = torch.round(y / q) * q
    if float(sensor.get("dropout", 0.0)) > 0:
        hold = ru(E, N1) < float(sensor["dropout"])
        for k in range(1, N1):
            y[:, k] = torch.where(hold[:, k, None], y[:, k - 1], y[:, k])
    if sensor.get("burst"):
        if dt is None:
            raise ValueError("a burst outage needs the sample time dt")
        k0, L = burst_window(sensor["burst"], N1, dt, t_ref)
        if L > 0:
            y[:, k0 + 1:k0 + L + 1] = y[:, k0:k0 + 1]
    return y
