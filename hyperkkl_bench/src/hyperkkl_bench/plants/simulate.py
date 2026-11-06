"""The integrator that generated the episodes (x part only).

RK4 on xdot = f(x, u) with the input held constant over each sample interval (zero-order hold:
u(t) = u_k on [t_k, t_{k+1})), float64. The data additionally adds Euler-Maruyama process noise
after every step: x_{k+1} += sigma_p * sqrt(dt) * N(0, I), sigma_p = process_noise_rel * x_std
(metadata: process_noise_rel = 0.005).
"""
import torch


def rk4_zoh_step(plant, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """x (B, nx), u (B, 1) held over the step -> x after dt."""
    a1 = plant.f(x, u)
    a2 = plant.f(x + dt / 2 * a1, u)
    a3 = plant.f(x + dt / 2 * a2, u)
    a4 = plant.f(x + dt * a3, u)
    return x + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
