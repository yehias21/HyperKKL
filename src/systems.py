"""
Dynamical systems for KKL observer design.

Each system defines:
    - function(t, u, x): dynamics dx/dt = f(x, u)
    - output(x): measurement y = h(x)
    - x_size, y_size, z_size: dimensions

All systems support batch mode (torch tensors with shape (batch, x_size))
and scalar mode (numpy arrays) for backward compatibility.

Standalone test:
    python -m src.systems
"""

from __future__ import annotations

import numpy as np
import torch
from smt.sampling_methods import LHS


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class System:
    """Base dynamical system."""

    def __init__(self, x_size: int, y_size: int, z_size: int,
                 add_noise: bool = False,
                 noise_std: float = 0.01,
                 noise_measurement_std: float = 0.01):
        self.x_size = x_size
        self.y_size = y_size
        self.z_size = z_size
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.noise_measurement_std = noise_measurement_std

    def function(self, t, u, x):
        raise NotImplementedError

    def output(self, x):
        raise NotImplementedError

    def sample_ic(self, limits: np.ndarray, n_samples: int, seed: int = 42):
        return LHS(xlimits=limits, random_state=seed)(n_samples)

    def _process_noise(self):
        return np.random.normal(0, self.noise_std, self.x_size)

    def _measurement_noise(self):
        return np.random.normal(0, self.noise_measurement_std, self.y_size).squeeze()


# ---------------------------------------------------------------------------
# Reversed Duffing
# ---------------------------------------------------------------------------

class RevDuff(System):
    """Reversed Duffing oscillator: x1' = x2, x2' = -x1 - x1^3 + u."""

    def __init__(self, zdim: int = 5, add_noise: bool = True, **kwargs):
        z = 5 if zdim == 5 else 3
        super().__init__(2, 1, z, add_noise,
                         kwargs.get("noise_std", 0.01),
                         kwargs.get("noise_measurement_std", 0.01))

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return torch.stack([x[:, 1], -x[:, 0] - x[:, 0] ** 3 + u], dim=1)
        x1, x2 = x[0], x[1]
        out = np.array([x2, -x1 - x1 ** 3 + (u if np.isscalar(u) else float(u))])
        return out

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0:1]
        return x[0] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# Van der Pol
# ---------------------------------------------------------------------------

class VdP(System):
    """Van der Pol oscillator."""

    def __init__(self, zdim: int = 5, my: float = 1.0,
                 add_noise: bool = True, **kwargs):
        z = 5 if zdim == 5 else 3
        super().__init__(2, 1, z, add_noise,
                         kwargs.get("noise_std", 0.01),
                         kwargs.get("noise_measurement_std", 0.01))
        self.my = my

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            x1, x2 = x[:, 0], x[:, 1]
            return torch.stack([x2, self.my * (1 - x1 ** 2) * x2 - x1 + u], dim=1)
        x1, x2 = x[0], x[1]
        u_val = u if np.isscalar(u) else float(u)
        return np.array([x2, self.my * (1 - x1 ** 2) * x2 - x1 + u_val])

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0:1]
        return x[0] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# Lorenz
# ---------------------------------------------------------------------------

class Lorenz(System):
    """Lorenz system."""

    def __init__(self, rho: float = 28, sigma: float = 10,
                 beta: float = 8 / 3, add_noise: bool = True, **kwargs):
        super().__init__(3, 1, 7, add_noise,
                         kwargs.get("noise_std", 0.01),
                         kwargs.get("noise_measurement_std", 0.01))
        self.rho = rho
        self.sigma = sigma
        self.beta = beta

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            return torch.stack([
                self.sigma * (x2 - x1),
                x1 * (self.rho - x3) - x2 + u,
                x1 * x2 - self.beta * x3,
            ], dim=1)
        x1, x2, x3 = x[0], x[1], x[2]
        u_val = u if np.isscalar(u) else float(u)
        return np.array([
            self.sigma * (x2 - x1),
            x1 * (self.rho - x3) - x2 + u_val,
            x1 * x2 - self.beta * x3,
        ])

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 1:2]
        return x[1] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# Rossler
# ---------------------------------------------------------------------------

class Rossler(System):
    """Rossler system."""

    def __init__(self, a: float = 0.1, b: float = 0.1, c: float = 14,
                 add_noise: bool = True, **kwargs):
        super().__init__(3, 1, 7, add_noise,
                         kwargs.get("noise_std", 0.01),
                         kwargs.get("noise_measurement_std", 0.01))
        self.a = a
        self.b = b
        self.c = c

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
            return torch.stack([
                -(x2 + x3),
                x1 + self.a * x2 + u,
                self.b + x3 * (x1 - self.c),
            ], dim=1)
        x1, x2, x3 = x[0], x[1], x[2]
        u_val = u if np.isscalar(u) else float(u)
        return np.array([
            -(x2 + x3),
            x1 + self.a * x2 + u_val,
            self.b + x3 * (x1 - self.c),
        ])

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 1:2]
        return x[1] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# FitzHugh-Nagumo
# ---------------------------------------------------------------------------

class FitzHughNagumo(System):
    """FitzHugh-Nagumo neuron model.

    v' = v - v^3/3 - w + I_ext + u
    w' = epsilon * (v + a - b*w)
    y = v (membrane potential measurement)
    """

    def __init__(self, a: float = 0.7, b: float = 0.8,
                 epsilon: float = 0.08, I_ext: float = 0.5,
                 zdim: int = 5, add_noise: bool = True, **kwargs):
        z = 5 if zdim == 5 else 3
        super().__init__(2, 1, z, add_noise,
                         kwargs.get("noise_std", 0.01),
                         kwargs.get("noise_measurement_std", 0.01))
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.I_ext = I_ext

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            v, w = x[:, 0], x[:, 1]
            v_dot = v - v ** 3 / 3 - w + self.I_ext + u
            w_dot = self.epsilon * (v + self.a - self.b * w)
            return torch.stack([v_dot, w_dot], dim=1)
        v, w = x[0], x[1]
        u_val = u if np.isscalar(u) else float(u)
        return np.array([
            v - v ** 3 / 3 - w + self.I_ext + u_val,
            self.epsilon * (v + self.a - self.b * w),
        ])

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0:1]
        return x[0] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# Highway Traffic (5-cell Greenshields)
# ---------------------------------------------------------------------------

class HighwayTraffic(System):
    """5-cell Greenshields highway traffic model."""

    def __init__(self, add_noise: bool = True,
                 u_main_base: float = 1.2, u_ramp_base: float = 0.3,
                 **kwargs):
        super().__init__(5, 1, 11, add_noise,
                         kwargs.get("noise_std", 0.001),
                         kwargs.get("noise_measurement_std", 0.001))
        self.v_free = 30.0
        self.rho_max = 0.25
        self.L = 500.0
        self.u_main_base = u_main_base
        self.u_ramp_base = u_ramp_base

        self._A = (self.v_free / self.L) * np.array([
            [-1, 0, 0, 0, 0],
            [1, -1, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, 1, -1, 0],
            [0, 0, 0, 1, -1],
        ], dtype=np.float64)

        self._B = (1.0 / self.L) * np.array([
            [1, 0], [0, 0], [0, 1], [0, 0], [0, 0],
        ], dtype=np.float64)

    def function(self, t, u, x):
        if torch.is_tensor(x) and x.ndim > 1:
            u_scalar = u if torch.is_tensor(u) else torch.tensor(u, dtype=x.dtype, device=x.device)
            if u_scalar.ndim == 0:
                u_scalar = u_scalar.expand(x.shape[0])
            phi = x - (1.0 / self.rho_max) * x ** 2
            A_t = torch.tensor(self._A, dtype=x.dtype, device=x.device)
            B_t = torch.tensor(self._B, dtype=x.dtype, device=x.device)
            u_vec = torch.stack([
                torch.full_like(u_scalar, self.u_main_base),
                self.u_ramp_base + u_scalar,
            ], dim=1)
            return (phi @ A_t.T) + (u_vec @ B_t.T)
        u_val = float(u) if not isinstance(u, (int, float)) else u
        u_vec = np.array([self.u_main_base, self.u_ramp_base + u_val])
        phi = np.asarray(x) - (1.0 / self.rho_max) * np.asarray(x) ** 2
        return self._A @ phi + self._B @ u_vec

    def output(self, x):
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 4:5]
        return x[4] + (self._measurement_noise() if self.add_noise else 0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SYSTEM_CLASSES = {
    "RevDuff": RevDuff,
    "VdP": VdP,
    "Lorenz": Lorenz,
    "Rossler": Rossler,
    "FitzHughNagumo": FitzHughNagumo,
    "HighwayTraffic": HighwayTraffic,
}


def create_system(config) -> System:
    """Create a system instance from a SystemConfig."""
    cls = SYSTEM_CLASSES[config.class_name]
    return cls(**config.init_args)


# ---------------------------------------------------------------------------
# RK4 integrator
# ---------------------------------------------------------------------------

def rk4_step(system, t, u, x, dt):
    """Single RK4 step. Works with both numpy scalars and torch batches."""
    k1 = system.function(t, u, x)
    k2 = system.function(t + dt / 2, u, x + dt / 2 * k1)
    k3 = system.function(t + dt / 2, u, x + dt / 2 * k2)
    k4 = system.function(t + dt, u, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_system(system, ic, t_eval, u_func=None):
    """Simulate system forward using RK4.

    Args:
        system: System instance
        ic: initial condition, shape (x_size,) or (batch, x_size)
        t_eval: time array (N+1,)
        u_func: callable(t) -> scalar/array, or None for u=0

    Returns:
        x_traj: (N+1, ..., x_size)
    """
    dt = t_eval[1] - t_eval[0]
    x = torch.tensor(ic, dtype=torch.float64)
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # Precompute u values
    if u_func is None:
        u_vals = np.zeros(len(t_eval))
    elif callable(u_func):
        u_vals = np.array([u_func(t) for t in t_eval])
    else:
        u_vals = np.asarray(u_func)

    u_t = torch.tensor(u_vals, dtype=torch.float64)

    traj = [x.clone()]
    N = len(t_eval) - 1
    for i in range(N):
        u_i = u_t[i] if x.shape[0] == 1 else u_t[i].expand(x.shape[0])
        u_half = (u_t[i] + u_t[min(i + 1, N)]) / 2.0
        if x.shape[0] > 1:
            u_half = u_half.expand(x.shape[0])
        u_next = u_t[min(i + 1, N)]
        if x.shape[0] > 1:
            u_next = u_next.expand(x.shape[0])

        k1 = system.function(t_eval[i], u_i, x)
        k2 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k1)
        k3 = system.function(t_eval[i] + dt / 2, u_half, x + dt / 2 * k2)
        k4 = system.function(t_eval[min(i + 1, N)], u_next, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj.append(x.clone())

    return torch.stack(traj)  # (N+1, batch, x_size)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems_demo = {
        "duffing": RevDuff(add_noise=False),
        "vdp": VdP(my=1.0, add_noise=False),
        "lorenz": Lorenz(add_noise=False),
    }

    for name, sys in systems_demo.items():
        t_eval = np.linspace(0, 20, 500)
        ic = np.random.uniform(-1, 1, sys.x_size)
        traj = simulate_system(sys, ic, t_eval).squeeze(1).numpy()
        print(f"{name}: trajectory shape {traj.shape}")

        fig, ax = plt.subplots(figsize=(8, 3))
        for d in range(sys.x_size):
            ax.plot(t_eval, traj[:, d], label=f"x{d + 1}")
        ax.set_title(f"{name} system")
        ax.legend()
        fig.savefig(f"/tmp/{name}_demo.png", dpi=100)
        plt.close(fig)
        print(f"  Plot saved to /tmp/{name}_demo.png")
