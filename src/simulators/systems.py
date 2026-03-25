"""
Dynamical system definitions for the HyperKKL pipeline.

Each concrete system implements:
    - ``function(t, u, x)`` — state dynamics (supports torch batch mode)
    - ``output(x)``         — measurement map   (supports torch batch mode)

Torch batch mode is detected via ``torch.is_tensor(x) and x.ndim > 1``.
When active, *x* has shape ``(batch, x_size)`` and every return keeps that
leading batch dimension.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from smt.sampling_methods import LHS

from hyperkkl.src.simulators.utils import RK4

# --------------------------------------------------------------------------- #
#  Decorators                                                                  #
# --------------------------------------------------------------------------- #


def add_process_noise(method: Callable) -> Callable:
    """Decorator that adds process noise to the output of ``function``."""

    def wrapper(self: "System", t: float, u, x):
        x_dot = method(self, t, u, x)
        if self.add_noise:
            w, _ = self.gen_noise()
            return x_dot + w
        return x_dot

    return wrapper


def add_measurement_noise(method: Callable) -> Callable:
    """Decorator that adds measurement noise to the output of ``output``."""

    def wrapper(self: "System", x):
        y = method(self, x)
        if self.add_noise:
            _, v = self.gen_noise()
            return y + v
        return y

    return wrapper


# --------------------------------------------------------------------------- #
#  Base class                                                                  #
# --------------------------------------------------------------------------- #


class System:
    """Base class for all dynamical systems.

    Parameters
    ----------
    function : callable
        State dynamics ``f(t, u, x)``.
    output : callable
        Measurement map ``h(x)``.
    add_noise : bool
        Whether to inject process / measurement noise.
    noise_process_mean, noise_process_std : float
        Mean and standard deviation for additive process noise.
    noise_measurement_mean, noise_measurement_std : float
        Mean and standard deviation for additive measurement noise.
    """

    x_size: int
    y_size: int
    z_size: int
    input: Optional[Callable]

    def __init__(
        self,
        function: Callable,
        output: Callable,
        add_noise: bool = False,
        noise_process_mean: float = 0.0,
        noise_process_std: float = 0.01,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.01,
    ) -> None:
        self.function = function
        self.output = output
        self.add_noise = add_noise
        self.noise: Union[float, NDArray] = 0
        self.noise_process_mean = noise_process_mean
        self.noise_process_std = noise_process_std
        self.noise_measurement_mean = noise_measurement_mean
        self.noise_measurement_std = noise_measurement_std

    # -- noise utilities ---------------------------------------------------- #

    def gen_noise(self) -> Tuple[NDArray, Union[float, NDArray]]:
        """Sample process and measurement noise vectors."""
        x_noise = np.random.normal(
            self.noise_process_mean, self.noise_process_std, (self.x_size,)
        )
        y_noise = np.random.normal(
            self.noise_measurement_mean, self.noise_measurement_std, (self.y_size,)
        )
        if self.y_size == 1:
            y_noise = y_noise[0]
        return x_noise, y_noise

    def toggle_noise(self) -> None:
        """Toggle the ``add_noise`` flag."""
        self.add_noise = not self.add_noise

    # -- initial-condition sampling ----------------------------------------- #

    def sample_ic(
        self,
        sample_space: NDArray,
        samples: int,
        seed: int,
    ) -> NDArray:
        """Latin-hypercube sample of initial conditions.

        Parameters
        ----------
        sample_space : ndarray of shape (x_size, 2)
            Lower / upper bounds for each state dimension.
        samples : int
            Number of ICs to draw.
        seed : int
            Random seed for reproducibility.
        """
        return LHS(xlimits=sample_space, random_state=seed)(samples)

    # -- simulation --------------------------------------------------------- #

    def simulate(
        self,
        a: float,
        b: float,
        N: int,
        v: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """Integrate the system from *a* to *b* in *N* steps starting at *v*.

        Returns
        -------
        x : ndarray
            State trajectory.
        t : ndarray
            Time vector.
        """
        x, t = RK4(self.function, a, b, N, v, self.input)
        return np.array(x), t

    def generate_data(
        self,
        ic: NDArray,
        a: float,
        b: float,
        N: int,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Run multiple simulations and collect state + output trajectories.

        Parameters
        ----------
        ic : ndarray of shape (n_traj, x_size)
            Initial conditions.
        a, b : float
            Integration interval ``[a, b]``.
        N : int
            Number of time-steps.

        Returns
        -------
        data : ndarray
            State trajectories, shape ``(n_traj, N+1, x_size)``.
        output : ndarray
            Output trajectories, shape ``(n_traj, N+1, y_size)`` or similar.
        time : ndarray
            Shared time vector.
        """
        data: List[NDArray] = []
        output: List[NDArray] = []
        for i in range(np.size(ic, axis=0)):
            x, time = self.simulate(a, b, N, ic[i])
            temp = [self.output(j) for j in x]
            data.append(x)
            output.append(np.array(temp))
        return np.array(data), np.array(output), time

    # -- helpers ------------------------------------------------------------ #

    def function_output(
        self, x: list
    ) -> Union[NDArray, torch.Tensor]:
        """Stack a list of scalars / tensors into a single array or tensor."""
        if isinstance(x[0], torch.Tensor):
            return torch.stack(x)
        return np.array(x)

    def __contains__(self, x: str) -> bool:
        y_variables = [f"y{i}" for i in range(1, self.y_size + 1)]
        return x in y_variables


# --------------------------------------------------------------------------- #
#  Concrete systems                                                            #
# --------------------------------------------------------------------------- #


class RevDuff(System):
    """Reversed Duffing oscillator.

    State: ``x = [x1, x2]``.
    Dynamics::

        x1' = x2
        x2' = -x1 - x1^3 + u

    Output: ``y = x1``.
    """

    def __init__(
        self,
        zdim: int,
        add_noise: bool = True,
        noise_mean: float = 0.0,
        noise_std: float = 0.01,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.01,
    ) -> None:
        self.y_size = 1
        self.x_size = 2
        if zdim == 5:
            self.z_size = self.y_size * (2 * self.x_size + 1)
        elif zdim == 3:
            self.z_size = self.y_size * (1 * self.x_size + 1)
        else:
            self.z_size = zdim
        self.input = None
        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_mean,
            noise_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1
        if is_batch:
            x1 = x[:, 0]
            x2 = x[:, 1]
        else:
            x1 = x[0]
            x2 = x[1]

        x1_dot = x2
        x2_dot = -x1 - x1 ** 3 + u

        if is_batch:
            return torch.stack([x1_dot, x2_dot], dim=1)

        out = self.function_output([x1_dot, x2_dot])

        if self.add_noise and not is_batch:
            self.noise = self.gen_noise()[0]
            return out + self.noise

        return out

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0].unsqueeze(1)
        y = x[0]
        noise = 0
        if self.add_noise:
            noise = self.gen_noise()[1]
        return y + noise


class VdP(System):
    """Van der Pol oscillator.

    State: ``x = [x1, x2]``.
    Dynamics::

        x1' = x2
        x2' = mu * (1 - x1^2) * x2 - x1 + u

    Output: ``y = x1``.
    """

    def __init__(
        self,
        zdim: int,
        mu: float = 3.0,
        add_noise: bool = True,
        noise_mean: float = 0.0,
        noise_std: float = 0.01,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.01,
    ) -> None:
        self.x_size = 2
        self.y_size = 1
        if zdim == 5:
            self.z_size = self.y_size * (2 * self.x_size + 1)
        elif zdim == 3:
            self.z_size = self.y_size * (1 * self.x_size + 1)
        else:
            self.z_size = zdim
        self.mu = mu
        self.input = None
        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_mean,
            noise_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1

        if is_batch:
            x1 = x[:, 0]
            x2 = x[:, 1]
        else:
            x1 = x[0]
            x2 = x[1]

        x1_dot = x2
        x2_dot = self.mu * (1 - x1 ** 2) * x2 - x1 + u

        if is_batch:
            return torch.stack([x1_dot, x2_dot], dim=1)

        return self.function_output([x1_dot, x2_dot])

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0].unsqueeze(1)
        y = x[0]
        noise = 0
        if self.add_noise:
            noise = self.gen_noise()[1]
        return y + noise


class Lorenz(System):
    """Lorenz attractor.

    State: ``x = [x1, x2, x3]``.
    Dynamics::

        x1' = sigma * (x2 - x1)
        x2' = x1 * (rho - x3) - x2 + u
        x3' = x1 * x2 - beta * x3

    Output: ``y = x2``.
    """

    def __init__(
        self,
        rho: float,
        sigma: float,
        beta: float,
        add_noise: bool = True,
        noise_process_mean: float = 0.0,
        noise_process_std: float = 0.01,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.01,
    ) -> None:
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size * (2 * self.x_size + 1)
        self.input = None
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_process_mean,
            noise_process_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1
        if is_batch:
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]
        else:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]

        x1_dot = self.sigma * (x2 - x1)
        x2_dot = x1 * (self.rho - x3) - x2 + u
        x3_dot = x1 * x2 - self.beta * x3

        if is_batch:
            return torch.stack([x1_dot, x2_dot, x3_dot], dim=1)
        return self.function_output([x1_dot, x2_dot, x3_dot])

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 1].unsqueeze(1)
        if self.add_noise:
            self.noise = self.gen_noise()[1]
        return x[1] + self.noise


class Rossler(System):
    """Rossler attractor.

    State: ``x = [x1, x2, x3]``.
    Dynamics::

        x1' = -(x2 + x3)
        x2' = x1 + a * x2 + u
        x3' = b + x3 * (x1 - c)

    Output: ``y = x2``.
    """

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        add_noise: bool = True,
        noise_mean: float = 0.0,
        noise_std: float = 0.1,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.1,
    ) -> None:
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size * (2 * self.x_size + 1)
        self.input = None
        self.a = a
        self.b = b
        self.c = c
        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_mean,
            noise_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1
        if is_batch:
            x1 = x[:, 0]
            x2 = x[:, 1]
            x3 = x[:, 2]
        else:
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]

        x1_dot = -(x2 + x3)
        x2_dot = x1 + self.a * x2 + u
        x3_dot = self.b + x3 * (x1 - self.c)

        if is_batch:
            return torch.stack([x1_dot, x2_dot, x3_dot], dim=1)
        return self.function_output([x1_dot, x2_dot, x3_dot])

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 1].unsqueeze(1)
        y = x[1]
        if self.add_noise:
            self.noise = self.gen_noise()[1]
        return y + self.noise


class FitzHughNagumo(System):
    """FitzHugh-Nagumo neuron model.

    State: ``x = [v, w]`` where *v* is membrane potential and *w* is the
    recovery variable.
    Dynamics::

        v' = v - v^3/3 - w + I_ext + u
        w' = epsilon * (v + a - b * w)

    Output: ``y = v`` (membrane potential).
    """

    def __init__(
        self,
        zdim: int,
        a: float = 0.7,
        b: float = 0.8,
        epsilon: float = 0.08,
        I_ext: float = 0.5,
        add_noise: bool = True,
        noise_mean: float = 0.0,
        noise_std: float = 0.01,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.01,
    ) -> None:
        self.x_size = 2
        self.y_size = 1
        if zdim == 5:
            self.z_size = self.y_size * (2 * self.x_size + 1)
        elif zdim == 3:
            self.z_size = self.y_size * (1 * self.x_size + 1)
        else:
            self.z_size = zdim
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.I_ext = I_ext
        self.input = None
        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_mean,
            noise_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1
        if is_batch:
            v = x[:, 0]
            w = x[:, 1]
        else:
            v = x[0]
            w = x[1]

        v_dot = v - v ** 3 / 3.0 - w + self.I_ext + u
        w_dot = self.epsilon * (v + self.a - self.b * w)

        if is_batch:
            return torch.stack([v_dot, w_dot], dim=1)
        return self.function_output([v_dot, w_dot])

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 0].unsqueeze(1)
        y = x[0]
        noise = 0
        if self.add_noise:
            noise = self.gen_noise()[1]
        return y + noise


class HighwayTraffic(System):
    """5-cell Greenshields highway traffic model.

    State: ``x in R^5`` — traffic density per cell (veh/m), each in
    ``[0, rho_max]``.

    Input: scalar *u* perturbs the on-ramp merging flow at cell 3.

    Output: ``y = x_5`` (camera sensor at the last cell).

    With ``u = 0`` the system uses constant base inputs:
    ``u_main = 1.2 veh/s``, ``u_ramp = 0.3 veh/s``.
    """

    def __init__(
        self,
        add_noise: bool = True,
        noise_mean: float = 0.0,
        noise_std: float = 0.001,
        noise_measurement_mean: float = 0.0,
        noise_measurement_std: float = 0.001,
        u_main_base: float = 1.2,
        u_ramp_base: float = 0.3,
    ) -> None:
        self.x_size = 5
        self.y_size = 1
        self.z_size = self.y_size * (2 * self.x_size + 1)  # 11
        self.input = None

        # Physical parameters
        self.v_free: float = 30.0      # free-flow speed (m/s)
        self.rho_max: float = 0.25     # jam density (veh/m)
        self.L: float = 500.0          # cell length (m)
        self.u_main_base = u_main_base
        self.u_ramp_base = u_ramp_base

        # Precompute system matrices
        self._A = (self.v_free / self.L) * np.array(
            [
                [-1, 0, 0, 0, 0],
                [1, -1, 0, 0, 0],
                [0, 1, -1, 0, 0],
                [0, 0, 1, -1, 0],
                [0, 0, 0, 1, -1],
            ],
            dtype=np.float64,
        )

        self._B = (1.0 / self.L) * np.array(
            [
                [1, 0],
                [0, 0],
                [0, 1],
                [0, 0],
                [0, 0],
            ],
            dtype=np.float64,
        )

        super().__init__(
            self.function,
            self.output,
            add_noise,
            noise_mean,
            noise_std,
            noise_measurement_mean,
            noise_measurement_std,
        )

    # -- helpers ------------------------------------------------------------ #

    def _flux(self, x) -> Union[NDArray, torch.Tensor]:
        """Greenshields flux (works with both numpy and torch)."""
        return self.v_free * x * (1.0 - x / self.rho_max)

    # -- dynamics ----------------------------------------------------------- #

    def function(self, t: float, u, x) -> Union[NDArray, torch.Tensor]:
        """State dynamics with torch batch support."""
        is_batch = torch.is_tensor(x) and x.ndim > 1

        if is_batch:
            # u is a (batch,) scalar perturbation to u_ramp
            if torch.is_tensor(u):
                u_scalar = u
            else:
                u_scalar = torch.tensor(u, dtype=x.dtype, device=x.device)
            if u_scalar.ndim == 0:
                u_scalar = u_scalar.expand(x.shape[0])

            # Nonlinear Greenshields term: x - (1/rho_max) * x^2
            phi = x - (1.0 / self.rho_max) * x ** 2

            A_t = torch.tensor(self._A, dtype=x.dtype, device=x.device)
            B_t = torch.tensor(self._B, dtype=x.dtype, device=x.device)

            # Input vector: [u_main_base, u_ramp_base + u_perturbation]
            u_vec = torch.stack(
                [
                    torch.full_like(u_scalar, self.u_main_base),
                    self.u_ramp_base + u_scalar,
                ],
                dim=1,
            )  # (batch, 2)

            dxdt = (phi @ A_t.T) + (u_vec @ B_t.T)
            return dxdt
        else:
            # Scalar mode (numpy)
            if isinstance(u, (int, float)):
                u_scalar_np = float(u)
            elif isinstance(u, np.ndarray):
                u_scalar_np = float(u.item()) if u.size == 1 else float(u)
            else:
                u_scalar_np = float(u)

            u_vec_np = np.array([self.u_main_base, self.u_ramp_base + u_scalar_np])
            phi_np = np.asarray(x) - (1.0 / self.rho_max) * np.asarray(x) ** 2
            dxdt = self._A @ phi_np + self._B @ u_vec_np

            return self.function_output(list(dxdt))

    def output(self, x) -> Union[float, torch.Tensor]:
        """Measurement map with torch batch support."""
        if torch.is_tensor(x) and x.ndim > 1:
            return x[:, 4].unsqueeze(1)
        y = x[4]
        noise = 0
        if self.add_noise:
            noise = self.gen_noise()[1]
        return y + noise
