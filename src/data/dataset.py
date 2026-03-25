"""Dataset class for autonomous KKL observer training data generation."""

import multiprocessing
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from hyperkkl.src.simulators.utils import RK4, calc_neg_t, sample_circular, sample_spherical, calc_p


def _system_simulation_worker(
    args: Tuple["System", np.ndarray, float, float, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a chunk of initial conditions in a worker process.

    This function is defined at module level so it can be pickled by
    :mod:`multiprocessing`.

    Args:
        args: Tuple of (system, ic_chunk, a, b, N).

    Returns:
        Tuple of (x_data, output_y, time).
    """
    system, ic_chunk, a, b, N = args
    return system.generate_data(ic_chunk, a, b, N)


class DataSet(Dataset):
    """Torch Dataset that generates synthetic (x, z, y) trajectories for KKL
    observer training.

    The dataset samples initial conditions, simulates the true system forward in
    time (parallelised across CPU cores), computes the corresponding
    z-dynamics via a vectorised RK4 solver, and exposes matched
    ``[x, z, y, x_ph, y_ph]`` tuples for data / physics loss splits.

    Args:
        system: Dynamical system object with ``generate_data``, ``sample_ic``,
            ``x_size``, ``z_size`` attributes.
        M: State matrix of the z-dynamics (z_dim, z_dim).
        K: Input matrix of the z-dynamics (z_dim, y_dim).
        a: Start time of the simulation interval.
        b: End time of the simulation interval.
        N: Number of integration steps.
        samples: Number of initial conditions to sample.
        limits_normal: Bounds used by ``system.sample_ic``.
        seed: Random seed for reproducibility.
        PINN_sample_mode: One of ``'split traj'``, ``'split set'``, or
            ``'no physics'``.
        data_gen_mode: One of ``'negative forward'``, ``'forward'``, or
            ``'negative backward'``.
        pretrained_T: Optional pre-trained forward map used to initialise z0.
    """

    def __init__(
        self,
        system,
        M: np.ndarray,
        K: np.ndarray,
        a: float,
        b: float,
        N: int,
        samples: int,
        limits_normal: np.ndarray,
        seed: int,
        PINN_sample_mode: str = "split traj",
        data_gen_mode: str = "negative forward",
        pretrained_T: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.M = M
        self.K = K
        self.system = system
        self.a = a
        self.b = b
        self.N = N
        self.samples = samples
        self.limits_normal = limits_normal
        self.seed = seed
        self.PINN_sample_mode = PINN_sample_mode
        self.data_gen_mode = data_gen_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate raw trajectory data
        if pretrained_T is not None:
            pretrained_T = pretrained_T.to(self.device)
            self.train_data = self._generate_data_pretrained(pretrained_T)
        else:
            self.train_data = self._generate_data()

        # Flatten trajectories for item-level access
        self.data_length = self.train_data[0].shape[0] * self.train_data[0].shape[1]

        x = torch.from_numpy(self.train_data[0]).reshape(
            self.data_length, system.x_size
        )
        z = torch.from_numpy(self.train_data[1]).reshape(
            self.data_length, system.z_size
        )

        y_np = self.train_data[2]
        if y_np.ndim > 2:
            y = torch.from_numpy(y_np).reshape(self.data_length, y_np.shape[2])
        else:
            y = torch.from_numpy(y_np).reshape(self.data_length)

        # -------------------- Split logic --------------------
        if PINN_sample_mode == "split set":
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.ic_normal = self.train_data[4]

            # Generate a separate physics set
            self.train_data_ph = self._generate_data()
            data_len_ph = (
                self.train_data_ph[0].shape[0] * self.train_data_ph[0].shape[1]
            )
            self.x_data_ph = torch.from_numpy(self.train_data_ph[0]).reshape(
                data_len_ph, system.x_size
            )
            self.z_data_ph = torch.from_numpy(self.train_data_ph[1]).reshape(
                data_len_ph, system.z_size
            )
            self.ic_ph = self.train_data_ph[4]

            y_ph_np = self.train_data_ph[2]
            if y_ph_np.ndim > 2:
                self.output_data_ph = torch.from_numpy(y_ph_np).reshape(
                    data_len_ph, y_ph_np.shape[2]
                )
            else:
                self.output_data_ph = torch.from_numpy(y_ph_np).reshape(data_len_ph)

        elif PINN_sample_mode == "split traj":
            self.data_length = self.data_length // 2
            self.x_data = x[::2]
            self.z_data = z[::2]
            self.output_data = y[::2]
            self.x_data_ph = x[1::2]
            self.z_data_ph = z[1::2]
            self.output_data_ph = y[1::2]
            self.ic = self.train_data[4]

        elif PINN_sample_mode == "no physics":
            self.x_data = x
            self.z_data = z
            self.output_data = y
            self.x_data_ph = x
            self.z_data_ph = y  # placeholder
            self.output_data_ph = y
            self.ic = self.train_data[4]

        else:
            raise ValueError(
                "PINN_sample_mode must be 'split set', 'split traj', or 'no physics'."
            )

        # -------------------- Statistics --------------------
        self.mean_x = torch.mean(self.x_data, dim=0)
        self.mean_z = torch.mean(self.z_data, dim=0)
        self.mean_output = torch.mean(self.output_data, dim=0)
        self.std_x = torch.std(self.x_data, dim=0)
        self.std_z = torch.std(self.z_data, dim=0)
        self.std_output = torch.std(self.output_data, dim=0)

        self.mean_x_ph = torch.mean(self.x_data_ph, dim=0)
        self.mean_z_ph = torch.mean(self.z_data_ph, dim=0)
        self.mean_output_ph = torch.mean(self.output_data_ph, dim=0)
        self.std_x_ph = torch.std(self.x_data_ph, dim=0)
        self.std_z_ph = torch.std(self.z_data_ph, dim=0)
        self.std_output_ph = torch.std(self.output_data, dim=0)

        self.time = torch.from_numpy(self.train_data[3])

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        """Return ``[x, z, y, x_ph, y_ph]`` for the given index."""
        return [
            self.x_data[idx].float(),
            self.z_data[idx].float(),
            self.output_data[idx].float(),
            self.x_data_ph[idx].float(),
            self.output_data_ph[idx].float(),
        ]

    def normalize(self) -> None:
        """Normalise data tensors to zero-mean, unit-variance."""
        self.x_data = (self.x_data - self.mean_x) / self.std_x
        self.z_data = (self.z_data - self.mean_z) / self.std_z
        self.output_data = (self.output_data - self.mean_output) / self.std_output

    def set_physics_limits(self, limits: np.ndarray) -> None:
        """Set physics-loss sampling limits (only valid for ``'split traj'``)."""
        if self.PINN_sample_mode == "split traj":
            self.limit_physics = limits
        else:
            raise ValueError("Can only set limits when PINN_sample_mode is 'split traj'.")

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def parallel_system_simulation(
        self,
        ic_total: np.ndarray,
        a: float,
        b: float,
        N: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run system simulation in parallel across available CPU cores.

        Args:
            ic_total: Array of initial conditions, shape ``(n_ic, x_dim)``.
            a: Start time.
            b: End time.
            N: Number of integration steps.

        Returns:
            Tuple of ``(x_data, output, time)`` arrays.
        """
        num_processes = min(multiprocessing.cpu_count(), 16)
        chunk_size = int(np.ceil(len(ic_total) / num_processes))
        chunks = [
            (self.system, ic_total[i : i + chunk_size], a, b, N)
            for i in range(0, len(ic_total), chunk_size)
        ]

        if num_processes > 1:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_system_simulation_worker, chunks)
        else:
            results = [_system_simulation_worker(c) for c in chunks]

        x_data = np.concatenate([r[0] for r in results], axis=0)
        output = np.concatenate([r[1] for r in results], axis=0)
        time = results[0][2]
        return x_data, output, time

    def vectorized_z_simulation(
        self,
        output: np.ndarray,
        a: float,
        b: float,
        ic_z: np.ndarray,
        N: int,
        scalar_y: bool = False,
    ) -> np.ndarray:
        """Vectorised RK4 solver for the linear z-dynamics ``z_dot = Mz + Ky``.

        Args:
            output: Observation trajectories, shape ``(samples, steps, y_dim)``
                or ``(samples, steps)`` when *scalar_y* is True.
            a: Start time.
            b: End time.
            ic_z: Initial z-states, shape ``(samples, z_dim)``.
            N: Number of integration steps.
            scalar_y: Whether ``output`` is scalar-valued.

        Returns:
            Z trajectories, shape ``(samples, N+1, z_dim)``.
        """
        h = (b - a) / N
        M_T = self.M.T
        K_T = self.K.T

        if scalar_y:
            Y = output[:, 1:, np.newaxis]
        else:
            Y = output[:, 1:, :]

        Z = ic_z
        z_traj = [Z]

        # Pre-compute K @ y for every time step
        if scalar_y:
            Ky_all = (
                (Y @ K_T)
                if self.K.ndim > 1
                else np.outer(Y, self.K).reshape(Y.shape[0], Y.shape[1], -1)
            )
        else:
            Ky_all = np.matmul(Y, K_T)

        steps = Ky_all.shape[1]
        for i in range(steps):
            ky = Ky_all[:, i, :]

            k1 = Z @ M_T + ky
            k2 = (Z + 0.5 * h * k1) @ M_T + ky
            k3 = (Z + 0.5 * h * k2) @ M_T + ky
            k4 = (Z + h * k3) @ M_T + ky

            Z = Z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_traj.append(Z)

        return np.stack(z_traj, axis=1)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def _generate_data_pretrained(
        self, pretrained_T: torch.nn.Module
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate training data using a pre-trained forward T map for z0.

        Args:
            pretrained_T: Neural network that maps x0 -> z0.

        Returns:
            Tuple of ``(x_data, z_data, output, time, ic)``.
        """
        ic = self.system.sample_ic(self.limits_normal, self.samples, seed=self.seed)

        x_data_fw, output_fw, t_fw = self.parallel_system_simulation(
            ic, self.a, self.b, self.N
        )

        ic_tensor = torch.from_numpy(ic).float().to(self.device)
        with torch.no_grad():
            z0 = pretrained_T(ic_tensor).cpu().numpy()

        scalar_y = output_fw.ndim == 2
        z_data_fw = self.vectorized_z_simulation(
            output_fw, self.a, self.b, z0, self.N, scalar_y
        )
        return x_data_fw, z_data_fw, output_fw, t_fw, ic

    def _generate_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate (x, z, y) trajectory data from sampled initial conditions.

        Supports three data-generation modes:

        * ``'forward'`` -- random z0 with transient cutoff.
        * ``'negative forward'`` -- simulate x backward in negative time to
          converge z, then simulate forward.
        * ``'negative backward'`` -- like *negative forward* but with a
          backward (time-reversed) simulation for the convergence phase.

        Returns:
            Tuple of ``(x_data, z_data, output, time, ic)``.
        """
        t_back = calc_neg_t(self.M, 10, 1e-6)
        h = (self.b - self.a) / self.N

        ic = self.system.sample_ic(self.limits_normal, self.samples, seed=self.seed)

        # Forward x simulation (parallel)
        x_data_fw, output_fw, t_fw = self.parallel_system_simulation(
            ic, self.a, self.b, self.N
        )
        num_traj_samples = x_data_fw.shape[1]

        # ----- 'forward' mode: random z0 with transient cutoff -----
        if self.data_gen_mode == "forward":
            cutoff_ind = int(0.1 * num_traj_samples)
            ic_z = np.random.rand(self.samples, self.system.z_size)

            scalar_y = output_fw.ndim == 2
            z_data_fw = self.vectorized_z_simulation(
                output_fw, self.a, self.b, ic_z, self.N, scalar_y
            )

            x_data_fw = x_data_fw[:, cutoff_ind:, :]
            z_data_fw = z_data_fw[:, cutoff_ind:, :]

            if output_fw.ndim == 2:
                output_fw = output_fw[:, cutoff_ind:]
            else:
                output_fw = output_fw[:, cutoff_ind:, :]

            t_fw = t_fw[cutoff_ind:]
            return x_data_fw, z_data_fw, output_fw, t_fw, ic

        # ----- backward modes: converge z via negative-time simulation -----
        N_back = int(np.abs(t_back / h))

        if self.data_gen_mode == "negative backward":
            _, output_bw, _ = self.parallel_system_simulation(
                ic, self.a, t_back, N_back
            )
            output_bw = np.flip(output_bw, axis=1)
        elif self.data_gen_mode == "negative forward":
            _, output_bw, _ = self.parallel_system_simulation(
                ic, t_back, self.a, N_back
            )
        else:
            raise ValueError(
                "data_gen_mode must be 'forward', 'negative forward', or "
                "'negative backward'."
            )

        ic_z_bw = np.random.rand(self.samples, self.system.z_size)
        scalar_y_bw = output_bw.ndim == 2
        z_data_bw = self.vectorized_z_simulation(
            output_bw, t_back, self.a, ic_z_bw, N_back, scalar_y_bw
        )

        # Use converged z at t=a as initial condition for forward pass
        ic_z = z_data_bw[:, -1, :]

        scalar_y_fw = output_fw.ndim == 2
        z_data_fw = self.vectorized_z_simulation(
            output_fw, self.a, self.b, ic_z, self.N, scalar_y_fw
        )

        return x_data_fw, z_data_fw, output_fw, t_fw, ic
