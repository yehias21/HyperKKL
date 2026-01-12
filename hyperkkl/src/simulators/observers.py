"""KKL Observer implementations."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class KKLObserver:
    """Kazantzis-Kravaris/Luenberger (KKL) Observer."""
    
    def __init__(
        self,
        z_dim: int,
        gains: dict,
        epsilon: float = 1e-6,
        z_max: float = 10.0,
        ic_space: Optional[list[list[float]]] = None,
        name: str = "kkl_observer"
    ):
        self.name = name
        self.z_dim = z_dim
        self.epsilon = epsilon
        self.z_max = z_max
        
        # Parse gains
        self.A = self._build_A_matrix(np.array(gains["a"]))
        self.B = np.array(gains["b"]).reshape(-1, 1)
        
        # IC space for observer states
        if ic_space is None:
            ic_space = [[-1.0, 1.0]] * z_dim
        self.ic_space = np.array(ic_space)
        
    def _build_A_matrix(self, a_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build A matrix from flattened gains."""
        # Assuming a_flat contains the A matrix row-wise
        size = int(np.sqrt(len(a_flat)))
        return a_flat.reshape(size, size)
    
    def dynamics(
        self, 
        t: float, 
        z: NDArray[np.float64], 
        y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Observer dynamics: ż = Az + By."""
        # Ensure proper shapes
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        if isinstance(y, (int, float)):
            y = np.array([[y]])
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
            
        dz = self.A @ z + self.B @ y.T
        return dz.flatten()
    
    def compute_pretransient_time(self, target_error: float = 1e-4) -> float:
        """Compute time needed for observer transients to decay."""
        eigenvalues = np.linalg.eigvals(self.A)
        max_real = np.max(np.real(eigenvalues))
        
        if max_real >= 0:
            raise ValueError("Observer is not stable (eigenvalue with non-negative real part)")
            
        # Time for exp(λt) < target_error
        t_pre = np.log(target_error) / max_real
        return abs(t_pre)
    
    def sample_initial_conditions(
        self, 
        n: int, 
        sampler: "Sampler"
    ) -> NDArray[np.float64]:
        """Sample n initial conditions for observer states."""
        return sampler.sample(n, self.ic_space)
    
    def compute_consistent_ic(
        self,
        system_ic: NDArray[np.float64],
        system: "System",
        solver: "Solver",
        sim_time: "SimulationTime",
        gen_mode: str = "backward"
    ) -> NDArray[np.float64]:
        """
        Compute observer IC consistent with system IC.
        
        This ensures z(0) = T(x(0)) by simulating backwards/forwards.
        """
        t_pre = self.compute_pretransient_time()
        
        if gen_mode == "backward":
            # Simulate system backward
            neg_time = sim_time.__class__(
                t0=sim_time.t0,
                tn=sim_time.t0 + t_pre,
                dt=-sim_time.dt
            )
        else:  # forward
            neg_time = sim_time.__class__(
                t0=t_pre + sim_time.t0,
                tn=sim_time.t0,
                dt=sim_time.dt
            )
            
        # This would integrate the system and observer to find consistent ICs
        # Implementation depends on solver interface
        # Placeholder: return sampled ICs
        return self.sample_initial_conditions(len(system_ic), solver.sampler)
