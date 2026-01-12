"""Core types and dataclasses for the simulation framework."""
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationTime:
    """Time configuration for simulation."""
    t0: float
    tn: float
    dt: float
    
    @property
    def time_points(self) -> NDArray[np.float64]:
        return np.arange(self.t0, self.tn, self.dt)
    
    @property
    def num_steps(self) -> int:
        return len(self.time_points)
    
    def reversed(self) -> "SimulationTime":
        """Return time config for backward simulation."""
        return SimulationTime(self.t0, self.tn, -self.dt)


@dataclass
class SystemConfig:
    """Configuration for a dynamical system."""
    state_dim: int
    output_dim: int
    C: NDArray[np.float64]  # Output matrix
    observable_index: list[int]
    coefficients: dict = field(default_factory=dict)


@dataclass
class ObserverConfig:
    """Configuration for KKL observer."""
    z_dim: int
    gains_a: NDArray[np.float64]
    gains_b: NDArray[np.float64]
    epsilon: float = 1e-6
    z_max: float = 10.0


@dataclass  
class SimulationResult:
    """Container for simulation results."""
    states: NDArray[np.float64]      # (num_traj, num_steps, state_dim)
    outputs: NDArray[np.float64]     # (num_traj, num_steps, output_dim)
    time: NDArray[np.float64]        # (num_steps,)
    inputs: Optional[NDArray] = None # (num_signals, num_steps, input_dim)
    
    def split_time(self, val_time: float, dt: float) -> tuple["SimulationResult", "SimulationResult"]:
        """Split results into train/validation by time horizon."""
        val_steps = int(val_time / dt)
        train = SimulationResult(
            states=self.states[:, :-val_steps],
            outputs=self.outputs[:, :-val_steps],
            time=self.time[:-val_steps],
            inputs=self.inputs[:, :-val_steps] if self.inputs is not None else None
        )
        val = SimulationResult(
            states=self.states[:, -val_steps:],
            outputs=self.outputs[:, -val_steps:],
            time=self.time[-val_steps:],
            inputs=self.inputs[:, -val_steps:] if self.inputs is not None else None
        )
        return train, val
