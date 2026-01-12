"""Exogenous signal generators."""
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from .types import SimulationTime


class Signal(ABC):
    """Base class for exogenous signals."""
    
    @abstractmethod
    def generate(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """Generate signal samples.
        
        Returns:
            Array of shape (num_samples, num_steps, signal_dim)
        """
        pass
    
    def __call__(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        return self.generate(sim_time, num_samples, seed)


class SquareWave(Signal):
    """Alternating square wave signal."""
    
    def __init__(
        self, 
        period: float = 10.0, 
        amplitude: list[float] = [-1.0, 1.0],
        name: str = "square"
    ):
        self.name = name
        self.period = period
        self.amp_low, self.amp_high = amplitude
        
    def generate(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        t = sim_time.time_points
        # Square wave: alternates every half period
        signal = np.where(
            (t % self.period) < (self.period / 2),
            self.amp_high,
            self.amp_low
        )
        # Broadcast to (num_samples, num_steps, 1)
        return np.tile(signal[np.newaxis, :, np.newaxis], (num_samples, 1, 1))


class Harmonics(Signal):
    """Sum of sinusoidal harmonics with random parameters."""
    
    def __init__(
        self,
        num_components: int = 3,
        omega_range: list[float] = [0.0, 0.6],
        amplitude_range: list[float] = [-3.0, 3.0],
        name: str = "harmonics"
    ):
        self.name = name
        self.num_components = num_components
        self.omega_range = omega_range
        self.amplitude_range = amplitude_range
        
    def generate(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        if seed is not None:
            np.random.seed(seed)
            
        t = sim_time.time_points
        signals = np.zeros((num_samples, len(t), 1))
        
        for i in range(num_samples):
            # Random frequencies and amplitudes for each sample
            omegas = np.random.uniform(*self.omega_range, self.num_components)
            amps = np.random.uniform(*self.amplitude_range, self.num_components)
            phases = np.random.uniform(0, 2 * np.pi, self.num_components)
            
            # Sum of sinusoids
            for omega, amp, phase in zip(omegas, amps, phases):
                signals[i, :, 0] += amp * np.sin(omega * t + phase)
                
        return signals


class SystemAsSignal(Signal):
    """Use output of a dynamical system as exogenous signal."""
    
    def __init__(
        self,
        system: "System",
        solver: Optional["Solver"] = None,
        sampler: Optional["Sampler"] = None,
        name: str = "system_signal"
    ):
        self.name = name
        self.system = system
        self._solver = solver
        self._sampler = sampler
        
    def generate(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        from .samplers import UniformSampler
        from .solvers import RK4Solver
        
        # Use provided or default solver/sampler
        solver = self._solver or RK4Solver()
        sampler = self._sampler or UniformSampler(seed=seed)
        
        # Sample ICs and simulate
        ics = self.system.sample_initial_conditions(num_samples, sampler)
        
        t = sim_time.time_points
        signals = np.zeros((num_samples, len(t), self.system.output_dim))
        
        for i, ic in enumerate(ics):
            # Simulate system
            states = solver.solve(
                self.system.dynamics,
                sim_time,
                ic
            )
            # Get output
            signals[i] = self.system.output(states)
            
        return signals


class CompositeSignal(Signal):
    """Combine multiple signals."""
    
    def __init__(self, signals: list[Signal], mode: str = "concat"):
        """
        Args:
            signals: List of signal generators
            mode: "concat" (along signal_dim) or "add" (sum signals)
        """
        self.signals = signals
        self.mode = mode
        self.name = "composite"
        
    def generate(
        self, 
        sim_time: SimulationTime, 
        num_samples: int,
        seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        generated = [s.generate(sim_time, num_samples, seed) for s in self.signals]
        
        if self.mode == "concat":
            return np.concatenate(generated, axis=-1)
        elif self.mode == "add":
            return sum(generated)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
