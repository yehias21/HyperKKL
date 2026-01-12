"""Dynamical system implementations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class System(ABC):
    """Base class for dynamical systems."""
    
    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        observation: dict,
        coefficients: dict,
        ic_space: list[list[float]],
        noise: Optional[dict] = None,
        name: str = "system"
    ):
        self.name = name
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.C = np.array(observation["C"])
        self.observable_index = observation["observable_index"]
        self.coefficients = coefficients
        self.ic_space = np.array(ic_space)
        self.noise_config = noise
        
    @abstractmethod
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """Compute dx/dt = f(t, x, u)."""
        pass
    
    def output(self, x: NDArray[np.float64], add_noise: bool = False) -> NDArray[np.float64]:
        """Compute y = Cx."""
        y = x @ self.C.T if x.ndim > 1 else self.C @ x
        if add_noise and self.noise_config:
            noise = self._sample_noise("measurement", y.shape)
            y = y + noise
        return y
    
    def sample_initial_conditions(
        self, 
        n: int, 
        sampler: "Sampler"
    ) -> NDArray[np.float64]:
        """Sample n initial conditions from ic_space."""
        return sampler.sample(n, self.ic_space)
    
    def _sample_noise(self, noise_type: str, shape: tuple) -> NDArray[np.float64]:
        """Sample noise for process or measurement."""
        if self.noise_config is None:
            return np.zeros(shape)
        cfg = self.noise_config.get(noise_type, {})
        return np.random.normal(cfg.get("mean", 0), cfg.get("std", 0), shape)


class Duffing(System):
    """Duffing oscillator: ẍ + δẋ + αx + βx³ = γcos(ωt) + u(t)."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        c = self.coefficients
        alpha = c.get("alpha", 1.0)
        beta = c.get("beta", -1.0)
        delta = c.get("delta", 0.3)
        gamma = c.get("gamma", 0.5)
        omega = c.get("omega", 1.2)
        
        x1, x2 = x[..., 0], x[..., 1]
        
        dx1 = x2
        dx2 = -delta * x2 - alpha * x1 - beta * x1**3 + gamma * np.cos(omega * t)
        
        if u is not None:
            dx2 = dx2 + u[..., 0] if u.ndim > 0 else dx2 + u
            
        return np.stack([dx1, dx2], axis=-1)


class Lorenz(System):
    """Lorenz attractor."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        c = self.coefficients
        sigma = c.get("sigma", 10.0)
        rho = c.get("rho", 28.0)
        beta = c.get("beta", 8/3)
        
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        
        dx1 = sigma * (x2 - x1)
        dx2 = x1 * (rho - x3) - x2
        dx3 = x1 * x2 - beta * x3
        
        return np.stack([dx1, dx2, dx3], axis=-1)


class VanDerPol(System):
    """Van der Pol oscillator."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        mu = self.coefficients.get("mu", 1.0)
        
        x1, x2 = x[..., 0], x[..., 1]
        
        dx1 = x2
        dx2 = mu * (1 - x1**2) * x2 - x1
        
        if u is not None:
            dx2 = dx2 + u[..., 0] if u.ndim > 0 else dx2 + u
            
        return np.stack([dx1, dx2], axis=-1)


class Rossler(System):
    """Rössler attractor."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        c = self.coefficients
        a = c.get("a", 0.2)
        b = c.get("b", 0.2)
        c_param = c.get("c", 5.7)
        
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        
        dx1 = -x2 - x3
        dx2 = x1 + a * x2
        dx3 = b + x3 * (x1 - c_param)
        
        return np.stack([dx1, dx2, dx3], axis=-1)


class Chua(System):
    """Chua's circuit."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        c = self.coefficients
        alpha = c.get("alpha", 9.0)
        beta = c.get("beta", 14.0)
        m0 = c.get("m0", -1.143)
        m1 = c.get("m1", -0.714)
        
        x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2]
        
        # Chua's diode nonlinearity
        h = m1 * x1 + 0.5 * (m0 - m1) * (np.abs(x1 + 1) - np.abs(x1 - 1))
        
        dx1 = alpha * (x2 - x1 - h)
        dx2 = x1 - x2 + x3
        dx3 = -beta * x2
        
        return np.stack([dx1, dx2, dx3], axis=-1)


class SIR(System):
    """SIR epidemiological model."""
    
    def dynamics(
        self, 
        t: float, 
        x: NDArray[np.float64], 
        u: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        c = self.coefficients
        beta = c.get("beta", 0.5)
        gamma = c.get("gamma", 0.1)
        N = c.get("N", 1000)
        
        S, I, R = x[..., 0], x[..., 1], x[..., 2]
        
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        
        return np.stack([dS, dI, dR], axis=-1)
