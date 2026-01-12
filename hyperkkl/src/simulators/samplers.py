"""Sampling strategies for initial conditions and parameters."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray


class Sampler(ABC):
    """Base class for samplers."""
    
    def __init__(self, seed: Optional[int] = None, name: str = "sampler"):
        self.name = name
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        
    @abstractmethod
    def sample(
        self, 
        n: int, 
        bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Sample n points within bounds.
        
        Args:
            n: Number of samples
            bounds: Array of shape (dim, 2) with [low, high] for each dimension
            
        Returns:
            Samples of shape (n, dim)
        """
        pass
    
    def __call__(self, n: int, bounds: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.sample(n, bounds)


class UniformSampler(Sampler):
    """Uniform random sampling."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed, name="uniform")
    
    def sample(
        self, 
        n: int, 
        bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        bounds = np.atleast_2d(bounds)
        dim = bounds.shape[0]
        low = bounds[:, 0]
        high = bounds[:, 1]
        
        return self._rng.uniform(low, high, size=(n, dim))


class LatinHypercubeSampler(Sampler):
    """Latin Hypercube Sampling for better space coverage."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed, name="lhs")
    
    def sample(
        self, 
        n: int, 
        bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        bounds = np.atleast_2d(bounds)
        dim = bounds.shape[0]
        low = bounds[:, 0]
        high = bounds[:, 1]
        
        # Create Latin Hypercube
        samples = np.zeros((n, dim))
        
        for d in range(dim):
            # Divide [0, 1] into n equal intervals
            intervals = np.linspace(0, 1, n + 1)
            
            # Sample uniformly within each interval
            lower = intervals[:-1]
            upper = intervals[1:]
            points = self._rng.uniform(lower, upper)
            
            # Shuffle to get Latin Hypercube property
            self._rng.shuffle(points)
            
            # Scale to bounds
            samples[:, d] = low[d] + points * (high[d] - low[d])
            
        return samples


class GridSampler(Sampler):
    """Regular grid sampling."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed, name="grid")
    
    def sample(
        self, 
        n: int, 
        bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        bounds = np.atleast_2d(bounds)
        dim = bounds.shape[0]
        
        # Points per dimension (approximate)
        points_per_dim = int(np.ceil(n ** (1/dim)))
        
        # Create grid
        grids = [
            np.linspace(bounds[d, 0], bounds[d, 1], points_per_dim)
            for d in range(dim)
        ]
        
        mesh = np.meshgrid(*grids, indexing='ij')
        samples = np.stack([m.flatten() for m in mesh], axis=-1)
        
        # Return exactly n samples (subsample if needed)
        if len(samples) > n:
            indices = self._rng.choice(len(samples), n, replace=False)
            samples = samples[indices]
            
        return samples


class SobolSampler(Sampler):
    """Sobol quasi-random sequence for low-discrepancy sampling."""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed, name="sobol")
    
    def sample(
        self, 
        n: int, 
        bounds: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("scipy >= 1.7 required for Sobol sampling")
            
        bounds = np.atleast_2d(bounds)
        dim = bounds.shape[0]
        
        sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
        samples_unit = sampler.random(n)
        
        # Scale to bounds
        low = bounds[:, 0]
        high = bounds[:, 1]
        samples = qmc.scale(samples_unit, low, high)
        
        return samples
