"""ODE Solvers for simulation."""
from abc import ABC, abstractmethod
from typing import Callable, Optional
import numpy as np
from numpy.typing import NDArray

from .types import SimulationTime


class Solver(ABC):
    """Base class for ODE solvers."""
    
    @abstractmethod
    def solve(
        self,
        dynamics: Callable,
        sim_time: SimulationTime,
        initial_condition: NDArray[np.float64],
        exogenous_input: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Solve ODE dx/dt = f(t, x, u).
        
        Args:
            dynamics: Function (t, x, u) -> dx/dt
            sim_time: Time configuration
            initial_condition: Initial state x(t0)
            exogenous_input: Optional input signal u(t), shape (num_steps, input_dim)
            
        Returns:
            State trajectory, shape (num_steps, state_dim)
        """
        pass
    
    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)


class RK4Solver(Solver):
    """4th-order Runge-Kutta solver."""
    
    def __init__(self, name: str = "rk4"):
        self.name = name
    
    def solve(
        self,
        dynamics: Callable,
        sim_time: SimulationTime,
        initial_condition: NDArray[np.float64],
        exogenous_input: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        t = sim_time.time_points
        dt = sim_time.dt
        n_steps = len(t)
        state_dim = len(initial_condition)
        
        # Allocate trajectory
        trajectory = np.zeros((n_steps, state_dim))
        trajectory[0] = initial_condition
        
        x = initial_condition.copy()
        
        for i in range(n_steps - 1):
            ti = t[i]
            u = exogenous_input[i] if exogenous_input is not None else None
            u_mid = exogenous_input[i] if exogenous_input is not None else None  # Approximate
            u_next = exogenous_input[i + 1] if exogenous_input is not None else None
            
            # RK4 steps
            k1 = dynamics(ti, x, u)
            k2 = dynamics(ti + dt/2, x + dt*k1/2, u_mid)
            k3 = dynamics(ti + dt/2, x + dt*k2/2, u_mid)
            k4 = dynamics(ti + dt, x + dt*k3, u_next)
            
            x = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory[i + 1] = x
            
        return trajectory


class EulerSolver(Solver):
    """Simple Euler method (for debugging/comparison)."""
    
    def __init__(self, name: str = "euler"):
        self.name = name
    
    def solve(
        self,
        dynamics: Callable,
        sim_time: SimulationTime,
        initial_condition: NDArray[np.float64],
        exogenous_input: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        t = sim_time.time_points
        dt = sim_time.dt
        n_steps = len(t)
        state_dim = len(initial_condition)
        
        trajectory = np.zeros((n_steps, state_dim))
        trajectory[0] = initial_condition
        
        x = initial_condition.copy()
        
        for i in range(n_steps - 1):
            u = exogenous_input[i] if exogenous_input is not None else None
            dx = dynamics(t[i], x, u)
            x = x + dt * dx
            trajectory[i + 1] = x
            
        return trajectory


class Dopri5Solver(Solver):
    """Dormand-Prince 5(4) adaptive solver using scipy."""
    
    def __init__(
        self, 
        rtol: float = 1e-6, 
        atol: float = 1e-8,
        name: str = "dopri5"
    ):
        self.name = name
        self.rtol = rtol
        self.atol = atol
    
    def solve(
        self,
        dynamics: Callable,
        sim_time: SimulationTime,
        initial_condition: NDArray[np.float64],
        exogenous_input: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        from scipy.integrate import solve_ivp
        
        t_eval = sim_time.time_points
        
        # Wrap dynamics to handle exogenous input via interpolation
        if exogenous_input is not None:
            from scipy.interpolate import interp1d
            u_interp = interp1d(
                t_eval, exogenous_input, axis=0, 
                bounds_error=False, fill_value="extrapolate"
            )
            def wrapped_dynamics(t, x):
                return dynamics(t, x, u_interp(t))
        else:
            def wrapped_dynamics(t, x):
                return dynamics(t, x, None)
        
        sol = solve_ivp(
            wrapped_dynamics,
            (sim_time.t0, sim_time.tn),
            initial_condition,
            method='DOP853',
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol
        )
        
        return sol.y.T  # Transpose to (n_steps, state_dim)
