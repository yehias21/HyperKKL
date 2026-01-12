"""Simulators module - dynamical systems, observers, and signals."""
from .types import SimulationTime, SimulationResult, SystemConfig, ObserverConfig
from .systems import System, Duffing, Lorenz, VanDerPol, Rossler, Chua, SIR
from .observers import KKLObserver
from .signals import Signal, SquareWave, Harmonics, SystemAsSignal, CompositeSignal
from .solvers import Solver, RK4Solver, EulerSolver, Dopri5Solver
from .samplers import Sampler, UniformSampler, LatinHypercubeSampler, GridSampler, SobolSampler

__all__ = [
    # Types
    "SimulationTime",
    "SimulationResult", 
    "SystemConfig",
    "ObserverConfig",
    # Systems
    "System",
    "Duffing",
    "Lorenz", 
    "VanDerPol",
    "Rossler",
    "Chua",
    "SIR",
    # Observers
    "KKLObserver",
    # Signals
    "Signal",
    "SquareWave",
    "Harmonics",
    "SystemAsSignal",
    "CompositeSignal",
    # Solvers
    "Solver",
    "RK4Solver",
    "EulerSolver",
    "Dopri5Solver",
    # Samplers
    "Sampler",
    "UniformSampler",
    "LatinHypercubeSampler",
    "GridSampler",
    "SobolSampler",
]
