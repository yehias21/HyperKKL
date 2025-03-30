import numpy as np

from src.simulators.systems import System
from src.simulators.types import SimTime
from tqdm import tqdm
from typing import Callable, Union, Tuple


def get_solver(name: str):
    match name.lower():
        case "rk4":
            return runge_kutta_4th_order
        case _:
            raise ValueError(f"{name} is not a valid solver")


def runge_kutta_4th_order(diff_eq: System.diff_eq, sim_time: SimTime, x0: np.ndarray,
                          exogenous_input: Union[np.ndarray, Callable] = None) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    exogenous_input: can be a callable or a numpy array, if numpy array it should be of shape  (t, sig_dim)
    """
    states = x0
    time = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
    h = sim_time.eps

    exogenous_input = [None]*time.shape[0] if exogenous_input is None else exogenous_input
    if isinstance(exogenous_input, np.ndarray):
        assert exogenous_input.shape[0] == time.shape[0], "Input signal should have the same length as time"
    exo_inp = lambda idx, t_current: exogenous_input(t_current) if callable(exogenous_input) else exogenous_input[idx]

    for ind, t in tqdm(enumerate(time)):
        k1 = diff_eq(x=x0, t=t, inp=exo_inp(ind, t))
        k2 = diff_eq(x=x0 + h / 2 * k1, t=t + h / 2, inp=exo_inp(ind, t))
        k3 = diff_eq(x=x0 + h / 2 * k2, t=t + h / 2, inp=exo_inp(ind, t))
        k4 = diff_eq(x=x0 + h * k3, t=t + h, inp=exo_inp(ind, t))
        # state n+1 at time t+h
        x0 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        states = np.vstack((states, x0))
        x0 = states[ind + 1]

    return states, time
