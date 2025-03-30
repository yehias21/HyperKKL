from abc import abstractmethod, ABC
from typing import Optional, Callable, Union

import numpy as np

from src.simulators.types import SysDim, SysParam


def _default_noise(state: np.ndarray, t: float = 0) -> np.ndarray:
    return np.zeros_like(state)


class System(ABC):
    """
    Abstract class for the system, this class should be inherited by the system class.
    """

    @abstractmethod
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, p_noise_flag: bool = True) -> None:
        """
        In our implementation, we chose to follow dependency injection design pattern, where the sampler, and
        the noise classes are initialized outside the system class, their callables (same paradigm of forward functions
        in pytorch) are passed to the system class.
        Therefore, a change in the sampler or the noise class will not affect the system class.
        :param sampler: is Callable, which samples the initial conditions of the system
        :param system_param: is a DataClass, that has the system parameters of the differential equation,\
        the C matrix (gain of the sensors), the observable index (index of the states that are observable)
        :param num_samples: is the number of the initial conditions we want the system to have (number of trajectories)
        :param p_noise, m_noise: are 2 callables, first that adds noise to the system dynamics (process noise),
        and the second that adds noise to the output of the system (measurement/sensor noise)
        The noise functions is designed to be non-stationary (i.e. time-varying, state dependent) for generalization
        :param p_noise_flag: boolean indicating whether to add noise
        """
        self._sampler = sampler
        self.p_noise = _default_noise if p_noise is None else p_noise
        self.m_noise = _default_noise if m_noise is None else m_noise
        self.p_noise_flag = p_noise_flag
        self._ic = sampler(num_samples)
        self.system_param = system_param


    @abstractmethod
    def diff_eq(self, x: list[float], t: float, inp: Optional[Union[float, Callable]]) -> np.ndarray:
        """
        Set of differential equation that describes the system dynamics
        :param t: time
        :param x: current states of the system
        :param inp: exogenous input of the system in case it's non-autonomous
        :return: numpy array of the derivatives of the states at time t (+ noise)
        """
        pass

    def generate_ic(self, num_samples: int) -> None:
        """
           Generate the initial conditions of the system
           :param num_samples: number of initial conditions
           :return: numpy array of initial conditions of the system
       """
        self._ic = self._sampler(num_samples)

    def get_output(self, states: np.ndarray, noise_flag=True, multi_inp=False) -> np.ndarray:
        """ Returns the output of the system """
        noise = self.m_noise(state=states) if self.p_noise_flag else np.zeros_like(states)
        return (np.array(self.system_param.C) * states + noise) if multi_inp else (
            (np.array(self.system_param.C) * states + noise)[..., self.system_param.ObservableIndex])

    @property
    def ic(self) -> np.ndarray:
        return self._ic

    @ic.setter
    def ic(self, ic):
        self._ic = ic

    def sampler(self, sampler: Callable) -> None:
        self._sampler = sampler

    @property
    def sys_dim(self) -> SysDim:
        """
        Calculated as mentioned in https://arxiv.org/pdf/2210.01476.pdf
        nz = ny(2nx + 1) # in case of autonomous system
        """
        x_dim, y_dim = len(self.system_param.C), len(self.system_param.ObservableIndex)
        z_dim = y_dim * (2 * x_dim + 1)
        return SysDim(x_dim=x_dim, z_dim=z_dim, y_dim=y_dim)


class Duffing(System):

    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)

    def diff_eq(self, x: list[float], t: float = 0, inp: list[float] = None) -> np.ndarray:
        if inp is None:
            inp = np.zeros(1)
        x1_dot = x[1] ** 3
        x2_dot = - x[0] + inp[0]
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))

        return np.array([x1_dot, x2_dot]) + noise


class Lorenz(System):

    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)
        # Lorenz system parameters
        self.sigma = eval(str(self.system_param.system_coeff.sigma))
        self.rho = eval(str(self.system_param.system_coeff.rho))
        self.beta = eval(str(self.system_param.system_coeff.beta))

    def diff_eq(self, x: list[float], t: float = 0, inp: list[float] = None) -> np.ndarray:
        if inp is None:
            inp = np.zeros(3)
        x1_dot = self.sigma * (x[1] - x[0]) + inp[0]
        x2_dot = x[0] * (self.rho - x[2]) - x[1] + inp[1]
        x3_dot = x[0] * x[1] - self.beta * x[2] + inp[2]
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))

        return np.array([x1_dot, x2_dot, x3_dot]) + noise


# Van der Pol Oscillator
class VdP(System):
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)
        self.mu = eval(str(system_param.system_coeff.mu))

    def diff_eq(self, x: list[float], t: float = 0, inp: Optional[Union[float, Callable]] = None) -> np.ndarray:
        if inp is None:
            inp = np.zeros(1)
        x1_dot = x[1]
        x2_dot = self.mu * (1 - x[0] ** 2) * x[1] - x[0] + inp[0]
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))
        return np.array([x1_dot, x2_dot]) + noise


# Rössler's System
class Rossler(System):
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)
        self.a = eval(str(system_param.system_coeff.a))
        self.b = eval(str(system_param.system_coeff.b))
        self.c = eval(str(system_param.system_coeff.c))

    def diff_eq(self, x: list[float], t: float = 0, inp: Optional[Union[float, Callable]] = None) -> np.ndarray:
        if inp is None:
            inp = np.zeros(3)
        x1_dot = -(x[1] + x[2]) + inp[0]
        x2_dot = x[0] + self.a * x[1] + inp[1]
        x3_dot = self.b + x[2] * (x[0] - self.c) + inp[2]
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))
        return np.array([x1_dot, x2_dot, x3_dot]) + noise


# SIR
class SIR(System):
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)
        self.beta = eval(str(system_param.system_coeff.beta))
        self.gamma = eval(str(system_param.system_coeff.gamma))
        self.N = eval(str(system_param.system_coeff.N))

    def diff_eq(self, x: list[float], t: float = 0, inp: Optional[Union[float, Callable]] = None) -> np.ndarray:
        S, I, R = x
        S_dot = -self.beta * I * S / self.N
        I_dot = self.beta * I * S / self.N - self.gamma * I
        R_dot = self.gamma * I
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))
        return np.array([S_dot, I_dot, R_dot]) + noise


class Chua(System):
    def __init__(self, sampler: Callable, system_param: SysParam, num_samples: int,
                 p_noise: Callable = None, m_noise: Callable = None, noise_flag: bool = True) -> None:
        super().__init__(sampler, system_param, num_samples, p_noise, m_noise, noise_flag)
        self.alpha = eval(str(self.system_param.system_coeff['alpha']))
        self.beta = eval(str(self.system_param.system_coeff['beta']))
        self.R = eval(str(self.system_param.system_coeff['R']))
        self.C2 = eval(str(self.system_param.system_coeff['C2']))
        self.m0 = eval(str(self.system_param.system_coeff['m0']))
        self.m1 = eval(str(self.system_param.system_coeff['m1']))

    def nonlinearity(self, x1: float) -> float:
        return self.m1 * x1 + 0.5 * (self.m0 - self.m1) * (abs(x1 + 1) - abs(x1 - 1))

    def diff_eq(self, x: list[float], t: float = 0, inp: list[float] = None) -> np.ndarray:
        if inp is None:
            inp = np.zeros(3)
        x1_dot = self.alpha * (x[1] - x[0] - self.nonlinearity(x[0])) + inp[0]
        x2_dot = (x[0] - x[1] + self.R * x[2]) / (self.R * self.C2) + inp[1]
        x3_dot = -self.beta * x[1] + inp[2]
        noise = self.p_noise(state=np.array(x), t=t) if self.p_noise_flag else np.zeros(len(x))

        return np.array([x1_dot, x2_dot, x3_dot]) + noise
