from abc import abstractmethod, ABC
from typing import Callable
import numpy as np


class StateObserver(ABC):
    def __init__(self, sampler: Callable, num_samples: int) -> None:
        self._ic = sampler(num_samples)
        self._sampler = sampler
    @abstractmethod
    def diff_eq(self, x: list[float], t: float, inp: list[float]) -> np.ndarray:
        pass

    def generate_ic(self, num_samples: int) -> None:
        """
           Generate the initial conditions of the system
           :param num_samples: number of initial conditions
           :return: numpy array of initial conditions of the system
       """
        self._ic = self._sampler(num_samples)

    @property
    def ic(self) -> np.ndarray:
        return self._ic

    def sampler(self, sampler: Callable) -> None:
        self._sampler = sampler


class KKLObserver(StateObserver):
    def __init__(self, observer_param: dict, sampler, num_samples=2) -> None:
        super().__init__(sampler, num_samples)
        self.A = np.array(observer_param['a']).reshape(observer_param['z_dim'], observer_param['z_dim'])
        self.B = np.array(observer_param['b'])
        self.z_dim = observer_param['z_dim']
        self.e = observer_param['e']
        self.z_max = observer_param['z_max']

    def diff_eq(self, x: list[float], t: float, inp: list[float], **kwargs) -> np.ndarray:
        x_hat_dot = np.matmul(self.A, x) + self.B * inp
        return x_hat_dot

    def calc_pret0(self):
        w, v = np.linalg.eig(self.A)
        min_ev = np.min(np.abs(np.real(w)))
        kappa = np.linalg.cond(v)
        s = np.sqrt(
            self.z_max * self.A.shape[0])  # check this: https://en.wikipedia.org/wiki/Norm_(mathematics)#Properties
        t = 1 / min_ev * np.log(self.e / (kappa * s))
        return t
