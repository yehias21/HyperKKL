import pickle as pk
from abc import abstractmethod, ABC
from concurrent.futures import ProcessPoolExecutor
from typing import Union, Dict, Any

import numpy as np
from hydra.utils import instantiate

from src.simulators.types import SimTime


# adjust the signal generation for the latest edits

class InputSignal(ABC):
    def __init__(self, seed: int = 0, params=None, num_samples: int = 1):
        if params is None:
            params = {}
        self.num_samples = num_samples
        self.seed = seed
        self.params = params

    @abstractmethod
    def generate_signal(self, t: Union[float, np.ndarray], inp_param: list) -> Union[float, np.ndarray]:
        pass

    @abstractmethod
    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        pass


class SignalGenerator:
    def __init__(self, saved_path=None, signals: Dict[str, Dict[str, Any]] = None, sys_states: int = 0,
                 num_samples: int = 1, seed: int = 17):
        self.saved_path = saved_path
        self.num_samples = num_samples
        self.sys_states = sys_states
        self.seed = seed
        self.signal_objects = self._initialize_signals(signals)

    def _initialize_signals(self, signals):
        signal_objects = {}
        for state, signal_config in signals.items():
            signal_objects[state] = instantiate(signal_config)
        for i in range(1, self.sys_states + 1):
            state_name = f"state_{i}"
            if state_name not in signal_objects:
                signal_objects[state_name] = NoSignal(seed=self.seed, num_samples=self.num_samples)
        # sort dictionary by key number
        signal_objects = dict(sorted(signal_objects.items(), key=lambda x: int(x[0].split('_')[-1])))
        return signal_objects

    def generate_signals(self, sim_time: SimTime = None) -> np.ndarray:
        if self.saved_path:
            try:
                with open(self.saved_path, 'rb') as f:
                    trajectories = pk.load(f)
                    t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
                    assert trajectories.shape[1] == t.shape[0], "The saved trajectories do not match the time vector"
                    return trajectories
            except FileNotFoundError:
                print("File not found, generating new signals")

        """ Generate the input signal for each state """
        trajectories = []
        for state, signal in self.signal_objects.items():
            trajectories.append(signal.generate_traj(sim_time))
        return np.concatenate(trajectories, axis=-1)  # returned dimension (input_num_samples, t, sys_dim)


class SquareSignal(InputSignal):
    def generate_signal(self, t: Union[float, np.ndarray], ic: list) -> np.ndarray:
        period, amp = ic[0], ic[1]
        signal = np.piecewise(t, [t % period < period / 2, t % period >= period / 2], amp)
        return np.array(signal).reshape(1, t.shape[0], 1)

    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        gen = np.random.RandomState(self.seed)
        period_range, amp = self.params.signal_data.get('period', [2, 3]), self.params.signal_data.get('amp', [1, -1])
        trajs = []
        for period in gen.uniform(period_range[0], period_range[1], self.num_samples):
            trajs.append(self.generate_signal(t, [period, amp]))
        return np.concatenate(trajs, axis=0)


class AlternatingSquareSignal(InputSignal):
    def generate_signal(self, t: Union[float, np.ndarray], ic: list) -> np.ndarray:
        period, amp = ic[0], ic[1]
        signal = np.repeat(np.random.choice(amp, int(t.shape[0] // period + 1)), int(period))[:t.shape[0]]
        return np.array(signal).reshape(1, t.shape[0], 1)

    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        period = self.params.signal_data.get('period', 5)
        amp = self.params.signal_data.get('amp', [1, 0])
        trajs = []
        for _ in range(self.num_samples):
            trajs.append(self.generate_signal(t, [period, amp]))
        return np.concatenate(trajs, axis=0)


class SinSignal(InputSignal):
    def generate_signal(self, t: Union[float, np.ndarray], ic: list) -> np.ndarray:
        input_signal = 0
        for amp, omega, phase in zip(ic[0], ic[1], ic[2]):
            input_signal += amp * np.sin(2 * np.pi * omega * t + phase * np.pi)
        return np.array(input_signal).reshape(1, t.shape[0], 1)

    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        trajs = []
        gen = np.random.RandomState(self.seed)
        if self.params.signal_type == 'harmonics_range':
            for i in range(self.num_samples):
                amps = gen.uniform(*self.params.signal_data.get('amps', [-2, 2]),
                                   size=self.params.signal_data.get('sin_component', 1))
                omegas = gen.uniform(*self.params.signal_data.get('omegas', [0.1, 1]),
                                     size=self.params.signal_data.get('sin_component', 1))
                phases = gen.uniform(*self.params.signal_data.get('phases', [0, 1]),
                                     size=self.params.signal_data.get('sin_component', 1))
                trajs.append(self.generate_signal(t, [amps, omegas, phases]))
            return np.concatenate(trajs, axis=0)

        elif self.params.signal_type == 'harmonics':
            amps = np.array(self.params.signal_data.get('amps', [1]))
            omegas = np.array(self.params.signal_data.get('omegas', [1]))
            phases = np.array(self.params.signal_data.get('phases', gen.uniform(amps)))
            # some assertion to do
            assert amps.shape == omegas.shape == phases.shape == (self.num_samples,
                                                                  self.params.signal_data.get('sin_component',
                                                                                              -1)), "The number of harmonics should be the same"
            for amp, omega, phase in zip(amps, omegas, phases):
                trajs.append(self.generate_signal(t, [amp, omega, phase]))
            return np.concatenate(trajs, axis=0)
        else:
            raise ValueError('Signal type not recognized')


class SystemAsSignal(InputSignal):
    def generate_signal(self, t: Union[float, np.ndarray], ic: list) -> np.ndarray:
        raise NotImplementedError

    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        trajectories = []
        solver, system = self.params.signal_data.solver, self.params.signal_data.system
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = map(lambda ic: executor.submit(solver, system.diff_eq, sim_time, ic), system.ic)
            temp_trajs = [result.result()[0] for result in results]
            trajectories.append(temp_trajs)
        # make it with for loop
        trajectories = np.squeeze(system.get_output(np.delete(np.array(trajectories), 0, -2)), 0)
        return trajectories


class NoSignal(InputSignal):
    def generate_signal(self, t: Union[float, np.ndarray], ic: float = 0) -> Union[float, np.ndarray]:
        return np.zeros_like(t).reshape(1, t.shape[0], 1)

    def generate_traj(self, sim_time: SimTime) -> np.ndarray:
        t = np.arange(sim_time.t0, sim_time.tn, sim_time.eps)
        return np.zeros((self.num_samples, t.shape[0], 1))
