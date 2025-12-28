# Library imports
import numpy as np


# Abstract noise class
class Noise:
    def __init__(self):
        pass

    def __call__(self, variable: np.ndarray, t=0) -> np.ndarray:
        """
        :description: The noise generator is crafted to be able to generate state and time dependent noise, aka non-stationary noise
        :param variable:
        :param t:
        :return:
        """
        raise NotImplementedError("Subclasses must implement __call__ method")


# Gaussian noise generator
class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, seed: int = 0):
        super().__init__()
        self.mean = mean
        self.std = std
        self.random = np.random.RandomState(seed)

    def __call__(self, state: np.ndarray, t=0) -> np.ndarray:
        noise = self.mean + self.std * self.random.randn(*np.array(state).shape)
        return noise


# White noise generator
class WhiteNoise(Noise):
    def __init__(self, amplitude: float, seed: int = 0):
        super().__init__()
        self.amplitude = amplitude
        self.random = np.random.RandomState(seed)

    def __call__(self, state: np.ndarray, t=0) -> np.ndarray:
        noise = self.amplitude * self.random.randn(*np.array(state).shape)
        return noise


# Sampling functions
def get_noise(noise_type: str, **kwargs) -> Noise:
    if noise_type.lower() == "gaussian":
        return GaussianNoise(**kwargs)
    elif noise_type.lower() == "white":
        return WhiteNoise(**kwargs)
    else:
        raise NotImplementedError(f"{noise_type} is not a valid noise type")
