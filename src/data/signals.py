"""Input signal generators for non-autonomous system simulation.

Provides zero, constant, sinusoidal, square-wave, step, and traffic-scenario
input types, each with train / test_id / test_ood parameter ranges.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState

# ---------------------------------------------------------------------------
# Parameter ranges used by generators that support train / test splits.
# ---------------------------------------------------------------------------

INPUT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "constant": {
        "train": (-1.0, 1.0),
        "test_id": (-1.0, 1.0),
        "test_ood": (-2.0, 2.0),
    },
    "sinusoid": {
        "train": {"A": (0.1, 1.0), "w": (0.5, 3.0)},
        "test_id": {"A": (0.1, 1.0), "w": (0.5, 3.0)},
        "test_ood": {"A": (1.0, 2.0), "w": (3.0, 6.0)},
    },
    "square": {
        "train": {"A": (0.1, 1.0), "w": (0.5, 2.0)},
        "test_id": {"A": (0.1, 1.0), "w": (0.5, 2.0)},
        "test_ood": {"A": (1.0, 2.0), "w": (2.0, 4.0)},
    },
    "step": {
        "train": {"A": (0.1, 1.0), "t_step": (5.0, 25.0)},
        "test_id": {"A": (0.1, 1.0), "t_step": (5.0, 25.0)},
        "test_ood": {"A": (1.0, 2.0), "t_step": (5.0, 40.0)},
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = Union[RandomState, np.random.Generator, None]


def _resolve_rng(rng: _RNG) -> Any:
    """Return a usable RNG instance (fall back to ``np.random``)."""
    return rng if rng is not None else np.random


def _mode_key(mode: str) -> str:
    """Map *mode* to the key used in :data:`INPUT_CONFIGS`."""
    return "train" if mode == "train" else f"test_{mode}"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class InputGenerator:
    """Abstract base class for all input signal generators.

    Parameters
    ----------
    input_type:
        Short string tag identifying the signal kind (e.g. ``"sinusoid"``).
    mode:
        One of ``"train"``, ``"id"`` or ``"ood"``.  The mode selects the
        parameter range that :meth:`sample_params` draws from.
    """

    def __init__(self, input_type: str, mode: str = "train") -> None:
        self.input_type = input_type
        self.mode = mode
        self.params: Dict[str, float] = {}

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        """Randomly sample signal parameters and store them in ``self.params``.

        Parameters
        ----------
        rng:
            A NumPy random state or ``None`` (uses the global RNG).

        Returns
        -------
        dict:
            The sampled parameters.
        """
        raise NotImplementedError

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the signal at time(s) *t*."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete generators
# ---------------------------------------------------------------------------


class ZeroInput(InputGenerator):
    """Identically-zero input signal."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("zero", mode)

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        return {}

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if np.isscalar(t):
            return 0.0
        return np.zeros_like(t)


class ConstantInput(InputGenerator):
    """Constant (DC) input signal."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("constant", mode)
        key = _mode_key(mode)
        self.range: Tuple[float, float] = INPUT_CONFIGS["constant"].get(
            key, INPUT_CONFIGS["constant"]["train"]
        )

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {"c": rng.uniform(*self.range)}
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        c = self.params.get("c", 0.0)
        if np.isscalar(t):
            return c
        return np.full_like(t, c)


class SinusoidInput(InputGenerator):
    """Sinusoidal input signal: ``A * sin(w * t + phi)``."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("sinusoid", mode)
        key = _mode_key(mode)
        self.config: Dict[str, Tuple[float, float]] = INPUT_CONFIGS["sinusoid"].get(
            key, INPUT_CONFIGS["sinusoid"]["train"]
        )

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {
            "A": rng.uniform(*self.config["A"]),
            "w": rng.uniform(*self.config["w"]),
            "phi": rng.uniform(0, 2 * np.pi),
        }
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.params["A"] * np.sin(self.params["w"] * t + self.params["phi"])


class SquareInput(InputGenerator):
    """Square-wave input signal: ``A * sign(sin(w * t))``."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("square", mode)
        key = _mode_key(mode)
        self.config: Dict[str, Tuple[float, float]] = INPUT_CONFIGS["square"].get(
            key, INPUT_CONFIGS["square"]["train"]
        )

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {
            "A": rng.uniform(*self.config["A"]),
            "w": rng.uniform(*self.config["w"]),
        }
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.params["A"] * np.sign(np.sin(self.params["w"] * t))


class StepInput(InputGenerator):
    """Unit-step input signal that jumps from 0 to *A* at ``t_step``."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("step", mode)
        key = _mode_key(mode)
        self.config: Dict[str, Tuple[float, float]] = INPUT_CONFIGS["step"].get(
            key, INPUT_CONFIGS["step"]["train"]
        )

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {
            "A": rng.uniform(*self.config["A"]),
            "t_step": rng.uniform(*self.config["t_step"]),
        }
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if np.isscalar(t):
            return self.params["A"] if t >= self.params["t_step"] else 0.0
        return np.where(t >= self.params["t_step"], self.params["A"], 0.0)


# ---------------------------------------------------------------------------
# Traffic-scenario generators
# ---------------------------------------------------------------------------


class TrafficRushHourInput(InputGenerator):
    """AM-modulated ramp perturbation mimicking rush-hour merging."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("traffic_rush_hour", mode)

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {"amp": rng.uniform(0.5, 1.5)}
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        amp = self.params.get("amp", 1.0)
        return (
            0.25 * amp
            * np.sin(2 * np.pi * t / 200)
            * (1 + 0.5 * np.sin(2 * np.pi * t / 80))
        )


class TrafficCongestionInput(InputGenerator):
    """Elevated ramp perturbation pushing cells toward capacity."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("traffic_congestion", mode)

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {"amp": rng.uniform(0.5, 1.5)}
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        amp = self.params.get("amp", 1.0)
        return 0.15 * amp + 0.15 * amp * np.sin(2 * np.pi * t / 150)


class TrafficPulseInput(InputGenerator):
    """Bursty on-ramp merging with a square-wave pattern."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("traffic_pulse", mode)

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {
            "amp": rng.uniform(0.5, 1.5),
            "period": rng.uniform(150, 300),
        }
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        amp = self.params.get("amp", 1.0)
        period = self.params.get("period", 200.0)
        if np.isscalar(t):
            phase = (t % period) / period
            return 0.7 * amp if 0.5 < phase < 0.7 else -0.2 * amp
        phase = (t % period) / period
        return np.where((phase > 0.5) & (phase < 0.7), 0.7 * amp, -0.2 * amp)


class TrafficLightInput(InputGenerator):
    """Very low ramp perturbation -- near-equilibrium easy baseline."""

    def __init__(self, mode: str = "train") -> None:
        super().__init__("traffic_light", mode)

    def sample_params(self, rng: _RNG = None) -> Dict[str, float]:
        rng = _resolve_rng(rng)
        self.params = {"amp": rng.uniform(0.5, 1.5)}
        return self.params

    def __call__(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        amp = self.params.get("amp", 1.0)
        return 0.05 * amp * np.sin(2 * np.pi * t / 400)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GENERATORS: Dict[str, type] = {
    "zero": ZeroInput,
    "constant": ConstantInput,
    "sinusoid": SinusoidInput,
    "square": SquareInput,
    "step": StepInput,
    "traffic_rush_hour": TrafficRushHourInput,
    "traffic_congestion": TrafficCongestionInput,
    "traffic_pulse": TrafficPulseInput,
    "traffic_light": TrafficLightInput,
}


def get_input_generator(input_type: str, mode: str = "train") -> InputGenerator:
    """Instantiate an :class:`InputGenerator` by name.

    Parameters
    ----------
    input_type:
        Key into :data:`_GENERATORS` (e.g. ``"sinusoid"``).
    mode:
        ``"train"``, ``"id"``, or ``"ood"``.

    Returns
    -------
    InputGenerator:
        A freshly constructed generator (parameters not yet sampled).

    Raises
    ------
    KeyError:
        If *input_type* is not recognised.
    """
    return _GENERATORS[input_type](mode)
