"""
Input signal generators for non-autonomous simulation.

Each generator has train/id/ood parameter ranges and a sample_params method
for random parameter sampling.

Standalone test:
    python -m src.signals
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Parameter ranges: {input_type: {mode: range_dict}}
# ---------------------------------------------------------------------------

INPUT_RANGES = {
    "constant": {
        "train": (-1.0, 1.0),
        "id": (-1.0, 1.0),
        "ood": (-2.0, 2.0),
    },
    "sinusoid": {
        "train": {"A": (0.1, 1.0), "w": (0.5, 3.0)},
        "id": {"A": (0.1, 1.0), "w": (0.5, 3.0)},
        "ood": {"A": (1.0, 2.0), "w": (3.0, 6.0)},
    },
    "square": {
        "train": {"A": (0.1, 1.0), "w": (0.5, 2.0)},
        "id": {"A": (0.1, 1.0), "w": (0.5, 2.0)},
        "ood": {"A": (1.0, 2.0), "w": (2.0, 4.0)},
    },
    "step": {
        "train": {"A": (0.1, 1.0), "t_step": (5.0, 25.0)},
        "id": {"A": (0.1, 1.0), "t_step": (5.0, 25.0)},
        "ood": {"A": (1.0, 2.0), "t_step": (5.0, 40.0)},
    },
}


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class SignalGenerator:
    """Base input signal generator."""

    def __init__(self, mode: str = "train"):
        self.mode = mode
        self.params = {}

    def sample_params(self, rng: np.random.RandomState = None):
        """Sample random parameters. Call before __call__."""
        raise NotImplementedError

    def __call__(self, t):
        """Evaluate signal at time t (scalar or array)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete generators
# ---------------------------------------------------------------------------

class ZeroSignal(SignalGenerator):
    def sample_params(self, rng=None):
        return {}

    def __call__(self, t):
        return np.zeros_like(t) if not np.isscalar(t) else 0.0


class ConstantSignal(SignalGenerator):
    def __init__(self, mode="train"):
        super().__init__(mode)
        self._range = INPUT_RANGES["constant"].get(mode, INPUT_RANGES["constant"]["train"])

    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {"c": rng.uniform(*self._range)}

    def __call__(self, t):
        c = self.params.get("c", 0.0)
        return np.full_like(t, c) if not np.isscalar(t) else c


class SinusoidSignal(SignalGenerator):
    def __init__(self, mode="train"):
        super().__init__(mode)
        self._cfg = INPUT_RANGES["sinusoid"].get(mode, INPUT_RANGES["sinusoid"]["train"])

    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {
            "A": rng.uniform(*self._cfg["A"]),
            "w": rng.uniform(*self._cfg["w"]),
            "phi": rng.uniform(0, 2 * np.pi),
        }

    def __call__(self, t):
        return self.params["A"] * np.sin(self.params["w"] * t + self.params["phi"])


class SquareSignal(SignalGenerator):
    def __init__(self, mode="train"):
        super().__init__(mode)
        self._cfg = INPUT_RANGES["square"].get(mode, INPUT_RANGES["square"]["train"])

    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {
            "A": rng.uniform(*self._cfg["A"]),
            "w": rng.uniform(*self._cfg["w"]),
        }

    def __call__(self, t):
        return self.params["A"] * np.sign(np.sin(self.params["w"] * t))


class StepSignal(SignalGenerator):
    def __init__(self, mode="train"):
        super().__init__(mode)
        self._cfg = INPUT_RANGES["step"].get(mode, INPUT_RANGES["step"]["train"])

    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {
            "A": rng.uniform(*self._cfg["A"]),
            "t_step": rng.uniform(*self._cfg["t_step"]),
        }

    def __call__(self, t):
        A, ts = self.params["A"], self.params["t_step"]
        if np.isscalar(t):
            return A if t >= ts else 0.0
        return np.where(t >= ts, A, 0.0)


# ---------------------------------------------------------------------------
# Traffic-specific signals
# ---------------------------------------------------------------------------

class TrafficRushHour(SignalGenerator):
    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {"amp": rng.uniform(0.5, 1.5)}

    def __call__(self, t):
        amp = self.params.get("amp", 1.0)
        return 0.25 * amp * np.sin(2 * np.pi * t / 200) * (
            1 + 0.5 * np.sin(2 * np.pi * t / 80)
        )


class TrafficCongestion(SignalGenerator):
    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {"amp": rng.uniform(0.5, 1.5)}

    def __call__(self, t):
        amp = self.params.get("amp", 1.0)
        return 0.15 * amp + 0.15 * amp * np.sin(2 * np.pi * t / 150)


class TrafficPulse(SignalGenerator):
    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {"amp": rng.uniform(0.5, 1.5), "period": rng.uniform(150, 300)}

    def __call__(self, t):
        amp = self.params.get("amp", 1.0)
        period = self.params.get("period", 200.0)
        phase = (t % period) / period
        if np.isscalar(t):
            return 0.7 * amp if 0.5 < phase < 0.7 else -0.2 * amp
        return np.where((phase > 0.5) & (phase < 0.7), 0.7 * amp, -0.2 * amp)


class TrafficLight(SignalGenerator):
    def sample_params(self, rng=None):
        rng = rng or np.random
        self.params = {"amp": rng.uniform(0.5, 1.5)}

    def __call__(self, t):
        amp = self.params.get("amp", 1.0)
        return 0.05 * amp * np.sin(2 * np.pi * t / 400)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SIGNAL_MAP = {
    "zero": ZeroSignal,
    "constant": ConstantSignal,
    "sinusoid": SinusoidSignal,
    "square": SquareSignal,
    "step": StepSignal,
    "traffic_rush_hour": TrafficRushHour,
    "traffic_congestion": TrafficCongestion,
    "traffic_pulse": TrafficPulse,
    "traffic_light": TrafficLight,
}


def create_signal(signal_type: str, mode: str = "train") -> SignalGenerator:
    """Create a signal generator by name and mode."""
    if signal_type not in _SIGNAL_MAP:
        raise ValueError(f"Unknown signal type: {signal_type}. "
                         f"Available: {list(_SIGNAL_MAP.keys())}")
    return _SIGNAL_MAP[signal_type](mode)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, 50, 1000)
    rng = np.random.RandomState(42)

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    for ax, (name, cls) in zip(axes.flat, _SIGNAL_MAP.items()):
        gen = cls("train")
        gen.sample_params(rng)
        ax.plot(t, gen(t))
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("/tmp/signals_demo.png", dpi=100)
    plt.close(fig)
    print("Signal demo saved to /tmp/signals_demo.png")
