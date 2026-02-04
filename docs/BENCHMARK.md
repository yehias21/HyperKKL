# HyperKKL-Bench: A Comprehensive Benchmark for Neural KKL Observers

## Executive Summary

**Goal**: Build a rigorous benchmark for evaluating neural KKL observers on non-autonomous dynamical systems, executable in **one week**.

**Inspired by**: PDEBench (Scientific ML), GIFT-Eval (foundation model evaluation), fev-bench (statistical rigor), TFB (comprehensive coverage), TimeSeriesExam (reasoning assessment), DynaBench (sparse observations), Monash Archive (forecasting standards).

---

## Table of Contents

1. [Benchmark Philosophy](#1-benchmark-philosophy)
2. [Core Evaluation Categories](#2-core-evaluation-categories)
3. [Systems & Signals Specification](#3-systems--signals-specification)
4. [Metrics Framework](#4-metrics-framework)
5. [Data Format & Infrastructure](#5-data-format--infrastructure)
6. [One-Week Implementation Plan](#6-one-week-implementation-plan)
7. [Appendix A: Extended Categories](#appendix-a-extended-categories)
8. [Appendix B: Benchmark Comparison Matrix](#appendix-b-benchmark-comparison-matrix)
9. [Appendix C: Additional Systems & Signals](#appendix-c-additional-systems--signals)
10. [Appendix D: Advanced Metrics](#appendix-d-advanced-metrics)

---

## 1. Benchmark Philosophy

### 1.1 Design Principles (Adapted from Leading Benchmarks)

| Principle | Source | Implementation |
|-----------|--------|----------------|
| **Systematic Coverage** | PDEBench | Orthogonal evaluation axes |
| **Statistical Rigor** | fev-bench | Bootstrap confidence intervals |
| **Domain Diversity** | GIFT-Eval, TFB | Multiple dynamical system families |
| **Non-leaking Splits** | GIFT-Eval | Strict train/val/test separation |
| **Reproducibility** | fev-bench | Fixed seeds, YAML task specs |
| **Sparse Observations** | DynaBench | Missing data, low-resolution tests |
| **Covariates Support** | fev-bench | Exogenous inputs as first-class citizens |
| **Reasoning Assessment** | TimeSeriesExam | Structured difficulty levels |

### 1.2 What Makes This Benchmark Unique

1. **Non-autonomous focus**: Unlike existing KKL benchmarks that target autonomous systems
2. **Observer-specific metrics**: Beyond MSE—convergence time, transient peak, estimation delay
3. **Input-system interaction**: Tests resonance, amplitude regimes, input complexity
4. **Multi-scale evaluation**: From clean signals to 50% dropout, from full to 10x downsampled

---

## 2. Core Evaluation Categories

### 2.1 Primary Categories (Must Implement in Week 1)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HyperKKL-Bench Categories                    │
├─────────────────────────────────────────────────────────────────┤
│  CAT-1: Initial Conditions     │  CAT-2: Exogenous Inputs       │
│  ├── ID-IC (in-distribution)   │  ├── Simple (harmonics, step)  │
│  ├── OOD-IC-Near (1.5x)        │  ├── Medium (VDP, Duffing out) │
│  └── OOD-IC-Far (2-3x)         │  ├── Complex (Lorenz, Chua)    │
│                                │  └── Adversarial (resonant)    │
├─────────────────────────────────────────────────────────────────┤
│  CAT-3: System Parameters      │  CAT-4: Temporal Horizon       │
│  ├── ID-Param (training)       │  ├── Short (≤ T_train)         │
│  ├── OOD-Param-Interp          │  ├── Medium (≤ 2×T_train)      │
│  └── OOD-Param-Extrap          │  └── Long (> 5×T_train)        │
├─────────────────────────────────────────────────────────────────┤
│  CAT-5: Noise (SNR)            │  CAT-6: Sampling Resolution    │
│  ├── Clean (∞ dB)              │  ├── Full (1x)                 │
│  ├── Low (40 dB)               │  ├── 2x downsample             │
│  ├── Medium (20 dB)            │  ├── 5x downsample             │
│  └── High (10 dB)              │  └── 10x downsample            │
├─────────────────────────────────────────────────────────────────┤
│  CAT-7: Missing Data           │  CAT-8: Cross-System           │
│  ├── Complete (0%)             │  ├── Same system               │
│  ├── Light (5%)                │  ├── Same family               │
│  ├── Moderate (20%)            │  └── Cross-class               │
│  └── Heavy (50%)               │                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Category Specifications

#### CAT-1: Initial Condition Generalization

```yaml
category: initial_conditions
levels:
  - name: ID-IC
    description: In-distribution initial conditions
    sampling: uniform within training bounds
    
  - name: OOD-IC-Near
    description: Near out-of-distribution
    sampling: 1.0x to 1.5x training bounds
    
  - name: OOD-IC-Far
    description: Far out-of-distribution  
    sampling: 2.0x to 3.0x training bounds
```

#### CAT-2: Exogenous Input Complexity

```yaml
category: exogenous_inputs
levels:
  - name: Simple
    signals: [harmonics, square, step]
    characteristics: single_frequency, predictable
    
  - name: Medium
    signals: [vdp_output, duffing_output]
    characteristics: quasi_periodic, bounded
    
  - name: Complex
    signals: [lorenz_output, chua_output, rossler_output]
    characteristics: broadband, aperiodic, chaotic
    
  - name: Adversarial
    signals: [resonant, switching, chirp]
    characteristics: designed_to_maximize_error
```

#### CAT-5: SNR Levels

```yaml
category: noise_robustness
levels:
  - name: Clean
    snr_db: inf
    noise_type: none
    
  - name: Low
    snr_db: 40
    noise_type: gaussian_white
    application: high_quality_sensors
    
  - name: Medium
    snr_db: 20
    noise_type: gaussian_white
    application: typical_industrial
    
  - name: High
    snr_db: 10
    noise_type: gaussian_white
    application: harsh_environments
```

#### CAT-6: Sampling Resolution (PDEBench-inspired)

```yaml
category: sampling_resolution
# Inspired by PDEBench's reduced_resolution_t parameter
levels:
  - name: Full
    factor: 1
    description: Native simulation timestep
    
  - name: 2x_Downsample
    factor: 2
    description: Every 2nd sample
    
  - name: 5x_Downsample
    factor: 5
    description: Every 5th sample
    
  - name: 10x_Downsample
    factor: 10
    description: Sparse observations
    
sampling_techniques:
  - uniform: Fixed Δt throughout
  - jittered: Uniform + random perturbation (±10%)
  - event_triggered: Sample when |y(t) - y(t_last)| > threshold
```

#### CAT-7: Missing Data (DynaBench-inspired)

```yaml
category: missing_data
levels:
  - name: Complete
    dropout_rate: 0.0
    pattern: none
    
  - name: Light
    dropout_rate: 0.05
    pattern: random_single_point
    
  - name: Moderate
    dropout_rate: 0.20
    pattern: random
    
  - name: Heavy
    dropout_rate: 0.50
    pattern: random
    
  - name: Burst
    dropout_rate: variable
    pattern: contiguous_gaps
    gap_length: [10, 50, 100]  # timesteps
```

---

## 3. Systems & Signals Specification

### 3.1 Core Dynamical Systems

| System | Dim | Type | Equations | Key Parameters |
|--------|-----|------|-----------|----------------|
| **Lorenz** | 3D | Chaotic | ẋ=σ(y-x), ẏ=x(ρ-z)-y, ż=xy-βz | σ=10, ρ=28, β=8/3 |
| **Rössler** | 3D | Chaotic | ẋ=-y-z, ẏ=x+ay, ż=b+z(x-c) | a=0.2, b=0.2, c=5.7 |
| **Chua** | 3D | Chaotic | ẋ=α(y-x-f(x)), ẏ=x-y+z, ż=-βy | α=15.6, β=28 |
| **VDP** | 2D | Limit Cycle | ẍ-μ(1-x²)ẋ+x=u(t) | μ=1.0 |
| **Duffing** | 2D | Oscillator | ẍ+δẋ+αx+βx³=γcos(ωt)+u(t) | α=1, β=1, δ=0.3 |
| **SIR** | 3D | Epidem. | Ṡ=-βSI, İ=βSI-γI, Ṙ=γI | β=0.3, γ=0.1 |

### 3.2 Exogenous Input Signals

```python
# Signal Generation Specifications
signals = {
    "harmonics": {
        "class": "Simple",
        "formula": "A * sin(ω*t + φ)",
        "params": {"A": [0.5, 2.0], "ω": [0.5, 5.0], "φ": [0, 2π]}
    },
    "square": {
        "class": "Simple", 
        "formula": "A * sign(sin(ω*t))",
        "params": {"A": [0.5, 2.0], "period": [1.0, 10.0]}
    },
    "vdp_output": {
        "class": "Medium",
        "formula": "x₁(t) from VDP system",
        "params": {"μ": [0.5, 3.0]}
    },
    "lorenz_output": {
        "class": "Complex",
        "formula": "x₁(t) from Lorenz system",
        "params": {"σ": 10, "ρ": 28, "β": 8/3}
    },
    "chirp": {
        "class": "Adversarial",
        "formula": "A * sin(2π*(f₀ + kt)*t)",
        "params": {"f₀": 0.1, "k": 0.1, "A": 1.0}
    }
}
```

### 3.3 System-Signal Compatibility Matrix

```
              │ Harmonics │ Square │ VDP │ Duff │ Lorenz │ Chua │ Rössler │
──────────────┼───────────┼────────┼─────┼──────┼────────┼──────┼─────────┤
Lorenz        │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
Rössler       │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
Chua          │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
VDP           │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
Duffing       │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
SIR           │     ✓     │   ✓    │  ✓  │  ✓   │   ✓    │  ✓   │    ✓    │
```

---

## 4. Metrics Framework

### 4.1 Primary Metrics (fev-bench style)

```python
metrics = {
    # Point Metrics
    "RMSE": "√(mean((x̂-x)²))",
    "NRMSE": "RMSE / std(x)",  # Normalized
    "MAE": "mean(|x̂-x|)",
    "MaxError": "max(|x̂-x|)",
    
    # Scaled Metrics (Monash-style)
    "MASE": "MAE / MAE_naive",  # vs naive baseline
}
```

### 4.2 Observer-Specific Metrics

```python
observer_metrics = {
    # Temporal Metrics
    "convergence_time": {
        "description": "Time to reach ε-ball around true state",
        "formula": "t_conv = min{t : ||x̂(t) - x(t)|| < ε for all t' > t}",
        "epsilon": 0.1  # relative to signal range
    },
    
    "transient_peak_error": {
        "description": "Maximum error before convergence",
        "formula": "max(||x̂(t) - x(t)||) for t < t_conv"
    },
    
    "steady_state_error": {
        "description": "Mean error after transient",
        "formula": "mean(||x̂(t) - x(t)||) for t > t_trans"
    },
    
    "estimation_delay": {
        "description": "Phase lag in tracking",
        "formula": "argmax(cross_correlation(x, x̂))"
    }
}
```

### 4.3 Frequency-Domain Metrics (PDEBench-inspired)

```python
frequency_metrics = {
    "FFT_RMSE_low": {
        "description": "Error in low frequency band",
        "range": "[0, f_max/4]"
    },
    "FFT_RMSE_mid": {
        "description": "Error in mid frequency band", 
        "range": "[f_max/4, 3*f_max/4]"
    },
    "FFT_RMSE_high": {
        "description": "Error in high frequency band",
        "range": "[3*f_max/4, f_max]"
    }
}
```

### 4.4 Aggregation Methods (fev-bench style)

```python
# Win Rate: How often model A beats model B
def win_rate(model_a_scores, model_b_scores, tasks):
    wins = sum(1 for t in tasks if model_a_scores[t] < model_b_scores[t])
    return wins / len(tasks)

# Skill Score: Relative improvement over baseline
def skill_score(model_scores, baseline_scores, tasks):
    return 1 - np.mean([model_scores[t] / baseline_scores[t] for t in tasks])

# Bootstrap Confidence Intervals
def bootstrap_ci(scores, n_bootstrap=1000, ci=0.95):
    bootstrap_means = [np.mean(np.random.choice(scores, len(scores))) 
                       for _ in range(n_bootstrap)]
    return np.percentile(bootstrap_means, [(1-ci)/2*100, (1+ci)/2*100])
```

---

## 5. Data Format & Infrastructure

### 5.1 Directory Structure

```
hyperkkl_bench/
├── data/
│   ├── raw/                    # Generated trajectories
│   │   ├── lorenz/
│   │   │   ├── train.h5
│   │   │   ├── val.h5
│   │   │   └── test/
│   │   │       ├── ic_id.h5
│   │   │       ├── ic_ood_near.h5
│   │   │       ├── input_simple.h5
│   │   │       ├── input_complex.h5
│   │   │       ├── snr_20db.h5
│   │   │       └── ...
│   │   └── rossler/
│   └── processed/              # Ready for training
├── configs/
│   ├── systems/
│   │   ├── lorenz.yaml
│   │   └── ...
│   ├── signals/
│   │   ├── harmonics.yaml
│   │   └── ...
│   └── tasks/
│       ├── cat1_ic.yaml
│       └── ...
├── src/
│   ├── data_generation/
│   ├── evaluation/
│   └── baselines/
├── results/
└── benchmark.yaml              # Master config
```

### 5.2 HDF5 Data Schema

```python
# Per-file structure
{
    "x": np.ndarray,          # [N_samples, N_timesteps, N_states]
    "y": np.ndarray,          # [N_samples, N_timesteps, N_outputs]
    "u": np.ndarray,          # [N_samples, N_timesteps, N_inputs]
    "t": np.ndarray,          # [N_timesteps]
    "metadata": {
        "system": str,
        "input_signal": str,
        "category": str,
        "level": str,
        "snr_db": float,
        "downsample_factor": int,
        "dropout_rate": float,
        "seed": int
    },
    "params": {
        "system_params": dict,
        "signal_params": dict
    }
}
```

### 5.3 Task Specification (fev-bench style)

```yaml
# configs/tasks/cat5_snr.yaml
task_name: snr_robustness
category: CAT-5
description: "Evaluate observer robustness to measurement noise"

systems: [lorenz, rossler, vdp, duffing]
signals: [harmonics, lorenz_output]

levels:
  - name: clean
    snr_db: inf
    n_samples: 100
    
  - name: low_noise
    snr_db: 40
    n_samples: 100
    
  - name: medium_noise
    snr_db: 20
    n_samples: 100
    
  - name: high_noise
    snr_db: 10
    n_samples: 100

evaluation:
  metrics: [NRMSE, convergence_time, steady_state_error]
  horizon: 1000  # timesteps
  
splitting:
  train_ratio: 0.7
  val_ratio: 0.1
  test_ratio: 0.2
  seed: 42
```

---

## 6. One-Week Implementation Plan

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Week 1 Implementation                         │
├─────────┬───────────────────────────────────────────────────────┤
│ Day 1-2 │ Infrastructure + Data Generation Pipeline             │
│ Day 3-4 │ Core Categories (CAT 1-4) Data Generation             │
│ Day 5   │ Extended Categories (CAT 5-8) + Metrics               │
│ Day 6   │ Baseline Evaluation + Results Aggregation             │
│ Day 7   │ Documentation + Validation + Release Prep             │
└─────────┴───────────────────────────────────────────────────────┘
```

---

### Day 1-2: Infrastructure Setup

#### Morning Day 1: Project Skeleton

```bash
# Create project structure
mkdir -p hyperkkl_bench/{data/{raw,processed},configs/{systems,signals,tasks},src/{data_generation,evaluation,baselines},results}

# Initialize Python environment
cd hyperkkl_bench
python -m venv venv
source venv/bin/activate
pip install numpy scipy h5py pyyaml tqdm matplotlib torch
```

#### Afternoon Day 1: System Simulators

```python
# src/data_generation/systems.py
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class DynamicalSystem:
    name: str
    dim: int
    f: Callable  # dx/dt = f(t, x, u)
    h: Callable  # y = h(x)
    default_params: dict
    default_bounds: np.ndarray  # [dim, 2] for IC sampling

class SystemRegistry:
    """Registry of all dynamical systems"""
    
    @staticmethod
    def lorenz(sigma=10, rho=28, beta=8/3):
        def f(t, x, u):
            return np.array([
                sigma * (x[1] - x[0]) + u,
                x[0] * (rho - x[2]) - x[1],
                x[0] * x[1] - beta * x[2]
            ])
        def h(x):
            return x[0:1]  # Observe x only
        return DynamicalSystem(
            name="lorenz",
            dim=3,
            f=f,
            h=h,
            default_params={"sigma": sigma, "rho": rho, "beta": beta},
            default_bounds=np.array([[-20, 20], [-30, 30], [0, 50]])
        )
    
    @staticmethod
    def rossler(a=0.2, b=0.2, c=5.7):
        def f(t, x, u):
            return np.array([
                -x[1] - x[2] + u,
                x[0] + a * x[1],
                b + x[2] * (x[0] - c)
            ])
        def h(x):
            return x[0:1]
        return DynamicalSystem(
            name="rossler",
            dim=3,
            f=f,
            h=h,
            default_params={"a": a, "b": b, "c": c},
            default_bounds=np.array([[-10, 10], [-10, 10], [0, 25]])
        )
    
    @staticmethod
    def vdp(mu=1.0):
        def f(t, x, u):
            return np.array([
                x[1],
                mu * (1 - x[0]**2) * x[1] - x[0] + u
            ])
        def h(x):
            return x[0:1]
        return DynamicalSystem(
            name="vdp",
            dim=2,
            f=f,
            h=h,
            default_params={"mu": mu},
            default_bounds=np.array([[-3, 3], [-3, 3]])
        )
    
    @staticmethod
    def duffing(alpha=1, beta=1, delta=0.3):
        def f(t, x, u):
            return np.array([
                x[1],
                -delta * x[1] - alpha * x[0] - beta * x[0]**3 + u
            ])
        def h(x):
            return x[0:1]
        return DynamicalSystem(
            name="duffing",
            dim=2,
            f=f,
            h=h,
            default_params={"alpha": alpha, "beta": beta, "delta": delta},
            default_bounds=np.array([[-2, 2], [-2, 2]])
        )
    
    @staticmethod  
    def chua(alpha=15.6, beta=28, m0=-1.143, m1=-0.714):
        def g(x):
            return m1*x + 0.5*(m0-m1)*(np.abs(x+1) - np.abs(x-1))
        def f(t, x, u):
            return np.array([
                alpha * (x[1] - x[0] - g(x[0])) + u,
                x[0] - x[1] + x[2],
                -beta * x[1]
            ])
        def h(x):
            return x[0:1]
        return DynamicalSystem(
            name="chua",
            dim=3,
            f=f,
            h=h,
            default_params={"alpha": alpha, "beta": beta, "m0": m0, "m1": m1},
            default_bounds=np.array([[-3, 3], [-0.5, 0.5], [-4, 4]])
        )
    
    @staticmethod
    def sir(beta_param=0.3, gamma=0.1):
        def f(t, x, u):
            S, I, R = x
            return np.array([
                -beta_param * S * I + u,
                beta_param * S * I - gamma * I,
                gamma * I
            ])
        def h(x):
            return x[1:2]  # Observe I only
        return DynamicalSystem(
            name="sir",
            dim=3,
            f=f,
            h=h,
            default_params={"beta": beta_param, "gamma": gamma},
            default_bounds=np.array([[0, 1], [0, 1], [0, 1]])
        )
```

#### Morning Day 2: Signal Generators

```python
# src/data_generation/signals.py
import numpy as np
from typing import Callable
from scipy.integrate import solve_ivp

class SignalGenerator:
    """Generate exogenous input signals"""
    
    @staticmethod
    def harmonics(t: np.ndarray, A=1.0, omega=1.0, phi=0.0) -> np.ndarray:
        """Simple harmonic signal"""
        return A * np.sin(omega * t + phi)
    
    @staticmethod
    def square(t: np.ndarray, A=1.0, period=2.0) -> np.ndarray:
        """Square wave signal"""
        return A * np.sign(np.sin(2 * np.pi * t / period))
    
    @staticmethod
    def step(t: np.ndarray, A=1.0, t_step=5.0) -> np.ndarray:
        """Step signal"""
        return A * (t >= t_step).astype(float)
    
    @staticmethod
    def chirp(t: np.ndarray, A=1.0, f0=0.1, k=0.1) -> np.ndarray:
        """Frequency sweep signal"""
        return A * np.sin(2 * np.pi * (f0 + k * t) * t)
    
    @staticmethod
    def vdp_output(t: np.ndarray, mu=1.0, x0=None) -> np.ndarray:
        """VDP system output as driving signal"""
        if x0 is None:
            x0 = [2.0, 0.0]
        def vdp(t, x):
            return [x[1], mu * (1 - x[0]**2) * x[1] - x[0]]
        sol = solve_ivp(vdp, [t[0], t[-1]], x0, t_eval=t, method='RK45')
        return sol.y[0]
    
    @staticmethod
    def lorenz_output(t: np.ndarray, sigma=10, rho=28, beta=8/3, x0=None) -> np.ndarray:
        """Lorenz system output as driving signal (chaotic)"""
        if x0 is None:
            x0 = [1.0, 1.0, 1.0]
        def lorenz(t, x):
            return [sigma*(x[1]-x[0]), x[0]*(rho-x[2])-x[1], x[0]*x[1]-beta*x[2]]
        sol = solve_ivp(lorenz, [t[0], t[-1]], x0, t_eval=t, method='RK45')
        # Normalize to [-1, 1] range
        x_out = sol.y[0]
        return 2 * (x_out - x_out.min()) / (x_out.max() - x_out.min()) - 1
    
    @staticmethod
    def chua_output(t: np.ndarray, alpha=15.6, beta=28, x0=None) -> np.ndarray:
        """Chua system output as driving signal (chaotic)"""
        if x0 is None:
            x0 = [0.1, 0.0, 0.0]
        m0, m1 = -1.143, -0.714
        def g(x):
            return m1*x + 0.5*(m0-m1)*(np.abs(x+1) - np.abs(x-1))
        def chua(t, x):
            return [alpha*(x[1]-x[0]-g(x[0])), x[0]-x[1]+x[2], -beta*x[1]]
        sol = solve_ivp(chua, [t[0], t[-1]], x0, t_eval=t, method='RK45')
        x_out = sol.y[0]
        return 2 * (x_out - x_out.min()) / (x_out.max() - x_out.min()) - 1
    
    @staticmethod
    def rossler_output(t: np.ndarray, a=0.2, b=0.2, c=5.7, x0=None) -> np.ndarray:
        """Rossler system output as driving signal"""
        if x0 is None:
            x0 = [1.0, 1.0, 1.0]
        def rossler(t, x):
            return [-x[1]-x[2], x[0]+a*x[1], b+x[2]*(x[0]-c)]
        sol = solve_ivp(rossler, [t[0], t[-1]], x0, t_eval=t, method='RK45')
        x_out = sol.y[0]
        return 2 * (x_out - x_out.min()) / (x_out.max() - x_out.min()) - 1

# Signal class mapping
SIGNAL_CLASSES = {
    "Simple": ["harmonics", "square", "step"],
    "Medium": ["vdp_output", "duffing_output"],
    "Complex": ["lorenz_output", "chua_output", "rossler_output"],
    "Adversarial": ["chirp", "resonant", "switching"]
}
```

#### Afternoon Day 2: Data Generation Pipeline

```python
# src/data_generation/generator.py
import numpy as np
import h5py
from scipy.integrate import solve_ivp
from tqdm import tqdm
from typing import List, Dict, Optional
import yaml

class TrajectoryGenerator:
    """Generate trajectories for benchmark"""
    
    def __init__(self, system, signal_func, dt=0.01, T=100.0):
        self.system = system
        self.signal_func = signal_func
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T, dt)
        
    def generate_single(self, x0: np.ndarray, signal_params: dict,
                       snr_db: float = np.inf,
                       downsample: int = 1,
                       dropout_rate: float = 0.0) -> Dict:
        """Generate a single trajectory"""
        
        # Generate input signal
        u = self.signal_func(self.t, **signal_params)
        
        # Simulate system
        def f_with_input(t, x):
            idx = int(t / self.dt)
            idx = min(idx, len(u) - 1)
            return self.system.f(t, x, u[idx])
        
        sol = solve_ivp(f_with_input, [0, self.T], x0, 
                       t_eval=self.t, method='RK45')
        x = sol.y.T  # [N_timesteps, N_states]
        
        # Generate observations
        y = np.array([self.system.h(x[i]) for i in range(len(x))])
        
        # Add noise
        if snr_db < np.inf:
            signal_power = np.mean(y**2)
            noise_power = signal_power / (10**(snr_db/10))
            y = y + np.sqrt(noise_power) * np.random.randn(*y.shape)
        
        # Downsample
        if downsample > 1:
            x = x[::downsample]
            y = y[::downsample]
            u = u[::downsample]
            t = self.t[::downsample]
        else:
            t = self.t
        
        # Apply dropout
        if dropout_rate > 0:
            mask = np.random.rand(len(y)) > dropout_rate
            y_dropout = y.copy()
            y_dropout[~mask] = np.nan
        else:
            y_dropout = y
            mask = np.ones(len(y), dtype=bool)
        
        return {
            "x": x,
            "y": y_dropout,
            "y_clean": y,
            "u": u.reshape(-1, 1),
            "t": t,
            "mask": mask
        }
    
    def generate_batch(self, n_samples: int, 
                      ic_sampler,
                      signal_params_sampler,
                      **kwargs) -> Dict:
        """Generate batch of trajectories"""
        
        trajectories = []
        for i in tqdm(range(n_samples), desc="Generating trajectories"):
            x0 = ic_sampler()
            signal_params = signal_params_sampler()
            traj = self.generate_single(x0, signal_params, **kwargs)
            trajectories.append(traj)
        
        # Stack into arrays
        return {
            "x": np.stack([t["x"] for t in trajectories]),
            "y": np.stack([t["y"] for t in trajectories]),
            "u": np.stack([t["u"] for t in trajectories]),
            "t": trajectories[0]["t"],
            "mask": np.stack([t["mask"] for t in trajectories])
        }
    
    def save_to_hdf5(self, data: Dict, filepath: str, metadata: Dict):
        """Save trajectory data to HDF5"""
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
            
            # Save metadata
            meta_group = f.create_group("metadata")
            for key, value in metadata.items():
                if isinstance(value, dict):
                    subgroup = meta_group.create_group(key)
                    for k, v in value.items():
                        subgroup.attrs[k] = v
                else:
                    meta_group.attrs[key] = value
```

---

### Day 3-4: Core Categories Data Generation

#### Day 3 Morning: CAT-1 (Initial Conditions) & CAT-2 (Exogenous Inputs)

```python
# scripts/generate_cat1_cat2.py
import numpy as np
from src.data_generation.systems import SystemRegistry
from src.data_generation.signals import SignalGenerator
from src.data_generation.generator import TrajectoryGenerator
import os

# Configuration
SYSTEMS = ["lorenz", "rossler", "vdp", "duffing", "chua", "sir"]
N_SAMPLES = 100
T = 100.0
DT = 0.01
SEED = 42

np.random.seed(SEED)

def generate_cat1_ic(system_name: str, output_dir: str):
    """Generate CAT-1: Initial Condition tests"""
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    
    # Use harmonics as default input for IC tests
    signal_func = SignalGenerator.harmonics
    signal_params = {"A": 1.0, "omega": 1.0, "phi": 0.0}
    
    generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
    bounds = system.default_bounds
    
    # ID-IC: In-distribution
    def id_sampler():
        return np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    
    def signal_sampler():
        return signal_params
    
    data_id = generator.generate_batch(N_SAMPLES, id_sampler, signal_sampler)
    generator.save_to_hdf5(data_id, f"{output_dir}/{system_name}/test_ic_id.h5",
                          {"system": system_name, "category": "CAT-1", "level": "ID-IC"})
    
    # OOD-IC-Near: 1.0x to 1.5x bounds
    def ood_near_sampler():
        scale = np.random.uniform(1.0, 1.5)
        return np.array([np.random.uniform(b[0]*scale, b[1]*scale) for b in bounds])
    
    data_ood_near = generator.generate_batch(N_SAMPLES, ood_near_sampler, signal_sampler)
    generator.save_to_hdf5(data_ood_near, f"{output_dir}/{system_name}/test_ic_ood_near.h5",
                          {"system": system_name, "category": "CAT-1", "level": "OOD-IC-Near"})
    
    # OOD-IC-Far: 2.0x to 3.0x bounds
    def ood_far_sampler():
        scale = np.random.uniform(2.0, 3.0)
        return np.array([np.random.uniform(b[0]*scale, b[1]*scale) for b in bounds])
    
    data_ood_far = generator.generate_batch(N_SAMPLES, ood_far_sampler, signal_sampler)
    generator.save_to_hdf5(data_ood_far, f"{output_dir}/{system_name}/test_ic_ood_far.h5",
                          {"system": system_name, "category": "CAT-1", "level": "OOD-IC-Far"})
    
    print(f"✓ CAT-1 generated for {system_name}")

def generate_cat2_inputs(system_name: str, output_dir: str):
    """Generate CAT-2: Exogenous Input tests"""
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    bounds = system.default_bounds
    
    def ic_sampler():
        return np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    
    # Simple inputs
    for signal_name in ["harmonics", "square"]:
        signal_func = getattr(SignalGenerator, signal_name)
        
        def signal_sampler():
            if signal_name == "harmonics":
                return {"A": np.random.uniform(0.5, 2.0),
                       "omega": np.random.uniform(0.5, 3.0),
                       "phi": np.random.uniform(0, 2*np.pi)}
            else:
                return {"A": np.random.uniform(0.5, 2.0),
                       "period": np.random.uniform(1.0, 5.0)}
        
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
        data = generator.generate_batch(N_SAMPLES, ic_sampler, signal_sampler)
        generator.save_to_hdf5(data, 
                              f"{output_dir}/{system_name}/test_input_{signal_name}.h5",
                              {"system": system_name, "category": "CAT-2", 
                               "level": "Simple", "signal": signal_name})
    
    # Complex inputs (chaotic)
    for signal_name in ["lorenz_output", "rossler_output"]:
        signal_func = getattr(SignalGenerator, signal_name)
        
        def signal_sampler():
            return {}  # Use defaults
        
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
        data = generator.generate_batch(N_SAMPLES, ic_sampler, signal_sampler)
        generator.save_to_hdf5(data,
                              f"{output_dir}/{system_name}/test_input_{signal_name}.h5",
                              {"system": system_name, "category": "CAT-2",
                               "level": "Complex", "signal": signal_name})
    
    print(f"✓ CAT-2 generated for {system_name}")

# Run generation
if __name__ == "__main__":
    output_dir = "data/raw"
    for system in SYSTEMS:
        os.makedirs(f"{output_dir}/{system}", exist_ok=True)
        generate_cat1_ic(system, output_dir)
        generate_cat2_inputs(system, output_dir)
```

#### Day 3 Afternoon & Day 4 Morning: CAT-3 (Parameters) & CAT-4 (Temporal)

```python
# scripts/generate_cat3_cat4.py
import numpy as np
from src.data_generation.systems import SystemRegistry
from src.data_generation.signals import SignalGenerator
from src.data_generation.generator import TrajectoryGenerator

def generate_cat3_params(system_name: str, output_dir: str):
    """Generate CAT-3: System Parameter variation tests"""
    
    # Parameter variation ranges per system
    PARAM_VARIATIONS = {
        "lorenz": {
            "param": "rho",
            "training": 28,
            "interp_range": [25, 31],
            "extrap_range": [32, 40]
        },
        "rossler": {
            "param": "c",
            "training": 5.7,
            "interp_range": [5.0, 6.5],
            "extrap_range": [7.0, 10.0]
        },
        "vdp": {
            "param": "mu",
            "training": 1.0,
            "interp_range": [0.5, 1.5],
            "extrap_range": [2.0, 4.0]
        },
        "duffing": {
            "param": "beta",
            "training": 1.0,
            "interp_range": [0.5, 1.5],
            "extrap_range": [2.0, 5.0]
        }
    }
    
    if system_name not in PARAM_VARIATIONS:
        print(f"Skipping CAT-3 for {system_name} (no param variation defined)")
        return
    
    config = PARAM_VARIATIONS[system_name]
    
    # Generate for interpolation
    for level, param_range in [("interp", config["interp_range"]), 
                               ("extrap", config["extrap_range"])]:
        
        for _ in range(N_SAMPLES):
            param_value = np.random.uniform(*param_range)
            system = getattr(SystemRegistry, system_name)(**{config["param"]: param_value})
            # ... generate and save
    
    print(f"✓ CAT-3 generated for {system_name}")

def generate_cat4_temporal(system_name: str, output_dir: str):
    """Generate CAT-4: Temporal horizon tests"""
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    signal_func = SignalGenerator.harmonics
    
    HORIZONS = {
        "short": 50.0,    # T_train / 2
        "medium": 200.0,  # 2 * T_train
        "long": 500.0,    # 5 * T_train
        "ultra_long": 2000.0  # 20 * T_train
    }
    
    for level, T_horizon in HORIZONS.items():
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T_horizon)
        # ... generate and save
    
    print(f"✓ CAT-4 generated for {system_name}")
```

#### Day 4 Afternoon: CAT-5 (Noise) & CAT-6 (Sampling)

```python
# scripts/generate_cat5_cat6.py

def generate_cat5_noise(system_name: str, output_dir: str):
    """Generate CAT-5: SNR tests"""
    
    SNR_LEVELS = {
        "clean": np.inf,
        "low_noise": 40,
        "medium_noise": 20,
        "high_noise": 10
    }
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    signal_func = SignalGenerator.harmonics
    
    for level, snr_db in SNR_LEVELS.items():
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
        
        def ic_sampler():
            return np.array([np.random.uniform(b[0], b[1]) 
                           for b in system.default_bounds])
        
        def signal_sampler():
            return {"A": 1.0, "omega": 1.0, "phi": 0.0}
        
        data = generator.generate_batch(N_SAMPLES, ic_sampler, signal_sampler,
                                        snr_db=snr_db)
        generator.save_to_hdf5(data,
                              f"{output_dir}/{system_name}/test_snr_{level}.h5",
                              {"system": system_name, "category": "CAT-5",
                               "level": level, "snr_db": snr_db})
    
    print(f"✓ CAT-5 generated for {system_name}")

def generate_cat6_sampling(system_name: str, output_dir: str):
    """Generate CAT-6: Sampling resolution tests (PDEBench-inspired)"""
    
    DOWNSAMPLE_FACTORS = {
        "full": 1,
        "2x": 2,
        "5x": 5,
        "10x": 10
    }
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    signal_func = SignalGenerator.harmonics
    
    for level, factor in DOWNSAMPLE_FACTORS.items():
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
        
        def ic_sampler():
            return np.array([np.random.uniform(b[0], b[1]) 
                           for b in system.default_bounds])
        
        def signal_sampler():
            return {"A": 1.0, "omega": 1.0, "phi": 0.0}
        
        data = generator.generate_batch(N_SAMPLES, ic_sampler, signal_sampler,
                                        downsample=factor)
        generator.save_to_hdf5(data,
                              f"{output_dir}/{system_name}/test_sampling_{level}.h5",
                              {"system": system_name, "category": "CAT-6",
                               "level": level, "downsample_factor": factor})
    
    print(f"✓ CAT-6 generated for {system_name}")
```

---

### Day 5: Extended Categories & Metrics

#### Morning: CAT-7 (Missing Data) & CAT-8 (Cross-System)

```python
# scripts/generate_cat7_cat8.py

def generate_cat7_missing(system_name: str, output_dir: str):
    """Generate CAT-7: Missing data tests (DynaBench-inspired)"""
    
    DROPOUT_RATES = {
        "complete": 0.0,
        "light": 0.05,
        "moderate": 0.20,
        "heavy": 0.50
    }
    
    system_factory = getattr(SystemRegistry, system_name)
    system = system_factory()
    signal_func = SignalGenerator.harmonics
    
    for level, rate in DROPOUT_RATES.items():
        generator = TrajectoryGenerator(system, signal_func, dt=DT, T=T)
        
        def ic_sampler():
            return np.array([np.random.uniform(b[0], b[1]) 
                           for b in system.default_bounds])
        
        def signal_sampler():
            return {"A": 1.0, "omega": 1.0, "phi": 0.0}
        
        data = generator.generate_batch(N_SAMPLES, ic_sampler, signal_sampler,
                                        dropout_rate=rate)
        generator.save_to_hdf5(data,
                              f"{output_dir}/{system_name}/test_dropout_{level}.h5",
                              {"system": system_name, "category": "CAT-7",
                               "level": level, "dropout_rate": rate})
    
    print(f"✓ CAT-7 generated for {system_name}")

# CAT-8 is handled in evaluation by using models trained on one system
# and tested on another - no special data generation needed
```

#### Afternoon: Metrics Implementation

```python
# src/evaluation/metrics.py
import numpy as np
from typing import Dict, Tuple
from scipy import signal as sig

class ObserverMetrics:
    """Comprehensive metrics for KKL observer evaluation"""
    
    @staticmethod
    def rmse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
        """Root Mean Square Error"""
        return np.sqrt(np.mean((x_true - x_pred)**2))
    
    @staticmethod
    def nrmse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
        """Normalized RMSE"""
        return ObserverMetrics.rmse(x_true, x_pred) / np.std(x_true)
    
    @staticmethod
    def mae(x_true: np.ndarray, x_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(x_true - x_pred))
    
    @staticmethod
    def max_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
        """Maximum Error"""
        return np.max(np.abs(x_true - x_pred))
    
    @staticmethod
    def convergence_time(x_true: np.ndarray, x_pred: np.ndarray, 
                        t: np.ndarray, epsilon: float = 0.1) -> float:
        """Time to converge within epsilon-ball"""
        # Normalize epsilon by signal range
        signal_range = np.max(x_true) - np.min(x_true)
        eps_abs = epsilon * signal_range
        
        error = np.linalg.norm(x_true - x_pred, axis=-1)
        
        # Find first time where error stays below threshold
        below_threshold = error < eps_abs
        
        # Look for sustained convergence
        for i in range(len(below_threshold)):
            if np.all(below_threshold[i:min(i+100, len(below_threshold))]):
                return t[i]
        
        return t[-1]  # Never converged
    
    @staticmethod
    def transient_peak_error(x_true: np.ndarray, x_pred: np.ndarray,
                            t_transient: float, t: np.ndarray) -> float:
        """Maximum error during transient phase"""
        mask = t < t_transient
        if not np.any(mask):
            return 0.0
        error = np.linalg.norm(x_true[mask] - x_pred[mask], axis=-1)
        return np.max(error)
    
    @staticmethod
    def steady_state_error(x_true: np.ndarray, x_pred: np.ndarray,
                          t_transient: float, t: np.ndarray) -> float:
        """Mean error after transient phase"""
        mask = t >= t_transient
        if not np.any(mask):
            return np.nan
        error = np.linalg.norm(x_true[mask] - x_pred[mask], axis=-1)
        return np.mean(error)
    
    @staticmethod
    def estimation_delay(x_true: np.ndarray, x_pred: np.ndarray,
                        dt: float) -> float:
        """Estimation delay via cross-correlation"""
        # Use first dimension for delay estimation
        x1 = x_true[:, 0] if x_true.ndim > 1 else x_true
        x2 = x_pred[:, 0] if x_pred.ndim > 1 else x_pred
        
        correlation = sig.correlate(x1, x2, mode='full')
        lags = sig.correlation_lags(len(x1), len(x2), mode='full')
        
        delay_idx = lags[np.argmax(correlation)]
        return delay_idx * dt
    
    @staticmethod
    def fft_rmse_bands(x_true: np.ndarray, x_pred: np.ndarray,
                      dt: float) -> Dict[str, float]:
        """Frequency-domain RMSE in low/mid/high bands (PDEBench-inspired)"""
        
        # Use first dimension
        x1 = x_true[:, 0] if x_true.ndim > 1 else x_true
        x2 = x_pred[:, 0] if x_pred.ndim > 1 else x_pred
        
        # Compute FFT
        fft_true = np.fft.rfft(x1)
        fft_pred = np.fft.rfft(x2)
        freqs = np.fft.rfftfreq(len(x1), dt)
        
        f_max = freqs[-1]
        
        # Band boundaries
        low_mask = freqs < f_max / 4
        mid_mask = (freqs >= f_max / 4) & (freqs < 3 * f_max / 4)
        high_mask = freqs >= 3 * f_max / 4
        
        def band_rmse(mask):
            if not np.any(mask):
                return 0.0
            return np.sqrt(np.mean(np.abs(fft_true[mask] - fft_pred[mask])**2))
        
        return {
            "fft_rmse_low": band_rmse(low_mask),
            "fft_rmse_mid": band_rmse(mid_mask),
            "fft_rmse_high": band_rmse(high_mask)
        }
    
    @staticmethod
    def compute_all(x_true: np.ndarray, x_pred: np.ndarray,
                   t: np.ndarray, dt: float,
                   t_transient: float = 10.0) -> Dict[str, float]:
        """Compute all metrics"""
        
        metrics = {
            "rmse": ObserverMetrics.rmse(x_true, x_pred),
            "nrmse": ObserverMetrics.nrmse(x_true, x_pred),
            "mae": ObserverMetrics.mae(x_true, x_pred),
            "max_error": ObserverMetrics.max_error(x_true, x_pred),
            "convergence_time": ObserverMetrics.convergence_time(x_true, x_pred, t),
            "transient_peak_error": ObserverMetrics.transient_peak_error(
                x_true, x_pred, t_transient, t),
            "steady_state_error": ObserverMetrics.steady_state_error(
                x_true, x_pred, t_transient, t),
            "estimation_delay": ObserverMetrics.estimation_delay(x_true, x_pred, dt)
        }
        
        # Add frequency metrics
        fft_metrics = ObserverMetrics.fft_rmse_bands(x_true, x_pred, dt)
        metrics.update(fft_metrics)
        
        return metrics


class AggregationMethods:
    """Statistical aggregation methods (fev-bench style)"""
    
    @staticmethod
    def win_rate(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
        """Fraction of tasks where model A beats model B"""
        return np.mean(scores_a < scores_b)
    
    @staticmethod
    def skill_score(model_scores: np.ndarray, 
                   baseline_scores: np.ndarray) -> float:
        """Relative improvement over baseline"""
        return 1 - np.mean(model_scores / baseline_scores)
    
    @staticmethod
    def bootstrap_ci(scores: np.ndarray, 
                    n_bootstrap: int = 1000,
                    ci: float = 0.95) -> Tuple[float, float]:
        """Bootstrap confidence interval"""
        bootstrap_means = np.array([
            np.mean(np.random.choice(scores, len(scores), replace=True))
            for _ in range(n_bootstrap)
        ])
        lower = np.percentile(bootstrap_means, (1-ci)/2 * 100)
        upper = np.percentile(bootstrap_means, (1+ci)/2 * 100)
        return (lower, upper)
```

---

### Day 6: Baseline Evaluation & Results

#### Morning: Baseline Implementation

```python
# src/baselines/naive_observer.py
import numpy as np

class NaiveObserver:
    """Baseline: use measurement directly as state estimate"""
    
    def __init__(self, n_states: int, n_outputs: int):
        self.n_states = n_states
        self.n_outputs = n_outputs
    
    def estimate(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        y: [N_timesteps, N_outputs]
        u: [N_timesteps, N_inputs]
        Returns: [N_timesteps, N_states]
        """
        # Pad with zeros for unobserved states
        x_hat = np.zeros((len(y), self.n_states))
        x_hat[:, :self.n_outputs] = y
        return x_hat


class SeasonalNaiveObserver:
    """Baseline: use lagged measurement"""
    
    def __init__(self, n_states: int, n_outputs: int, lag: int = 10):
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.lag = lag
    
    def estimate(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_hat = np.zeros((len(y), self.n_states))
        for i in range(len(y)):
            lag_idx = max(0, i - self.lag)
            x_hat[i, :self.n_outputs] = y[lag_idx]
        return x_hat


class LinearObserver:
    """Baseline: Luenberger observer (linear approx)"""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, L: np.ndarray):
        """
        A: System matrix [n, n]
        B: Input matrix [n, m]
        C: Output matrix [p, n]
        L: Observer gain [n, p]
        """
        self.A = A
        self.B = B
        self.C = C
        self.L = L
        self.n = A.shape[0]
    
    def estimate(self, y: np.ndarray, u: np.ndarray, 
                x0: np.ndarray = None, dt: float = 0.01) -> np.ndarray:
        
        if x0 is None:
            x0 = np.zeros(self.n)
        
        x_hat = np.zeros((len(y), self.n))
        x_hat[0] = x0
        
        for i in range(1, len(y)):
            # Observer dynamics: dx̂/dt = Ax̂ + Bu + L(y - Cx̂)
            y_hat = self.C @ x_hat[i-1]
            innovation = y[i-1] - y_hat
            
            dx = self.A @ x_hat[i-1] + self.B @ u[i-1] + self.L @ innovation
            x_hat[i] = x_hat[i-1] + dt * dx
        
        return x_hat
```

#### Afternoon: Evaluation Pipeline

```python
# src/evaluation/evaluator.py
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from .metrics import ObserverMetrics, AggregationMethods

class BenchmarkEvaluator:
    """Evaluate observers across all benchmark categories"""
    
    def __init__(self, data_dir: str, results_dir: str):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_data(self, filepath: str) -> Dict:
        """Load test data from HDF5"""
        with h5py.File(filepath, 'r') as f:
            data = {
                "x": f["x"][:],
                "y": f["y"][:],
                "u": f["u"][:],
                "t": f["t"][:]
            }
            # Load metadata
            data["metadata"] = {}
            for key in f["metadata"].attrs:
                data["metadata"][key] = f["metadata"].attrs[key]
        return data
    
    def evaluate_observer(self, observer, test_files: List[str],
                         dt: float = 0.01) -> Dict:
        """Evaluate observer on multiple test files"""
        
        all_results = {}
        
        for filepath in tqdm(test_files, desc="Evaluating"):
            data = self.load_test_data(filepath)
            
            file_results = []
            for i in range(len(data["x"])):
                x_true = data["x"][i]
                y = data["y"][i]
                u = data["u"][i]
                t = data["t"]
                
                # Handle missing data (NaN values)
                if np.any(np.isnan(y)):
                    y_interp = self._interpolate_missing(y, t)
                else:
                    y_interp = y
                
                # Get estimate
                x_pred = observer.estimate(y_interp, u)
                
                # Compute metrics
                metrics = ObserverMetrics.compute_all(x_true, x_pred, t, dt)
                file_results.append(metrics)
            
            # Aggregate results for this file
            all_results[Path(filepath).stem] = {
                "mean": {k: np.mean([r[k] for r in file_results]) 
                        for k in file_results[0]},
                "std": {k: np.std([r[k] for r in file_results])
                       for k in file_results[0]},
                "ci_95": {k: AggregationMethods.bootstrap_ci(
                    np.array([r[k] for r in file_results]))
                         for k in file_results[0]}
            }
        
        return all_results
    
    def _interpolate_missing(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Interpolate missing values (NaN)"""
        y_interp = y.copy()
        for dim in range(y.shape[1] if y.ndim > 1 else 1):
            col = y[:, dim] if y.ndim > 1 else y
            mask = ~np.isnan(col)
            if np.any(~mask):
                y_interp[:, dim] = np.interp(t, t[mask], col[mask])
        return y_interp
    
    def generate_report(self, results: Dict, output_file: str):
        """Generate evaluation report"""
        
        # Save raw results
        with open(self.results_dir / f"{output_file}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary table
        summary = []
        for test_name, test_results in results.items():
            row = {"test": test_name}
            for metric, value in test_results["mean"].items():
                row[f"{metric}_mean"] = f"{value:.4f}"
                row[f"{metric}_ci"] = f"[{test_results['ci_95'][metric][0]:.4f}, {test_results['ci_95'][metric][1]:.4f}]"
            summary.append(row)
        
        # Save summary as CSV
        import csv
        with open(self.results_dir / f"{output_file}_summary.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)
        
        print(f"✓ Report saved to {self.results_dir / output_file}")
```

---

### Day 7: Documentation & Release

#### Morning: Documentation

```python
# scripts/generate_docs.py
"""Generate documentation for the benchmark"""

README_TEMPLATE = """
# HyperKKL-Bench Results

## Overview

This repository contains evaluation results for the HyperKKL-Bench benchmark.

## Categories Evaluated

| Category | Description | Levels |
|----------|-------------|--------|
| CAT-1 | Initial Conditions | ID, OOD-Near, OOD-Far |
| CAT-2 | Exogenous Inputs | Simple, Medium, Complex, Adversarial |
| CAT-3 | System Parameters | ID, Interpolation, Extrapolation |
| CAT-4 | Temporal Horizon | Short, Medium, Long, Ultra-Long |
| CAT-5 | Noise (SNR) | Clean, 40dB, 20dB, 10dB |
| CAT-6 | Sampling | 1x, 2x, 5x, 10x |
| CAT-7 | Missing Data | 0%, 5%, 20%, 50% |
| CAT-8 | Cross-System | Same, Same-Family, Cross-Class |

## Results Summary

{results_table}

## Usage

```python
from hyperkkl_bench import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(data_dir="data/raw", results_dir="results")
results = evaluator.evaluate_observer(your_observer, test_files)
```

## Citation

If you use this benchmark, please cite:
```
@misc{{hyperkkl_bench,
  title={{HyperKKL-Bench: A Comprehensive Benchmark for Neural KKL Observers}},
  author={{Your Name}},
  year={{2025}}
}}
```
"""

def generate_readme(results: Dict) -> str:
    # Format results as table
    table_rows = ["| Test | NRMSE | Conv. Time | Steady Error |",
                  "|------|-------|------------|--------------|"]
    
    for test_name, metrics in results.items():
        row = f"| {test_name} | {metrics['mean']['nrmse']:.4f} | "
        row += f"{metrics['mean']['convergence_time']:.2f} | "
        row += f"{metrics['mean']['steady_state_error']:.4f} |"
        table_rows.append(row)
    
    return README_TEMPLATE.format(results_table="\n".join(table_rows))
```

#### Afternoon: Validation & Release Checklist

```markdown
## Release Checklist

### Data Validation
- [ ] All HDF5 files are readable
- [ ] Metadata is complete for all files
- [ ] No NaN in state trajectories (only in observations for missing data tests)
- [ ] Trajectory lengths match specifications

### Code Validation  
- [ ] All scripts run without errors
- [ ] Unit tests pass for metrics
- [ ] Baseline evaluations complete

### Documentation
- [ ] README complete with usage examples
- [ ] YAML configs documented
- [ ] Metric definitions clear

### Release
- [ ] Create release tag
- [ ] Upload data to Zenodo/HuggingFace
- [ ] Update paper with benchmark link
```

---

## Appendix A: Extended Categories

### A.1 Input-Dynamics Interaction (Non-Autonomous Specific)

```yaml
category: input_dynamics_interaction
description: "Tests specific to non-autonomous systems"

levels:
  - name: Resonant
    description: "Input frequency near system natural frequency"
    implementation: |
      For VDP/Duffing: ω_input ≈ ω_natural
      Expect large amplitude oscillations, potential chaos
    
  - name: Anti-Resonant
    description: "Input frequency far from natural frequency"
    implementation: |
      ω_input >> ω_natural or ω_input << ω_natural
      Expect attenuated response
    
  - name: Amplitude_Small
    description: "Small amplitude forcing (linear regime)"
    amplitude_range: [0.01, 0.1]
    
  - name: Amplitude_Large
    description: "Large amplitude forcing (strongly nonlinear)"
    amplitude_range: [2.0, 10.0]
```

### A.2 Observability Configuration

```yaml
category: observability
description: "Tests different measurement configurations"

levels:
  - name: Full_State
    description: "All states measured"
    C_matrix: identity
    
  - name: Partial_Easy
    description: "Observable subsystem measured"
    example_lorenz: "y = x" (first state only)
    observability_index: 3
    
  - name: Partial_Hard  
    description: "High observability index"
    example: "y = z" (third state only for Lorenz)
    observability_index: 5+
    
  - name: MIMO
    description: "Multiple outputs"
    example: "y = [x, z]"
```

### A.3 Attractor Topology

```yaml
category: attractor_topology
description: "Tests across different dynamical regimes"

levels:
  - name: Fixed_Point
    systems: [damped_oscillator, overdamped_sir]
    
  - name: Limit_Cycle
    systems: [vdp, duffing_undamped]
    
  - name: Quasi_Periodic
    systems: [coupled_oscillators]
    
  - name: Chaotic
    systems: [lorenz, rossler, chua]
    lyapunov_exponent: "> 0"
```

---

## Appendix B: Benchmark Comparison Matrix

| Feature | HyperKKL-Bench | PDEBench | GIFT-Eval | fev-bench | TFB | DynaBench |
|---------|---------------|----------|-----------|-----------|-----|-----------|
| **Domain** | Dynamical Systems | PDEs | Time Series | Time Series | Time Series | PDEs |
| **Task** | State Estimation | Forward/Inverse | Forecasting | Forecasting | Forecasting | Prediction |
| **Systems** | ODEs | PDEs | Real-world | Real-world | Real-world | PDEs |
| **Covariates/Inputs** | ✓ (exogenous) | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Noise Levels** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Missing Data** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **Resolution Tests** | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ |
| **OOD Tests** | ✓ | Limited | Limited | ✗ | Limited | ✗ |
| **Bootstrap CI** | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Pretraining Data** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |

---

## Appendix C: Additional Systems & Signals

### C.1 Suggested Additional Systems

```python
# Future expansion systems

class AdditionalSystems:
    
    @staticmethod
    def fitzhugh_nagumo(a=0.7, b=0.8, tau=12.5, I_ext=0.5):
        """Excitable neuron dynamics"""
        def f(t, x, u):
            v, w = x
            return np.array([
                v - v**3/3 - w + I_ext + u,
                (v + a - b*w) / tau
            ])
        # ...
    
    @staticmethod
    def hindmarsh_rose(a=1, b=3, c=1, d=5, r=0.001, s=4, x_r=-8/5, I=3.2):
        """Bursting neuron dynamics"""
        def f(t, x, u):
            x1, x2, x3 = x
            return np.array([
                x2 - a*x1**3 + b*x1**2 - x3 + I + u,
                c - d*x1**2 - x2,
                r*(s*(x1 - x_r) - x3)
            ])
        # ...
    
    @staticmethod
    def chen_system(a=35, b=3, c=28):
        """Chen attractor (Lorenz family)"""
        def f(t, x, u):
            return np.array([
                a*(x[1] - x[0]) + u,
                (c - a)*x[0] - x[0]*x[2] + c*x[1],
                x[0]*x[1] - b*x[2]
            ])
        # ...
    
    @staticmethod
    def glycolytic_oscillator(k1=100, k2=6, k3=16, k4=100, k5=1.28, k6=12, 
                              kappa=13, q=4, K1=0.52, psi=0.1, N=1):
        """Biochemical oscillator"""
        # Selkov model or full glycolytic pathway
        # ...
```

### C.2 Suggested Additional Signals

```python
class AdditionalSignals:
    
    @staticmethod
    def ornstein_uhlenbeck(t, theta=1.0, mu=0.0, sigma=0.3, x0=0.0):
        """Colored noise (mean-reverting)"""
        dt = t[1] - t[0]
        x = np.zeros(len(t))
        x[0] = x0
        for i in range(1, len(t)):
            dx = theta * (mu - x[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
            x[i] = x[i-1] + dx
        return x
    
    @staticmethod
    def filtered_noise(t, cutoff=1.0, order=4):
        """Butterworth filtered white noise"""
        from scipy.signal import butter, filtfilt
        noise = np.random.randn(len(t))
        b, a = butter(order, cutoff / (0.5 / (t[1]-t[0])), btype='low')
        return filtfilt(b, a, noise)
    
    @staticmethod
    def switching(t, signals, switch_times):
        """Piecewise different signals"""
        output = np.zeros(len(t))
        for i, (t_start, t_end, sig_func) in enumerate(zip(
            [0] + switch_times, switch_times + [t[-1]], signals)):
            mask = (t >= t_start) & (t < t_end)
            output[mask] = sig_func(t[mask])
        return output
```

---

## Appendix D: Advanced Metrics

### D.1 Physics-Informed Metrics

```python
class PhysicsMetrics:
    """Metrics that incorporate physical constraints"""
    
    @staticmethod
    def energy_consistency(x_true, x_pred, energy_func):
        """
        Check if predicted trajectory conserves/dissipates energy correctly
        
        energy_func: Function that computes energy from state
        """
        E_true = np.array([energy_func(x) for x in x_true])
        E_pred = np.array([energy_func(x) for x in x_pred])
        
        # For conservative systems: check energy conservation
        # For dissipative systems: check monotonic decrease
        
        return {
            "energy_rmse": np.sqrt(np.mean((E_true - E_pred)**2)),
            "energy_drift": E_pred[-1] - E_pred[0] - (E_true[-1] - E_true[0])
        }
    
    @staticmethod
    def constraint_violation(x_pred, constraint_func):
        """
        Check constraint violations (e.g., SIR: S+I+R=1)
        
        constraint_func: Returns 0 if constraint satisfied
        """
        violations = np.array([constraint_func(x) for x in x_pred])
        return {
            "max_violation": np.max(np.abs(violations)),
            "mean_violation": np.mean(np.abs(violations))
        }
```

### D.2 Information-Theoretic Metrics

```python
class InformationMetrics:
    """Information-theoretic assessment"""
    
    @staticmethod
    def mutual_information(x_true, x_pred, bins=50):
        """MI between true and predicted trajectories"""
        from sklearn.metrics import mutual_info_score
        
        # Discretize for MI computation
        x_true_disc = np.digitize(x_true.flatten(), 
                                   np.linspace(x_true.min(), x_true.max(), bins))
        x_pred_disc = np.digitize(x_pred.flatten(),
                                   np.linspace(x_pred.min(), x_pred.max(), bins))
        
        return mutual_info_score(x_true_disc, x_pred_disc)
    
    @staticmethod
    def transfer_entropy(x_true, x_pred, lag=1):
        """TE from observations to estimates"""
        # Measures information flow
        # ...
        pass
```

---

## Quick Reference Card

```
╔══════════════════════════════════════════════════════════════════╗
║                    HyperKKL-Bench Quick Reference                 ║
╠══════════════════════════════════════════════════════════════════╣
║ SYSTEMS:  lorenz | rossler | chua | vdp | duffing | sir          ║
║ SIGNALS:  harmonics | square | vdp | duffing | lorenz | chua     ║
╠══════════════════════════════════════════════════════════════════╣
║ CATEGORIES                                                        ║
║ CAT-1: IC         ID | OOD-Near (1.5x) | OOD-Far (3x)            ║
║ CAT-2: Inputs     Simple | Medium | Complex | Adversarial        ║
║ CAT-3: Params     ID | Interpolation | Extrapolation             ║
║ CAT-4: Horizon    Short | Medium | Long | Ultra-Long             ║
║ CAT-5: SNR        Clean | 40dB | 20dB | 10dB                     ║
║ CAT-6: Sampling   1x | 2x | 5x | 10x                             ║
║ CAT-7: Missing    0% | 5% | 20% | 50%                            ║
║ CAT-8: Transfer   Same | Family | Cross-Class                    ║
╠══════════════════════════════════════════════════════════════════╣
║ KEY METRICS                                                       ║
║ • NRMSE (normalized RMSE)                                        ║
║ • Convergence Time (to ε-ball)                                   ║
║ • Steady-State Error (after transient)                           ║
║ • FFT-RMSE (low/mid/high bands)                                  ║
╠══════════════════════════════════════════════════════════════════╣
║ ONE-WEEK PLAN                                                     ║
║ D1-2: Infrastructure + Simulators                                 ║
║ D3-4: Data Generation (CAT 1-4, then 5-8)                        ║
║ D5:   Metrics Implementation                                      ║
║ D6:   Baseline Evaluation                                         ║
║ D7:   Documentation + Release                                     ║
╚══════════════════════════════════════════════════════════════════╝
```