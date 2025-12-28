# HyperKKL Project Structure

A research framework for learning neural observers on non-autonomous dynamical systems using Spectral Curriculum Learning.

---

## Overview

This project implements **HyperKKL**: a method for learning observer mappings (T: x → z and T⁻¹: z → x) that work across varying exogenous inputs u(t). The key contribution is using **Spectral Curriculum Learning** to overcome spectral bias in hypernetworks.

**Core problem:** Given a non-autonomous system dx/dt = f(x, u) with partial observation y = h(x), learn to estimate the full state x from measurements y.

---

## Project Layout

```
hyperkkl/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/                      # Hydra configuration
├── scripts/                      # CLI entry points
├── src/                          # Source code
├── data/                         # Generated data (gitignored)
├── outputs/                      # Training outputs (gitignored)
├── docs/                         # Documentation
├── tests/                        # Unit tests
└── notebooks/                    # Exploration (placeholder)
```

---

## Source Code Structure

```
src/
├── __init__.py
│
├── simulation/                   # Simulation engine
│   ├── __init__.py
│   ├── base.py                   # Abstract base classes
│   ├── systems.py                # Dynamical systems
│   ├── wrappers.py               # Wrappers for torchdiffeq/scipy
│   ├── samplers.py               # Initial condition samplers
│   ├── signals.py                # Exogenous input generators
│   ├── noise.py                  # Noise models
│   └── simulator.py              # Orchestration
│
├── observers/                    # State estimators (INFERENCE ONLY)
│   ├── __init__.py
│   ├── base.py                   # AbstractObserver
│   ├── kkl_inference.py          # KKL runtime observer
│   ├── high_gain.py              # PLACEHOLDER
│   └── kalman.py                 # PLACEHOLDER
│
├── data/                         # Data pipeline
│   ├── __init__.py
│   ├── generation.py             # Dataset generation
│   ├── datasets.py               # PyTorch datasets (Map + Iterable)
│   └── curriculum.py             # Curriculum BatchSampler + scheduler
│
├── models/                       # Neural networks
│   ├── __init__.py
│   ├── functional.py             # Stateless T, T⁻¹ definitions
│   ├── networks.py               # MLP, Transformer (standard)
│   ├── hypernetwork.py           # Weight generation
│   ├── spectral.py               # FFT feature extractors
│   └── xlstm.py                  # PLACEHOLDER
│
├── learners/                     # Training objectives
│   ├── __init__.py
│   ├── base.py                   # AbstractLearner
│   ├── supervised.py             # Supervised learning
│   ├── pinn.py                   # Physics-informed
│   ├── curriculum.py             # Curriculum wrapper
│   ├── neural_ode.py             # PLACEHOLDER
│   └── rl/                       # PLACEHOLDER
│       └── ...
│
├── evaluation/                   # Metrics and analysis
│   ├── __init__.py
│   ├── metrics.py                # Loss metrics
│   ├── visualization.py          # Plotting
│   ├── uncertainty.py            # PLACEHOLDER
│   └── statistics.py             # PLACEHOLDER
│
└── utils/
    ├── __init__.py               # Seeding, logging helpers
    └── callbacks.py              # Callback implementations
```

---

## Key Design Decisions (Reviewer Feedback Incorporated)

### 1. ODE Solvers: Use Libraries, Not Custom

**Problem:** Custom solvers introduce subtle bugs and are slower than optimized libraries.

**Solution:** `simulation/wrappers.py` wraps existing libraries:

```python
# simulation/wrappers.py

from scipy.integrate import solve_ivp
import torchdiffeq

class ScipyIntegrator:
    """Wrapper for scipy.integrate.solve_ivp (data generation)."""
    
    def __init__(self, method: str = 'RK45', rtol: float = 1e-6):
        self.method = method
        self.rtol = rtol
    
    def integrate(self, f, x0, t_span, t_eval, args=None):
        """Integrate ODE using scipy."""
        sol = solve_ivp(f, t_span, x0, method=self.method, t_eval=t_eval, args=args)
        return sol.y.T  # Shape: (T, state_dim)


class TorchdiffeqIntegrator:
    """Wrapper for torchdiffeq (differentiable, for Neural ODE)."""
    
    def __init__(self, method: str = 'rk4'):
        self.method = method
    
    def integrate(self, f, x0, t_span):
        """Integrate ODE with gradient support."""
        return torchdiffeq.odeint(f, x0, t_span, method=self.method)
```

| Use Case | Library | Wrapper |
|----------|---------|---------|
| Data generation | scipy.integrate.solve_ivp | `ScipyIntegrator` |
| Neural ODE training | torchdiffeq | `TorchdiffeqIntegrator` |

---

### 2. Curriculum Pipeline: IterableDataset + BatchSampler

**Problem:** Pre-generating all levels causes memory issues. On-the-fly generation has RNG issues with multiprocessing.

**Solution:** Two options:

#### Option A: IterableDataset (streaming)

```python
# data/datasets.py

class StreamingTrajectoryDataset(IterableDataset):
    """Infinite streaming of trajectories at specified level."""
    
    def __init__(self, simulator: Simulator, level: int, seed: int):
        self.simulator = simulator
        self.level = level
        self.seed = seed
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = np.random.default_rng(seed)
        
        while True:
            traj = self.simulator.run(level=self.level, rng=rng)
            for point in traj:
                yield point
```

#### Option B: CurriculumBatchSampler (pre-generated)

```python
# data/curriculum.py

class CurriculumBatchSampler(BatchSampler):
    """Constructs batches with configurable level mixing.
    
    Example: 50% Level 1 + 50% Level 2 per batch.
    """
    
    def __init__(
        self,
        level_indices: Dict[int, List[int]],  # {level: [indices]}
        level_weights: Dict[int, float],       # {level: proportion}
        batch_size: int,
        drop_last: bool = False,
    ):
        self.level_indices = level_indices
        self.level_weights = level_weights
        self.batch_size = batch_size
    
    def __iter__(self):
        # Construct batches with mixed levels
        for _ in range(len(self)):
            batch = []
            for level, weight in self.level_weights.items():
                n = int(self.batch_size * weight)
                batch.extend(random.sample(self.level_indices[level], n))
            yield batch
    
    def update_weights(self, new_weights: Dict[int, float]):
        """Update mixing ratios (called by scheduler)."""
        self.level_weights = new_weights
```

**Recommendation:** Start with Option B (simpler). Switch to Option A if memory becomes an issue.

---

### 3. Training vs Inference Separation in Observers

**Problem:** KKL has two distinct modes that shouldn't be mixed:
- **Training:** Optimize T, T⁻¹ networks (supervised/PINN loss)
- **Inference:** Run observer dynamics dz/dt = Az + By, then x̂ = T⁻¹(z)

**Solution:** 

| Module | Responsibility |
|--------|----------------|
| `models/functional.py` | Define T, T⁻¹ as stateless functions |
| `learners/*.py` | Training logic (loss computation, optimization) |
| `observers/kkl_inference.py` | Runtime observer (loads trained model) |

```python
# observers/kkl_inference.py

class KKLInferenceObserver(AbstractObserver):
    """Runtime KKL observer. Loads trained T⁻¹ model."""
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        T_inverse: nn.Module,  # Pre-trained, frozen
        dt: float = 0.05,
    ):
        self.A = A
        self.B = B
        self.T_inverse = T_inverse.eval()
        self.dt = dt
        self.z = None
    
    @torch.no_grad()
    def estimate(self, y: np.ndarray, u: np.ndarray = None) -> np.ndarray:
        """Runtime state estimation."""
        # Update z dynamics
        dz = self.A @ self.z + self.B @ y
        self.z = self.z + self.dt * dz
        
        # Map back to state space
        z_tensor = torch.from_numpy(self.z).float()
        x_hat = self.T_inverse(z_tensor).numpy()
        return x_hat
    
    @classmethod
    def from_checkpoint(cls, path: str, A: np.ndarray, B: np.ndarray):
        """Load from trained checkpoint."""
        checkpoint = torch.load(path)
        T_inverse = ...  # Reconstruct model
        T_inverse.load_state_dict(checkpoint['T_inverse'])
        return cls(A, B, T_inverse)
```

**Training does NOT use the Observer class.** Training treats it as supervised regression or PINN.

---

### 4. Hypernetwork: Functional/Stateless Design

**Problem:** Standard nn.Module weight hacking is messy for hypernetworks.

**Solution:** Stateless functional models using `torch.func.functional_call`:

```python
# models/functional.py

import torch
from torch import nn
from torch.func import functional_call

class FunctionalMLP:
    """Stateless MLP definition. Weights passed as argument."""
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
        self.layer_sizes = layer_sizes
        self.activation = getattr(F, activation)
        
        # Define parameter shapes (no actual parameters)
        self.param_shapes = {}
        for i in range(len(layer_sizes) - 1):
            self.param_shapes[f'layers.{i}.weight'] = (layer_sizes[i+1], layer_sizes[i])
            self.param_shapes[f'layers.{i}.bias'] = (layer_sizes[i+1],)
    
    def __call__(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """Forward pass with external parameters."""
        for i in range(len(self.layer_sizes) - 1):
            x = F.linear(x, params[f'layers.{i}.weight'], params[f'layers.{i}.bias'])
            if i < len(self.layer_sizes) - 2:
                x = self.activation(x)
        return x
    
    def num_params(self) -> int:
        """Total number of parameters."""
        return sum(np.prod(s) for s in self.param_shapes.values())


# models/hypernetwork.py

class HyperNetwork(nn.Module):
    """Generates weights for T/T⁻¹ based on input signal."""
    
    def __init__(
        self,
        spectral_encoder: nn.Module,
        target_param_shapes: Dict[str, Tuple],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.spectral_encoder = spectral_encoder
        self.target_param_shapes = target_param_shapes
        
        # Decoder: embedding → flattened weights
        total_params = sum(np.prod(s) for s in target_param_shapes.values())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_params),
        )
    
    def forward(self, u: Tensor) -> Dict[str, Tensor]:
        """Generate parameters conditioned on input signal.
        
        Args:
            u: Input signal, shape (batch, T) or spectral features
            
        Returns:
            params: Dict of parameter tensors matching target_param_shapes
        """
        # Extract spectral features
        embedding = self.spectral_encoder(u)
        
        # Generate flattened weights
        flat_params = self.decoder(embedding)
        
        # Reshape to parameter dict
        params = {}
        offset = 0
        for name, shape in self.target_param_shapes.items():
            size = np.prod(shape)
            params[name] = flat_params[..., offset:offset+size].view(*shape)
            offset += size
        
        return params


# models/spectral.py

class SpectralEncoder(nn.Module):
    """Extract spectral features from input signal u(t)."""
    
    def __init__(self, n_fft: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.encoder = nn.Sequential(
            nn.Linear(n_fft // 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, u: Tensor) -> Tensor:
        """Extract spectral embedding.
        
        Args:
            u: Input signal, shape (batch, T)
            
        Returns:
            embedding: Shape (batch, hidden_dim)
        """
        # Compute magnitude spectrum
        spectrum = torch.fft.rfft(u, n=self.n_fft)
        magnitude = torch.abs(spectrum)
        
        # Encode
        return self.encoder(magnitude)
```

**Usage in training:**

```python
# learners/supervised.py

class SupervisedLearner(AbstractLearner):
    
    def compute_loss(self, batch):
        x, z, u = batch['x'], batch['z'], batch['u']
        
        # Generate weights conditioned on u
        params = self.hypernetwork(u)
        
        # Forward pass with generated weights
        z_pred = self.T_functional(x, params['T'])
        x_pred = self.T_inv_functional(z, params['T_inv'])
        
        # Reconstruction loss
        loss_T = F.mse_loss(z_pred, z)
        loss_T_inv = F.mse_loss(x_pred, x)
        
        return {'loss_T': loss_T, 'loss_T_inv': loss_T_inv}
```

---

### 5. Configuration Structure

```
configs/
├── config.yaml                  # Main defaults
│
├── experiment/                  # Complete experiments
│   ├── baseline_duffing.yaml
│   ├── hyperkkl_curriculum.yaml
│   └── ablation_lambda.yaml
│
├── system/                      # System configs
│   ├── duffing.yaml
│   ├── lorenz.yaml
│   └── vdp.yaml
│
├── learner/                     # Learner configs
│   ├── supervised.yaml
│   ├── pinn.yaml
│   └── curriculum.yaml
│
├── model/                       # Model configs
│   ├── mlp.yaml
│   ├── hypernetwork_lora.yaml
│   └── hypernetwork_none.yaml
│
├── data/                        # Data generation configs
│   ├── training.yaml
│   └── benchmark.yaml
│
├── callbacks/                   # Callback configs (NEW)
│   ├── wandb.yaml
│   ├── checkpoint.yaml
│   ├── curriculum.yaml
│   └── none.yaml               # For debugging (no callbacks)
│
└── sweep/                       # HPO configs (NEW)
    ├── lambda_sweep.yaml
    └── lora_rank_sweep.yaml
```

**Usage:**

```bash
# Disable W&B for debugging
python scripts/train.py callbacks=none

# Run hyperparameter sweep
python scripts/sweep.py --config configs/sweep/lambda_sweep.yaml
```

---

### 6. Scripts

```
scripts/
├── train.py               # Main training
├── evaluate.py            # Benchmark evaluation
├── generate_data.py       # Generate training data
├── generate_benchmark.py  # Generate benchmark splits
├── sweep.py               # Hyperparameter optimization (Optuna + Hydra)
└── visualize.py           # Visualization
```

---

### 7. Critical Tests

```
tests/
├── __init__.py
├── test_kkl_condition.py      # PRIORITY: Verify KKL math
├── test_systems.py
├── test_observers.py
├── test_data.py
└── test_learners.py
```

**`test_kkl_condition.py`** — Must pass before any training:

```python
"""Test KKL condition: L_f T(x) = A T(x) + B h(x)

If this fails, all training is useless.
"""

import torch
import numpy as np

def test_kkl_condition_analytical():
    """Test with known analytical T for simple system."""
    
    # Simple linear system: dx/dt = Ax, y = Cx
    A_sys = np.array([[0, 1], [-1, 0]])  # Harmonic oscillator
    C = np.array([[1, 0]])                # y = x1
    
    # Known T for this system (observer canonical form)
    # T(x) = [y, ẏ] = [x1, x2]
    # So T = I (identity for this case)
    
    # Observer: A_obs = [[0, 1], [-1, 0]], B = [[0], [1]]
    A_obs = np.array([[0, 1], [-1, 0]])
    B_obs = np.array([[0], [1]])
    
    # Test: L_f T(x) = A T(x) + B h(x)
    # L_f T(x) = dT/dx * f(x) = I * Ax = Ax
    # A T(x) + B h(x) = A*x + B*Cx
    
    x_test = np.array([1.0, 0.5])
    
    lhs = A_sys @ x_test  # L_f T(x) = Ax (since T=I)
    rhs = A_obs @ x_test + B_obs @ (C @ x_test)
    
    np.testing.assert_allclose(lhs, rhs, rtol=1e-6)


def test_kkl_condition_learned():
    """Test learned T satisfies KKL condition approximately."""
    
    # Load trained model
    # ...
    
    # Sample test points
    x_samples = ...
    
    # Compute LHS: L_f T(x) = dT/dx * f(x)
    # Use autograd for dT/dx
    x_tensor = torch.tensor(x_samples, requires_grad=True)
    T_x = model.T(x_tensor)
    
    # Compute Jacobian
    jacobian = torch.autograd.functional.jacobian(model.T, x_tensor)
    f_x = system.f(x_samples)
    lhs = jacobian @ f_x
    
    # Compute RHS: A T(x) + B h(x)
    rhs = A @ T_x + B @ system.h(x_samples)
    
    # Should be approximately equal
    error = torch.mean((lhs - rhs) ** 2)
    assert error < 0.01, f"KKL condition violated: MSE = {error}"
```

---

## Module Specifications

### `simulation/` — Simulation Engine

| File | Purpose |
|------|---------|
| `base.py` | AbstractSystem, AbstractSampler, AbstractNoise |
| `systems.py` | Duffing, Lorenz, VDP, Rossler, Chua |
| `wrappers.py` | ScipyIntegrator, TorchdiffeqIntegrator |
| `samplers.py` | UniformSampler, LHSSampler, GridSampler |
| `signals.py` | SignalGenerator (levels 0-3) |
| `noise.py` | GaussianNoise, WhiteNoise |
| `simulator.py` | Orchestrates all components |

---

### `observers/` — State Estimators (Inference Only)

| File | Purpose | Status |
|------|---------|--------|
| `base.py` | AbstractObserver interface | ✅ |
| `kkl_inference.py` | Runtime KKL observer | ✅ |
| `high_gain.py` | High-gain observer | PLACEHOLDER |
| `kalman.py` | EKF, UKF | PLACEHOLDER |

**Note:** Observers are for **inference only**. Training logic lives in `learners/`.

---

### `data/` — Data Pipeline

| File | Purpose |
|------|---------|
| `generation.py` | DataGenerator class |
| `datasets.py` | PointwiseDataset, SequentialDataset, StreamingDataset |
| `curriculum.py` | CurriculumBatchSampler, CurriculumScheduler |

---

### `models/` — Neural Networks

| File | Purpose |
|------|---------|
| `functional.py` | Stateless T, T⁻¹ (weights as args) |
| `networks.py` | Standard MLP, Transformer |
| `hypernetwork.py` | Weight generation |
| `spectral.py` | FFT feature extraction |
| `xlstm.py` | PLACEHOLDER |

**Key pattern:** `functional.py` defines stateless models. `hypernetwork.py` generates weights. Forward pass uses `functional_call` or manual weight injection.

---

### `learners/` — Training Objectives

| File | Loss | Status |
|------|------|--------|
| `supervised.py` | ‖T(x) - z‖² + ‖T⁻¹(z) - x‖² | ✅ |
| `pinn.py` | + λ‖L_f T - AT - Bh‖² | ✅ |
| `curriculum.py` | Wraps learner + scheduler | ✅ |
| `neural_ode.py` | Adjoint-based | PLACEHOLDER |
| `rl/` | Policy gradient | PLACEHOLDER |

---

### `evaluation/` — Metrics

| File | Purpose | Status |
|------|---------|--------|
| `metrics.py` | MSE, MAE, spectral error, divergence time | ✅ |
| `visualization.py` | Trajectory plots, FFT, phase portraits | ✅ |
| `uncertainty.py` | MC Dropout, Ensemble | PLACEHOLDER |
| `statistics.py` | Permutation test, CI | PLACEHOLDER |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA GENERATION                              │
│                   (uses scipy.solve_ivp)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SignalGenerator ──→ u(t)                                        │
│         │                                                        │
│         ▼                                                        │
│  Sampler ──→ x₀                                                  │
│         │                                                        │
│         ▼                                                        │
│  System.f(x, u) + ScipyIntegrator ──→ x(t)                      │
│         │                                                        │
│         ▼                                                        │
│  System.h(x) + Noise ──→ y(t)                                   │
│         │                                                        │
│         ▼                                                        │
│  Save as HuggingFace Dataset                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                  │
│              (models + learners, NO observer class)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Dataset ──→ DataLoader + CurriculumBatchSampler                │
│         │                                                        │
│         ▼                                                        │
│  Batch{x, z, y, u}                                              │
│         │                                                        │
│         ▼                                                        │
│  SpectralEncoder(u) ──→ embedding                               │
│         │                                                        │
│         ▼                                                        │
│  HyperNetwork(embedding) ──→ params                             │
│         │                                                        │
│         ▼                                                        │
│  FunctionalMLP(x, params['T']) ──→ z_pred                       │
│  FunctionalMLP(z, params['T_inv']) ──→ x_pred                   │
│         │                                                        │
│         ▼                                                        │
│  Learner.compute_loss() ──→ loss                                │
│         │                                                        │
│         ▼                                                        │
│  Backward + Optimize + Callbacks                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE                                  │
│               (observers/kkl_inference.py)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Load trained T⁻¹ from checkpoint                               │
│         │                                                        │
│         ▼                                                        │
│  KKLInferenceObserver(A, B, T_inverse)                          │
│         │                                                        │
│         ▼                                                        │
│  y(t), u(t) ──→ observer.estimate() ──→ x̂(t)                    │
│                     │                                            │
│                     ├── Update z: dz/dt = Az + By                │
│                     └── Estimate: x̂ = T⁻¹(z)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Placeholder Modules

| Module | Purpose | Priority |
|--------|---------|----------|
| `observers/high_gain.py` | High-gain observer | Low |
| `observers/kalman.py` | EKF, UKF | Low |
| `models/xlstm.py` | Extended LSTM | Medium |
| `learners/neural_ode.py` | Adjoint-based training | Medium |
| `learners/rl/` | RL-based learning | Low |
| `evaluation/uncertainty.py` | MC Dropout, Ensemble | Medium |
| `evaluation/statistics.py` | Statistical tests | Medium |
| `notebooks/` | Exploration | As needed |

---

## Dependencies

```toml
[project]
name = "hyperkkl"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "torch>=2.0",
    "scipy>=1.10",
    "torchdiffeq>=0.2",         # Differentiable ODE solvers
    "hydra-core>=1.3",
    "wandb>=0.15",
    "matplotlib>=3.7",
    "datasets>=2.14",            # HuggingFace datasets
    "optuna>=3.0",               # Hyperparameter optimization
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "ruff>=0.1",
]
```

---

## Summary of Reviewer Feedback Incorporated

| Feedback | Resolution |
|----------|------------|
| Don't write custom ODE solvers | `simulation/wrappers.py` wraps scipy/torchdiffeq |
| Curriculum data bottleneck | `CurriculumBatchSampler` + optional `StreamingDataset` |
| Separate training vs inference | `observers/` is inference only; training in `learners/` |
| Functional hypernetwork design | `models/functional.py` with stateless forward pass |
| Callbacks in config | `configs/callbacks/` directory |
| Sweep script | `scripts/sweep.py` with Optuna + Hydra |
| KKL condition test | `tests/test_kkl_condition.py` as priority |

---

## References

- KKL Observer: Kazantzis & Kravaris (1998), Luenberger (1964)
- Spectral Bias: Rahaman et al. (2019)
- Hypernetworks: Ha et al. (2016)
- Physics-Informed Neural Networks: Raissi et al. (2019)
- torchdiffeq: Chen et al. (2018)