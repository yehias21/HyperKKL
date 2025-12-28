# HyperKKL Project Structure

Complete project organization for the HyperKKL research framework.

## TL;DR

```
hyperkkl/
├── configs/          # Hydra configuration files
├── scripts/          # CLI entry points (train, evaluate, generate)
├── src/              # Main source code
│   ├── dynamics/     # Dynamical systems (Duffing, Lorenz, ...)
│   ├── observers/    # Observer structures (KKL, + placeholders)
│   ├── simulation/   # ODE solvers, samplers, noise
│   ├── data/         # Data generation, datasets, curriculum
│   ├── models/       # Networks, hypernetworks, lifting maps
│   ├── learners/     # Training methods (supervised, PINN, curriculum, + placeholders)
│   ├── evaluation/   # Metrics, visualization, + placeholders
│   └── utils/        # Callbacks, logging, reproducibility
├── data/             # Generated data (gitignored)
├── outputs/          # Training outputs (gitignored)
├── docs/             # Documentation
├── tests/            # Unit tests (placeholder)
└── notebooks/        # Exploration notebooks (placeholder)
```

---

## Full Structure

```
hyperkkl/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── experiment/
│   │   ├── baseline_duffing.yaml
│   │   ├── hyperkkl_curriculum.yaml
│   │   └── ablation_lambda.yaml
│   ├── system/
│   │   ├── duffing.yaml
│   │   ├── lorenz.yaml
│   │   ├── vdp.yaml
│   │   ├── rossler.yaml
│   │   └── chua.yaml
│   ├── learner/
│   │   ├── supervised.yaml
│   │   ├── pinn.yaml
│   │   └── curriculum.yaml
│   ├── model/
│   │   ├── mlp.yaml
│   │   ├── transformer.yaml
│   │   ├── hypernetwork_lora.yaml
│   │   ├── hypernetwork_full.yaml
│   │   └── hypernetwork_none.yaml
│   └── data/
│       ├── training.yaml
│       └── benchmark.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── generate_data.py
│   └── visualize.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── systems.py
│   │
│   ├── observers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── kkl.py
│   │   ├── high_gain.py              # PLACEHOLDER
│   │   └── kalman.py                 # PLACEHOLDER
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── solvers.py
│   │   ├── samplers.py
│   │   └── noise.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generation.py
│   │   ├── datasets.py
│   │   └── curriculum.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── networks.py
│   │   ├── xlstm.py                  # PLACEHOLDER
│   │   ├── hypernetwork.py
│   │   └── lifting.py
│   │
│   ├── learners/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── supervised.py
│   │   ├── pinn.py
│   │   ├── curriculum.py
│   │   ├── neural_ode.py             # PLACEHOLDER
│   │   └── rl/                       # PLACEHOLDER
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── env.py
│   │       └── policies.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── uncertainty.py            # PLACEHOLDER
│   │   └── statistics.py             # PLACEHOLDER
│   │
│   └── utils/
│       ├── __init__.py
│       └── callbacks.py
│
├── data/
│   ├── .gitkeep
│   ├── training/
│   │   └── {system}/
│   │       └── {level}/
│   └── benchmark/
│       └── {system}/
│           ├── in_distribution/
│           ├── ood_ic/
│           ├── ood_params/
│           ├── ood_input/
│           ├── ood_horizon/
│           ├── ood_sampling/
│           └── cross_system/
│
├── outputs/
│   └── {date}/{time}/
│       ├── .hydra/
│       ├── checkpoints/
│       └── wandb/
│
├── docs/
│   ├── BENCHMARK.md
│   └── PROJECT_STRUCTURE.md
│
├── tests/                            # PLACEHOLDER
│   ├── __init__.py
│   ├── test_dynamics.py
│   ├── test_observers.py
│   ├── test_data.py
│   └── test_learners.py
│
└── notebooks/                        # PLACEHOLDER
    └── README.md
```

---

## Placeholder Modules

These modules are scaffolded but not implemented. Each contains:

```python
"""
TODO: Implement [module name]

Planned features:
- Feature 1
- Feature 2

References:
- Paper/link
"""

raise NotImplementedError("This module is not yet implemented")
```

| Module | Purpose | Priority |
|--------|---------|----------|
| `observers/high_gain.py` | High-gain observer | Low |
| `observers/kalman.py` | EKF, UKF | Low |
| `models/xlstm.py` | Extended LSTM network | Medium |
| `learners/neural_ode.py` | Adjoint-based training | Medium |
| `learners/rl/` | RL-based observer learning | Low |
| `evaluation/uncertainty.py` | MC Dropout, Ensemble | Medium |
| `evaluation/statistics.py` | Permutation test, CI | Medium |
| `tests/` | Unit tests | After MVP |
| `notebooks/` | Exploration notebooks | As needed |

---

## Directory Details

### `configs/` - Hydra Configuration

All configuration is managed via Hydra YAML files.

| Directory | Purpose |
|-----------|---------|
| `config.yaml` | Main defaults, references other configs |
| `experiment/` | Complete self-contained experiments (for paper) |
| `system/` | Dynamical system parameters |
| `learner/` | Training method configs |
| `model/` | Network architecture configs |
| `data/` | Data generation configs |

**Usage:**
```bash
# Use defaults
python scripts/train.py

# Override system
python scripts/train.py system=lorenz

# Run complete experiment
python scripts/train.py experiment=hyperkkl_curriculum

# Ablation sweep
python scripts/train.py learner.physics_weight=0.1,1.0,10.0 --multirun
```

---

### `scripts/` - Entry Points

CLI entry points for all operations.

| Script | Purpose | Example |
|--------|---------|---------|
| `train.py` | Main training | `python scripts/train.py experiment=X` |
| `evaluate.py` | Run benchmark | `python scripts/evaluate.py --checkpoint X` |
| `generate_data.py` | Generate training/benchmark data | `python scripts/generate_data.py --system duffing` |
| `visualize.py` | Plot systems, results, tables | `python scripts/visualize.py --system lorenz` |

---

### `src/` - Source Code

#### `src/dynamics/` - Dynamical Systems

| File | Purpose |
|------|---------|
| `base.py` | `AbstractSystem` base class |
| `systems.py` | Duffing, Lorenz, VDP, Rossler, Chua (all systems) |

**AbstractSystem interface:**
```python
class AbstractSystem(ABC):
    @abstractmethod
    def f(self, x: np.ndarray, t: float, u: float = None) -> np.ndarray:
        """dx/dt = f(x, t, u)"""
        
    @property
    @abstractmethod
    def state_dim(self) -> int: ...
    
    @property
    @abstractmethod
    def output_dim(self) -> int: ...
    
    def h(self, x: np.ndarray) -> np.ndarray:
        """y = h(x) - output/measurement function"""
```

---

#### `src/observers/` - Observer Structures

| File | Purpose | Status |
|------|---------|--------|
| `base.py` | `AbstractObserver` base class | ✅ |
| `kkl.py` | KKL observer (linear z-dynamics) | ✅ |
| `high_gain.py` | High-gain observer | PLACEHOLDER |
| `kalman.py` | EKF, UKF | PLACEHOLDER |

**AbstractObserver interface:**
```python
class AbstractObserver(ABC):
    @abstractmethod
    def g(self, z: np.ndarray, t: float, y: np.ndarray, u: float = None) -> np.ndarray:
        """dz/dt = g(z, t, y, u)"""
```

**Implemented observers:**

| Observer | Structure | Parameters |
|----------|-----------|------------|
| KKL | dz/dt = Az + By | A (Hurwitz), B |
| HyperKKL | dz/dt = A(u)z + B(u)y | Hypernetwork generates A, B |

---

#### `src/simulation/` - ODE Solving

| File | Purpose |
|------|---------|
| `solvers.py` | RK4, RK45 adaptive, Euler |
| `samplers.py` | Initial condition sampling (LHS, uniform, grid) |
| `noise.py` | Process noise, measurement noise |

---

#### `src/data/` - Data Pipeline

| File | Purpose |
|------|---------|
| `generation.py` | Trajectory simulation + input signal generation |
| `datasets.py` | PointwiseDataset + SequentialDataset |
| `curriculum.py` | CurriculumSampler + level scheduler |

**Difficulty levels (spectral curriculum):**

| Level | Name | Signal Type |
|-------|------|-------------|
| 0 | DC | u(t) = constant |
| 1 | Low Frequency | Single sine, ω < 0.5 Hz |
| 2 | High Frequency | Multi-sine, chirp |
| 3 | Chaos/Broadband | Chaotic system, PRBS |

**Dataset types:**

| Dataset | Returns | Use Case |
|---------|---------|----------|
| `PointwiseDataset` | Single (x, z, y, u, t) | MLP training |
| `SequentialDataset` | Sequence [(x, z, y, u, t), ...] | Transformer/xLSTM |

---

#### `src/models/` - Neural Networks

| File | Purpose | Status |
|------|---------|--------|
| `networks.py` | MLP + Transformer | ✅ |
| `xlstm.py` | Extended LSTM | PLACEHOLDER |
| `hypernetwork.py` | Encoder + all decoders (LoRA, Full, Chunked) | ✅ |
| `lifting.py` | T, T⁻¹ wrapper (forward/inverse maps) | ✅ |

**Hypernetwork structure:**
```
u(t) → [Encoder] → embedding → [Decoder] → weights for T, T⁻¹
```

---

#### `src/learners/` - Training Methods

| File | Method | Status |
|------|--------|--------|
| `base.py` | AbstractLearner | ✅ |
| `supervised.py` | \|\|T(x) - z\|\|² + \|\|T⁻¹(z) - x\|\|² | ✅ |
| `pinn.py` | + \|\|∂T/∂x · f(x) - Az - By\|\|² | ✅ |
| `curriculum.py` | Wraps other learners + level scheduler | ✅ |
| `neural_ode.py` | Adjoint-based training | PLACEHOLDER |
| `rl/` | Policy gradient (PPO/SAC) | PLACEHOLDER |

**AbstractLearner interface:**
```python
class AbstractLearner(ABC):
    @abstractmethod
    def train_step(self, batch) -> dict:
        """Single training step, returns losses"""
        
    @abstractmethod
    def validate(self, val_loader) -> dict:
        """Validation loop"""
```

---

#### `src/evaluation/` - Metrics & Analysis

| File | Purpose | Status |
|------|---------|--------|
| `metrics.py` | MSE, MAE, spectral error, trajectory divergence | ✅ |
| `visualization.py` | Trajectory plots, FFT, loss curves | ✅ |
| `uncertainty.py` | MC Dropout, Ensemble methods | PLACEHOLDER |
| `statistics.py` | Permutation test, bootstrap CI | PLACEHOLDER |

---

#### `src/utils/` - Utilities

| File | Purpose |
|------|---------|
| `__init__.py` | Seeding, logging helpers |
| `callbacks.py` | W&B logging, checkpointing, curriculum progression |

---

### `data/` - Generated Data

```
data/
├── training/
│   └── {system}/
│       ├── level_0/
│       ├── level_1/
│       ├── level_2/
│       └── level_3/
│
└── benchmark/
    └── {system}/
        ├── in_distribution/
        │   ├── standard/
        │   └── long_horizon/
        ├── ood_ic/
        │   ├── mild/
        │   └── extreme/
        ├── ood_params/
        ├── ood_input/
        ├── ood_horizon/
        ├── ood_sampling/
        └── cross_system/
```

**Storage format:** HuggingFace `datasets` (local + cloud sync)

---

### `outputs/` - Training Outputs

Hydra automatically organizes outputs by date/time:

```
outputs/
└── 2025-01-15/
    └── 14-30-22/
        ├── .hydra/
        │   ├── config.yaml      # Resolved config
        │   ├── hydra.yaml
        │   └── overrides.yaml
        ├── checkpoints/
        │   ├── best.pt
        │   └── last.pt
        ├── wandb/
        └── train.log
```

---

## Extension Points

### Adding a New System

1. Add class to `src/dynamics/systems.py`:
```python
class NewSystem(AbstractSystem):
    ...
```

2. Create config `configs/system/new_system.yaml`

3. Generate benchmark: `python scripts/generate_data.py --system new_system --benchmark`

---

### Adding a New Observer

1. Add class to `src/observers/`:
```python
class NewObserver(AbstractObserver):
    ...
```

---

### Adding a New Learner

1. Add class to `src/learners/`:
```python
class NewLearner(AbstractLearner):
    ...
```

2. Create config `configs/learner/new_learner.yaml`

---

### Adding a New Hypernetwork Decoder

1. Add decoder class to `src/models/hypernetwork.py`

2. Create config `configs/model/hypernetwork_new.yaml`

---

## Common Workflows

### Training a Model

```bash
# Basic training
python scripts/train.py system=duffing learner=pinn

# Full experiment
python scripts/train.py experiment=hyperkkl_curriculum

# With overrides
python scripts/train.py experiment=hyperkkl_curriculum learner.physics_weight=10.0
```

### Running Ablations

```bash
# Single axis sweep
python scripts/train.py experiment=baseline learner.physics_weight=0.1,1.0,10.0 --multirun
```

### Generating Data

```bash
# Training data
python scripts/generate_data.py --system duffing --levels 0,1,2,3

# Benchmark
python scripts/generate_data.py --system duffing --benchmark
```

### Evaluation

```bash
# Run benchmark
python scripts/evaluate.py --checkpoint outputs/2025-01-15/14-30-22/checkpoints/best.pt

# Visualize
python scripts/visualize.py --run outputs/2025-01-15/14-30-22
```

---

## Environment Setup

### Dependencies

```bash
# Create environment
conda create -n hyperkkl python=3.10
conda activate hyperkkl

# Install package
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

### Environment Variables

Create `.env` from `.env.example`:

```bash
# W&B
WANDB_API_KEY=your_key
WANDB_PROJECT=hyperkkl
WANDB_ENTITY=your_username

# HuggingFace
HF_TOKEN=your_token
```

---

## File Descriptions

### Root Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, quick start |
| `pyproject.toml` | Package config, dependencies |
| `.env.example` | Template for environment variables |
| `.gitignore` | Git ignore patterns |

### Documentation

| File | Purpose |
|------|---------|
| `docs/BENCHMARK.md` | Benchmark specification |
| `docs/PROJECT_STRUCTURE.md` | This file |

---

## Git Ignore Patterns

```gitignore
# Data
data/
!data/.gitkeep

# Outputs
outputs/
wandb/

# Python
__pycache__/
*.egg-info/
.eggs/
dist/
build/

# Environment
.env
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Jupyter
.ipynb_checkpoints/
```