# Implementation Details

This document describes the architecture, data flow, and key design decisions
of the HyperKKL pipeline.

## Two-Phase Training Pipeline

### Phase 1: Autonomous KKL Observer

**Goal:** Learn the bijective transformation pair (T, T*) between the original
state space and the linear observer space.

**Mathematical formulation:**

Given a dynamical system dx/dt = f(x) with output y = h(x), the KKL observer
operates in transformed coordinates z = T(x) where the dynamics become linear:

```
dz/dt = Mz + Ky
```

M and K are pre-designed observer gain matrices (stable eigenvalues).

**Training procedure (`src/training/phase1.py`):**

1. **Data generation** -- Sample initial conditions via Latin Hypercube
   Sampling, simulate the system forward with RK4, compute z-trajectories
   by integrating the linear observer dynamics backward/forward in time
   until convergence.

2. **Train encoder T: x -> z** -- Feedforward network with PDE-informed loss:
   ```
   L_T = MSE(T(x), z) + lambda * ||dT/dx * f(x) - Mz - Ky||^2 / ||f(x)||^2
   ```
   The PDE weight lambda is linearly warmed up over the first 5 epochs.

3. **Train decoder T*: z -> x** -- Uses pre-trained T to generate (z, x) pairs:
   ```
   L_T* = MSE(T*(T(x)), x)
   ```

**Key modules:**
- `src/data/dataset.py` -- Autonomous dataset with vectorized z-dynamics solver
- `src/simulators/pde_utils.py` -- Jacobian computation and PDE residual loss
- `src/models/normalizer.py` -- Per-variable standardization stored as nn.Module buffers

### Phase 2: Non-Autonomous Extension

**Goal:** Extend the autonomous observer to handle exogenous inputs u(t).

Four methods are implemented in `src/training/phase2.py`:

#### 1. Static HyperKKL (`train_hyperkkl`)

Freezes T and T*. Learns an input encoder and injection network:

```
dz/dt = Mz + Ky + phi(z, encoder(u_window))
```

- **Input encoder:** WindowEncoder (CNN) or LSTMEncoder processes a window
  of recent inputs into a latent code.
- **Injection network:** InputInjectionNet maps (z, latent) to a correction
  term added to the observer dynamics.
- **Loss:** JVP-based dynamics matching -- dz/dt predicted vs actual.

#### 2. Curriculum Learning (`train_curriculum`)

Unfreezes T and T*. Trains with progressively complex data:

| Stage | Input | Epochs | PDE Loss |
|-------|-------|--------|----------|
| 1 | Zero (autonomous) | configurable | Yes |
| 2 | Constant | configurable | No |
| 3 | Sinusoid | configurable | No |
| 4 | All signal types | configurable | No |

PDE loss is only valid for autonomous data (u=0), so stages 2-4 use
autoencoder reconstruction loss only: ||T*(T(x)) - x||^2.

#### 3. Dynamic Weight Modulation (`train_dynamic_hyperkkl`)

Freezes T and T*. A hypernetwork generates time-varying weight residuals:

```
theta_T(t)  = theta*_T  + alpha * delta_T(u_window)
theta_T*(t) = theta*_T* + alpha * delta_T*(u_window)
```

Two architectures:
- **DualHyperNetwork** (window-based) -- CNN encoder -> shared MLP -> low-rank
  weight residuals via U * v^T factorization.
- **ResidualHyperNetwork** (recurrent) -- LSTM/GRU processes input history
  step-by-step, MLP heads generate encoder/decoder deltas.

**Loss:** Full time-varying Luenberger PDE + reconstruction:
```
L = ||dT/dt + dT/dx * f - Mz - Ky|| + ||T*(T(x)) - x||^2
```

where dT/dt is approximated by finite difference between T_t(x) and T_{t-1}(x).

#### 4. Dynamic LoRA (`train_dynamic_lora`)

Like dynamic weight modulation but uses per-layer low-rank adaptation:

- **PerLayerLoRAHyperNetwork** -- RNN processes input, per-layer embeddings
  specialize MLP outputs, generates rank-r factored deltas A_l * B_l for each
  linear layer.
- **Biases are frozen** from Phase 1 (only weights are modulated).
- **Input energy gate:** delta weights are scaled by u_rms, ensuring they
  vanish exactly when u=0 (recovers the autonomous solution architecturally).
- **Physics normalization:** PDE residual divided by ||dx/dt||^2 for
  scale invariance.

## Data Generation

### Autonomous Data (`src/data/dataset.py`)

1. Sample N initial conditions from the state space bounds (LHS).
2. Simulate x-trajectories forward using RK4 (parallelized across CPU cores).
3. Compute z-trajectories via vectorized observer dynamics:
   - **Forward mode:** Random z0, discard transient (first 10%).
   - **Negative-forward mode:** Simulate backward in negative time to converge
     z, then simulate forward (most accurate).
4. Split into regression and physics subsets:
   - `split traj` -- alternating time steps (default).
   - `split set` -- separate IC samples.
   - `no physics` -- single dataset, PDE loss disabled.

### Non-Autonomous Data (`src/data/data_gen.py`)

Multiprocess workers generate (x, y, u, dxdt) samples:

1. Each worker creates a system instance, samples ICs, generates input signals.
2. Vectorized RK4 in torch float64 batch mode for efficiency.
3. Collects sliding-window samples: (x_t, y_t, u_window, u_window_prev, dxdt).
4. Diverged trajectories filtered by FLOAT32_SAFE bound (1e6).
5. Results concatenated across workers and input types.

## Model Architectures

### Base Network (`src/models/nn.py`)

Feedforward MLP with integrated normalizer:
- Input is standardized before the first layer.
- Output is de-standardized after the last layer.
- `mode` attribute switches between 'normal' and 'physics' statistics.

### Hypernetwork Architectures (`src/models/hypernetworks.py`)

| Architecture | Input Processing | Weight Generation | Parameters |
|-------------|-----------------|-------------------|------------|
| WindowEncoder | 1D CNN + AdaptivePool | bias-free linear | ~5K |
| LSTMEncoder | LSTM hidden state | bias-free linear | ~10K |
| DualHyperNetwork | Window -> shared MLP | low-rank U*v^T | ~50K |
| ResidualHyperNetwork | LSTM/GRU step-by-step | MLP heads per mapper | ~100K |
| PerLayerLoRAHyperNetwork | LSTM/GRU + layer embeddings | rank-r A*B per layer | ~80K |

All bias-free output layers ensure zero input produces zero correction,
recovering the autonomous solution exactly.

### Weight Modulation Functions

- `apply_weight_modulation(model, delta)` -- Adds delta to all parameters.
- `apply_weight_modulation_skip_bias(model, delta)` -- Adds delta to weights
  only, biases remain frozen from Phase 1.

Both use `torch.func.functional_call` for differentiable parameter substitution.

## Evaluation (`src/evaluation/`)

### Observer Simulation

Three observer types share the same RK4 integration loop:

1. **Autonomous:** dz/dt = Mz + Ky, decode via T*(z).
2. **Static HyperKKL:** dz/dt = Mz + Ky + phi(z, latent), decode via T*(z).
3. **Dynamic:** dz/dt = Mz + Ky, decode via T*_t(z) with time-varying weights.

All simulations use the same torch float64 RK4 as training data generation
for consistency.

### Metrics

- **RMSE total** -- Root mean square error over the full trajectory.
- **RMSE steady** -- RMSE after settling time (t > 5s), excluding transient.
- **Max error** -- Peak estimation error.
- **SMAPE** -- Symmetric mean absolute percentage error (evaluation script).

Metrics are computed over N trials with LHS-sampled initial conditions.

### Visualization

- **Loss curves** -- Separate encoder/decoder for Phase 1, stage boundaries
  for curriculum.
- **Time-series** -- Per-state true vs estimated overlays.
- **Phase portraits** -- 2D or 3D attractor comparison.
- **Density heatmaps** -- Space-time plots for traffic systems.
- **Boxplots** -- Cross-method metric distributions.
- **Overlay plots** (evaluate.py) -- All methods on one plot with zoom insets.

## Configuration

### System Configs (`src/training/configs.py`)

Each system is registered with:
- `class` / `init_args` -- System constructor.
- `M`, `K` -- Observer gain matrices.
- `x_size`, `z_size` -- State and observer dimensions.
- `num_hidden`, `hidden_size` -- Default network architecture.
- `limits` -- State space bounds for IC sampling.
- `a`, `b`, `N` -- Simulation time interval and step count.
- `natural_inputs` -- Input signal types for evaluation.

### Signal Generators (`src/data/signals.py`)

Nine signal types with train/test_id/test_ood parameter ranges:

| Signal | Parameters | Description |
|--------|-----------|-------------|
| zero | None | Autonomous baseline |
| constant | c in [-1, 1] | DC offset |
| sinusoid | A, w, phi | Harmonic |
| square | A, w | Square wave |
| step | A, t_step | Unit step |
| traffic_rush_hour | amp | AM-modulated ramp |
| traffic_congestion | amp | Elevated ramp |
| traffic_pulse | amp, period | Bursty square |
| traffic_light | amp | Low-amplitude sinusoid |

## Design Decisions

1. **Self-contained package** -- All imports use `hyperkkl.src.*` paths.
   The package does not depend on or modify external code.

2. **Torch batch mode in systems** -- All system `function(t, u, x)` and
   `output(x)` methods detect batched tensors via `torch.is_tensor(x) and
   x.ndim > 1` and return stacked results. This enables vectorized RK4
   across trajectories without Python loops.

3. **Normalizer as nn.Module** -- Statistics stored as registered buffers,
   so they are saved/loaded with `state_dict` and move across devices
   automatically.

4. **Modular Phase 1** -- `load_autonomous()` reconstructs networks from
   checkpoint files using a `_DummyNormalizer` whose buffers are overwritten
   by the saved state_dict. This allows Phase 2 to start from any Phase 1
   checkpoint without re-running data generation or training.

5. **Input energy gate** -- LoRA weight deltas are multiplied by u_rms,
   guaranteeing zero correction when u=0. This is an architectural
   (not learned) guarantee that the autonomous solution is preserved.

6. **Parallel data generation** -- `multiprocessing.Pool` with per-worker
   system instantiation avoids pickling torch models across processes.
   Each worker runs vectorized RK4 in float64 for numerical stability.
