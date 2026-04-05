# HyperKKL

Neural KKL (Kazantzis-Kravaris-Luenberger) observers for non-autonomous dynamical systems, extended with hypernetwork-based parameter adaptation.

## Overview

HyperKKL learns a coordinate transformation pair (T, T*) that maps a nonlinear system into a linear observer space where state estimation is straightforward:

- **Phase 1 (Autonomous):** Train forward map T: x -> z and inverse map T*: z -> x using physics-informed neural networks. The PDE constraint dT/dx * f(x) = Mz + Ky is enforced as a regularizer.
- **Phase 2 (Non-Autonomous):** Extend the maps to handle exogenous inputs u(t) via four methods:
  - **Augmented Observer** (`augmented`) -- learn a correction phi(z, u) injected into the observer ODE
  - **Curriculum** -- fine-tune T/T* with progressively complex inputs
  - **Dynamic Full** (`full`) -- hypernetwork generates time-varying weight residuals for T/T*
  - **Dynamic LoRA** (`lora`) -- per-layer low-rank adaptation of T/T* weights via recurrent hypernetwork

  Encoder type (LSTM or GRU) is a config parameter (`phase2.encoder_type`), not part of the method name.

## Supported Systems

| System | Dim | Observer Dim | Description |
|--------|-----|-------------|-------------|
| Duffing | 2 | 5 | Reversed Duffing oscillator |
| Van der Pol | 2 | 5 | Nonlinear oscillator (mu=1) |
| Lorenz | 3 | 7 | Chaotic attractor (sigma=10, rho=28) |
| Rossler | 3 | 7 | Chaotic oscillator (a=0.1, b=0.1, c=14) |
| FitzHugh-Nagumo | 2 | 5 | Neuron model |
| Highway Traffic | 5 | 11 | 5-cell Greenshields traffic model |

## Quick Start

```bash
# Full pipeline (Phase 1 + Phase 2)
python -m scripts.run_pipeline --system duffing --method all

# Single method
python -m scripts.run_pipeline --system lorenz --method lora

# All systems in parallel
python -m scripts.run_pipeline --all_systems --parallel

# Evaluate saved checkpoints
python -m scripts.evaluate --results_dir results/duffing/v1

# Hyperparameter search
python -m scripts.sweep --system duffing --method lora --n_trials 10
```

## Project Structure

```
HyperKKl/
├── configs/
│   ├── default.yaml                # Default hyperparameters
│   └── systems/                    # Per-system configs (M, K, limits, etc.)
│       ├── duffing.yaml
│       ├── vdp.yaml
│       ├── lorenz.yaml
│       ├── rossler.yaml
│       ├── fhn.yaml
│       └── highway_traffic.yaml
├── src/
│   ├── config.py                   # Typed dataclass configs + YAML loading
│   ├── logger.py                   # Unified TensorBoard + W&B logger
│   ├── systems.py                  # 6 dynamical systems (batch torch + numpy)
│   ├── signals.py                  # 9 input signal generators
│   ├── dataset.py                  # Unified data generation (Phase 1 + 2)
│   ├── models.py                   # NN, encoders, hypernetworks, weight ops
│   ├── training.py                 # Unified training (Phase 1 + 2, all methods)
│   ├── evaluation.py               # Unified observer simulation + metrics
│   └── plotting.py                 # Loss, time-series, attractor, density plots
├── scripts/
│   ├── run_pipeline.py             # Main training pipeline
│   ├── evaluate.py                 # Checkpoint evaluation with overlay plots
│   └── sweep.py                    # Hyperparameter search (grid / random)
└── kkl.yml                         # Conda environment
```

### Design Principles

- **Each `src/` module is independently runnable** (`python -m src.dataset --system duffing`)
- **Unified data path**: training and evaluation use the same `generate_phase2_data` function, differing only in signal mode (`train` / `id` / `ood`)
- **No redundancy**: one `simulate_observer` handles all method types; one `train_dynamic` handles all dynamic variants
- **Configs saved with results**: every run saves `config.yaml` alongside checkpoints for reproducibility
- **Structured logging**: TensorBoard and/or W&B via `ExperimentLogger`

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--system` | duffing | System name (duffing, vdp, lorenz, rossler, fhn, highway_traffic) |
| `--method` | all | Method (autonomous, curriculum, augmented, full, lora, all) |
| `--epochs_phase1` | 20 | Phase 1 training epochs |
| `--epochs_phase2` | 30 | Phase 2 training epochs |
| `--batch_size` | 2049 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--out_dir` | ./results | Output directory |
| `--parallel` | False | Run systems in parallel across GPUs |
| `--use_wandb` | False | Enable Weights & Biases logging |

## Output Structure

Each run creates a versioned, self-contained directory:

```
results/{system}/v{N}/
├── config.yaml                # Full experiment config (reproducible)
├── T_encoder.pt               # Phase 1: forward map T
├── T_inv_decoder.pt           # Phase 1: inverse map T*
├── {method}.pt                # Phase 2: method checkpoint
├── results.json               # Evaluation metrics per signal type
├── loss_histories.json        # Training loss curves
├── logs/                      # TensorBoard logs
│   └── tb/
└── plots/
    ├── {method}_loss.png
    └── {signal}/
        ├── {method}_timeseries.png
        └── {method}_attractor.png
```

## Hyperparameter Search

```bash
# Random search (default space)
python -m scripts.sweep --system duffing --method lora --n_trials 10

# Grid search with custom space
python -m scripts.sweep --system lorenz --method full --search grid \
    --sweep_config configs/my_sweep.yaml
```

Results include a leaderboard and per-trial configs for full reproducibility.
