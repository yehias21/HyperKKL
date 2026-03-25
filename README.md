# HyperKKL

Neural KKL (Kazantzis-Kravaris-Luenberger) observers for non-autonomous dynamical systems, extended with hypernetwork-based parameter adaptation.

## Overview

HyperKKL learns a coordinate transformation pair (T, T*) that maps a nonlinear system into a linear observer space where state estimation is straightforward:

- **Phase 1 (Autonomous):** Train forward map T: x -> z and inverse map T*: z -> x using physics-informed neural networks. The PDE constraint dT/dx * f(x) = Mz + Ky is enforced as a regularizer.
- **Phase 2 (Non-Autonomous):** Extend the static maps to handle exogenous inputs u(t) via four methods:
  - **Static HyperKKL** -- learn an injection term phi(z, u) added to the observer dynamics
  - **Curriculum Learning** -- fine-tune T/T* with progressively complex inputs
  - **Dynamic Weight Modulation** -- hypernetwork generates time-varying weight residuals for T/T*
  - **Dynamic LoRA** -- per-layer low-rank adaptation of T/T* weights via recurrent hypernetwork

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
# Train Phase 1 independently
python -m hyperkkl.scripts.train_phase1 --system duffing --epochs 20

# Run Phase 2 using saved Phase 1 (skips retraining)
python -m hyperkkl.scripts.run_pipeline --system duffing --method dynamic_lora \
    --phase1_dir results/phase1/duffing

# Full pipeline (Phase 1 + Phase 2 together)
python -m hyperkkl.scripts.run_pipeline --system duffing --method all

# Evaluate saved checkpoints
python -m hyperkkl.scripts.evaluate --results_dir results/hyperkkl/duffing/v1

# All systems in parallel
python -m hyperkkl.scripts.run_pipeline --all_systems --method all --parallel
```

## Project Structure

```
hyperkkl/
├── configs/                     # Hydra-compatible YAML configs
│   ├── config.yaml              # Main configuration
│   ├── system/                  # System definitions (duffing, lorenz, vdp)
│   ├── observer/                # Observer configs (gains, dimensions)
│   └── signal/                  # Input signal configs
├── docs/
│   ├── BENCHMARK.md             # Benchmark specification
│   └── IMPLEMENTATION.md        # Architecture and implementation details
├── scripts/
│   ├── run_pipeline.py          # Full training pipeline (Phase 1 + 2)
│   ├── train_phase1.py          # Standalone Phase 1 training
│   └── evaluate.py              # Post-training checkpoint evaluation
└── src/
    ├── data/                    # Data generation
    │   ├── dataset.py           # Autonomous KKL dataset (Phase 1)
    │   ├── data_gen.py          # Parallel non-autonomous data (Phase 2)
    │   └── signals.py           # Input signal generators (9 types)
    ├── models/                  # Neural network architectures
    │   ├── nn.py                # Feedforward network with normalizer
    │   ├── hypernetworks.py     # Encoders + 4 hypernetwork architectures
    │   └── normalizer.py        # Dataset statistics for standardization
    ├── simulators/              # Dynamical systems and solvers
    │   ├── systems.py           # 6 systems with torch batch mode
    │   ├── utils.py             # RK4 integration, sampling, convergence
    │   └── pde_utils.py         # PDE constraint loss (Jacobian-based)
    ├── training/                # Training logic
    │   ├── phase1.py            # Autonomous training + checkpoint loading
    │   ├── phase2.py            # 4 non-autonomous training methods
    │   └── configs.py           # System configuration registry
    └── evaluation/              # Evaluation and visualization
        ├── evaluate.py          # Observer simulation + metrics (RMSE, SMAPE)
        └── plot.py              # Loss curves, time-series, attractors, density
```

## Modular Phase 1

Phase 1 training is the most expensive step. It can be run once and reused:

```bash
# Train Phase 1 for Lorenz (saves T_encoder.pt + T_inv_decoder.pt)
python -m hyperkkl.scripts.train_phase1 --system lorenz --epochs 50 \
    --out_dir results/phase1/lorenz

# Later: run any Phase 2 method without retraining Phase 1
python -m hyperkkl.scripts.run_pipeline --system lorenz --method dynamic_lstm \
    --phase1_dir results/phase1/lorenz

python -m hyperkkl.scripts.run_pipeline --system lorenz --method curriculum \
    --phase1_dir results/phase1/lorenz
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--system` | duffing | System to train on |
| `--method` | all | Training method (autonomous, curriculum, static_lstm, dynamic_lstm, dynamic_lora, all) |
| `--phase1_dir` | None | Path to pre-trained Phase 1 checkpoints (skips Phase 1) |
| `--epochs_phase1` | 20 | Phase 1 training epochs |
| `--epochs_phase2` | 100 | Phase 2 training epochs |
| `--batch_size` | 2049 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--out_dir` | ./results/hyperkkl | Output directory |
| `--parallel` | False | Run multiple systems in parallel |

## Output Structure

Each run produces a versioned directory:

```
results/hyperkkl/{system}/v{N}/
├── T_encoder.pt           # Phase 1: forward map T
├── T_inv_decoder.pt       # Phase 1: inverse map T*
├── {method}.pt            # Phase 2: method checkpoint
├── results.json           # Evaluation metrics
├── loss_histories.json    # Training loss curves
└── plots/
    ├── {method}_loss.png          # Loss history
    ├── boxplot_{metric}.png       # Cross-method comparison
    └── {signal}/
        ├── {method}_timeseries.png
        └── {method}_attractor.png
```
