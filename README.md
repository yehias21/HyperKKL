# HyperKKL

**Learning KKL Observers for Non-Autonomous Nonlinear Systems via Hypernetwork-Based Input Conditioning**

<p align="center">
  <img src="assets/Teaser_spaces.png" alt="HyperKKL teaser" width="85%">
</p>

HyperKKL extends Kazantzis–Kravaris/Luenberger (KKL) observers to controlled (non-autonomous) nonlinear systems. A hypernetwork conditions the observer on the exogenous input history `u(t)`, yielding transformation maps that track the time-varying geometry of the system's attractor — something static autonomous maps provably cannot do.

## Why HyperKKL

Classical neural KKL observers are trained under `u ≡ 0` and treat inputs as unmodeled disturbances. Under forcing, this induces a persistent bounded estimation error. HyperKKL converts that disturbance into structured information via two complementary strategies:

- **HyperKKL<sub>obs</sub>** (`augmented`) — keeps the pretrained encoder/decoder frozen and injects an input-dependent correction `Φ(ẑ, u)` into the latent observer dynamics. Lightweight; best in 13/16 benchmark settings.
- **HyperKKL<sub>dyn</sub>** (`lora` / `full`) — a hypernetwork generates input-conditioned weight perturbations `θ̃(t), η̃(t)` for the encoder and decoder, producing genuinely time-varying transformation maps as prescribed by the non-autonomous KKL theory.

Both variants collapse exactly to the autonomous observer when `u ≡ 0` (by construction — bias-free projection / zero-initialized LoRA deltas).

## Pipelines

<p align="center">
  <img src="assets/training_pipeline.png" alt="Training pipeline" width="45%">
  &nbsp;
  <img src="assets/inference_pipeline.png" alt="Inference pipeline" width="45%">
  <br>
  <em>Left: Phase 1 pretrains autonomous T/T*; Phase 2 trains the hypernetwork with frozen base weights. Right: at inference, one GRU pass over <code>u[t-ω, t]</code> conditions the observer — no retraining per input.</em>
</p>

## Method Overview

| Phase | What is trained | Loss |
|-------|----------------|------|
| **Phase 1 — Autonomous** | Base encoder `T: x→z` and decoder `T*: z→x` (MLPs). | `‖ẑ − z‖² + ν·‖∂T/∂x·f(x) − Az − By‖²` |
| **Phase 2 — `augmented`** | GRU + injection MLP `Φ_ϖ`. Base maps frozen. | `‖ż_obs − ż_true‖²` |
| **Phase 2 — `lora` / `full`** | Hypernetwork `H_ψ` producing weight deltas. Base weights frozen. | `‖x − T̂*(T̂(x))‖² + ν·‖PDE residual‖²` |
| `curriculum` (baseline) | Fine-tunes base maps on staged non-autonomous data. | Reconstruction only. |

Observer matrices: `A = −diag(1, 2, …, n_z)`, `B = 𝟙`, with `n_z = n_y(2n_x + 1)`.

## Results

SMAPE (%) over 100 trials, four input regimes per system. Lower is better; **bold** = best; ↓ = improvement over Autonomous.

| Method | Duffing Zero | Const | Sin | Sqr | Van der Pol Zero | Const | Sin | Sqr |
|---|---|---|---|---|---|---|---|---|
| Autonomous | 19.3 | 74.6 | 37.1 | 40.7 | 17.7 | 32.6 | 20.2 | 22.2 |
| Curriculum | 33.0 | 80.5 | 67.3 | 71.9 | 51.4 | 76.3 | 81.2 | 81.4 |
| **HyperKKL<sub>obs</sub>** | **5.6**↓ | **21.1**↓ | **22.7**↓ | **21.3**↓ | 5.3↓ | **26.0**↓ | **15.5**↓ | **17.9**↓ |
| HyperKKL<sub>dyn</sub> | 8.2↓ | 57.3↓ | 29.7↓ | 30.7↓ | **5.0**↓ | 32.7 | 18.8↓ | 20.8↓ |

| Method | Rössler Zero | Const | Sin | Sqr | FHN Zero | Const | Sin | Sqr |
|---|---|---|---|---|---|---|---|---|
| Autonomous | 64.9 | 66.9 | 64.8 | 64.9 | **9.8** | 61.4 | 41.3 | 45.0 |
| Curriculum | 98.0 | 103.1 | 103.0 | 103.3 | 59.6 | 67.0 | 55.6 | 54.9 |
| HyperKKL<sub>obs</sub> | 64.4↓ | 68.0 | 66.5 | 68.8 | 9.9 | **42.1**↓ | **30.9**↓ | **32.5**↓ |
| **HyperKKL<sub>dyn</sub>** | **52.7**↓ | **55.4**↓ | **53.5**↓ | **54.0**↓ | 17.0 | 54.6↓ | 40.8↓ | 44.5↓ |

Average **29% SMAPE reduction** across all non-zero input regimes.

<p align="center">
  <img src="assets/smape_duffing.png" width="48%">
  <img src="assets/smape_vdp.png" width="48%"><br>
  <img src="assets/smape_rossler.png" width="48%">
  <img src="assets/smape_fhn.png" width="48%">
</p>

## Supported Systems

| System | `n_x` | `n_z` | Description |
|--------|-----|-------------|-------------|
| Duffing | 2 | 5 | Reversed Duffing oscillator |
| Van der Pol | 2 | 5 | Nonlinear oscillator (μ=1) |
| Lorenz | 3 | 7 | Chaotic attractor (σ=10, ρ=28) |
| Rössler | 3 | 7 | Chaotic oscillator (a=0.1, b=0.1, c=14) |
| FitzHugh–Nagumo | 2 | 5 | Neuron model |
| Highway Traffic | 5 | 11 | 5-cell Greenshields traffic model |

## Quick Start

```bash
# Full pipeline (Phase 1 + Phase 2, all methods)
python -m scripts.run_pipeline --system duffing --method all

# Single method
python -m scripts.run_pipeline --system lorenz --method lora

# All systems in parallel across GPUs
python -m scripts.run_pipeline --all_systems --parallel

# Evaluate saved checkpoints
python -m scripts.evaluate --results_dir results/duffing/v1

# Hyperparameter search
python -m scripts.sweep --system duffing --method lora --n_trials 10
```

Environment: `conda env create -f kkl.yml`.

## Project Structure

```
HyperKKL/
├── configs/
│   ├── default.yaml                # Default hyperparameters
│   └── systems/                    # Per-system configs (A, B, limits, etc.)
├── src/
│   ├── config.py                   # Typed dataclass configs + YAML loading
│   ├── logger.py                   # Unified TensorBoard + W&B logger
│   ├── systems.py                  # 6 dynamical systems (batch torch + numpy)
│   ├── signals.py                  # 9 input signal generators
│   ├── dataset.py                  # Unified data generation (Phase 1 + 2)
│   ├── models.py                   # MLPs, encoders, hypernetworks, weight ops
│   ├── training.py                 # Unified training (all phases / methods)
│   ├── evaluation.py               # Unified observer simulation + metrics
│   └── plotting.py                 # Loss, time-series, attractor, density plots
├── scripts/
│   ├── run_pipeline.py             # Main training pipeline
│   ├── evaluate.py                 # Checkpoint evaluation with overlay plots
│   └── sweep.py                    # Hyperparameter search (grid / random)
├── assets/                         # Figures used in the paper and README
└── kkl.yml                         # Conda environment
```

### Design Principles

- Each `src/` module is independently runnable (e.g. `python -m src.dataset --system duffing`).
- Unified data path — training and evaluation use the same `generate_phase2_data` with signal modes `train` / `id` / `ood`.
- One `simulate_observer` covers every method; one `train_dynamic` covers all dynamic variants.
- Every run saves `config.yaml` alongside checkpoints for reproducibility.
- Structured logging via `ExperimentLogger` (TensorBoard and/or W&B).

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--system` | `duffing` | `duffing`, `vdp`, `lorenz`, `rossler`, `fhn`, `highway_traffic` |
| `--method` | `all` | `autonomous`, `curriculum`, `augmented`, `full`, `lora`, `all` |
| `--epochs_phase1` | 20 | Phase 1 training epochs |
| `--epochs_phase2` | 30 | Phase 2 training epochs |
| `--batch_size` | 2049 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--out_dir` | `./results` | Output directory |
| `--parallel` | False | Run systems in parallel across GPUs |
| `--use_wandb` | False | Enable Weights & Biases logging |

The Phase-2 encoder type (LSTM or GRU) is set via `phase2.encoder_type` in the config, not in the method name.

## Output Structure

```
results/{system}/v{N}/
├── config.yaml                # Full experiment config (reproducible)
├── T_encoder.pt               # Phase 1: forward map T
├── T_inv_decoder.pt           # Phase 1: inverse map T*
├── {method}.pt                # Phase 2: method checkpoint
├── results.json               # Evaluation metrics per signal type
├── loss_histories.json        # Training loss curves
├── logs/tb/                   # TensorBoard logs
└── plots/
    ├── {method}_loss.png
    └── {signal}/
        ├── {method}_timeseries.png
        └── {method}_attractor.png
```

## Theoretical Guarantee (informal)

The asymptotic state-estimation error satisfies

> `limsup ‖ξ(t)‖ ≤ ε_dec + ℓ_dec · (ε_pde · κ/λ + ε_enc)`

where `ε_enc, ε_dec` are encoder/decoder approximation errors, `ε_pde` is the PDE residual, `ℓ_dec` is the decoder's Lipschitz constant, and `κ, λ` come from the Hurwitz matrix `A`. HyperKKL<sub>obs</sub> keeps `ℓ_dec` at its pretrained value; HyperKKL<sub>dyn</sub> lowers `ε_pde` at the cost of potentially inflating `ℓ_dec` — which explains why `obs` often wins empirically.

## Citation

```bibtex
@inproceedings{shaaban2026hyperkkl,
  title     = {HyperKKL: Learning KKL Observers for Non-Autonomous Nonlinear
               Systems via Hypernetwork-Based Input Conditioning},
  author    = {Shaaban, Yahia Salaheldin and Sayed, Abdelrahman Sayed
               and Niazi, M. Umar B. and Johansson, Karl Henrik},
  booktitle = {IEEE Conference on Decision and Control (CDC)},
  year      = {2026}
}
```
