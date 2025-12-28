# HyperKKL

A Python-based framework for implementing and training neural networks, with a focus on dynamical systems.

## Project Structure

```
├── src/                    # Core source code
│   ├── data_loader/       # Dataset handling and preparation
│   ├── model/            # Neural network models and hypernetworks
│   ├── simulators/       # System simulators and solvers
│   ├── trainer/          # Training logic
│   └── utils/            # Helper functions and utilities
│
├── runners/               # Execution scripts and configurations
│   ├── config/           # Configuration files
│   │   ├── data/        # Data configurations
│   │   │   ├── exo_input/    # External input configs
│   │   │   ├── observer/     # Observer configs
│   │   │   └── system/       # System configs
│   │   ├── model/       # Model configurations
│   │   └── trainer/     # Training configurations
│   ├── run.py           # Main execution script
│   ├── train.py         # Training script
│   └── test.py          # Testing script
│
├── data/                 # Data storage
├── docs/                 # Documentation
│   └── img/             # Documentation images
├── envs/                 # Environment configuration
├── notebooks/           # Jupyter notebooks
├── results/             # Output results and visualizations
└── test/                # Test files and configurations
```

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f envs/kkl.yml
conda activate kkl
```

2. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training

To train a model using the default configuration:

```bash
python Scripts/train.py
```

### Testing

To run tests:

```bash
python Scripts/test.py
```

### Data Generation

To generate datasets:

```bash
python Scripts/generate_dataset.py
```

## Configuration

The project uses YAML configuration files located in `runners/config/`:

- `baseline.yaml`: Main configuration file
- `data generation.yaml`: Data generation settings
- System-specific configurations in `config/data/system/`
- Model configurations in `config/model/`
- Training configurations in `config/trainer/`

## Supported Systems

The framework supports various dynamical systems:
- Duffing
- Lorenz
- Rossler
- Chua
- Van der Pol (VDP)
- SIR

## Documentation

- `THEORY.md`: Theoretical background
- `IMPLEMENTATION.md`: Implementation details
- `docs/`: Additional documentation and visualizations

## Results

Results and visualizations are stored in `results/visualizations/` with timestamps for each experiment.

## Development

For development and testing, Jupyter notebooks are available in the `notebooks/` directory.

## License

[Add your license information here]

## Model
The aim of HyperKKL is to take advantage of information provided in the model:
- **Temporal aspect of the data itself:** data is trained in random sample fashion (aka assume no correlation between the points) mathematically presentation as 
P(xt+1|xt ) = 0
- **Online error**: output error produced during inference: y_hat-y error signal
- **Input signal (u)** 
- **Training and sampling techniques**: Employing curriculum learning and autoregressive (teacher forcing) has been shown to enhance the model accuracy.

### Extended Idea

- We want to generalize for any kind of Input, Idea proposed:
  - If input is very complex in nature to be learned, take the spectral decomposition of it (e.g Fourier basis)
- The problem with the input is in the encoder and not the decoder, as the decoder maps from Z space to X space. The encoder is the one that needs to be more complex (have a dynamic weights), because it deals with the input.
- Use the differential equation as a hard constraint not soft constraint --> Can I prove that the hypernetwork guarantee such thing?
### Extended Obstacles and solutions
- **Problem:** What about the sampling of the signals, different sampling rates, will need retraining. \
**Solution:** Fix a maximum sampling, as in reality the sampling rate of the sensor/oscillator is physically limited.
### Baseline



## Data generation and creation

### Data Generation 

- The goal is to generate a dataset, a mapping between the system_state and the observer_state (X, Z)
- These points are sampled from multiple trajectories of the system, note that each trajectory represents a new distribution of 


### Code Design
## Evaluation technique
### Uncertainty Quantification
### Hypothesis testing - Permutation test - Ablation study
## Implementation Timeline
### Steps
- [X] Generate the data as proposed
- [X] Develop the Hyper-network model
- [ ] Finish the training loop
- [ ] Simulation
- [ ] Evaluation
- [ ] Iteration
### Refinement 
- [ ] Test code for each code part + visualization technique
- [ ] Refactoring