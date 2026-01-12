 # HyperKKL Data Generation Pipeline

Modular, Hydra-based data generation for KKL observer training.

## Structure

```
hyperkkl/
├── configs/
│   ├── config.yaml          # Main config (single source of truth)
│   ├── system/              # Dynamical systems
│   │   ├── duffing.yaml
│   │   ├── lorenz.yaml
│   │   └── vdp.yaml
│   ├── observer/            # KKL observers
│   │   └── kkl_duffing.yaml
│   ├── signal/              # Exogenous signals
│   │   ├── square.yaml
│   │   ├── harmonics.yaml
│   │   └── lorenz.yaml      # System-as-signal
│   ├── solver/              # ODE solvers
│   │   ├── rk4.yaml
│   │   └── dopri5.yaml
│   ├── sampler/             # IC samplers
│   │   ├── lhs.yaml
│   │   └── uniform.yaml
│   └── experiment/          # Full experiment configs
│       └── duffing_lorenz.yaml
├── src/
│   ├── simulators/          # Core simulation code
│   │   ├── systems.py       # Dynamical systems
│   │   ├── observers.py     # KKL observer
│   │   ├── signals.py       # Signal generators
│   │   ├── solvers.py       # ODE solvers
│   │   └── samplers.py      # IC sampling
│   └── data/                # Dataset handling
│       ├── generator.py     # Data generation engine
│       └── dataset.py       # PyTorch datasets
└── generate_data.py         # Main script
```

## Usage

### Basic generation
```bash
python generate_data.py
```

### Override system
```bash
python generate_data.py system=lorenz
```

### Add signals
```bash
python generate_data.py signal@signals.0=lorenz signal@signals.1=harmonics
```

### Use experiment preset
```bash
python generate_data.py +experiment=duffing_lorenz
```

### Multi-run sweep
```bash
python generate_data.py --multirun system=duffing,lorenz,vdp
```

## Key Design Principles

1. **Single source of truth**: Global params in main config, referenced via `${...}`
2. **Composition over inheritance**: Use Hydra's defaults list
3. **Consistent paths**: All interpolations use same prefix (`${simulation.tn}`, not `${data.sim_time.tn}`)
4. **CLI-first**: Any config can be overridden from command line
5. **No manual glob**: Let Hydra handle config discovery

## Adding New Systems

1. Create `configs/system/my_system.yaml`:
```yaml
name: my_system
_target_: src.simulators.systems.MySystem
state_dim: 3
output_dim: 1
observation:
  C: [1.0, 0.0, 0.0]
  observable_index: [0]
coefficients:
  param1: 1.0
ic_space: [[-1, 1], [-1, 1], [-1, 1]]
```

2. Implement in `src/simulators/systems.py`:
```python
class MySystem(System):
    def dynamics(self, t, x, u=None):
        # Your dynamics here
        return dx
```

3. Use it:
```bash
python generate_data.py system=my_system
```
