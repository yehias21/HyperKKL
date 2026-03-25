"""System configurations: M, K matrices, limits, and default hyperparameters."""

import numpy as np
from hyperkkl.src.simulators.systems import RevDuff, VdP, Lorenz, Rossler, HighwayTraffic, FitzHughNagumo


def get_system_config(system_name: str) -> dict:
    """Get system configuration including M, K matrices."""

    if system_name == 'duffing':
        return {
            'name': 'duffing',
            'class': RevDuff,
            'init_args': {'zdim': 5, 'add_noise': True},
            'x_size': 2,
            'y_size': 1,
            'z_size': 5,
            'num_hidden': 3,
            'hidden_size': 150,
            'M': np.diag([-1, -2, -3, -4, -5]),
            'K': np.array([[1, 1, 1, 1, 1]]).T,
            'limits': np.array([[-1, 1], [-1, 1]]),
            'a': 0, 'b': 50, 'N': 1000,
            'natural_inputs': ['zero', 'sinusoid', 'square', 'constant'],
        }

    elif system_name == 'vdp':
        return {
            'name': 'vdp',
            'class': VdP,
            'init_args': {'zdim': 5, 'mu': 1.0, 'add_noise': True},
            'x_size': 2,
            'y_size': 1,
            'z_size': 5,
            'num_hidden': 2,
            'hidden_size': 350,
            'M': np.diag([-1, -2, -3, -4, -5]),
            'K': np.array([[1, 1, 1, 1, 1]]).T,
            'limits': np.array([[-3, 3], [-3, 3]]),
            'a': 0, 'b': 50, 'N': 1000,
            'natural_inputs': ['zero', 'sinusoid', 'square', 'constant'],
        }

    elif system_name == 'lorenz':
        M = np.array([
            [-2.12116939,  2.73877907, -0.75338041,  3.13947511,  1.00581224, -5.34548426, -5.34544528],
            [-5.91811463, -1.99160105, -0.26953991,  0.45654273, -0.56750459, -3.79023002, -3.00480715],
            [-1.27814843, -2.08375366, -2.60781906, -4.11186517,  0.33087248,  0.08191241, -2.81998688],
            [ 0.69080313,  2.00687664,  3.02478991, -3.53231915, -0.89704804, -5.19153027,  3.16796666],
            [-1.16566857, -0.53979687, -0.30300461, -3.79753229, -2.7939018,  -0.30176078, -2.61480264],
            [ 2.97899741,  3.97265572, -2.04951338, -1.08716448, -0.98741222, -4.7047881,  -1.38218644],
            [ 0.95887295,  0.43056828, -1.39564903, -3.18608794,  3.74388666, -3.88509855, -3.07468653]
        ])
        return {
            'name': 'lorenz',
            'class': Lorenz,
            'init_args': {'rho': 28, 'sigma': 10, 'beta': 8/3, 'add_noise': True},
            'x_size': 3,
            'y_size': 1,
            'z_size': 7,
            'num_hidden': 2,
            'hidden_size': 350,
            'M': M,
            'K': np.ones([7, 1]),
            'limits': np.array([[-20, 20], [-30, 30], [0, 50]]),
            'a': 0, 'b': 50, 'N': 1000,
            'natural_inputs': ['zero', 'sinusoid', 'square', 'constant'],
        }

    elif system_name == 'rossler':
        M = np.array([
            [-2.12116939,  2.73877907, -0.75338041,  3.13947511,  1.00581224, -5.34548426, -5.34544528],
            [-5.91811463, -1.99160105, -0.26953991,  0.45654273, -0.56750459, -3.79023002, -3.00480715],
            [-1.27814843, -2.08375366, -2.60781906, -4.11186517,  0.33087248,  0.08191241, -2.81998688],
            [ 0.69080313,  2.00687664,  3.02478991, -3.53231915, -0.89704804, -5.19153027,  3.16796666],
            [-1.16566857, -0.53979687, -0.30300461, -3.79753229, -2.7939018,  -0.30176078, -2.61480264],
            [ 2.97899741,  3.97265572, -2.04951338, -1.08716448, -0.98741222, -4.7047881,  -1.38218644],
            [ 0.95887295,  0.43056828, -1.39564903, -3.18608794,  3.74388666, -3.88509855, -3.07468653]
        ])
        return {
            'name': 'rossler',
            'class': Rossler,
            'init_args': {'a': 0.1, 'b': 0.1, 'c': 14, 'add_noise': True, 'noise_std': 0.01, 'noise_measurement_std': 0.01},
            'x_size': 3,
            'y_size': 1,
            'z_size': 7,
            'num_hidden': 3,
            'hidden_size': 250,
            'M': M,
            'K': np.ones([7, 1]),
            'limits': np.array([[-10, 10], [-10, 10], [-10, 10]]),
            'a': 0, 'b': 50, 'N': 1000,
            'natural_inputs': ['zero', 'sinusoid', 'square', 'constant'],
        }

    elif system_name == 'fitzhugh_nagumo':
        return {
            'name': 'fitzhugh_nagumo',
            'class': FitzHughNagumo,
            'init_args': {'zdim': 5, 'a': 0.7, 'b': 0.8, 'epsilon': 0.08, 'I_ext': 0.5, 'add_noise': True},
            'x_size': 2,
            'y_size': 1,
            'z_size': 5,
            'num_hidden': 3,
            'hidden_size': 150,
            'M': np.diag([-1, -2, -3, -4, -5]),
            'K': np.array([[1, 1, 1, 1, 1]]).T,
            'limits': np.array([[-2.5, 2.5], [-1.5, 1.5]]),
            'a': 0, 'b': 100, 'N': 2000,
            'natural_inputs': ['zero', 'sinusoid', 'square', 'constant'],
        }

    elif system_name == 'highway_traffic':
        return {
            'name': 'highway_traffic',
            'class': HighwayTraffic,
            'init_args': {'add_noise': True, 'noise_std': 0.001, 'noise_measurement_std': 0.001},
            'x_size': 5,
            'y_size': 1,
            'z_size': 11,
            'num_hidden': 3,
            'hidden_size': 200,
            'M': np.diag([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0]),
            'K': np.ones([11, 1]),
            'limits': np.array([
                [0.01, 0.10],   # cell 1 density (free-flow regime)
                [0.01, 0.10],   # cell 2 density
                [0.01, 0.10],   # cell 3 density
                [0.01, 0.10],   # cell 4 density
                [0.01, 0.10],   # cell 5 density
            ]),
            'a': 0, 'b': 100, 'N': 2000,
            'natural_inputs': ['zero', 'traffic_rush_hour', 'traffic_congestion', 'traffic_pulse', 'traffic_light'],
        }

    else:
        raise ValueError(f"Unknown system: {system_name}")
