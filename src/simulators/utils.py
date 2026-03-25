"""Utility functions for the KKL observer pipeline.

Provides numerical integration (RK4), KKL observer simulation,
initial-condition sampling on circles and spheres, and helpers for
computing negative-time bounds and covering numbers.
"""

import math
from typing import Callable, List, Optional, Tuple

import numpy as np


def rk4(
    f: Callable,
    a: float,
    b: float,
    n_steps: int,
    x0: np.ndarray,
    inputs: Optional[Callable] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Integrate an ODE with the classical fourth-order Runge-Kutta method.

    Parameters
    ----------
    f : Callable
        Right-hand side ``f(t, u, x)`` of the ODE.
    a : float
        Start time.
    b : float
        End time.
    n_steps : int
        Number of integration steps.
    x0 : np.ndarray
        Initial state vector.
    inputs : Callable, optional
        Function returning the exogenous input at a given time.

    Returns
    -------
    states : list[np.ndarray]
        State trajectory including the initial condition.
    t : np.ndarray
        Time grid.
    """
    h = (b - a) / n_steps
    t = np.arange(a, b + h, h)
    v = x0.copy()
    states: List[np.ndarray] = [v.copy()]
    u: np.ndarray | float = 0

    for i in range(n_steps):
        time = t[i]
        if inputs is not None:
            u = np.array(inputs(t[-1]))

        k1 = f(time, u, v)
        k2 = f(time, u, v + h / 2 * k1)
        k3 = f(time, u, v + h / 2 * k2)
        k4 = f(time, u, v + h * k3)

        v = v + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        states.append(v.tolist())

    return states, t


def kkl_observer_data(
    m_matrix: np.ndarray,
    k_matrix: np.ndarray,
    y: np.ndarray,
    a: float,
    b: float,
    ic: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Simulate the KKL observer dynamics using RK4 integration.

    Parameters
    ----------
    m_matrix : np.ndarray
        Observer state matrix *M*.
    k_matrix : np.ndarray
        Observer gain matrix *K*.
    y : np.ndarray
        Output trajectories used to drive the observer.
    a : float
        Start time.
    b : float
        End time.
    ic : np.ndarray
        Initial conditions for each trajectory.
    n_steps : int
        Number of integration steps.

    Returns
    -------
    np.ndarray
        Simulated observer state trajectories.
    """
    scalar_y = False
    data: List[np.ndarray] = []
    size_z = m_matrix.shape[0]
    h = (b - a) / n_steps

    if y.ndim > 2:
        f = lambda yi, z: m_matrix @ z + k_matrix @ np.expand_dims(yi, 1)
    else:
        f = lambda yi, z: m_matrix @ z + k_matrix * yi
        scalar_y = True

    for output, init in zip(y, ic):
        x = [init.tolist()]
        v = np.array(x).T

        truncated_output = np.delete(output, 0) if scalar_y else output[1:, :]

        for i in truncated_output:
            k1 = f(i, v)
            k2 = f(i, v + h / 2 * k1)
            k3 = f(i, v + h / 2 * k2)
            k4 = f(i, v + h * k3)

            v = v + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            x.append(np.reshape(v.T, size_z).tolist())

        data.append(np.array(x))

    return np.array(data)


def calc_neg_t(
    m_matrix: np.ndarray,
    z_max: float,
    epsilon: float,
) -> float:
    """Compute the negative-time bound for observer convergence.

    Parameters
    ----------
    m_matrix : np.ndarray
        Observer state matrix *M*.
    z_max : float
        Maximum norm of the observer state.
    epsilon : float
        Desired accuracy threshold.

    Returns
    -------
    float
        Negative-time bound *t*.
    """
    eigenvalues, eigenvectors = np.linalg.eig(m_matrix)
    min_ev = np.min(np.abs(np.real(eigenvalues)))
    kappa = np.linalg.cond(eigenvectors)
    s = np.sqrt(z_max * m_matrix.shape[0])
    t = (1 / min_ev) * np.log(epsilon / (kappa * s))
    return float(t)


def sample_circular(
    delta: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Sample initial conditions from concentric circles.

    Parameters
    ----------
    delta : np.ndarray
        Array of radial offsets for each circle.
    num_samples : int
        Number of points per circle.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(delta), num_samples, 1, 2)``.
    """
    ic = []
    for distance in delta:
        r = distance + np.sqrt(2)
        angles = np.arange(0, 2 * np.pi, 2 * np.pi / num_samples)
        x = r * np.cos(angles, np.zeros([1, num_samples])).T
        y = r * np.sin(angles, np.zeros([1, num_samples])).T
        init_cond = np.concatenate((x, y), axis=1)
        ic.append(np.expand_dims(init_cond, axis=1))
    return np.array(ic)


def sample_spherical(
    delta: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Sample initial conditions from expanding spherical shells.

    Parameters
    ----------
    delta : np.ndarray
        Array of radial offsets for each shell.
    num_samples : int
        Number of angular divisions for both theta and phi.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(delta), num_samples, num_samples, 3)``.
    """
    r = delta + np.sqrt(0.02)
    theta = np.arange(0, 2 * np.pi, 2 * np.pi / num_samples)
    phi = np.arange(0, np.pi, np.pi / num_samples)

    x_fn = lambda r, th, ph: r * np.cos(th) * np.sin(ph)
    y_fn = lambda r, th, ph: r * np.sin(th) * np.sin(ph)
    z_fn = lambda r, ph: r * np.cos(ph)

    sphere = []
    for radius in r:
        circles = []
        for angle in phi:
            ones = np.ones(len(theta))
            x_coord = x_fn(radius, theta, ones * angle).reshape(-1, 1)
            y_coord = y_fn(radius, theta, ones * angle).reshape(-1, 1)
            z_coord = z_fn(radius, ones * angle).reshape(-1, 1)
            circle_coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)
            circles.append(circle_coord)
        sphere.append(circles)

    return np.array(sphere)


def calc_p(
    epsilon: float,
    n_x: int,
    bounds: List[Tuple[float, float]],
) -> float:
    """Compute the covering number for a given state-space volume.

    Parameters
    ----------
    epsilon : float
        Radius of the covering balls.
    n_x : int
        Dimension of the state space.
    bounds : list of tuple[float, float]
        Lower and upper bounds for each state dimension.

    Returns
    -------
    float
        Covering number *p*.
    """
    vol = 1.0
    for lo, hi in bounds:
        vol *= hi - lo

    p = vol * math.gamma(n_x / 2 + 1) / (math.pi ** (n_x / 2) * epsilon ** n_x)
    return p


# Backwards-compatible aliases (original code uses uppercase names)
RK4 = rk4
KKL_observer_data = kkl_observer_data
