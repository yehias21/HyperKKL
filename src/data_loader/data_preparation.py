from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
from hydra.utils import instantiate
from tqdm import tqdm

from src.simulators.estimators import KKLObserver
from src.simulators.systems import System
from src.simulators.types import SimTime


def _observer_ic_methods(gen_mode: str, system, solver, observer, sim_time):
    t_neg = observer.calc_pret0()
    system.p_noise_flag = False
    match gen_mode:
        case 'backward':
            sim_neg = SimTime(sim_time.t0, sim_time.t0 + t_neg, -sim_time.eps)
            # simulate the system in the negative time, with same initial condition of forward time
            neg_states, _ = simulate_system_data(system, solver, sim_neg)
            neg_out = system.get_output(neg_states, noise_flag=False)
            # converge to the initial condition of the observer so that Z0 = T(X0)
            neg_out = np.squeeze(neg_out, 0)
            neg_out = np.flip(neg_out, axis=-2)
        case 'forward':
            sim_neg = SimTime(t_neg + sim_time.t0, sim_time.t0, sim_time.eps)
            # simulate the system in the negative time, with same initial condition of forward time
            neg_states, _ = simulate_system_data(system, solver, sim_neg)
            neg_out = system.get_output(neg_states, noise_flag=False)
            # converge to the initial condition of the observer so that Z0 = T(X0)
            neg_out = np.squeeze(neg_out, 0)

        case 'time_convergence':
            sim_neg = SimTime(t_neg + sim_time.t0, sim_time.t0, sim_time.eps)
            # simulate the system in the negative time, with same initial condition of forward time
            temp_ic = system.ic
            system.ic = system.system_param.system_coeff['ic']
            neg_states, _ = simulate_system_data(system, solver, sim_neg)
            system.ic = temp_ic
            neg_out = system.get_output(neg_states, noise_flag=False)
            # converge to the initial condition of the observer so that Z0 = T(X0)
            neg_out = np.squeeze(neg_out, 0)
        case _:
            raise ValueError(f"{gen_mode} is not a valid generation mode")
    z_init = []
    sim_neg = SimTime(t_neg + sim_time.t0, sim_time.t0, sim_time.eps)
    for z0, y in zip(observer.ic, neg_out):
        z_temp, _ = solver(observer.diff_eq, sim_neg, z0, exogenous_input=y)
        z_init.append(z_temp)
    z_init = np.array(z_init)
    z_init = z_init[:, -1, :]
    assert z_init.shape[0] == system.ic.shape[
        0], "The initial conditions for the observer should be the same as the output"
    return z_init


def simulate_system_data(system, solver, sim_time, input_data: Optional[np.ndarray] = None):
    """
     input_trajs: input signal to the system, dimension (n, t, sig_dim)
     :returns: system states, dimension (inp, sys_ic, t, x_dim)
     """
    trajectories = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        if input_data is not None:
            # Case with exogenous input data
            for input_traj in tqdm(input_data, desc="Exogenous Input IC"):
                results = map(lambda ic: executor.submit(solver, system.diff_eq, sim_time, ic, input_traj), system.ic)
                temp_trajs = [result.result()[0] for result in results]
                trajectories.append(temp_trajs)
        else:
            # Case without exogenous input data
            results = map(lambda ic: executor.submit(solver, system.diff_eq, sim_time, ic), system.ic)
            temp_trajs = [result.result()[0] for result in results]
            trajectories.append(temp_trajs)
    # drop the initial conditions from system states
    trajectories = np.delete(np.array(trajectories), 0, -2)
    return trajectories, np.arange(sim_time.t0, sim_time.tn, sim_time.eps)


def simulate_kklobserver_data(observer: KKLObserver, system: System, y_out: np.ndarray,
                              solver, sim_time: SimTime, gen_mode='forward'):
    z_init = _observer_ic_methods(gen_mode, system, solver, observer, sim_time)
    # simulate the observer
    z_states = []
    # First loop over the input signal
    for y_in in tqdm(y_out, desc="Observer Exogenous Input Loop"):
        # Inner loop for each initial condition Z0
        with ProcessPoolExecutor(max_workers=8) as executor:
            results = [executor.submit(solver, observer.diff_eq, sim_time, z0, exogenous_input=y_traj) for z0, y_traj in
                       zip(z_init, y_in)]

            # make it list comprehension to avoid the generator to be exhausted
            z_states_temp = [result.result()[0] for result in results]
        z_states.append(z_states_temp)

    # drop the initial conditions from system states
    z_states = np.delete(np.array(z_states), 0, -2)
    return z_states


def generate_ph_points(cfg, system, observer, solver, sim_time, input_trajectories, states, observer_states):
    match cfg.pinn_sampling:
        case 'split_set':
            system.sampler(instantiate(cfg.ph_sampler))
            ph_x_states, _ = simulate_system_data(system=system, solver=solver,
                                                  sim_time=sim_time, input_data=input_trajectories)
            ph_out = system.get_output(ph_x_states)
            ph_observer_states = simulate_kklobserver_data(observer=observer, system=system, y_out=ph_out,
                                                           solver=solver, sim_time=sim_time, gen_mode=cfg.gen_mode)
            x_states = {
                'x_regress': states,
                'x_physics': ph_x_states
            }
            z_states = {
                'z_regress': observer_states,
                'z_physics': ph_observer_states
            }
        case 'split_traj':
            assert states.shape[1] > 1, ValueError("Cannot split along the first dimension if its size is 1")
            x_states = {
                'x_regress': states[:, ::2],
                'x_physics': states[:, 1::2],
            }

            z_states = {
                'z_regress': observer_states[:, ::2],
                'z_physics': observer_states[:, 1::2],
            }

        case 'no_physics':
            x_states = {
                'x_regress': states,
            }
            z_states = {
                'z_regress': observer_states,
            }
        case _:
            raise ValueError(f"{cfg.pinn_sample_mode} is not a valid sample mode")
    return x_states, z_states
