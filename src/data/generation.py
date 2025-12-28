import hydra, h5py, os
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.utils.helpers import get_exp_dir
def generate_data(cfg: DictConfig, data_dir:str="") -> None:
    if not (data_dir and os.path.exists(data_dir)):
        print(f"The output directory {data_dir} does not exist. Using default path.")
        os.makedirs("../../data/raw", exist_ok=True)

    # Instantiate the components
    system = instantiate(cfg.system)
    observer = instantiate(cfg.observer)
    sim_time = instantiate(cfg.sim_time)
    solver = instantiate(cfg.solver)
    input_trajectories = None
    strategy = instantiate(cfg.strategy)     # Strategy

    if 'input_signal' in cfg:
        print("--- Generating Exogenous Input Signals ---")
        input_signal = instantiate(cfg.exo_input, _recursive_=False)
        input_trajectories = input_signal.generate_signals(sim_time)

    generated_data = strategy.generate(system, observer, solver, sim_time) # generate the data

    # Save the generated data
    output_dir = os.path.join(data_dir, get_exp_dir(cfg))
    os.makedirs(output_dir)
    # with h5py.File(os.path.join(output_dir, "training.h5"), "w") as f:

