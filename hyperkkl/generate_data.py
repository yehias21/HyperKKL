#!/usr/bin/env python3
"""Generate KKL observer training datasets using Hydra configuration."""
import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.simulators import SimulationTime
from src.data import DataGenerator, DataGenerationConfig, save_dataset

log = logging.getLogger(__name__)


def build_signals(cfg: DictConfig) -> list:
    """Build signal generators from config."""
    signals = []
    if "signals" in cfg:
        for key, sig_cfg in cfg.signals.items():
            if sig_cfg is not None:
                signals.append(instantiate(sig_cfg))
    return signals


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for dataset generation."""
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Build simulation time
    sim_time = SimulationTime(
        t0=cfg.simulation.t0,
        tn=cfg.simulation.tn,
        dt=cfg.simulation.dt
    )
    
    # Instantiate components
    log.info("Building system: %s", cfg.system.name)
    system = instantiate(cfg.system)
    
    log.info("Building observer: %s", cfg.observer.name)
    observer = instantiate(cfg.observer)
    
    log.info("Building solver: %s", cfg.solver.name)
    solver = instantiate(cfg.solver)
    
    # Samplers
    system_sampler = instantiate(cfg.system_sampler)
    observer_sampler = instantiate(cfg.observer_sampler)
    
    # Build signals
    signals = build_signals(cfg)
    if signals:
        log.info("Using %d exogenous signals", len(signals))
    
    # Data generation config
    gen_config = DataGenerationConfig(
        num_trajectories=cfg.simulation.num_trajectories,
        gen_mode=cfg.data.gen_mode,
        pinn_sampling=cfg.data.pinn_sampling,
        validation_method=cfg.data.validation.method if cfg.data.validation.enabled else None,
        validation_time=cfg.data.validation.time,
    )
    
    # Create generator
    generator = DataGenerator(
        system=system,
        observer=observer,
        solver=solver,
        system_sampler=system_sampler,
        observer_sampler=observer_sampler,
        signals=signals,
        config=gen_config
    )
    
    # Generate dataset
    log.info("Starting data generation...")
    dataset = generator.generate(sim_time, seed=cfg.seed)
    
    # Save dataset
    output_dir = Path(cfg.data.output_dir)
    output_path = output_dir / f"{cfg.data.name}.npz"
    save_dataset(dataset, output_path)
    log.info("Dataset saved to: %s", output_path)
    
    # Log statistics
    if "train" in dataset:
        train_data = dataset["train"]
        log.info("Training samples: %d trajectories, %d timesteps", 
                 train_data["x_regress"].shape[0], 
                 train_data["x_regress"].shape[1])
        if dataset.get("val"):
            val_data = dataset["val"]
            log.info("Validation samples: %d trajectories, %d timesteps",
                     val_data["x_regress"].shape[0],
                     val_data["x_regress"].shape[1])


if __name__ == "__main__":
    main()
