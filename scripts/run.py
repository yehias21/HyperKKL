from hydra import compose, initialize
import hydra, os
from omegaconf import DictConfig, OmegaConf
from train import run_experiment
from glob import glob

def run_baselines():
    signals_conf = {os.path.basename(conf).split(".")[0]: conf.split("config/")[1]
                    for conf in glob("config/data/exo_input/signals/*.yaml")}
    # how to cut a string from a certain substring
    with initialize(config_path="config"):
        state_0 = compose(config_name=signals_conf['duff'])
        cfg = compose(config_name="baseline")
        cfg.data.exo_input.signals.state_0 = state_0
        run_experiment(cfg)

def run_lora():
    with initialize(config_path="config"):
        cfg = compose(config_name="lora")
        OmegaConf.to_yaml(cfg)

if __name__ == "__main__":
    run_baselines()
