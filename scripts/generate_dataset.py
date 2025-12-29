from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import os
from glob import glob

CONFIG_PATH = 'config'
with initialize(config_path=CONFIG_PATH,version_base=None):
    cfg = compose(config_name="data_generation.yaml")
    signals = glob(os.path.join(cfg.data.signals_path, '*.yaml'))
    signals = {os.path.basename(conf).split('.')[0]: "/".join(conf.split('\\')[-4:]) for conf in signals}
    signal = compose(config_name=signals['square'])
    cfg.data.exo_input.signals.state_0 = signal
    print(OmegaConf.to_yaml(cfg,resolve=True))