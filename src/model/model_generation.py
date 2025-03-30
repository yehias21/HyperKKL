# imports for model generation
import copy
from omegaconf import DictConfig, OmegaConf
from src.model.mapper import KKLObserverNetwork
import hydra
from src.model.models import get_model
from src.model.hypernetwork import HyperNetwork, get_decoder


def create_hypernetwork(model_dict, encoder, cfg):
    if not cfg.shared:
        # Clone encoder module
        enc = copy.deepcopy(encoder)
    else:  # Shared hypernetwork
        enc = encoder

    # Todo: Filter stage
    model_dict = {k: v for k, v in model_dict.items() if 'weight' in k}

    decoder = get_decoder(cfg.decoder, model_dict)

    return HyperNetwork(enc, decoder)


def init_models(cfg: DictConfig) -> KKLObserverNetwork:
    # 1. Initialize the forward and inverse mapper
    forward_mapper = get_model(cfg.forward_mapper)
    inverse_mapper = get_model(cfg.inverse_mapper)

    # 2. Initialize the hypernetwork (if required)
    forward_hypernetwork, inverse_hypernetwork = None, None
    if 'hypernetwork' in cfg:
        encoder = get_model(cfg.hypernetwork.encoder)

        if cfg.forward_mapper.update_method != 'backprop':
            forward_hypernetwork = create_hypernetwork(forward_mapper.state_dict(), encoder, cfg.hypernetwork)

        if cfg.inverse_mapper.update_method != 'backprop':
            inverse_hypernetwork = create_hypernetwork(inverse_mapper.state_dict(), encoder, cfg.hypernetwork)

    return KKLObserverNetwork(forward_mapper, inverse_mapper, forward_hypernetwork, inverse_hypernetwork)
