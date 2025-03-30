from typing import Optional
import torch.nn as nn, torch
from src.model.hypernetwork import HyperNetwork
from abc import ABC, abstractmethod


class KKLObserverNetwork(nn.Module):
    def __init__(self, forward_mapper, inverse_mapper,
                 forward_hypernetwork: Optional[HyperNetwork],
                 inverse_hypernetwork: Optional[HyperNetwork]):
        super(KKLObserverNetwork, self).__init__()
        self.forward_mapper = forward_mapper
        self.inverse_mapper = inverse_mapper
        self.forward_hypernetwork = forward_hypernetwork
        self.inverse_hypernetwork = inverse_hypernetwork
        self.normalizer = None

    @property
    def learnable_params(self):
        params = []
        # Parameters from forward hypernetwork (if it exists)
        if self.forward_hypernetwork is None:
            params.extend(self.forward_mapper.parameters())
        else:
            params.extend(self.forward_hypernetwork.parameters())

            # Parameters from inverse hypernetwork (if it exists)
        if self.inverse_hypernetwork is None:
            params.extend(self.inverse_mapper.parameters())
        else:
            params.extend(self.inverse_hypernetwork.parameters())

        return params

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _normalize_data(self, data):
        if self.normalizer:
            norm_data = {}
            for key in data.keys():
                if key in ['time', 'y_out']:
                    continue
                if isinstance(data[key], dict):
                    norm_data[key] = {}
                    for inner_key in data[key].keys():
                        norm_data[key][inner_key] = self.normalizer.normalize(data[key][inner_key], inner_key)
                else:
                    norm_data[key] = self.normalizer.normalize(data[key], key)

            return norm_data

    def _denormalize_data(self, data):
        if self.normalizer:
            denorm_data = {}
            for key in data.keys():
                if key in ['time', 'y_out']:
                    continue
                if isinstance(data[key], dict):
                    denorm_data[key] = {}
                    for inner_key in data[key].keys():
                        denorm_data[key][inner_key] = self.normalizer.denormalize(data[key][inner_key], inner_key)
                else:
                    denorm_data[key] = self.normalizer.denormalize(data[key], key)

            return denorm_data

    def forward(self, inputs: dict):
        # Todo: Need to adjust if states['physics'] is not present
        result = {'x_states': {}, 'z_states': {}}
        inputs = self._normalize_data(inputs)
        if self.forward_hypernetwork:
            weights = self.forward_hypernetwork(inputs['exo_input'])
            result['z_states']['z_regress'] = self.forward_mapper(inputs['x_states']['x_regress'],
                                                                  weights)
            if 'x_physics' in inputs['x_states']:
                result['z_states']['z_physics'] = self.forward_mapper(inputs['x_states']['x_physics'],
                                                                      weights)
        else:
            result['z_states']['z_regress'] = self.forward_mapper(inputs['x_states']['x_regress'])
            if 'x_physics' in inputs['x_states']:
                result['z_states']['z_physics'] = self.forward_mapper(inputs['x_states']['x_physics'])

        # INVERSE MAPPER
        if self.inverse_hypernetwork:
            weights = self.inverse_hypernetwork(inputs['exo_input'])
            result['x_states']['x_regress'] = self.inverse_mapper(inputs['z_states']['z_regress'],
                                                                  weights)
            if 'z_physics' in inputs['z_states']:
                result['x_states']['x_physics'] = self.inverse_mapper(inputs['z_states']['z_physics'],
                                                                      weights)
        else:
            result['x_states']['x_regress'] = self.inverse_mapper(inputs['z_states']['z_regress'])
            if 'z_physics' in inputs['z_states']:
                result['x_states']['x_physics'] = self.inverse_mapper(inputs['z_states']['z_physics'])

        self._denormalize_data(result)
        return result
