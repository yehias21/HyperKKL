import torch
import numpy as np
import torch.utils.data


class Normalizer:
    def __init__(self, dataset):
        self.means = {}
        self.stds = {}
        # Calculate mean and std for states
        for key, value in dataset.x_states.items():
            self.means[key] = np.mean(value, dtype=np.float32)
            self.stds[key] = np.std(value, dtype=np.float32)
        # Calculate mean and std for observer states
        for key, value in dataset.z_states.items():
            self.means[key] = np.mean(value, dtype=np.float32)
            self.stds[key] = np.std(value, dtype=np.float32)
        # Calculate mean and std for exo_input
        if dataset.exo_input is not None:
            self.means['exo_input'] = np.mean(dataset.exo_input, dtype=np.float32)
            self.stds['exo_input'] = np.std(dataset.exo_input, dtype=np.float32)

    # convert means and stds to np.float32

    def normalize(self, data, var_name):
        return (data - self.means[var_name]) / self.stds[var_name]

    def denormalize(self, normalized_data, var_name):
        return normalized_data * self.stds[var_name] + self.means[var_name]
