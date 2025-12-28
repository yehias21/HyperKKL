from typing import Optional

import numpy as np
from torch.utils.data import Dataset
from src.simulation.systems import System

class KKLDataset(Dataset):
    def __init__(self, system: System, observer, x_states: dict, z_states: dict, time,
                 exo_input: Optional[np.array] = None):
        self.system = system
        self.observer = observer
        self.x_states = x_states
        self.z_states = z_states
        self.exo_input = exo_input
        self.time = time
        self.y_out = {key.replace('x', 'y'): self.system.get_output(self.x_states[key]) for key in self.x_states}
        # Check dimensions and set parameters accordingly
        self.inp_ic, self.ic, self.t, _ = (self.x_states['x_regress'].shape)  # dimension of the regress must be equal  to dimension of physics

    def __len__(self):
        return self.inp_ic * self.ic * self.t

    def __getitem__(self, idx):
        # Calculate indices for each dimension
        inp_ic_idx = idx // (self.ic * self.t)
        rem = idx % (self.ic * self.t)
        ic_idx = rem // self.t
        t_idx = rem % self.t

        x_sample = {key: self.x_states[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.x_states}
        z_sample = {key: self.z_states[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.z_states}
        y_out = {key: self.y_out[key][inp_ic_idx, ic_idx, t_idx, :].astype(np.float32) for key in self.y_out}

        if self.exo_input is None:
            return {'x_states': x_sample, 'z_states': z_sample, 'time': self.time[t_idx].astype(np.float32),
                    'y_out': y_out}
        else:
            return {'x_states': x_sample, 'z_states': z_sample,
                    'exo_input': self.exo_input[inp_ic_idx, t_idx].astype(np.float32),
                    'time': self.time[t_idx].astype(np.float32), 'y_out': y_out}

