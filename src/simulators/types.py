from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class SysParam:
    """
    System parameters
    """
    C: list[float]
    ObservableIndex: list[int]
    system_coeff: Optional[Dict[str, Any]] = None


@dataclass
class SimTime:
    """
    Time interval for the simulation
    t0: Initial time
    tn: Final time
    eps: time resolution
    """
    t0: float
    tn: float
    eps: float


@dataclass
class SysDim:
    """
    System dimensions
    """
    x_dim: int
    z_dim: int
    y_dim: int


@dataclass
class NoiseParam:
    """
    Parameters for the noise
    """
    sys_noise: Optional[dict]
    out_noise: Optional[dict]
    inp_noise: Optional[dict]


## Input Signal dataclasses

@dataclass
class SigParam:
    signal_type: str
    signal_data: Dict


@dataclass
class ObserverParam:
    data: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)
