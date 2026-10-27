r"""Scalable FitzHugh-Nagumo ring (experiments/scaling): plant family, latent design, run configs.

This module never edits an existing plant class. It registers NEW plant names ``fhnring{N}``
(N nodes, n_x = 2N) into the package registry at import time, so the cache-hashed source of
the package (cache.DATA_MODULES) is untouched.

Plant (per node i = 0..N-1, ring neighbours j = i +- 1 mod N, adjacency a_ij):

    v_i' = v_i - v_i^3/3 - w_i + I_ext
           + kappa sum_j a_ij (v_j - v_i)                       gap junctions (linear, KNOWN)
           + g_syn (E_syn - v_i) sum_j a_ij sig((v_j - th)/s)   chemical synapse (UNMODELLED)
           + [i = 0] u
    w_i' = eps_i (v_i + a - b w_i),  eps_i = eps exp(spread (2 frac(i g) - 1)),  g = (sqrt 5 - 1)/2

The `network` plant (systems_extra.FHNRing, 5 nodes) is the template: same node parameters,
kappa = 0.5; recovery rates eps_i over exp(+-spread) with spread = 1.0 (rev 2; rev 1 used the
network's 0.6 and failed the pre-registered boundedness check (2a) at n_x = 16, 32 through slow
phase dispersion: S(5T)/S(T) up to 2.3, states never left |v| <= 2.5; spread 1.0: 1.16); rev 2
(spread 1.0, monotone ramp of eps_i along the ring) failed (2a) at n_x = 64 (S(5T)/S(T) 4.2);
rev 3 (this) places eps_i by the golden-ratio sequence (n_x = 64: 1.10 train, 0.88 shape_ood). New: the excitatory synapse (the declared
missing physics, identical per node at every N because every node has exactly two
neighbour slots; N = 2 uses a_01 = a_10 = 2 so the degree is 2 there too), and the
sensor rule. Sensors: v of every EVEN node (n_y = N/2, 25 % of the states; every
unmeasured node has two measured neighbours). Input on node 0 (saturating actuator as
`network`).

Latent design ("bank"): for n_y = 1 the package convention (Butterworth, n_z = 2 n_x + 1 = 9 at
n_x = 4). For n_y >= 2 the package convention (n_z = n_x + 2, one Butterworth circle with
n_z poles) does not scale: the pole radius alpha / sin(pi / 2 n_z) grows with n_z (dt and
N per episode grow ~ n_z). The bank design gives every measured channel its own m-pole
Butterworth bank (m = 5, alpha = 0.3): A = blockdiag(A_m, ..., A_m), B = blockdiag(1_m, ...),
n_z = 5 n_y = 1.25 n_x (= n_x + 2 at n_x = 8). Fixed grid for every N: dt = 0.05, T = 100.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from hyperkkl import data as Dm
from hyperkkl import design as D
from hyperkkl.config import deep_merge
from hyperkkl.systems import Plant, register

NX_GRID = (4, 8, 16, 32, 64)
NX_REF = 8                       # reference size: package defaults at n_x = 8
BANK_M = 5
DT = 0.05
LEVELS = ("exact", "off30", "partial", "realistic")


GOLDEN = 0.6180339887498949


def _sig(v):
    return 1.0 / (1.0 + torch.exp(-v))


class FHNRingN(Plant):
    N_NODES = 0
    PARAMS = {"a": 0.7, "b": 0.8, "eps": 0.08, "I_ext": 0.5, "kappa": 0.5, "spread": 1.0,
              "g_syn": 0.15, "E_syn": 2.0, "th_syn": 1.0, "s_syn": 0.25}
    NOT_PERTURBED = ("I_ext",)   # the applied bias current is an operating condition, not plant physics
    UNMODELLED = ("the excitatory chemical synapse g_syn (E_syn - v_i) sum_j a_ij sig((v_j - th)/s) "
                  "between ring neighbours (model: gap-junction coupling only)")

    def __init__(self, sensors=None, **params):
        super().__init__(sensors=sensors, **params)
        self._cache = {}

    @classmethod
    def adjacency(cls) -> np.ndarray:
        N = cls.N_NODES
        A = np.zeros((N, N))
        for i in range(N):
            A[i, (i + 1) % N] += 1.0
            A[i, (i - 1) % N] += 1.0
        return A

    def _consts(self, dtype, device):
        k = (dtype, str(device))
        if k not in self._cache:
            A = torch.as_tensor(self.adjacency(), dtype=dtype, device=device)
            L = A - torch.diag(A.sum(1))
            # rev 3: golden-ratio (low-discrepancy) layout of the recovery rates around the ring, so
            # ring neighbours always differ (a monotone ramp makes long runs of near-identical
            # neighbours that phase-lock in clusters: rev 2 failed boundedness (2a) at n_x = 64)
            i = np.arange(self.N_NODES)
            pos = torch.as_tensor(2 * ((i * GOLDEN) % 1.0) - 1, dtype=dtype, device=device)
            eps = self.eps * torch.exp(self.spread * pos)
            e0 = torch.zeros(self.N_NODES, dtype=dtype, device=device); e0[0] = 1.0
            self._cache[k] = (A, L, eps, e0)
        return self._cache[k]

    def f(self, x, u):
        return self._f(x, u, True)

    def f_known(self, x, u):
        return self._f(x, u, False)

    def _f(self, x, u, synapse: bool):
        A, L, eps, e0 = self._consts(x.dtype, x.device)
        v, w = x[:, 0::2], x[:, 1::2]
        dv = v - v ** 3 / 3 - w + self.I_ext + self.kappa * v @ L + u[:, :1] * e0
        if synapse:
            s = _sig((v - self.th_syn) / self.s_syn)
            dv = dv + self.g_syn * (self.E_syn - v) * (s @ A)          # A symmetric
        dw = eps * (v + self.a - self.b * w)
        return torch.stack([dv, dw], 2).reshape(x.shape[0], self.nx)


def sensors_for(N: int) -> tuple:
    return tuple(2 * i for i in range(0, N, 2))


PLANTS = {}
for _nx in NX_GRID + (128,):
    _N = _nx // 2
    _cls = type(f"FHNRing{_N}", (FHNRingN,), {"N_NODES": _N, "nx": _nx,
                                               "DEFAULT_SENSORS": sensors_for(_N)})
    PLANTS[_nx] = register(f"fhnring{_N}")(_cls)


# ---------------------------------------------------------------------------- latent design
def bank_AB(ny: int, m: int, alpha: float) -> tuple:
    Am, _ = D.butterworth_AB(m, alpha, 1)
    A = np.kron(np.eye(ny), Am)
    B = np.kron(np.eye(ny), np.ones((m, 1)))
    return A, B


_orig_make_design = Dm.make_design


def make_design(scfg: dict):
    if not scfg.get("bank"):
        return _orig_make_design(scfg)
    ny = len(scfg["sensors"])
    alpha = float(scfg["alpha"]) * float(scfg.get("alpha_scale", 1.0))
    m = int(scfg["bank"])
    if ny == 1:
        A, B = D.butterworth_AB(int(scfg["nz"]), alpha, 1)
    else:
        A, B = bank_AB(ny, m, alpha)
    dt = float(scfg["dt"])
    N = int(math.ceil(float(scfg["T"]) / dt))
    return Dm.Design(A, B, dt, N, alpha, D.latent_step(A, B, dt))


Dm.make_design = make_design


def design_check(scfg: dict) -> dict:
    """The package design gate, applied to ONE channel bank (the blocks are identical and
    decoupled, so controllability / normality / Gramian of the whole = those of a block)."""
    des = make_design(scfg)
    ny = len(scfg["sensors"])
    m = des.A.shape[0] if ny == 1 else int(scfg["bank"])
    v = D.validate(des.A[:m, :m], des.B[:m, :1], des.dt, des.alpha)
    v["nz"] = int(des.A.shape[0]); v["N"] = des.N; v["dt"] = des.dt
    return v


# ---------------------------------------------------------------------------- configs
BASE = yaml.safe_load(open(Path(__file__).resolve().parent / "pkg" / "configs" / "base.yaml"))

LEVEL_SET = {
    "exact": {"knowledge": "full", "mismatch": 0.0, "sensor": {"noise": 0.03}},
    "off30": {"knowledge": "full", "mismatch": 0.3, "sensor": {"noise": 0.03}},
    "partial": {"knowledge": "partial", "mismatch": 0.0, "sensor": {"noise": 0.03}},
    "realistic": {"knowledge": "partial", "mismatch": 0.3,
                  "sensor": {"noise": 0.03, "kind": "colored_t", "rho": 0.9, "nu": 3.0}},
}
# training data are shared by levels with the same real plant and sensor
DATASET = {"exact": "D0", "partial": "D0", "off30": "D30", "realistic": "DR"}
DATASET_LEVEL = {"D0": "exact", "D30": "off30", "DR": "realistic"}


def system_cfg(nx: int) -> dict:
    N = nx // 2
    sens = list(sensors_for(N))
    ny = len(sens)
    nz = 2 * nx + 1 if ny == 1 else BANK_M * ny
    box = []
    for _ in range(N):
        box += [[-2.5, 2.5], [-1.5, 1.5]]
    # every parameter is written into the config so that every cache key covers it (rev 2)
    return {"name": f"ring{nx}", "plant": f"fhnring{N}", "role": "scaling",
            "params": dict(FHNRingN.PARAMS),
            "sensors": sens, "ic_box": box, "T": 100, "nz": nz, "alpha": 0.3, "u_scale": 1.0,
            "t_ref": 1.5, "u_bounds": [], "actuator": {"kind": "saturating", "u_bar": 2.0},
            "mismatch": 0.0, "bank": BANK_M, "dt": DT, "knowledge": "full"}


def scale_factor(nx: int) -> float:
    return math.sqrt(nx / NX_REF)


def sizes(nx: int, data_nx: int = None) -> dict:
    """The pre-registered capacity / data rule (PREREG.md section 3). data_nx: the n_x whose
    data amount is used (fixed-data variant: data_nx = 4)."""
    s, r = scale_factor(nx), nx / NX_REF
    dn = (data_nx or nx) / NX_REF
    return {
        "enc_width": round(192 * s), "dec_width": round(128 * s),
        "budget": round(140_000 * r), "nlox_width": round(64 * s), "nlox_dec_width": round(128 * s),
        "gb_width": round(64 * s), "dyn_width": round(64 * s), "gru_budget": round(140_000 * r),
        "n_train": round(600 * dn), "n_p1": round(400 * dn), "n_coll": round(32768 * dn),
        "n_val": 64,
    }


def make_cfg(nx: int, level: str, seed: int, data_nx: int = None) -> dict:
    z = sizes(nx, data_nx)
    L = LEVEL_SET[level]
    cfg = copy.deepcopy(BASE)
    cfg["system"] = deep_merge(system_cfg(nx), {"knowledge": L["knowledge"], "mismatch": L["mismatch"]})
    cfg["data"] = deep_merge(cfg["data"], {"n_train": z["n_train"], "n_p1": z["n_p1"],
                                           "n_coll": z["n_coll"], "n_val": z["n_val"]})
    cfg["data"]["sensor"] = dict(L["sensor"])
    cfg["phase1"] = deep_merge(cfg["phase1"], {"enc_width": z["enc_width"], "dec_width": z["dec_width"]})
    cfg["phase2"] = deep_merge(cfg["phase2"], {"budget": z["budget"]})
    cfg["nlox"] = deep_merge(cfg["nlox"], {"width": z["nlox_width"], "dec_width": z["nlox_dec_width"]})
    cfg["data_baselines"] = {"gb_width": z["gb_width"], "gb_n_fit": 10 ** 9, "n_fit": 10 ** 9}
    cfg["eval"] = deep_merge(cfg["eval"], {"split": "test", "conditions": ["clean"], "save_traj": False,
                                           "memory_probe": False})
    cfg["seed"] = seed
    cfg["scaling"] = {"nx": nx, "level": level, "data_nx": data_nx or nx, "sizes": z}
    return cfg
