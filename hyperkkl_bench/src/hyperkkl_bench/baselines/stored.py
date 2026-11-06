"""Adapters for the STORED research models (no retraining). Requires the research repository
(the ``hyperkkl`` package and experiments/baselines2/code) next to this package; outside the
repository these are unavailable and the bundled standalone baselines must be used instead.

    est = StoredModel("ude5k1_ekf", seed=0)        # corrected grey-box (UDE residual + EKF)
    rep = hkb.evaluate(est, "vdp", "partial", fit=True)   # fit() only LOADS the checkpoint

Available (experiments/baselines2/models/<node>/<plant>/seed<k>/):
    ude5k1_ekf, ude5k1_ukf   corrected grey-box: f_known + MLP residual trained THROUGH RK4 (P1 fixed)
    ude5k_ekf,  node5k_ekf   UDE / NODE (5k parameters) + EKF
    gru140, gru500           GRU sequence observers (140k / 500k parameters)
Cells: partial + mismatch (6 dev plants), realistic (5 test plants: fhn, highway_traffic, network,
rossler, two_link_arm; and cartpole_b). Seeds 0-4. The ``exact`` level was not trained.

HyperKKL (the paper's method): ``StoredHyperKKL`` runs the experiments/jitter_fix checkpoints
(battery/partial, cartpole_b/realistic, quad_planar/realistic) in a subprocess with the experiment's
own package copy: ``repro_A3_D0`` (A3 retrained on the standard data: equal data), ``base`` (A3 on
the mixed D0+D1 data: UNEQUAL data, PITFALLS P6), ``C5_j3_s0.3_k0.5`` (the jitter-fixed C5 recipe,
mixed data). CPU reproduction of the stored per-episode NRMSE: ~5e-7. The beat_greybox A3
checkpoints of the other cells were kept only in a session scratch folder and are not wrapped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..estimator import BaseEstimator
from ..paths import REPO

NODES = {"partial": "C3-partial", "mismatch": "C3-mismatch30"}
TEST_NODE = {"cartpole_b": "TEST-C3-B"}
DEV = ("battery", "bioreactor", "duffing", "lorenz", "quad_tank", "vdp")


def _node(plant, knowledge):
    if plant in DEV and knowledge in NODES:
        return NODES[knowledge]
    if plant not in DEV and knowledge == "realistic":
        return TEST_NODE.get(plant, "TEST-C3")
    raise ValueError(f"no stored baselines2 model for {plant}/{knowledge} "
                     "(dev plants: partial, mismatch; test plants: realistic)")


def _import_research():
    for p in (REPO, REPO / "experiments" / "baselines2" / "code"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from hyperkkl import systems_extra, systems_bounded  # noqa: F401
        from b2 import models as M  # noqa: F401
    except ImportError as e:
        raise ImportError("stored models need the research repository (hyperkkl package + "
                          f"experiments/baselines2/code) at {REPO}: {e}") from e


class StoredModel(BaseEstimator):
    def __init__(self, model: str = "ude5k1_ekf", seed: int = 0):
        self.model, self.seed = model, int(seed)
        self.name = f"stored:{model}[s{seed}]"

    def fit(self, train, val, context):
        _import_research()
        import torch
        from b2 import models as M
        from hyperkkl import baselines_data as BD
        from hyperkkl.observers import EKF, UKF
        self.context = context
        plant, k = context["plant"], context["knowledge"]
        node = _node(plant, k)
        f = REPO / "experiments" / "baselines2" / "models" / node / plant / f"seed{self.seed}" / f"{self.model}.pt"
        d = torch.load(f, weights_only=False, map_location="cpu")
        meta = context["meta"]
        model = context["model"]                      # same modeller's model as the research runs
        sens = list(context["sensors"])
        nx, ny = int(meta["n_x"]), len(sens)
        u_scale = float(meta["input_box"]["u_scale"])
        xm = np.asarray(context["x_mean"])
        if self.model.startswith("gru"):
            norm = M._Norm(u_scale, xm[sens], context["y_std"], xm, context["x_std"])
            net = M.GRUNet(1, ny, nx, d["arch"]["hidden"], d["arch"]["layers"], norm)
            net.load_state_dict(d["state"])
            self.ob = M.SeqObserver(self.model, net)
            self._np = sum(p.numel() for p in net.parameters())
        else:
            net = BD.ResidualMLP(nx, 1, d["width"], context["x_mean"], context["x_std"], u_scale,
                                 d["t_mean"], d["t_std"])
            net.load_state_dict(d["state"])
            Fk = EKF if self.model.endswith("_ekf") else UKF
            p0 = context["prior_std"]
            self.ob = Fk(M.deployed_dynamics(model, net, d["hybrid"]), context["dt"], context["q_std"],
                         context["r_std"], p0)
            self._np = sum(p.numel() for p in net.parameters())
        self.checkpoint = str(f)

    # the research observers are batched: run_batch is the native path
    def run_batch(self, U, Y, X0, P0=None, chunk=64):
        import torch
        from hyperkkl.observers import run
        out = []
        for c in range(0, len(U), chunk):
            y = torch.as_tensor(Y[c:c + chunk], dtype=torch.float32)
            u = torch.as_tensor(U[c:c + chunk], dtype=torch.float32)[..., None]
            x0 = torch.as_tensor(X0[c:c + chunk], dtype=torch.float32)
            with torch.no_grad():
                out.append(run(self.ob, y, u, x0).double().numpy())
        return np.concatenate(out)

    def reset(self, x0_prior, P0_prior=None):
        import torch
        self._buf = None
        self._x0 = torch.as_tensor(np.asarray(x0_prior, float)[None], dtype=torch.float32)
        self._k = 0

    def step(self, u, y):
        """Online adapter: the research observers advance on (y_k, u_k, y_{k+1}, u_{k+1}) pairs."""
        import torch
        yk = torch.as_tensor(np.asarray(y, float)[None], dtype=torch.float32)
        uk = torch.as_tensor([[float(u)]], dtype=torch.float32)
        with torch.no_grad():
            if self._k == 0:
                self.ob.reset(1, self._x0, uk)
            else:
                yp, up = self._buf
                self.ob.advance(yp, up, yk, uk)
        self._buf = (yk, uk)
        self._k += 1
        return self.ob.estimate()[0].double().numpy()

    def n_params(self):
        return int(self._np)


HKKL_CELLS = {("battery", "partial"): "battery", ("cartpole_b", "realistic"): "cartpole",
              ("quad_planar", "realistic"): "quad"}


class StoredHyperKKL(BaseEstimator):
    """Batch-only adapter (``step`` is not available: the model runs in a separate interpreter).
    Latency is therefore not measured by the harness; see resources/tracks_reference.json."""

    def __init__(self, model: str = "C5_j3_s0.3_k0.5", seed: int = 0):
        self.model, self.seed = model, int(seed)
        self.name = f"stored:hkkl_{model}[s{seed}]"

    def fit(self, train, val, context):
        self.context = context
        key = (context["plant"], context["knowledge"])
        if key not in HKKL_CELLS:
            raise ValueError(f"no stored HyperKKL model for {key}; available: {sorted(HKKL_CELLS)}")
        self.jf_plant = HKKL_CELLS[key]
        d = REPO / "experiments" / "jitter_fix" / "models" / self.jf_plant / self.model / f"seed{self.seed}"
        if not list(d.glob("p2_*.pt")):
            raise FileNotFoundError(f"no checkpoint in {d}")
        self._np = None

    def run_batch(self, U, Y, X0, P0=None):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            fi, fo = f"{td}/in.npz", f"{td}/out.npz"
            np.savez(fi, Y=Y, U=U, X0=X0)
            env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="")
            r = subprocess.run([sys.executable, "-m", "hyperkkl_bench.baselines._hkkl_runner", self.jf_plant,
                                self.model, str(self.seed), fi, fo], capture_output=True, text=True, env=env)
            if r.returncode != 0:
                raise RuntimeError(f"HyperKKL runner failed:\n{r.stderr[-3000:]}")
            d = np.load(fo)
            self._np = int(d["n_params"])
            return d["X_hat"]

    def reset(self, x0_prior, P0_prior=None):
        raise NotImplementedError("StoredHyperKKL is batch-only (run_batch); it runs in a subprocess")

    def step(self, u, y):
        raise NotImplementedError("StoredHyperKKL is batch-only (run_batch); it runs in a subprocess")

    def n_params(self):
        return self._np
