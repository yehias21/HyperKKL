"""SOAP optimiser (Vyas et al., arXiv:2409.11321), re-typed from the archived AAAI code
(``archive/HyperKKL/src/optimizers.py``), which produced every SOAP number in
``experiments/archive_recipe/STUDY.md``. Behaviour is kept identical; review notes inline.

Per 2-D parameter G of shape (m, n):
    L <- b L + (1 - b) G G^T,   R <- b R + (1 - b) G^T G           (shampoo_beta b)
    every ``precond_freq`` steps: Q_L, Q_R = eigenvectors of L, R; the Adam moments are
        rotated into the new basis (M exactly, V approximately via sqrt)
    G' = Q_L^T G Q_R;  Adam moments on G';  U' = M_hat / (sqrt(V_hat) + eps)
    update = Q_L U' Q_R^T
Parameters with ndim != 2 (biases, scalars) use plain Adam with the same betas/eps.

Review against the archive version:
  * the first eigendecomposition happens at step 1 (``(t - 1) % freq == 0``), from
    L = (1 - b) G G^T, as there;
  * the moment rotation uses P_L = Q_L_new^T Q_L_old and P_R = Q_R_old^T Q_R_new, so
    M_new = P_L M_old P_R expresses the same matrix in the new basis (checked in the tests);
  * eigh runs in float64 and falls back to the identity on failure, as there;
  * weight decay is L2 added to the gradient (default 0, unused here).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class SOAP(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=0.95,
                 eps=1e-8, weight_decay=0.0, precond_freq=10):
        defaults = dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta, eps=eps,
                        weight_decay=weight_decay, precond_freq=precond_freq)
        super().__init__(params, defaults)

    @staticmethod
    def _eigh(M: torch.Tensor) -> torch.Tensor:
        try:
            _, Q = torch.linalg.eigh(M.double())
            return Q.to(M.dtype)
        except Exception:                       # ill-posed: keep the identity basis
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            sb, eps, lr, wd = group["shampoo_beta"], group["eps"], group["lr"], group["weight_decay"]
            freq = int(group["precond_freq"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["M"] = torch.zeros_like(p)
                    st["V"] = torch.zeros_like(p)
                    if G.ndim == 2:
                        m, n = G.shape
                        st["L"] = torch.zeros(m, m, device=G.device, dtype=G.dtype)
                        st["R"] = torch.zeros(n, n, device=G.device, dtype=G.dtype)
                        st["QL"] = torch.eye(m, device=G.device, dtype=G.dtype)
                        st["QR"] = torch.eye(n, device=G.device, dtype=G.dtype)
                st["step"] += 1
                t = st["step"]
                if wd:
                    G = G.add(p, alpha=wd)
                if G.ndim == 2:
                    st["L"].mul_(sb).add_(G @ G.T, alpha=1 - sb)
                    st["R"].mul_(sb).add_(G.T @ G, alpha=1 - sb)
                    if (t - 1) % freq == 0:
                        QLn, QRn = self._eigh(st["L"]), self._eigh(st["R"])
                        if t > 1:               # carry the moments into the new basis
                            PL = QLn.T @ st["QL"]
                            PR = st["QR"].T @ QRn
                            st["M"] = PL @ st["M"] @ PR
                            st["V"] = (PL @ st["V"].sqrt() @ PR) ** 2
                        st["QL"], st["QR"] = QLn, QRn
                    Gr = st["QL"].T @ G @ st["QR"]
                else:
                    Gr = G
                st["M"].mul_(b1).add_(Gr, alpha=1 - b1)
                st["V"].mul_(b2).addcmul_(Gr, Gr, value=1 - b2)
                Mh = st["M"] / (1 - b1 ** t)
                Vh = st["V"] / (1 - b2 ** t)
                Ur = Mh / (Vh.sqrt() + eps)
                U = (st["QL"] @ Ur @ st["QR"].T) if G.ndim == 2 else Ur
                p.add_(U, alpha=-lr)
        return loss


def make_optimizer(name: str, params, lr: float):
    """'adam' (default, = torch.optim.Adam with torch defaults) | 'adam95' (Adam, betas 0.95/0.95)
    | 'soap' (archive settings)."""
    name = (name or "adam").lower()
    params = list(params)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adam95":                 # control: Adam with SOAP's betas / eps, no preconditioner
        return torch.optim.Adam(params, lr=lr, betas=(0.95, 0.95), eps=1e-8)
    if name == "soap":
        return SOAP(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")
