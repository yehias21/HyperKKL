"""PDE constraint utilities for KKL observer training.

Provides the PDE loss enforcing the transformation map T satisfying:
    dT/dx · f(x) = M·z + K·y
where f is the system dynamics, M and K are observer gain matrices,
z = T(x) is the observer state, and y = h(x) is the measured output.
"""

from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def calc_J(x: Tensor, model: nn.Module) -> Tensor:
    """Compute the batch Jacobian dT/dx via ``torch.autograd.grad``.

    For each output dimension *i* of the model, a backward pass with a
    unit selector vector isolates ∂T_i/∂x for every sample in the batch.
    This is efficient when the output dimension ``n_z`` is small.

    Parameters
    ----------
    x : Tensor, shape ``[B, n_x]``
        Input batch.  Must have ``requires_grad=True``.
    model : nn.Module
        The transformation network T: R^{n_x} -> R^{n_z}.

    Returns
    -------
    Tensor, shape ``[B, n_z, n_x]``
        Batch Jacobian matrices.
    """
    y = model(x)
    n_outputs = y.shape[1]
    jacobian_rows: list[Tensor] = []

    for i in range(n_outputs):
        grad_output = torch.zeros_like(y)
        grad_output[:, i] = 1.0

        grads = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]  # [B, n_x]

        jacobian_rows.append(grads.unsqueeze(1))  # [B, 1, n_x]

    return torch.cat(jacobian_rows, dim=1)  # [B, n_z, n_x]


def pde_loss(
    T_net: nn.Module,
    x: Tensor,
    y: Tensor,
    z_hat: Tensor,
    time: float,
    system,
    M: Union[np.ndarray, Tensor],
    K: Union[np.ndarray, Tensor],
    device: torch.device,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Compute the PDE constraint loss for the KKL observer.

    The loss measures the residual of the PDE:

        dT/dx · f(x)  −  M·z  −  K·y  =  0

    normalised by ``||f(x)||^2`` for scale invariance across dynamical
    systems with different magnitudes.

    Parameters
    ----------
    T_net : nn.Module
        Transformation network T: R^{n_x} -> R^{n_z}.
    x : Tensor, shape ``[B, n_x]``
        State-space samples.
    y : Tensor, shape ``[B, n_y]`` or ``[B]``
        Measured output h(x).
    z_hat : Tensor, shape ``[B, n_z]``
        Observer state estimates T(x).
    time : float
        Time argument forwarded to ``system.function`` (typically 0 for
        autonomous Phase-1 training).
    system : object
        Dynamical system exposing ``system.function(idx, t, x) -> Tensor``.
    M : ndarray or Tensor, shape ``[n_z, n_z]``
        Observer dynamics matrix.
    K : ndarray or Tensor, shape ``[n_z, n_y]``
        Output injection matrix.
    device : torch.device
        Computation device.
    reduction : ``'mean'`` | ``'sum'`` | ``'none'``
        How to reduce the per-sample losses.

    Returns
    -------
    Tensor
        Scalar loss (if *reduction* is ``'mean'`` or ``'sum'``), or a
        per-sample loss vector of shape ``[B]``.
    """
    # Convert numpy matrices to tensors if necessary
    if isinstance(M, np.ndarray):
        M = torch.from_numpy(M).to(device=device, dtype=torch.float32)
    else:
        M = M.to(device=device, dtype=torch.float32)

    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).to(device=device, dtype=torch.float32)
    else:
        K = K.to(device=device, dtype=torch.float32)

    x.requires_grad_()

    # --- 1. Jacobian dT/dx  [B, n_z, n_x] ---
    dTdx = calc_J(x, T_net)

    # --- 2. System dynamics f(x)  [B, n_x, 1] ---
    f_val = system.function(0, 0.0, x).to(device=device, dtype=torch.float32)
    f = f_val.unsqueeze(2)  # [B, n_x, 1]

    # --- 3. Lie derivative: dT/dx · f(x)  [B, n_z, 1] ---
    dTdx_mul_f = torch.bmm(dTdx, f)

    # --- 4. Observer dynamics: M·z + K·y ---
    z_hat = z_hat.unsqueeze(2)  # [B, n_z, 1]
    M_mul_z = torch.matmul(M, z_hat)  # [B, n_z, 1]

    y = y.to(torch.float32)
    if y.ndim == 1:
        y = y.unsqueeze(1)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.unsqueeze(2)  # [B, 1, 1]
    K_mul_y = torch.matmul(K, y)  # [B, n_z, 1]

    # --- 5. PDE residual ---
    pde = dTdx_mul_f - M_mul_z - K_mul_y  # [B, n_z, 1]
    pde_norm_sq = torch.linalg.norm(pde.squeeze(2), dim=1) ** 2  # [B]

    # Normalise by ||f(x)||^2 for scale invariance across systems
    f_norm_sq = torch.sum(f_val ** 2, dim=1) + 1e-8  # [B]
    loss_batch = pde_norm_sq / f_norm_sq  # [B]

    if reduction == "mean":
        return torch.mean(loss_batch)
    if reduction == "sum":
        return torch.sum(loss_batch)
    return loss_batch
