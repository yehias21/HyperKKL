"""Baseline modules and their deployed observers (package contract: observers.Observer).

GRUNet / NCDENet are sequence observers: ``seq(u, y)`` maps a whole (B, L, .) history to (B, L, nx)
estimates (training, validation); ``SeqObserver`` runs the same network one sample at a time online.
x̂_k uses (u_0..u_k, y_0..y_k) only: the GRU reads (u_k, y_k) at step k; the CDE step k -> k+1
integrates the linear interpolation of X between samples k and k+1 (so x̂_{k+1} uses y_{k+1}, u_{k+1}).

LearnedDynamics: f(x, u) = [f_known(x, u)] + r_theta(x, u) (UDE: with the modeller's model, NODE:
without), the MLP evaluated functionally like baselines_data.GreyBoxModel (the EKF's vmap(jacrev)
differentiates through it).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from hyperkkl.baselines_data import GreyBoxModel, ResidualMLP
from hyperkkl.nets import MLP
from hyperkkl.observers import Observer


def _f32(a):
    return torch.as_tensor(np.asarray(a, float), dtype=torch.float32)


class _Norm(nn.Module):
    """Fixed input / output standardisation shared by the sequence observers."""

    def __init__(self, u_scale, y_mean, y_std, x_mean, x_std):
        super().__init__()
        self.register_buffer("us", _f32(np.atleast_1d(u_scale)))
        self.register_buffer("ym", _f32(y_mean))
        self.register_buffer("ys", _f32(y_std))
        self.register_buffer("xm", _f32(x_mean))
        self.register_buffer("xs", _f32(x_std))

    def inp(self, u, y):
        return torch.cat([u / self.us, (y - self.ym) / self.ys], -1)

    def out(self, o):
        return self.xm + self.xs * o

    def out_norm(self, x):
        return (x - self.xm) / self.xs


# ---------------------------------------------------------------------------------------------- GRU

class GRUNet(nn.Module):
    kind = "gru"

    def __init__(self, nu, ny, nx, hidden, layers, norm: _Norm):
        super().__init__()
        self.norm = norm
        self.rnn = nn.GRU(nu + ny, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, nx)
        self.layers, self.hidden = int(layers), int(hidden)

    def init_state(self, B, device):
        return torch.zeros(self.layers, B, self.hidden, device=device)

    def seq_norm(self, u, y, h=None):
        """(B, L, .) -> normalised estimates (B, L, nx), final state."""
        if h is None:
            h = self.init_state(u.shape[0], u.device)
        o, h = self.rnn(self.norm.inp(u, y), h)
        return self.head(o), h

    def seq(self, u, y):
        return self.norm.out(self.seq_norm(u, y)[0])

    def step(self, state, u_k, y_k):
        """state, (B, nu), (B, ny) -> state', x̂_k (after reading sample k)."""
        if state is None:
            state = self.init_state(u_k.shape[0], u_k.device)
        o, h = self.rnn(self.norm.inp(u_k, y_k)[:, None], state)
        return h, self.norm.out(self.head(o[:, 0]))


def gru_params(nu, ny, nx, H, L=2) -> int:
    d = nu + ny
    p = 3 * (d * H + H * H + 2 * H) + (L - 1) * 3 * (2 * H * H + 2 * H)
    return p + H * nx + nx


def solve_gru(nu, ny, nx, budget, L=2) -> int:
    return min(range(4, 2048), key=lambda H: abs(gru_params(nu, ny, nx, H, L) - budget))


# ---------------------------------------------------------------------------------------------- CDE

class NCDENet(nn.Module):
    """Neural CDE (Kidger et al. 2020): dz = f_theta(z) dX, X = (t / t_ref, u, y) normalised and
    linearly interpolated between samples; one RK4 step per sample interval (dX/ds constant on it,
    s in [0, 1]); z_0 = zeta(X_0); x̂ = read-out(z)."""
    kind = "ncde"

    def __init__(self, nu, ny, nx, dz, width, norm: _Norm, dt_over_tref: float):
        super().__init__()
        self.norm = norm
        self.dX = 1 + nu + ny
        self.dz = int(dz)
        self.zeta = nn.Linear(self.dX, self.dz)
        self.f = MLP(self.dz, self.dz * self.dX, int(width), 2, "silu")
        self.head = nn.Linear(self.dz, nx)
        self.tau = float(dt_over_tref)

    def X(self, u, y, k0=0):
        """(B, L, nu), (B, L, ny) -> (B, L, dX) with the time channel t_k / t_ref (k from k0)."""
        L = u.shape[1]
        k = torch.arange(k0, k0 + L, device=u.device, dtype=u.dtype) if not torch.is_tensor(k0) else \
            k0[:, None].to(u.dtype) + torch.arange(L, device=u.device, dtype=u.dtype)[None]
        t = (k * self.tau).expand(u.shape[0], L)[..., None]
        return torch.cat([t, self.norm.inp(u, y)], -1)

    def field(self, z, dX):
        F = torch.tanh(self.f(z)).view(-1, self.dz, self.dX)
        return torch.bmm(F, dX[:, :, None])[:, :, 0]

    def cde_step(self, z, dX):
        k1 = self.field(z, dX)
        k2 = self.field(z + 0.5 * k1, dX)
        k3 = self.field(z + 0.5 * k2, dX)
        k4 = self.field(z + k3, dX)
        return z + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def seq_norm_from(self, z, X):
        """z at X[:, 0]; returns normalised estimates at samples 1..L-1 and the final z."""
        outs = []
        for j in range(X.shape[1] - 1):
            z = self.cde_step(z, X[:, j + 1] - X[:, j])
            outs.append(z)
        Z = torch.stack(outs, 1)
        return self.head(Z), z

    def seq(self, u, y):
        X = self.X(u, y)
        z0 = self.zeta(X[:, 0])
        o, _ = self.seq_norm_from(z0, X)
        return self.norm.out(torch.cat([self.head(z0)[:, None], o], 1))

    def step(self, state, u_k, y_k):
        """state = (z_{k-1}, X_{k-1}, k) or None."""
        if state is None:
            X0 = self.X(u_k[:, None], y_k[:, None])[:, 0]
            z = self.zeta(X0)
            return (z, X0, 0), self.norm.out(self.head(z))
        z, Xp, k = state
        Xk = self.X(u_k[:, None], y_k[:, None], k + 1)[:, 0]
        z = self.cde_step(z, Xk - Xp)
        return (z, Xk, k + 1), self.norm.out(self.head(z))


def ncde_params(nu, ny, nx, dz, W) -> int:
    dX = 1 + nu + ny
    return (dX * dz + dz) + (dz * W + W + W * W + W + W * dz * dX + dz * dX) + (dz * nx + nx)


def solve_ncde(nu, ny, nx, dz, budget) -> int:
    return min(range(4, 4096), key=lambda W: abs(ncde_params(nu, ny, nx, dz, W) - budget))


# ------------------------------------------------------------------------------------ deployed observer

class SeqObserver(Observer):
    """x̂_0 = the prior (no measurement read yet, like the filters' x̂_0); advance(y_k, u_k, y_{k+1},
    u_{k+1}) reads sample k on the first call, then sample k+1, and returns x̂_{k+1}."""
    learned = True

    def __init__(self, name, net):
        self.name, self.net = name, net.eval()

    @torch.no_grad()
    def reset(self, B, x0_hat, u0):
        self.xh = x0_hat.clone()
        self.state, self.fresh = None, True

    def estimate(self):
        return self.xh

    @torch.no_grad()
    def advance(self, y_k, u_k, y_k1, u_k1):
        if self.fresh:
            self.state, _ = self.net.step(None, u_k, y_k)
            self.fresh = False
        self.state, self.xh = self.net.step(self.state, u_k1, y_k1)


# ------------------------------------------------------------------------------------ dynamics models

def mlp_params(nx, nu, W) -> int:
    return (nx + nu) * W + W + W * W + W + W * nx + nx


def solve_mlp(nx, nu, budget) -> int:
    return min(range(4, 4096), key=lambda W: abs(mlp_params(nx, nu, W) - budget))


class NoPhysics:
    """A 'model' with f = 0 and the plant's known measurement map (NODE: everything is learned)."""

    def __init__(self, model):
        self.nx, self.nu, self.sensors, self.ny = model.nx, model.nu, model.sensors, model.ny
        self.name, self._h = "nophysics", model.h

    def f(self, x, u):
        return torch.zeros_like(x)

    def h(self, x):
        return self._h(x)

    def params(self):
        return {}


def deployed_dynamics(model, net: ResidualMLP, hybrid: bool) -> GreyBoxModel:
    """f = f_known + r (hybrid) or r alone, h known; functional MLP (EKF jacrev-able)."""
    gm = GreyBoxModel(model if hybrid else NoPhysics(model), net.cpu())
    gm.name = ("ude" if hybrid else "node") + f"[{sum(p.numel() for p in net.parameters())}]"
    return gm


def rk4_train(fm, net, x, u, dt):
    """One RK4 / ZOH step of f_known (fm, may be None) + net, differentiable in net."""
    def f(xx):
        out = net(xx, u)
        return out if fm is None else out + fm.f(xx, u)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
