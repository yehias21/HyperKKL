"""The Estimator protocol every method implements.

Online (causal) contract, one episode at a time:

    est.fit(train, val, context)        once per (plant, knowledge); may be a no-op
    est.reset(x0_prior, P0_prior)       start of an episode: the prior mean (n_x,) and prior std (n_x,)
    for k in 0..N:
        out = est.step(u[k], y[k])      returns x_hat[k] (n_x,), or (x_hat[k], P[k]) with P the
                                        (n_x, n_x) error covariance (or (n_x,) variances)

x_hat[k] may use y[0..k] and u[0..k] (u[k] is the command applied on [t_k, t_{k+1}); it is known
at t_k and carries no information about x[k]). Nothing else from the episode may be used: in
particular not x, not the episode's family / signal parameters, and not future samples.

``context`` (a dict built by the harness) is everything a method may legitimately know:
plant metadata (dt, sensors, x_std, y_std, x_mean, input box, IC box), the modeller's model
(``context['model']``: the knowledge level's model, nominal parameters; None for model-free
methods is fine), the nominal noise levels, and the knowledge level name. It never contains
the real plant's parameters at the mismatch / realistic levels.

Optional speed path: an estimator may also implement
    est.run_batch(U, Y, X0, P0) -> X_hat (E, N+1, n_x) [, P (E, N+1, n_x[, n_x])]
which MUST be equivalent to calling reset/step per episode (tests/test_protocol.py checks this
for the bundled baselines). The harness uses it when present.

Optional cost hooks: ``n_params()`` (int), ``state_bytes()`` (persistent memory in bytes),
``flops_per_step()`` (int, multiply-add = 2 FLOPs). Missing hooks are reported as null.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Estimator(Protocol):
    name: str

    def fit(self, train, val, context: dict) -> None: ...

    def reset(self, x0_prior: np.ndarray, P0_prior: np.ndarray | None = None) -> None: ...

    def step(self, u: float, y: np.ndarray): ...


class BaseEstimator:
    """Convenience base class: implement ``reset`` and ``step``; ``run_batch`` falls back to the
    per-episode loop."""
    name = "base"
    returns_covariance = False

    def fit(self, train, val, context: dict) -> None:
        self.context = context

    def reset(self, x0_prior, P0_prior=None):
        raise NotImplementedError

    def step(self, u, y):
        raise NotImplementedError


def run_episodes(est, U, Y, X0, P0=None):
    """Run an estimator over stacked episodes (U (E, N+1), or (E, N+1, n_u) for a vector input). Uses run_batch if available.
    Returns (X_hat (E, N+1, n_x), P or None)."""
    if hasattr(est, "run_batch"):
        out = est.run_batch(U, Y, X0, P0)
        if isinstance(out, tuple):
            return np.asarray(out[0], np.float64), (None if out[1] is None else np.asarray(out[1], np.float64))
        return np.asarray(out, np.float64), None
    E, N1 = U.shape[:2]
    nx = X0.shape[1]
    Xh = np.empty((E, N1, nx))
    P = None
    for e in range(E):
        est.reset(X0[e], None if P0 is None else P0[e])
        for k in range(N1):
            o = est.step(float(U[e, k]) if U.ndim == 2 else U[e, k], Y[e, k])
            if isinstance(o, tuple):
                xk, pk = o
                if P is None:
                    pk = np.asarray(pk)
                    P = np.full((E, N1) + pk.shape, np.nan)
                P[e, k] = pk
            else:
                xk = o
            Xh[e, k] = xk
    return Xh, P
