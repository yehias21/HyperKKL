"""Optimisation diagnostics: what an autoresearch loop needs to tell a bad recipe from a
bad idea (exploding / vanishing gradients, permanent clipping, a dead learning rate,
non-finite steps, weights that stopped moving).

Per-step statistics accumulate ON THE DEVICE (no host sync inside the loop); ``flush`` is
called at the existing logging points and returns one summary dict per window:

    grad_mean, grad_max   pre-clip global gradient norm over the window
    clip_frac             fraction of steps whose gradient was clipped
    nonfinite             steps with a NaN/Inf gradient (the optimiser step still ran)
    lr                    current learning rate
    param_norm            global parameter norm
    update_ratio          ||theta_now - theta_at_last_flush|| / ||theta_at_last_flush||
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TrainMonitor:
    def __init__(self, params, opt, max_norm: float = 1.0):
        self.params = [q for q in params if q.requires_grad]
        self.opt, self.max_norm = opt, float(max_norm)
        dev = self.params[0].device if self.params else torch.device("cpu")
        self._zero = lambda: torch.zeros((), device=dev)
        self._reset()
        self._ref = self._snapshot()
        self.windows: list[dict] = []

    def _reset(self):
        self.n = 0
        self.s_sum, self.s_max = self._zero(), self._zero()
        self.n_clip, self.n_bad = self._zero(), self._zero()

    @torch.no_grad()
    def _snapshot(self):
        return [q.detach().clone() for q in self.params]

    def clip(self) -> torch.Tensor:
        """Drop-in for ``nn.utils.clip_grad_norm_(params, max_norm)`` that also records."""
        g = nn.utils.clip_grad_norm_(self.params, self.max_norm)
        fin = torch.isfinite(g)
        gf = torch.where(fin, g, torch.zeros_like(g))
        self.s_sum += gf
        self.s_max = torch.maximum(self.s_max, gf)
        self.n_clip += (gf > self.max_norm).float()
        self.n_bad += (~fin).float()
        self.n += 1
        return g

    @torch.no_grad()
    def flush(self, step: int) -> dict:
        n = max(self.n, 1)
        pn = math.sqrt(sum(float(q.pow(2).sum()) for q in self.params))
        dn = math.sqrt(sum(float((q - r).pow(2).sum()) for q, r in zip(self.params, self._ref)))
        rn = math.sqrt(sum(float(r.pow(2).sum()) for r in self._ref))
        d = {"step": int(step),
             "grad_mean": float(self.s_sum) / n, "grad_max": float(self.s_max),
             "clip_frac": float(self.n_clip) / n, "nonfinite": int(self.n_bad),
             "lr": float(self.opt.param_groups[0]["lr"]),
             "param_norm": pn, "update_ratio": dn / max(rn, 1e-12)}
        self.windows.append(d)
        self._reset()
        self._ref = self._snapshot()
        return d

    @staticmethod
    def fmt(d: dict) -> str:
        return (f"|g| {d['grad_mean']:.2e}/{d['grad_max']:.2e}  clip {d['clip_frac']:.2f}  "
                f"lr {d['lr']:.1e}  dθ/θ {d['update_ratio']:.1e}"
                + (f"  NONFINITE {d['nonfinite']}" if d["nonfinite"] else ""))


def flags(windows: list[dict]) -> list[str]:
    """Plain-language warnings over a whole run, for reports and the autoresearch loop."""
    if not windows:
        return []
    out = []
    if any(w["nonfinite"] for w in windows):
        out.append(f"non-finite gradients in {sum(w['nonfinite'] for w in windows)} steps")
    late = windows[len(windows) // 2:]
    if all(w["clip_frac"] > 0.9 for w in late):
        out.append("gradient clipped on >90% of steps in the second half (lr too high or loss scale off)")
    if windows[-1]["update_ratio"] < 1e-5 and windows[-1]["lr"] > 0:
        out.append("parameters stopped moving (update ratio < 1e-5)")
    g0, g1 = windows[0]["grad_mean"], windows[-1]["grad_mean"]
    if g0 > 0 and g1 / g0 > 10:
        out.append(f"gradient norm grew {g1 / g0:.0f}x over training")
    return out
