"""Evaluation conditions of bench v2 (DESIGN.md section 1). Deterministic in (cell, split, condition).

sensor (both families): clean; outliers (2 % of samples per output +-12 y_std); burst_outage (all outputs frozen at
  0.6 T for 2 t_ref); poor_init (prior at a far corner, 0.9 half-width). Family A also: noise10 (white sensor noise
  raised to 10 % y_std); dropout (10 % of samples lost, previous sample held).
input record (family B, operating-profile tiers): input_realistic (the plant's preset, profiles_v2.SPECS[...]['imperf']);
  input_cmd_vs_applied (the observer is told the operator's COMMAND, the plant received the actuator output);
  input_quant (12-bit logged input over the actuator span). The plant always received the applied input.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch


def _seed(*k):
    return int(hashlib.sha256("|".join(map(str, k)).encode()).hexdigest(), 16) % (2 ** 31)


def _cmd(cell, es):
    import profiles_v2 as PV
    out = []
    for i, m in enumerate(es.meta):
        sp = m["profile_split"]
        pr = PV.generate(cell.plant, "train" if sp == "train" else sp, int(m["profile_seed"]), T=cell.des.T, dt=cell.des.dt)
        err = float(np.abs(pr.u - es.u[i, :, 0].double().numpy()).max())
        assert err < 1e-4 * max(1.0, float(np.abs(pr.u).max())), f"profile regeneration mismatch {err}"
        out.append(pr.cmd)
    return np.stack(out)


def apply(cell, es, split, c):
    from hyperkkl.evaluate import initial_guess
    s = _seed(cell.family, cell.plant, cell.variant, split, c)
    Y, U = es.y.clone(), es.u.clone()
    ys = torch.as_tensor(np.asarray(cell.sc["y_std"], float), dtype=Y.dtype)
    E, N1, ny = Y.shape
    x0 = initial_guess(cell.cfg, E, "center", 0, "cpu")
    g = torch.Generator().manual_seed(s)
    if c == "clean":
        pass
    elif c == "outliers":
        m = torch.rand(Y.shape, generator=g) < 0.02
        sg = torch.where(torch.rand(Y.shape, generator=g) < 0.5, -1.0, 1.0)
        Y = torch.where(m, Y + 12.0 * ys * sg, Y)
    elif c == "burst_outage":
        k0 = int(round(0.6 * (N1 - 1)))
        L = max(0, min(int(round(2.0 * cell.t_ref / cell.des.dt)), N1 - 1 - k0))
        if L > 0:
            Y[:, k0 + 1:k0 + L + 1] = Y[:, k0:k0 + 1]
    elif c == "poor_init":
        x0 = initial_guess(cell.cfg, E, "poor", s, "cpu")
    elif c == "noise10":
        Y = Y + math.sqrt(0.10 ** 2 - 0.03 ** 2) * ys * torch.randn(Y.shape, generator=g)
    elif c == "dropout":
        hold = torch.rand(E, N1, generator=g) < 0.10
        for k in range(1, N1):
            Y[:, k] = torch.where(hold[:, k, None], Y[:, k - 1], Y[:, k])
    elif c.startswith("input_"):
        import profiles_v2 as PV
        P = PV.SPECS[cell.plant]
        span = float(P["hi"] - P["lo"])
        if c == "input_realistic":
            imp = P["imperf"]
        elif c == "input_cmd_vs_applied":
            imp = PV.Imperfection(use_command=True)
        elif c == "input_quant":
            imp = PV.Imperfection(quant=span / 4096)
        else:
            raise KeyError(c)
        cmd = _cmd(cell, es) if imp.use_command else [None] * E
        Uo = [PV.observed_input(es.u[i, :, 0].double().numpy(), cell.des.dt, imp, span, s + i, cmd=cmd[i])
              for i in range(E)]
        U = torch.as_tensor(np.stack(Uo), dtype=es.u.dtype)[..., None]
    else:
        raise KeyError(c)
    return es, Y, U, x0
