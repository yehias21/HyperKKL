"""Conditioner capacity at each budget: what 'hyper' and 'lora' actually parameterise.

Uses hyperkkl.conditioning.match_width / build_conditioner on a frozen base decoder of the
phase-1 shape (BaseMap(nz, nx, dec_width, dec_depth, 'tanh')) with the SFM context dim. The
result does not depend on trained weights, only on shapes, so dummy normalisation stats are used.

    hyper: Delta W_l = H_l t(c), t(c) in R^k (k = solved width) -> every weight update lies in a
           FIXED k-dimensional linear subspace of decoder-weight space (Park, review section 3.1)
    lora : Delta W_l = U_l(t) V_l(t) / sqrt(r), t(c) in R^w (w = solved trunk width): rank <= r
           per layer, factors bilinear in a w-dim feature (U affine, V linear in t)
"""

from __future__ import annotations

import torch

from hyperkkl.conditioning import build_conditioner, match_width
from hyperkkl.nets import BaseMap, n_params

from .store import ROLES

BUDGETS = (35000, 140000, 560000)
RANKS = (1, 4, 16)


def _base(nz, nx, width=128, depth=3):
    return BaseMap(nz, nx, width, depth, "tanh", (torch.zeros(nz), torch.ones(nz)), (torch.zeros(nx), torch.ones(nx)))


def capacity(systems=("vdp", "battery"), dc=16, dec_width=128, dec_depth=3) -> dict:
    """{(system, kind, budget, rank): {...}} for kinds latent/concat/film/lora/hyper."""
    out = {}
    for s in systems:
        nz, nx = int(ROLES[s]["nz"]), int(ROLES[s]["nx"])
        base = _base(nz, nx, dec_width, dec_depth)
        dims = [(l.in_features, l.out_features) for l in base.layers]
        z_std = torch.ones(nz)
        for b in BUDGETS:
            for kind in ("latent", "concat", "film", "lora", "hyper"):
                for r in (RANKS if (kind == "lora" and b == 140000) else (4,)):
                    w, npar = match_width(kind, base, dc, b, r, z_std)
                    d = {"system": s, "kind": kind, "budget": b, "rank": r, "width": w, "n_params": npar,
                         "nz": nz, "nx": nx, "layers": dims}
                    if kind == "hyper":
                        d["trunk_hidden"] = max(w, 64)
                        d["subspace_dim"] = w
                        d["desc"] = (f"Delta W in a fixed {w}-dim linear subspace (trunk {dc}->{max(w, 64)}->{w}); "
                                     f"heads {sum(a * o + o for a, o in dims)} x {w}")
                    elif kind == "lora":
                        d["trunk_hidden"] = w
                        d["max_rank"] = [min(r, a, o) for a, o in dims]
                        d["desc"] = (f"rank <= {r} per layer (max ranks {d['max_rank']}), U(t)V(t) bilinear in a "
                                     f"{w}-dim trunk feature")
                    elif kind == "film":
                        d["desc"] = f"per-feature scale+shift of {sum(o for _, o in dims[:-1])} hidden units, generator width {w}"
                    elif kind == "concat":
                        d["desc"] = f"additive output residual R([z,c]) - R([z,0]), MLP width {w}"
                    else:
                        d["desc"] = f"latent injection Phi([z,c]) - Phi([z,0]), MLP width {w}"
                    cond = build_conditioner(kind, base, dc, w, r, z_std)
                    assert n_params(cond) == npar
                    out[(s, kind, b, r)] = d
    return out
