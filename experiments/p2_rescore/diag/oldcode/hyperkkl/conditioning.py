r"""Where the input enters a learned KKL observer (contribution C1).

The deployed observer always runs the exact latent filter  zhat' = A zhat + B y  and
decodes  xhat = T+(zhat; c),  c = E(u-history). The variants differ ONLY in where c acts:

    latent   zhat' = A zhat + B y + Phi(zhat, c)   decoder static        (latent space)
    concat   T+(z; c) = T0+(z) + R([z, c])         c joins the input     (input space)
    film     h_l <- h_l (1 + gamma_l(c)) + beta_l(c), pre-activation     (feature space)
    lora     W_l <- W_l + U_l(c) V_l(c) / sqrt(r)                        (weight space, low rank)
    hyper    W_l <- W_l + DeltaW_l(c),  b_l <- b_l + Deltab_l(c)         (weight space, full)

Controls that make the comparison causal (AAAI R2 W2, AAAI AI reviewer):
  * same frozen base maps T0, T0+ from phase 1;
  * same context encoder E (identical input history and architecture);
  * same trainable-parameter budget: each variant has one width knob, solved by
    ``match_width`` so the conditioner's parameter count hits the budget;
  * same zero-input behaviour: every Delta is exactly 0 at c = 0 (bias-free generators
    with zero-preserving activations -- for lora the V factor --, or an explicit
    R([z,c]) - R([z,0]) difference);
  * no parity constraint: generators use SiLU, which is zero-preserving but NOT odd. A
    bias-free tanh generator is odd in c, and the context encoders are odd in u, so FiLM
    and hyper could only represent input effects that flip sign with u (and the original
    lora was exactly even: blind to the sign of the input).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .nets import MLP, BaseMap, n_params

DECODER_KINDS = ("none", "concat", "film", "lora", "hyper")
ALL_KINDS = DECODER_KINDS + ("latent",)


def _layer_dims(base: BaseMap):
    return [(l.in_features, l.out_features) for l in base.layers]


class ConditionedDecoder(nn.Module):
    """x = T+(z; c). ``base`` is frozen; only the conditioning pathway trains."""

    def __init__(self, base: BaseMap, kind: str, dc: int, width: int, rank: int = 4):
        super().__init__()
        assert kind in DECODER_KINDS, kind
        self.base, self.kind, self.dc, self.width, self.rank = base, kind, dc, width, rank
        for p in base.parameters():
            p.requires_grad_(False)
        dims = _layer_dims(base)
        nz, nx = dims[0][0], dims[-1][1]
        if kind == "concat":
            self.R = MLP(nz + dc, nx, width, 2, "silu", zero_last=True)
        elif kind == "film":
            self.hidden = [d_out for _, d_out in dims[:-1]]
            self.G = MLP(dc, 2 * sum(self.hidden), width, 1, "silu", bias=False, zero_last=True)
        elif kind == "lora":
            self.trunk = MLP(dc, width, width, 1, "silu", bias=False)
            # U affine, V linear in the (odd) trunk features: Delta = 0 at c = 0 because
            # V(0) = 0. With both factors linear, U(t) V(t) = U(-t) V(-t), so the decoder
            # was exactly even in c (blind to the sign of the input) and flat to first order.
            self.U = nn.ModuleList(nn.Linear(width, a * rank) for a, _ in dims)
            self.V = nn.ModuleList(nn.Linear(width, rank * b, bias=False) for _, b in dims)
            for v in self.V:
                nn.init.zeros_(v.weight)        # Delta = 0 at init, gradient path open via U
        elif kind == "hyper":
            # trunk hidden width >= 64: at a matched budget the heads cost ~35k parameters
            # per trunk output, so width is only 1-16; the trunk itself must not be the
            # bottleneck as well (8 hidden units before)
            self.trunk = MLP(dc, width, max(width, 64), 1, "silu", bias=False)
            self.H = nn.ModuleList(nn.Linear(width, a * b + b, bias=False) for a, b in dims)
            for h in self.H:
                nn.init.zeros_(h.weight)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = self.base
        if self.kind == "none":
            return b(z)
        h = b.inp.norm(z)
        if self.kind == "concat":
            # two calls of identical shape, not one stacked batch: a CPU GEMM may round
            # rows differently by position, and then R([h,0]) - R([h,0]) != 0 (~1e-7)
            r = self.R(torch.cat([h, c], -1)) - self.R(torch.cat([h, torch.zeros_like(c)], -1))
            return b.out.denorm(b.mlp(h) + r)
        n = len(b.layers)
        if self.kind == "film":
            gb = self.G(c)
            off = 0
            for i, layer in enumerate(b.layers):
                h = layer(h)
                if i < n - 1:
                    d = self.hidden[i]
                    g, be = gb[:, off:off + d], gb[:, off + d:off + 2 * d]
                    off += 2 * d
                    h = b.act(h * (1 + g) + be)
            return b.out.denorm(h)
        t = self.trunk(c)
        for i, layer in enumerate(b.layers):
            a, o = layer.in_features, layer.out_features
            if self.kind == "lora":
                U = self.U[i](t).view(-1, a, self.rank)
                V = self.V[i](t).view(-1, self.rank, o)
                dh = torch.bmm(torch.bmm(h[:, None], U), V)[:, 0] / math.sqrt(self.rank)
            else:
                w = self.H[i](t)
                dW = w[:, :a * o].view(-1, o, a)
                dh = torch.bmm(dW, h[:, :, None])[:, :, 0] + w[:, a * o:]
            h = layer(h) + dh
            if i < n - 1:
                h = b.act(h)
        return b.out.denorm(h)


class LatentInjection(nn.Module):
    """Phi(z, c) = M([z, c]) - M([z, 0]) in latent units; enters zhat' additively."""

    def __init__(self, nz: int, dc: int, width: int, z_std: torch.Tensor):
        super().__init__()
        self.M = MLP(nz + dc, nz, width, 2, "silu", zero_last=True)
        self.register_buffer("z_std", torch.as_tensor(z_std, dtype=torch.float32))

    def forward(self, z, c):
        zn = z / self.z_std                     # two same-shape calls: exact 0 at c = 0 (see concat)
        return (self.M(torch.cat([zn, c], -1)) - self.M(torch.cat([zn, torch.zeros_like(c)], -1))) * self.z_std


def build_conditioner(kind: str, base_dec: BaseMap, dc: int, width: int, rank: int,
                      z_std) -> nn.Module:
    if kind == "latent":
        return LatentInjection(base_dec.layers[0].in_features, dc, width, z_std)
    return ConditionedDecoder(base_dec, kind, dc, width, rank)


def match_width(kind: str, base_dec: BaseMap, dc: int, budget: int, rank: int, z_std) -> tuple:
    """Smallest-error width so the conditioner's trainable parameters ~ budget."""
    if kind == "none":
        return 0, 0
    count = lambda w: n_params(build_conditioner(kind, base_dec, dc, w, rank, z_std))
    lo, hi = 1, 1
    while count(hi) < budget and hi < 1 << 14:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < budget:
            lo = mid + 1
        else:
            hi = mid
    best = min({max(lo - 1, 1), lo}, key=lambda w: abs(count(w) - budget))
    return best, count(best)
