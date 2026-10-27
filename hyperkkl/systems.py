"""Benchmark plants  xdot = f(x, u),  y = h(x).

Every plant is torch-batched: ``f(x, u)`` takes ``x`` of shape (B, nx) and ``u`` of shape
(B, nu) and returns (B, nx); ``h(x)`` returns the measured coordinates ``x[:, sensors]``.
There is no scalar/numpy path.

Physical parameters are plain attributes listed in ``PARAMS``, so a mismatched plant is
``plant.perturbed(rel)`` and nothing else changes.

The input enters each plant in a different way (CDC R1 item 3 asked for more than
additive forcing): additive (vdp, fhn, rossler, lorenz), state-dependent gain (duffing),
bilinear (bioreactor), matrix-valued G(x) (two_link_arm, cartpole), distributed
(quad_tank, highway_traffic). ``SaturatingActuator`` additionally makes any plant
non-affine in u.
"""

from __future__ import annotations

import math

import torch

REGISTRY: dict = {}


def register(name):
    def deco(cls):
        REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def _pos(v, eps: float = 1e-6):
    """Smooth positive part, C-infinity, equal to v for v >> sqrt(eps)."""
    return 0.5 * (v + torch.sqrt(v * v + eps))


class Plant:
    nx: int = 0
    nu: int = 1
    PARAMS: dict = {}            # name -> nominal value (scalars only)
    NOT_PERTURBED: tuple = ()    # parameters that are operating conditions, not plant physics
    DEFAULT_SENSORS: tuple = (0,)
    name = "plant"

    def __init__(self, sensors=None, **params):
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        for k, v in self.PARAMS.items():
            setattr(self, k, float(params.get(k, v)))
        self.sensors = tuple(self.DEFAULT_SENSORS if sensors is None else sensors)
        self.ny = len(self.sensors)

    # -- interface -------------------------------------------------------------
    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def h(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, list(self.sensors)]

    # Incomplete knowledge: the part of f a modeller would plausibly have, with one
    # physically identified term missing or simplified (UNMODELLED says which). Every
    # model-based component (filters, model-based KKL, the phase-1 PDE term) gets this
    # model when system.knowledge = partial; the real plant keeps f.
    UNMODELLED: str = ""

    def f_known(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.name}: no partial-knowledge model defined")

    # -- helpers ---------------------------------------------------------------
    def params(self) -> dict:
        return {k: getattr(self, k) for k in self.PARAMS}

    def perturbed(self, rel: float) -> "Plant":
        """The same plant with every physical parameter off nominal by exp(+-rel).

        Signs alternate over the sorted parameter names, so the perturbation is
        deterministic and never a pure rescaling of time.
        """
        p, sign = {}, 1.0
        for k in sorted(self.PARAMS):
            if k in self.NOT_PERTURBED:
                p[k] = getattr(self, k)
                continue
            p[k] = getattr(self, k) * math.exp(sign * rel)
            sign = -sign
        return type(self)(sensors=self.sensors, **p)

    def input_gain(self, x: torch.Tensor) -> torch.Tensor:
        """G(x) = f(x, 1) - f(x, 0), shape (B, nx). Exact for input-affine plants."""
        one = torch.ones(x.shape[0], self.nu, dtype=x.dtype, device=x.device)
        return self.f(x, one) - self.f(x, 0 * one)


# ---------------------------------------------------------------------------
# 2-D plants
# ---------------------------------------------------------------------------

@register("vdp")
class VanDerPol(Plant):
    """x1' = x2,  x2' = mu (1 - x1^2) x2 - x1 + u,  y = x1."""
    nx = 2
    PARAMS = {"mu": 1.0}

    def f(self, x, u):
        x1, x2 = x[:, 0], x[:, 1]
        return torch.stack([x2, self.mu * (1 - x1 ** 2) * x2 - x1 + u[:, 0]], 1)

    UNMODELLED = "the nonlinear damping law mu (1 - x1^2) x2 (model: undamped oscillator)"

    def f_known(self, x, u):
        return torch.stack([x[:, 1], -x[:, 0] + u[:, 0]], 1)


@register("duffing")
class Duffing(Plant):
    """Reversed Duffing with state-dependent input gain.

    x1' = x2,  x2' = -x1 - x1^3 + (1 + kappa x1^2) u,  y = x1.
    """
    nx = 2
    PARAMS = {"kappa": 0.5}

    def f(self, x, u):
        x1, x2 = x[:, 0], x[:, 1]
        return torch.stack([x2, -x1 - x1 ** 3 + (1 + self.kappa * x1 ** 2) * u[:, 0]], 1)

    UNMODELLED = "the cubic stiffness -x1^3 (model: linear spring)"

    def f_known(self, x, u):
        x1, x2 = x[:, 0], x[:, 1]
        return torch.stack([x2, -x1 + (1 + self.kappa * x1 ** 2) * u[:, 0]], 1)


@register("fhn")
class FitzHughNagumo(Plant):
    """v' = v - v^3/3 - w + I + u,  w' = eps (v + a - b w),  y = v."""
    nx = 2
    PARAMS = {"a": 0.7, "b": 0.8, "eps": 0.08, "I_ext": 0.5}

    def f(self, x, u):
        v, w = x[:, 0], x[:, 1]
        return torch.stack([v - v ** 3 / 3 - w + self.I_ext + u[:, 0],
                            self.eps * (v + self.a - self.b * w)], 1)

    UNMODELLED = "the recovery leak -eps b w"

    def f_known(self, x, u):
        v, w = x[:, 0], x[:, 1]
        return torch.stack([v - v ** 3 / 3 - w + self.I_ext + u[:, 0], self.eps * (v + self.a)], 1)


@register("bioreactor")
class Bioreactor(Plant):
    """Continuous stirred bioreactor, bilinear in (state, input).

    x1' = (mu(x2) - D) x1,  x2' = -mu(x2) x1 / Y + D (x2f - x2),  D = D_nom + u,
    mu(s) = mu_max s / (Ks + s) on the smooth positive part of s, y = x2 (substrate).
    Measuring substrate rather than biomass is both the standard instrument and the
    observable choice (empirical observability index 0.70 against 0.056).
    """
    nx = 2
    PARAMS = {"mu_max": 1.0, "Ks": 1.0, "Y": 0.4, "x2f": 4.0, "D_nom": 0.3}
    # feed concentration and dilution set point are the operating point, not the
    # biology: perturbing them by e^0.5 moved the biomass equilibrium from 1.4 to 3.4,
    # outside the envelope (every mismatch-50 episode was rejected)
    NOT_PERTURBED = ("x2f", "D_nom")
    DEFAULT_SENSORS = (1,)

    def f(self, x, u):
        x1, x2 = x[:, 0], x[:, 1]
        D = self.D_nom + u[:, 0]
        sp = _pos(x2)
        mu = self.mu_max * sp / (self.Ks + sp)
        return torch.stack([(mu - D) * x1, -mu * x1 / self.Y + D * (self.x2f - x2)], 1)

    UNMODELLED = "Monod saturation of the growth rate (model: first-order kinetics mu_max s / Ks)"

    def f_known(self, x, u):
        x1, x2 = x[:, 0], x[:, 1]
        D = self.D_nom + u[:, 0]
        mu = self.mu_max * _pos(x2) / self.Ks
        return torch.stack([(mu - D) * x1, -mu * x1 / self.Y + D * (self.x2f - x2)], 1)


# ---------------------------------------------------------------------------
# 3-D chaotic plants
# ---------------------------------------------------------------------------

@register("rossler")
class Rossler(Plant):
    """x1' = -(x2 + x3),  x2' = x1 + a x2 + u,  x3' = b + x3 (x1 - c),  y = x2."""
    nx = 3
    PARAMS = {"a": 0.1, "b": 0.1, "c": 14.0}
    DEFAULT_SENSORS = (1,)

    def f(self, x, u):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([-(x2 + x3), x1 + self.a * x2 + u[:, 0],
                            self.b + x3 * (x1 - self.c)], 1)

    UNMODELLED = "the folding nonlinearity x1 x3"

    def f_known(self, x, u):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([-(x2 + x3), x1 + self.a * x2 + u[:, 0], self.b - self.c * x3], 1)


@register("lorenz")
class Lorenz(Plant):
    """sigma (x2 - x1),  x1 (rho - x3) - x2 + u,  x1 x2 - beta x3,  y = x2."""
    nx = 3
    PARAMS = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    DEFAULT_SENSORS = (1,)

    def f(self, x, u):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([self.sigma * (x2 - x1),
                            x1 * (self.rho - x3) - x2 + u[:, 0],
                            x1 * x2 - self.beta * x3], 1)

    UNMODELLED = "the coupling -x1 x3 in the measured equation"

    def f_known(self, x, u):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([self.sigma * (x2 - x1), self.rho * x1 - x2 + u[:, 0],
                            x1 * x2 - self.beta * x3], 1)


# ---------------------------------------------------------------------------
# 4-D and 5-D plants
# ---------------------------------------------------------------------------

@register("quad_tank")
class QuadrupleTank(Plant):
    """Quadruple-tank process (Johansson, IEEE TCST 2000), non-minimum-phase setting.

    The two lower levels are measured, the upper two are not. One scalar command drives
    both pumps differentially, v1 = v1n + u, v2 = v2n - u/2. sqrt is taken on a smooth
    positive part so Jacobians stay finite as h -> 0.
    """
    nx = 4
    PARAMS = {"A1": 28.0, "A2": 32.0, "A3": 28.0, "A4": 32.0,
              "a1": 0.071, "a2": 0.057, "a3": 0.071, "a4": 0.057,
              "gam1": 0.43, "gam2": 0.34, "k1": 3.33, "k2": 3.35,
              "v1n": 3.0, "v2n": 3.0, "g": 981.0}
    # g is a constant of nature and v1n, v2n are the operating point (nominal pump
    # voltages): perturbing them moved tank 2's equilibrium from 10.9 to 29.3 cm at
    # rel = 0.3, outside the envelope, so every mismatch episode was rejected
    NOT_PERTURBED = ("g", "v1n", "v2n")
    DEFAULT_SENSORS = (0, 1)

    def f(self, x, u):
        uu = u[:, 0]
        v1, v2 = self.v1n + uu, self.v2n - 0.5 * uu
        c = math.sqrt(2.0 * self.g)
        s = [torch.sqrt(_pos(x[:, i]) + 1e-6) for i in range(4)]
        d1 = (-self.a1 * c * s[0] + self.a3 * c * s[2] + self.gam1 * self.k1 * v1) / self.A1
        d2 = (-self.a2 * c * s[1] + self.a4 * c * s[3] + self.gam2 * self.k2 * v2) / self.A2
        d3 = (-self.a3 * c * s[2] + (1 - self.gam2) * self.k2 * v2) / self.A3
        d4 = (-self.a4 * c * s[3] + (1 - self.gam1) * self.k1 * v1) / self.A4
        return torch.stack([d1, d2, d3, d4], 1)

    UNMODELLED = "the square-root orifice law (model: flows linearised at the nominal equilibrium)"
    H_OP = (11.66, 10.9, 5.36, 3.3)               # nominal zero-input equilibrium (cm)

    def f_known(self, x, u):
        uu = u[:, 0]
        v1, v2 = self.v1n + uu, self.v2n - 0.5 * uu
        c = math.sqrt(2.0 * self.g)
        s = [math.sqrt(h0) + (x[:, i] - h0) / (2 * math.sqrt(h0)) for i, h0 in enumerate(self.H_OP)]
        d1 = (-self.a1 * c * s[0] + self.a3 * c * s[2] + self.gam1 * self.k1 * v1) / self.A1
        d2 = (-self.a2 * c * s[1] + self.a4 * c * s[3] + self.gam2 * self.k2 * v2) / self.A2
        d3 = (-self.a3 * c * s[2] + (1 - self.gam2) * self.k2 * v2) / self.A3
        d4 = (-self.a4 * c * s[3] + (1 - self.gam1) * self.k1 * v1) / self.A4
        return torch.stack([d1, d2, d3, d4], 1)


@register("cartpole")
class CartPole(Plant):
    """Cart-pole about the upright equilibrium, pole-balancing loop closed.

    States (x, v, th, om); cart position and pole angle are measured. Only the pole
    states are fed back (gain K_bal in input units): a full LQR drives every initial
    condition to the same origin and makes the plant unobservable as a benchmark. u is
    an exogenous command on top of the balancing loop. th is smoothly limited.
    """
    nx = 4
    PARAMS = {"mc": 1.0, "mp": 0.1, "l": 0.5, "g": 9.81, "kd_v": 0.05, "kd_w": 0.02,
              "f_scale": 5.0}
    DEFAULT_SENSORS = (0, 2)
    TH_MAX = 0.6
    K_BAL = (0.0, 0.0, -9.889688, -2.181384)

    def f(self, x, u):
        v, th, om = x[:, 1], x[:, 2], x[:, 3]
        th_s = self.TH_MAX * torch.tanh(th / self.TH_MAX)
        s, c = torch.sin(th_s), torch.cos(th_s)
        k1, k2, k3, k4 = self.K_BAL
        F = self.f_scale * (u[:, 0] - (k1 * x[:, 0] + k2 * v + k3 * th_s + k4 * om))
        den = self.mc + self.mp * s * s
        av = (F + self.mp * s * (self.l * om * om - self.g * c)) / den - self.kd_v * v
        aw = (((self.mc + self.mp) * self.g * s - c * (F + self.mp * self.l * om * om * s))
              / (self.l * den)) - self.kd_w * om
        return torch.stack([v, av, om, aw], 1)

    UNMODELLED = "large-angle pendulum terms (model: small-angle sin th = th, cos th = 1, no om^2)"

    def f_known(self, x, u):
        v, th, om = x[:, 1], x[:, 2], x[:, 3]
        th_s = self.TH_MAX * torch.tanh(th / self.TH_MAX)
        k1, k2, k3, k4 = self.K_BAL
        F = self.f_scale * (u[:, 0] - (k1 * x[:, 0] + k2 * v + k3 * th_s + k4 * om))
        av = (F - self.mp * self.g * th_s) / self.mc - self.kd_v * v
        aw = ((self.mc + self.mp) * self.g * th_s - F) / (self.l * self.mc) - self.kd_w * om
        return torch.stack([v, av, om, aw], 1)


@register("two_link_arm")
class TwoLinkArm(Plant):
    """Lightly damped planar two-link arm with gravity, torque on joint 1.

    x = (q1, q2, dq1, dq2), both joint angles measured, G(x) = M(q)^{-1} e1. Damping is
    light (kd = 0.05): heavy damping collapses every trajectory onto the hanging
    equilibrium and destroys reconstructibility.
    """
    nx = 4
    PARAMS = {"m1": 1.0, "m2": 1.0, "l1": 1.0, "lc1": 0.5, "lc2": 0.5,
              "I1": 0.083, "I2": 0.083, "g": 9.81, "kd": 0.05}
    DEFAULT_SENSORS = (0, 1)

    def f(self, x, u):
        q1, q2, dq1, dq2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        a = self.m1 * self.lc1 ** 2 + self.I1 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + self.I2
        b = self.m2 * self.l1 * self.lc2
        d = self.m2 * self.lc2 ** 2 + self.I2
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g
        g2 = self.m2 * self.lc2 * self.g
        c2, s2 = torch.cos(q2), torch.sin(q2)
        m11, m12, m22 = a + 2 * b * c2, d + b * c2, d + 0 * c2
        det = m11 * m22 - m12 ** 2
        hh = b * s2
        r1 = u[:, 0] + hh * dq2 * (2 * dq1 + dq2) - g1 * torch.cos(q1) \
            - g2 * torch.cos(q1 + q2) - self.kd * dq1
        r2 = -hh * dq1 ** 2 - g2 * torch.cos(q1 + q2) - self.kd * dq2
        return torch.stack([dq1, dq2, (m22 * r1 - m12 * r2) / det,
                            (-m12 * r1 + m11 * r2) / det], 1)

    UNMODELLED = "Coriolis and centrifugal terms (model: M(q) qdd = tau - g(q) - kd qd)"

    def f_known(self, x, u):
        q1, q2, dq1, dq2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        a = self.m1 * self.lc1 ** 2 + self.I1 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2) + self.I2
        b = self.m2 * self.l1 * self.lc2
        d = self.m2 * self.lc2 ** 2 + self.I2
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g
        g2 = self.m2 * self.lc2 * self.g
        c2 = torch.cos(q2)
        m11, m12, m22 = a + 2 * b * c2, d + b * c2, d + 0 * c2
        det = m11 * m22 - m12 ** 2
        r1 = u[:, 0] - g1 * torch.cos(q1) - g2 * torch.cos(q1 + q2) - self.kd * dq1
        r2 = -g2 * torch.cos(q1 + q2) - self.kd * dq2
        return torch.stack([dq1, dq2, (m22 * r1 - m12 * r2) / det,
                            (-m12 * r1 + m11 * r2) / det], 1)


@register("highway_traffic")
class HighwayTraffic(Plant):
    """Five-cell Greenshields freeway, ramp inflow on cell 3 is the input.

    Three loop detectors (cells 1, 3, 5); with the downstream detector alone the chain
    is a five-fold differentiator and effectively unobservable.
    """
    nx = 5
    PARAMS = {"v_free": 30.0, "rho_max": 0.25, "L": 500.0,
              "u_main_base": 1.2, "u_ramp_base": 0.3}
    # demand is an operating condition: scaling it by e^{0.3} pushed the road past capacity
    # into congestion, where the Greenshields model used here is not valid
    NOT_PERTURBED = ("u_main_base", "u_ramp_base")
    DEFAULT_SENSORS = (0, 2, 4)

    def f(self, x, u):
        q = self.v_free * (x - x ** 2 / self.rho_max)           # cell outflows
        inflow = torch.cat([torch.full_like(x[:, :1], self.u_main_base), q[:, :-1]], 1)
        z = torch.zeros_like(x[:, :1])
        ramp = torch.cat([z, z, self.u_ramp_base + u[:, :1], z, z], 1)   # no in-place: vmap-safe
        return (inflow - q + ramp) / self.L

    UNMODELLED = "congestion in the fundamental diagram (model: free-flow q = v_free rho)"

    def f_known(self, x, u):
        q = self.v_free * x
        inflow = torch.cat([torch.full_like(x[:, :1], self.u_main_base), q[:, :-1]], 1)
        z = torch.zeros_like(x[:, :1])
        ramp = torch.cat([z, z, self.u_ramp_base + u[:, :1], z, z], 1)
        return (inflow - q + ramp) / self.L


# ---------------------------------------------------------------------------
# Non-affine actuator
# ---------------------------------------------------------------------------

class SaturatingActuator(Plant):
    r"""Make an input-affine plant non-affine in u (DC-motor form).

        u_eff = u_bar tanh((u - k (x_j - x_ref)) / u_bar),   f(x, u) = f0(x) + G(x) u_eff

    The actuator saturates and is opposed by a back-EMF term on the state it drives, so
    u and x are inseparable. The back-EMF acts on the deviation from the operating point
    x_ref (the IC-box centre, near but not at the equilibrium for bioreactor and quad_tank):
    the archive used the absolute state, which moved the
    bioreactor's dilution rate to D = -0.16 at zero input, a physically different plant.
    Model-based baselines differentiate this exact map; only the input structure changes.
    The back-EMF must not change the plant's character: a system YAML may fix ``j`` / ``k``
    where the automatic design would (two_link_arm: non-collocated, pumps energy; rossler:
    k = a cancels the term that makes it chaotic).
    """

    def __init__(self, base: Plant, u_bar: float, k: float, j: int, x_ref: float):
        self.base, self.u_bar, self.k, self.j, self.x_ref = base, float(u_bar), float(k), int(j), float(x_ref)
        self.nx, self.nu, self.ny = base.nx, base.nu, base.ny
        self.sensors = base.sensors
        self.name = base.name
        self.PARAMS = base.PARAMS

    def f(self, x, u):
        zero = torch.zeros_like(u)
        f0 = self.base.f(x, zero)
        g = self.base.f(x, zero + 1) - f0
        ue = self.u_bar * torch.tanh((u[:, 0] - self.k * (x[:, self.j] - self.x_ref)) / self.u_bar)
        return f0 + g * ue[:, None]

    def h(self, x):
        return self.base.h(x)

    def params(self):
        return self.base.params()

    def perturbed(self, rel):
        return SaturatingActuator(self.base.perturbed(rel), self.u_bar, self.k, self.j, self.x_ref)


def actuator_design(base: Plant, box: torch.Tensor, u_bar: float, j: int = None) -> tuple:
    """(k, j, x_ref): j is the state the input pushes hardest over the IC box (unless
    given); the back-EMF gain makes k * half-span(x_j) = u_bar / 2, i.e. the actuator
    enters saturation inside the operating set rather than staying linear."""
    gen = torch.Generator().manual_seed(0)
    lo, hi = box[:, 0], box[:, 1]
    X = lo + (hi - lo) * torch.rand(512, base.nx, generator=gen, dtype=torch.float64)
    if j is None:
        j = int(base.input_gain(X).abs().mean(0).argmax())
    j = int(j)
    half = float(hi[j] - lo[j]) / 2
    return 0.5 * u_bar / max(half, 1e-6), j, float(lo[j] + hi[j]) / 2


class KnownPart(Plant):
    """The modeller's model under incomplete knowledge: ``base.f_known`` in place of f."""

    def __init__(self, base: Plant):
        self.base, self.nx, self.nu = base, base.nx, base.nu
        self.sensors, self.ny, self.name = base.sensors, base.ny, base.name + "/known"
        self.PARAMS = {}

    def f(self, x, u):
        return self.base.f_known(x, u)

    def params(self):
        return self.base.params()


def build_plant(scfg: dict, perturb: float = 0.0, knowledge: str = "full") -> Plant:
    """Construct the plant described by a system config section. ``knowledge='partial'``
    returns the modeller's model (``f_known``, see Plant.UNMODELLED); the actuator model
    is known in either case."""
    mk = lambda: REGISTRY[scfg["plant"]](sensors=scfg.get("sensors"), **scfg.get("params", {}))
    base = mk().perturbed(perturb) if perturb else mk()
    if knowledge == "partial":
        base = KnownPart(base)
    elif knowledge != "full":
        raise ValueError(f"knowledge must be 'full' or 'partial', not {knowledge!r}")
    act = scfg.get("actuator") or {}
    if act.get("kind", "affine") == "saturating":
        u_bar = float(act.get("u_bar", 2.0)) * float(scfg.get("u_scale", 1.0))
        k, j, xr = actuator_design(mk(), torch.tensor(scfg["ic_box"], dtype=torch.float64), u_bar,
                                   act.get("j"))
        k = float(act.get("k", k))                      # YAML override (reason given there)
        return SaturatingActuator(base, u_bar, k, j, xr)
    return base
