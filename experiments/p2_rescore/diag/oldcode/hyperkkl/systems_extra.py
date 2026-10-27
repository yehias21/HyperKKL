"""Two further plants, same contract as ``systems.py`` (torch-batched f(x, u), h(x) = x[:, sensors]).

    battery   lithium-ion cell: 2-RC equivalent circuit, lumped thermal model, BMS voltage
              channel (5 states, input = current, nonlinear OCV, Joule + reversible heat)
    network   ring of five diffusively coupled, heterogeneous FitzHugh-Nagumo neurons
              (10 states, input on node 0, three membrane potentials measured)

They live in their own module so that adding them leaves the source of the cache-hashed
modules (cache.DATA_MODULES / MODEL_MODULES) untouched. ``hyperkkl/__init__.py`` imports this
module on the first registry lookup of a name ``systems.py`` does not define.
"""

from __future__ import annotations

import math

import torch

from .systems import Plant, _pos, register


def _sig(v):
    return 1.0 / (1.0 + torch.exp(-v))


@register("battery")
class LiIonCell(Plant):
    r"""Lithium-ion cell: 2-RC Thevenin model with SOC, lumped temperature and the BMS
    voltage channel. Time unit = 1 minute (``TIME_UNIT`` s); parameters are SI.

    States x = (SOC, V1, V2, dT, Vm). u = current demand I [A], positive = discharge.

        SOC' = -I_c / (3600 Q)                       I_c: current the BMS lets through
        Vi'  = I_c / Ci - Vi / (Ri(T) Ci),  i = 1, 2 (short / long transient RC pairs)
        C_th dT' = I_c^2 R0(T) + V1^2/R1(T) + V2^2/R2(T) - I_c T dUdT - hA dT
        tau_f Vm' = OCV(SOC) + dUdT dT - V1 - V2 - R0(T) I_c - V_REF - Vm
        y = (dT, Vm)                                 thermistor and BMS voltage channel

    OCV(SOC) and the asymptotic (SOC > 0.2) R/C values are Chen & Rincon-Mora, IEEE Trans.
    Energy Convers. 21(2), 2006 (850 mAh polymer Li-ion cell): tau1 = R1 C1 = 33 s,
    tau2 = R2 C2 = 223 s. Their exponential SOC dependence of R/C is dropped: below
    1.5 % effect on R and 7 % on C1 for SOC >= 0.2, and C_TS(SOC) turns negative for SOC < 0.005
    (and earlier under perturbation). Heat generation follows Bernardi, Pawlikowski & Newman,
    J. Electrochem. Soc. 132(1), 1985 (irreversible = dissipation in R0, R1, R2; reversible =
    -I T dU/dT); lumped thermal model as in Forgez et al., J. Power Sources 195(9), 2010.
    dUdT = -0.1 mV/K is within the range Viswanathan et al. (J. Power Sources 195(11), 2010)
    report for layered oxides; R_i(T) Arrhenius with E_a / R_gas = 3000 K (25 kJ/mol). C_th and
    hA are estimates for a 17 g pouch, cp ~ 1 kJ/kg/K and ~51 cm^2 under natural convection
    (h ~ 10 W/m^2/K): thermal time constant 340 s.

    Measured output. Terminal voltage V = OCV - V1 - V2 - R0 I has direct feedthrough of u,
    but h only sees x, and the EKF / PF (observers._selection) assume y = x[:, sensors]. So
    the measured quantity is the BMS voltage channel Vm, terminal voltage minus a fixed 3.7 V
    reference behind a first-order acquisition filter (tau_f = 2 s, faster than every plant
    mode by a factor of 16). It keeps the R0 I drop, so y still responds to u immediately at
    the scale of every other mode. The cell temperature is measured too, as in any BMS:
    from the voltage alone dT is nearly unobservable (it enters only through dU/dT and
    R(T); empirical Gramian lam_min / lam_max 1e-10..1e-6 at rest, ~1e-4 under load), with the
    thermistor the weakest direction is the V1 / V2 split (1.3e-3 to 1.7e-3).

    BMS current derating. A constant 1 C current over the 30 min horizon moves SOC by 0.5, so
    from the edges of the IC box it would leave [0, 1], where the OCV fit is meaningless. The BMS passes discharge current with gain
    sigmoid((SOC - 0.08) / 0.01) and charge current with sigmoid((0.92 - SOC) / 0.01), i.e.
    SOC-dependent current limits that are 1 to within 1e-3 on [0.15, 0.85]. This, Joule heat
    (I^2) and the T-dependent resistances make the plant non-affine in u by itself, so its
    config uses ``actuator: affine``.
    """
    nx = 5
    PARAMS = {"Q": 0.85, "R0": 0.07446, "R1": 0.04669, "C1": 703.6, "R2": 0.04984, "C2": 4475.0,
              "C_th": 17.0, "hA": 0.05, "Ea_R": 3000.0, "dUdT": -1.0e-4,
              "T_amb": 298.15, "tau_f": 2.0}
    # operating conditions: ambient temperature and cooling (hA; scaled by e^-0.3 it pushed
    # dT to 1.7x the envelope half-width at the 2.5 C inputs of input protocol v1); tau_f is
    # the BMS's own filter
    NOT_PERTURBED = ("T_amb", "hA", "tau_f")
    DEFAULT_SENSORS = (3, 4)
    TIME_UNIT = 60.0                                         # seconds per model time unit
    V_REF = 3.7                                              # V, subtracted in the BMS channel
    OCV_COEF = (-1.031, -35.0, 3.685, 0.2156, -0.1178, 0.3201)   # Chen & Rincon-Mora (2006)
    DERATE = (0.08, 0.92, 0.01)                              # SOC low / high limit, width

    def ocv(self, soc):
        """OCV(SOC) [V], strictly increasing. SOC is softly kept in [0.01, 0.99] (C^inf
        corners of width ~1e-3) so the exponential term cannot blow up off the domain."""
        c0, c1, k0, k1, k2, k3 = self.OCV_COEF
        s = 0.01 + _pos(soc - 0.01) - _pos(soc - 0.99)
        return c0 * torch.exp(c1 * s) + k0 + k1 * s + k2 * s ** 2 + k3 * s ** 3

    def cell_current(self, soc, u):
        lo, hi, w = self.DERATE
        return _sig((soc - lo) / w) * _pos(u) - _sig((hi - soc) / w) * _pos(-u)

    def f(self, x, u):
        return self._f(x, u, slow_branch=True)

    UNMODELLED = ("the slow RC branch R2 || C2 (model: the textbook 1-RC equivalent circuit; "
                  "V2 is not driven, not in the terminal voltage and dissipates no heat)")

    def f_known(self, x, u):
        return self._f(x, u, slow_branch=False)

    def _f(self, x, u, slow_branch: bool):
        soc, v1, v2, dT, vm = x.unbind(1)
        I = self.cell_current(soc, u[:, 0])
        if not slow_branch:              # the 1-RC model: V2 only relaxes, enters nothing else
            I2, v2_out = 0.0 * I, 0.0 * v2
        else:
            I2, v2_out = I, v2
        T = self.T_amb + dT
        arr = torch.exp(self.Ea_R * (1.0 / T - 1.0 / self.T_amb))        # R(T) / R(T_amb)
        r0, r1, r2 = self.R0 * arr, self.R1 * arr, self.R2 * arr
        heat = I ** 2 * r0 + v1 ** 2 / r1 + v2_out ** 2 / r2 - I * T * self.dUdT - self.hA * dT
        v_t = self.ocv(soc) + self.dUdT * dT - v1 - v2_out - r0 * I
        s = self.TIME_UNIT
        return torch.stack([-s * I / (3600.0 * self.Q),
                            s * (I - v1 / r1) / self.C1,
                            s * (I2 - v2 / r2) / self.C2,
                            s * heat / self.C_th,
                            s * (v_t - self.V_REF - vm) / self.tau_f], 1)


@register("network")
class FHNRing(Plant):
    r"""Ring of five diffusively coupled FitzHugh-Nagumo oscillators (FitzHugh, Biophys. J.
    1(6), 1961; Nagumo, Arimoto & Yoshizawa, Proc. IRE 50(10), 1962), node parameters as the
    ``fhn`` plant.

        v_i' = v_i - v_i^3/3 - w_i + I_ext + kappa sum_j a_ij (v_j - v_i) + [i = 0] u
        w_i' = eps_i (v_i + a - b w_i),   eps_i = eps exp(spread (i - 2) / 2)

    x = (v_0, w_0, ..., v_4, w_4); a_ij = 1 for ring neighbours (symmetric Laplacian, the
    gap-junction form of electrical coupling). Measured: v_0 (the driven node), v_1, v_3, so
    each unmeasured node (2, 4) has two measured neighbours.

    The recovery rates differ (eps_i from 0.044 to 0.15): identical nodes at this coupling
    collapse onto the synchronous manifold, where the ring is one FHN and nodes are
    interchangeable. With the spread the nodes frequency-lock with phase lags at zero input
    and are pulled apart by the input on node 0. kappa = 0.5, spread = 0.6 were chosen by the
    empirical observability Gramian over a 20-unit window (median lam_min / lam_max ~1e-4,
    5e-5 to 2e-4 across IC sets; 1.5e-5 at kappa = 0.3, spread = 0.3; 2e-6 at kappa = 1,
    spread = 0.9). The ratio is small for the reason tests/test_systems.py gives for rossler
    and lorenz: the weak directions are the fast transverse contraction of the relaxation
    oscillators (and the antisymmetric v_2 - v_4 mode of the unmeasured nodes). Every
    direction below 1e-3 shrinks to < 0.25 of its size within the window (detectable), while
    the well-observed ones grow up to 13x. Measuring four nodes only gives 1.4e-4.
    """
    nx = 10
    PARAMS = {"a": 0.7, "b": 0.8, "eps": 0.08, "I_ext": 0.5, "kappa": 0.5, "spread": 0.6}
    DEFAULT_SENSORS = (0, 2, 6)
    N_NODES = 5
    EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))       # ring

    UNMODELLED = "the coupling edge 2-3 of the ring (model: chain 3-4-0-1-2, graph partly unknown)"
    EDGES_KNOWN = tuple(e for e in EDGES if e != (2, 3))

    def laplacian(self, dtype=torch.float64, device=None, edges=None):
        L = torch.zeros(self.N_NODES, self.N_NODES, dtype=dtype, device=device)
        for i, j in (self.EDGES if edges is None else edges):
            L[i, j] = L[j, i] = 1.0
        return L - torch.diag(L.sum(1))

    def f(self, x, u):
        return self._f(x, u, self.EDGES)

    def f_known(self, x, u):
        return self._f(x, u, self.EDGES_KNOWN)

    def _f(self, x, u, edges):
        v, w = x[:, 0::2], x[:, 1::2]                                       # (B, 5) each
        eps = self.eps * torch.exp(self.spread * (torch.arange(self.N_NODES, dtype=x.dtype,
                                                               device=x.device) - 2) / 2)
        drive = torch.cat([u[:, :1], torch.zeros_like(v[:, 1:])], 1)       # no in-place: vmap-safe
        dv = v - v ** 3 / 3 - w + self.I_ext + self.kappa * v @ self.laplacian(x.dtype, x.device, edges) + drive
        dw = eps * (v + self.a - self.b * w)
        return torch.stack([dv, dw], 2).reshape(x.shape[0], self.nx)
