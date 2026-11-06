"""Partial (nominal) models of the real-data tracks.

``battery_pf``: the identified 2-RC equivalent-circuit model (ECM) of the Panasonic 18650PF cell,
vendored verbatim from the research code (hyperkkl/realdata.py, class Pan18650PF; the OCV basis and
its C^1 linear extension). Same state, input and output convention as the simulated ``battery``
plant (time unit 1 min, SI parameters):

    SOC' = -I / (3600 Q)
    Vi'  = I / Ci - Vi / (Ri(T) Ci),  i = 1, 2
    C_th dT' = I^2 R0(T) + V1^2 / R1(T) + V2^2 / R2(T) - I T dUdT - hA dT
    tau_f Vm' = OCV(SOC) + dUdT dT - V1 - V2 - R0(T) I - V_REF - Vm
    y = (dT, Vm)

It is a PARTIAL model of the real cell: no hysteresis, no SOC- or rate-dependence of R or Q,
constant parameters identified by least squares on the training cycles only.
"""

from __future__ import annotations

import numpy as np
import torch

from .systems import Plant, register

V_REF = 3.7
TAU_F = 2.0
TIME_UNIT = 60.0
OCV_DEG = 7
N_OCV = OCV_DEG + 3


def ocv_basis(s, w0, w1, xp=np):
    """[1, s, ..., s^DEG, exp(-s / w0), exp(-(1 - s) / w1)] on s in [0, 1]."""
    cols = [s ** i for i in range(OCV_DEG + 1)]
    cols += [xp.exp(-s / w0), xp.exp(-(1.0 - s) / w1)]
    return cols


def ocv_eval(s, coef, w0, w1, xp=np):
    """OCV with the linear extension beyond [0, 1] (slope of the edge), C^1."""
    if xp is np:
        sc = np.clip(s, 0.0, 1.0)
    else:
        sc = s.clamp(0.0, 1.0)
    v = sum(c * b for c, b in zip(coef, ocv_basis(sc, w0, w1, xp)))
    h = 1e-4
    lo = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + h, w0, w1, xp)))
    v0 = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc, w0, w1, xp)))
    hi = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + 1.0, w0, w1, xp)))
    hm = sum(c * b for c, b in zip(coef, ocv_basis(0.0 * sc + 1.0 - h, w0, w1, xp)))
    s0, s1 = (lo - v0) / h, (hi - hm) / h
    return v + s0 * xp.minimum(s - sc, 0.0 * s) + s1 * xp.maximum(s - sc, 0.0 * s) if xp is np else \
        v + s0 * torch.clamp(s - sc, max=0.0) + s1 * torch.clamp(s - sc, min=0.0)


@register("battery_pf")
class Pan18650PF(Plant):
    nx = 5
    PARAMS = {"Q": 2.9, "R0": 0.03, "R1": 0.01, "C1": 2000.0, "R2": 0.01, "C2": 50000.0,
              "C_th": 60.0, "hA": 0.1, "Ea_R": 3000.0, "dUdT": -1.0e-4, "T_amb": 298.15,
              "tau_f": TAU_F, "ow0": 0.02, "ow1": 0.05,
              **{f"o{i}": 0.0 for i in range(N_OCV)}}
    NOT_PERTURBED = ("T_amb", "tau_f", "ow0", "ow1") + tuple(f"o{i}" for i in range(N_OCV))
    DEFAULT_SENSORS = (3, 4)
    TIME_UNIT = TIME_UNIT
    V_REF = V_REF
    UNMODELLED = "identified ECM: no hysteresis, no SOC/rate dependence of R, Q"

    def ocv(self, soc):
        return ocv_eval(soc, [getattr(self, f"o{i}") for i in range(N_OCV)], self.ow0, self.ow1,
                        xp=torch)

    def f(self, x, u):
        soc, v1, v2, dT, vm = x.unbind(1)
        I = u[:, 0]
        T = self.T_amb + dT
        arr = torch.exp(self.Ea_R * (1.0 / T - 1.0 / self.T_amb))
        r0, r1, r2 = self.R0 * arr, self.R1 * arr, self.R2 * arr
        heat = I ** 2 * r0 + v1 ** 2 / r1 + v2 ** 2 / r2 - I * T * self.dUdT - self.hA * dT
        v_t = self.ocv(soc) + self.dUdT * dT - v1 - v2 - r0 * I
        s = self.TIME_UNIT
        return torch.stack([-s * I / (3600.0 * self.Q),
                            s * (I - v1 / r1) / self.C1,
                            s * (I - v2 / r2) / self.C2,
                            s * heat / self.C_th,
                            s * (v_t - self.V_REF - vm) / self.tau_f], 1)

    def f_known(self, x, u):
        return self.f(x, u)


def cold_params(params: dict, t_amb_c: float) -> tuple:
    """The 25 degC ECM moved to another ambient (research realdata_run.cold_plant): R scaled by the
    Arrhenius factor, C unchanged, T_amb = the new ambient [degC]. Returns (params, factor)."""
    import math
    p = dict(params)
    T_new = 273.15 + t_amb_c
    fac = math.exp(p["Ea_R"] * (1.0 / T_new - 1.0 / p["T_amb"]))
    for k in ("R0", "R1", "R2"):
        p[k] *= fac
    p["T_amb"] = T_new
    return p, fac


@register("emps_axis")
class EMPSAxis(Plant):
    r"""Electro-mechanical positioning system (Janot, Gautier, Brunot 2019, eq. 3): a rigid prismatic axis
    driven by a force tau with viscous + Coulomb friction and a force offset,

        q' = v,   v' = (tau - Fv v - Fc sign(v) - offset) / M,      y = q        (time unit 1 s)

    sign(v) is replaced by tanh(v / v_eps) (v_eps = 1 mm/s) so that the EKF Jacobian exists; below
    ~3 v_eps the friction is therefore smoothed (stated). M [kg], Fv [N s/m], Fc [N], offset [N] are
    identified on the training split by the authors' IDIM-LS (real/emps.py)."""
    nx = 2
    PARAMS = {"M": 95.1089, "Fv": 203.5034, "Fc": 20.3935, "offset": -3.1648, "v_eps": 1e-3}
    NOT_PERTURBED = ("v_eps",)
    DEFAULT_SENSORS = (0,)
    UNMODELLED = "identified rigid-body + symmetric smoothed Coulomb friction: no asymmetric friction, no ball-screw compliance"

    def f(self, x, u):
        q, v = x.unbind(1)
        tau = u[:, 0]
        fr = self.Fv * v + self.Fc * torch.tanh(v / self.v_eps) + self.offset
        return torch.stack([v, (tau - fr) / self.M], 1)

    def f_known(self, x, u):
        return self.f(x, u)


@register("silverbox_duffing")
class SilverboxDuffing(Plant):
    r"""Silverbox (Wigren & Schoukens 2013): an electronic implementation of a mass-spring-damper with a
    cubic hardening spring (forced Duffing oscillator), in the units of the measured voltages,

        y' = v,   v' = -d v - k1 y - k3 y^3 + g u + b0,      output y (V2 [V]); input u (V1 [V])

    with time in units of 10 ms (so the resonance near 60 Hz is ~3.8 rad per unit). d, k1, k3, g are
    identified on the training split by least squares on zero-phase smoothed derivatives (real/silverbox.py)."""
    nx = 2
    PARAMS = {"d": 0.1, "k1": 14.0, "k3": 0.0, "g": 14.0, "b0": 0.0}
    DEFAULT_SENSORS = (0,)
    UNMODELLED = "identified second-order Duffing structure: no higher-order dynamics of the electronic circuit"

    def f(self, x, u):
        y, v = x.unbind(1)
        return torch.stack([v, -self.d * v - self.k1 * y - self.k3 * y ** 3 + self.g * u[:, 0] + self.b0], 1)

    def f_known(self, x, u):
        return self.f(x, u)


NB_OM0 = 1350.0                     # rad/s, reference rotor speed: u = rotor speed - NB_OM0 (near hover)
NB_G = 9.80665


@register("neurobem_quad")
class NeuroBEMNominal(Plant):
    r"""Nominal rigid-body quadrotor of the ``neurobem`` track, without any yaw or position state (neither is observable
    from an IMU and rotor speeds):

        x = (v_b [m/s], g_b, w [rad/s], a_m [m/s^2])       n_x = 12
        u = rotor speeds - NB_OM0 [rad/s]: back right, front right, back left, front left
        y = (w, a_m) = x[6:12]: the onboard gyro and accelerometer

        s    = sqrt(mean rotor speed^2) / NB_OM0
        a_b  = (-dx s vx, -dy s vy, sum_i bT_i (om_i / NB_OM0)^2 - dz s vz)      specific force [m/s^2]
        v_b' = -w x v_b - g0 g_b + a_b                      rigid body, body frame (g_b: world 'up' in the body frame)
        g_b' = -w x g_b                                     unit vector: replaces roll and pitch, no singularity
        w'   = -J^-1 (w x J w) + B (om / NB_OM0)^2 + (Mx s vy, My s vx, 0) + c0     Euler; B: 3 x 4 control effectiveness
        a_m' = (C a_b + b - a_m) / tau_a                    accelerometer: affine calibration + firmware low-pass

    Rigid body with a quadratic rotor thrust / torque map (Mahony, Kumar, Corke, IEEE RAM 2012) and lumped linear rotor drag
    scaled by rotor speed (Faessler, Franchi, Scaramuzza, IEEE RA-L 2018; Leishman et al., IEEE CSM 2014). The drag term is
    kept because without it the accelerometer sees no tilt or horizontal speed at all. Mx, My are the first-order blade-flapping
    moment and c0 a constant trim moment: without them the steady aerodynamic moment biases B by a factor ~8 (REVIEW_6). J is the value in the dataset Readme;
    the other parameters are identified on the TRAINING flights (real/neurobem.py). It is a NOMINAL model: no blade-element
    aerodynamics, no NeuroBEM residual, no motor reaction torque, no speed-dependent thrust loss, no battery or ground effect."""
    nx = 12
    nu = 4
    PARAMS = {"Jx": 0.0025, "Jy": 0.0021, "Jz": 0.0043, "dx": 0.3, "dy": 0.3, "dz": 0.1, "tau_a": 0.04,
              **{f"bT{i}": 2.45 for i in range(4)}, "Mx": 0.0, "My": 0.0, "c0x": 0.0, "c0y": 0.0, "c0z": 0.0,
              **{f"B{a}{i}": 0.0 for a in "xyz" for i in range(4)},
              **{f"C{r}{c}": float(r == c) for r in "xyz" for c in "xyz"}, "bx": 0.0, "by": 0.0, "bz": 0.0}
    DEFAULT_SENSORS = (6, 7, 8, 9, 10, 11)
    UNMODELLED = "rigid body + quadratic rotor map + lumped linear rotor drag: no blade-element aerodynamics"

    def __init__(self, sensors=None, **params):
        super().__init__(sensors, **params)
        p = lambda names: torch.tensor([getattr(self, n) for n in names], dtype=torch.float64)
        self._bT = p([f"bT{i}" for i in range(4)])
        self._B = p([f"B{a}{i}" for a in "xyz" for i in range(4)]).reshape(3, 4)
        self._C = p([f"C{r}{c}" for r in "xyz" for c in "xyz"]).reshape(3, 3)
        self._b = p(["bx", "by", "bz"])

    def specific_force(self, x, u):
        om = (u + NB_OM0).clamp(min=0.0) / NB_OM0
        w2 = om ** 2
        s = torch.sqrt(w2.mean(1) + 1e-9)
        return torch.stack([-self.dx * s * x[:, 0], -self.dy * s * x[:, 1],
                            w2 @ self._bT.to(x) - self.dz * s * x[:, 2]], 1), w2

    def f(self, x, u):
        v, g, w, am = x[:, :3], x[:, 3:6], x[:, 6:9], x[:, 9:12]
        ab, w2 = self.specific_force(x, u)
        J = x.new_tensor([self.Jx, self.Jy, self.Jz])
        s = torch.sqrt(w2.mean(1) + 1e-9)
        aero = torch.stack([self.Mx * s * x[:, 1] + self.c0x, self.My * s * x[:, 0] + self.c0y, 0 * s + self.c0z], 1)
        dw = -torch.linalg.cross(w, J * w) / J + w2 @ self._B.to(x).T + aero
        return torch.cat([-torch.linalg.cross(w, v) - NB_G * g + ab, -torch.linalg.cross(w, g), dw,
                          (ab @ self._C.to(x).T + self._b.to(x) - am) / self.tau_a], 1)

    def f_known(self, x, u):
        return self.f(x, u)
