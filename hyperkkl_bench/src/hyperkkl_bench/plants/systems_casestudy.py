"""Deployment case-study plants (2026-10-27; experiments/casestudies/CASE_A.md, CASE_B.md).

    quad_planar   planar quadrotor (x-z plane) under its flight controller, 100 Hz estimator
                  rate; IMU (gyro, body-x accelerometer behind its DLPF) + GPS position.
                  Unmeasured: vx, vz, pitch. Input: path set-point (pilot / guidance) on an
                  inclined inspection path, |r| <= 3 m. UNMODELLED: rotor drag (the accelerometer's signal).
    cstr_exo      exothermic CSTR with cooling jacket (A -> B), Arrhenius kinetics; reactor
                  and jacket temperatures measured, concentration unmeasured. Input: coolant
                  flow (fraction of nominal, +-50 %). UNMODELLED: the high-activation-energy
                  side reaction A -> D. Time unit = 1 min.

Same contract as systems.py (torch-batched f(x, u); h(x) = x[:, sensors]). Own module, NOT in
cache.DATA_MODULES, so no existing cache key changes; hyperkkl/__init__.py imports it on the
first registry lookup of one of its names. Do not edit after runs exist (record sha1 in
experiments/casestudies/STATUS.md).
"""

from __future__ import annotations

import math

import torch

from .systems import Plant, _pos, register


def _sat(v, lim):
    """Smooth symmetric saturation lim tanh(v / lim)."""
    return lim * torch.tanh(v / lim)


@register("quad_planar")
class QuadPlanar(Plant):
    r"""Planar quadrotor (pitch plane) in closed loop with its flight controller.

    x = (px, pz, vx, vz, theta, q, a_m): world position [m] (z up), world velocity [m/s],
    pitch [rad] (thrust axis z_b = (sin th, cos th)), pitch rate [rad/s], and the IMU's
    body-x accelerometer channel a_m [m/s^2] behind its digital low-pass filter (tau_imu).
    u = path set-point r [m] (pilot / guidance) along an inclined inspection path: the horizontal
    set-point is r, the altitude set-point Z_SLOPE r (slope 1:3), |r| <= u_scale = 3 m.

    Rigid body:  m v' = T z_b - m g e_z + F_drag,   Iyy q' = tau,   theta' = q.
    Rotor drag (Martin & Salaun ICRA 2010; Leishman, Macdonald, Beard & McLain, IEEE CSM 34(1)
    2014): linear in the body-frame velocity, F_drag = -m (kd_x v_bx x_b + kd_z v_bz z_b),
    x_b = (cos th, -sin th). The accelerometer measures the specific force; its body-x
    component is the drag alone, a_bx = -kd_x v_bx (thrust is along z_b), which is why an
    accelerometer carries velocity information on a multirotor -- if the drag is modelled.
    IMU channel: tau_imu a_m' = a_bx - a_m (on-chip DLPF ~ 8 Hz).

    Flight controller (firmware constants: known exactly, NOT parameters):
      theta_cmd = TH_MAX tanh((K_PX (r - px) - K_VX vx) / TH_MAX)          (position loop)
      a_z_cmd   = K_PZ (Z_SLOPE r - pz) - K_VZ vz                          (altitude loop)
      T         = sat_[0, 2 M_CTRL g](M_CTRL (g + a_z_cmd) / cos th)      (thrust, actuator limit)
      tau       = IYY_CTRL sat_{ALPHA_MAX}(K_TH (theta_cmd - th) - K_Q q)  (attitude PD, torque limit)
    M_CTRL, IYY_CTRL are the firmware's nominal values; the true m, Iyy are PARAMS (payload,
    battery swap), so a mass mismatch shows up as an altitude offset, as on a real drone.
    The loops are closed on the true state (the flight controller's own estimator is
    abstracted away; disclosed in CASE_A.md).
    """
    nx = 7
    PARAMS = {"m": 0.5, "Iyy": 2.5e-3, "kd_x": 0.40, "kd_z": 0.20, "g": 9.81, "tau_imu": 0.02}
    NOT_PERTURBED = ("g", "tau_imu")           # gravity; the IMU DLPF is set in the IMU registers
    DEFAULT_SENSORS = (0, 1, 5, 6)             # GPS px, GPS pz, gyro q, accelerometer a_m
    M_CTRL, IYY_CTRL = 0.5, 2.5e-3
    K_PX, K_VX, TH_MAX = 0.25, 0.25, 0.5
    K_PZ, K_VZ, Z_SLOPE = 4.0, 3.0, 0.3
    K_TH, K_Q, ALPHA_MAX = 150.0, 20.0, 400.0

    UNMODELLED = ("rotor drag (kd_x, kd_z): the practitioner's rigid-body + thrust model has no "
                  "aerodynamic drag, so its accelerometer channel predicts a_bx = 0")

    def _f(self, x, u, drag: bool):
        px, pz, vx, vz, th, q, am = x.unbind(1)
        r = u[:, 0]
        g = self.g
        th_cmd = _sat(self.K_PX * (r - px) - self.K_VX * vx, self.TH_MAX)
        az_cmd = self.K_PZ * (self.Z_SLOPE * r - pz) - self.K_VZ * vz
        c, s = torch.cos(th), torch.sin(th)
        t_hi = 2.0 * self.M_CTRL * g
        T_raw = self.M_CTRL * (g + az_cmd) / c
        T = _pos(T_raw, 1e-4) - _pos(T_raw - t_hi, 1e-4)        # smooth clip to [0, 2 m g]
        tau = self.IYY_CTRL * _sat(self.K_TH * (th_cmd - th) - self.K_Q * q, self.ALPHA_MAX)
        vbx = vx * c - vz * s
        vbz = vx * s + vz * c
        kx, kz = (self.kd_x, self.kd_z) if drag else (0.0, 0.0)
        ax = T / self.m * s - kx * vbx * c - kz * vbz * s
        az = T / self.m * c - g + kx * vbx * s - kz * vbz * c
        abx = -kx * vbx
        return torch.stack([vx, vz, ax, az, q, tau / self.Iyy, (abx - am) / self.tau_imu], 1)

    def f(self, x, u):
        return self._f(x, u, drag=True)

    def f_known(self, x, u):
        return self._f(x, u, drag=False)


@register("cstr_exo")
class ExothermicCSTR(Plant):
    r"""Exothermic CSTR with a cooling jacket (time unit 1 min, SI-like units).

    x = (C_A [mol/L], T [K], T_j [K]); u = coolant flow change, F_c = F_c0 (1 + u), |u| <= 0.5.

        C_A' = (C_Af - C_A) / tau_res - r1 - r2
        T'   = (T_f - T) / tau_res + J1 r1 + J2 r2 - beta (T - T_j)
        T_j' = (F_c0 / V_j)(1 + u)(T_cin - T_j) + gamma (T - T_j)
        r_i  = k_i(T) C_A,  k_i(T) = k_i,ref exp(-E_i/R (1/T - 1/T_REF))     (Arrhenius)

    J_i = -dH_i / (rho c_p) [K L / mol] (adiabatic temperature rise per unit conversion),
    beta = UA / (V rho c_p), gamma = UA / (V_j rho_j c_pj) [1/min]. Reaction 1 (A -> B) is the
    product; reaction 2 (A -> D) is a side reaction with a HIGHER activation energy: negligible
    at the operating point, it grows faster with temperature and adds heat -- the classic
    runaway mechanism the practitioner's single-reaction model misses (UNMODELLED).
    Structure: Seborg, Edgar, Mellichamp & Doyle, Process Dynamics and Control (CSTR example),
    Uppal, Ray & Poore, Chem. Eng. Sci. 29, 1974 (jacketed CSTR). Numbers are chosen for an
    industrial-scale unit: residence time 30 min, jacket time constant ~2.5 min, adiabatic rise
    100 K (J1 C_Af); steady states (true plant): u = 0: T 333.8 K, C_A 0.91 mol/L (55 %
    conversion); u = -0.5 (min coolant): 357 K, 0.23 mol/L; u = +0.5: 328 K, 1.12 mol/L --
    one stable steady state at every constant input in the box, for the true, the partial and
    the 10 % / 30 % mismatched plants (slowest mode ~ 1/60 per min; checked in
    tests/test_systems_casestudy.py). The side reaction is 10 % of the main rate at 335 K and
    ~30 % at 357 K: the partial model puts the min-coolant steady state at 344 K / 0.63 mol/L
    instead of 357 K / 0.23 mol/L, i.e. it under-predicts the hot, runaway-prone corner.
    The reactor is parametrically sensitive: from hot, concentrated initial states it ignites
    transiently to 380-420 K before settling. Above T_KIN_MAX = 400 K the Arrhenius rates are
    frozen (smooth min): the kinetics are only trusted up to 400 K, and beyond it the side reaction's
    rate (k2 dt > 3 at 410 K) would make the fixed-step RK4 of every simulator / filter unstable.
    The IC box (cstr_exo.yaml) keeps transient peaks below ~390 K.

    Mismatch (perturbed): rate constants, heats of reaction and heat-transfer coefficients are
    off by exp(+-rel); activation energies by exp(+-rel / 4) (a 30 % parameter error would
    otherwise move k(T) by e^{0.3 E/RT} ~ 10^3, not a plausible prior; activation energies are
    typically known to ~5-10 %). Flows, feed composition and temperatures are measured operating
    conditions (NOT_PERTURBED).
    """
    nx = 3
    PARAMS = {"tau_res": 30.0, "C_Af": 2.0, "T_f": 330.0, "T_cin": 310.0, "Fc_Vj": 0.25,
              "k1_ref": 0.04, "E1_R": 8000.0, "J1": 50.0,
              "k2_ref": 0.004, "E2_R": 16000.0, "J2": 80.0,
              "beta": 0.12, "gamma": 0.15}
    NOT_PERTURBED = ("tau_res", "C_Af", "T_f", "T_cin", "Fc_Vj")
    E_PARAMS = ("E1_R", "E2_R")
    DEFAULT_SENSORS = (1, 2)
    T_REF = 335.0
    T_KIN_MAX = 400.0
    TIME_UNIT = 60.0

    UNMODELLED = ("the side reaction A -> D (k2, E2, dH2): the practitioner's model has the main "
                  "reaction A -> B only")

    def perturbed(self, rel: float) -> "Plant":
        p, sign = {}, 1.0
        for k in sorted(self.PARAMS):
            if k in self.NOT_PERTURBED:
                p[k] = getattr(self, k)
                continue
            r = rel / 4.0 if k in self.E_PARAMS else rel
            p[k] = getattr(self, k) * math.exp(sign * r)
            sign = -sign
        return type(self)(sensors=self.sensors, **p)

    def _k(self, kref, e_r, T):
        return kref * torch.exp(-e_r * (1.0 / T - 1.0 / self.T_REF))

    def _f(self, x, u, side: bool):
        ca, T, Tj = x.unbind(1)
        Tc = 200.0 + _pos(T - 200.0, 1e-2)                    # guard: Arrhenius off the domain
        Tc = self.T_KIN_MAX - _pos(self.T_KIN_MAX - Tc, 1e-2)  # rates frozen above T_KIN_MAX (see docstring)
        r1 = self._k(self.k1_ref, self.E1_R, Tc) * ca
        r2 = self._k(self.k2_ref, self.E2_R, Tc) * ca if side else 0.0 * ca
        dca = (self.C_Af - ca) / self.tau_res - r1 - r2
        dT = (self.T_f - T) / self.tau_res + self.J1 * r1 + self.J2 * r2 - self.beta * (T - Tj)
        dTj = self.Fc_Vj * (1.0 + u[:, 0]) * (self.T_cin - Tj) + self.gamma * (T - Tj)
        return torch.stack([dca, dT, dTj], 1)

    def f(self, x, u):
        return self._f(x, u, side=True)

    def f_known(self, x, u):
        return self._f(x, u, side=False)
