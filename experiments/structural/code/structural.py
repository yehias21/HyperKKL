r"""Structural model error (experiments/structural/PREREG.md): the TRUE plant carries hidden
dynamics that the modeller's model (data.model_plant: f_known at nominal parameters) and every
residual r(x, u) on the model's state cannot represent.

Nothing in the package is changed. ``install(Dm)`` patches, at run time and only in the process
that calls it, three functions of the (copied) package:

  data.build_plant      knowledge = full and system.structural set -> HiddenPlant (the true plant
                        with hidden states); the modeller's model (knowledge = partial) is untouched
  data.generate         integrates the FULL state (visible + hidden), measures y = h_full(x_full),
                        stores only the visible state as the label x (the observers' target)
  data.system_scales    computed with system.structural removed: NRMSE normalisation, envelope,
                        washout are those of the reference plant (identical across the two tiers)
  evaluate.build_plant  the oracle filter's model-error rate uses HiddenPlant.f (hidden state
                        sampled from its declared box), so the oracle is told about the structure

Structural variants (one per plant, constants fixed in PREREG.md; hidden-mechanism constants are
class constants, NOT in PARAMS, so ``perturbed(rel)`` perturbs exactly the reference parameters):

  quad_tank   tank5      pump 1's cross flow to tank 4 passes through an unmodelled fifth tank
  battery     hyst       one-state OCV hysteresis (Plett 2004) in the terminal voltage
  cartpole    motor      first-order motor/drive lag on the cart force (inside the balancing loop)
  highway     ramp_queue on-ramp queue: the ramp demand reaches cell 3 through a vehicle queue
  bioreactor  growth_lag the specific growth rate follows Monod kinetics with a metabolic lag
  vdp         sensor_lag the position sensor has first-order dynamics (y = lagged x1)
  lorenz      regime     the Rayleigh parameter rho steps up by 25% at a hidden random time
"""

from __future__ import annotations

import math

import numpy as np
import torch

from hyperkkl import systems as Sy
from hyperkkl import systems_extra as SE
from hyperkkl.systems import _pos

# ---------------------------------------------------------------------------
# Full-state true plants (visible states first, hidden states last)
# ---------------------------------------------------------------------------


class QuadTank5(Sy.QuadrupleTank):
    NAME_S = "tank5"
    NH = 1
    A5, a5 = 28.0, 0.071            # a copy of tank 3 at nominal values (never perturbed)
    HIDDEN_BOX = [(1.0, 6.0)]       # h5 [cm]; zero-input equilibrium at v1n: 3.27 cm

    def f(self, x, u):
        uu = u[:, 0]
        v1, v2 = self.v1n + uu, self.v2n - 0.5 * uu
        c = math.sqrt(2.0 * self.g)
        s = [torch.sqrt(_pos(x[:, i]) + 1e-6) for i in range(5)]
        d1 = (-self.a1 * c * s[0] + self.a3 * c * s[2] + self.gam1 * self.k1 * v1) / self.A1
        d2 = (-self.a2 * c * s[1] + self.a4 * c * s[3] + self.gam2 * self.k2 * v2) / self.A2
        d3 = (-self.a3 * c * s[2] + (1 - self.gam2) * self.k2 * v2) / self.A3
        d4 = (-self.a4 * c * s[3] + self.a5 * c * s[4]) / self.A4
        d5 = (-self.a5 * c * s[4] + (1 - self.gam1) * self.k1 * v1) / self.A5
        return torch.stack([d1, d2, d3, d4, d5], 1)


class BatteryHyst(SE.LiIonCell):
    NAME_S = "hyst"
    NH = 1
    M_H, GAMMA_H, I_W = 0.010, 50.0, 0.02     # V, 1/(unit SOC), A (smooth sign width)
    HIDDEN_BOX = [(-0.010, 0.010)]

    def f(self, x, u):
        soc, v1, v2, dT, vm, hy = x.unbind(1)
        I = self.cell_current(soc, u[:, 0])
        T = self.T_amb + dT
        arr = torch.exp(self.Ea_R * (1.0 / T - 1.0 / self.T_amb))
        r0, r1, r2 = self.R0 * arr, self.R1 * arr, self.R2 * arr
        heat = I ** 2 * r0 + v1 ** 2 / r1 + v2 ** 2 / r2 - I * T * self.dUdT - self.hA * dT
        v_t = self.ocv(soc) + hy + self.dUdT * dT - v1 - v2 - r0 * I
        s = self.TIME_UNIT
        absI = torch.sqrt(I * I + 1e-6)
        dh = -s * self.GAMMA_H * absI / (3600.0 * self.Q) * (hy + self.M_H * torch.tanh(I / self.I_W))
        return torch.stack([-s * I / (3600.0 * self.Q),
                            s * (I - v1 / r1) / self.C1,
                            s * (I - v2 / r2) / self.C2,
                            s * heat / self.C_th,
                            s * (v_t - self.V_REF - vm) / self.tau_f,
                            dh], 1)


class CartPoleMotor(Sy.CartPole):
    NAME_S = "motor"
    NH = 1
    TAU_M = 0.05                     # s, motor + belt drive (electrical + mechanical) lag
    HIDDEN_BOX = None                # F_0 = commanded force at u = 0 (drive settled), see hidden_ic

    def force_cmd(self, x, u, th_s):
        k1, k2, k3, k4 = self.K_BAL
        return self.f_scale * (u[:, 0] - (k1 * x[:, 0] + k2 * x[:, 1] + k3 * th_s + k4 * x[:, 3]))

    def f(self, x, u):
        v, th, om, F = x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        th_s = self.TH_MAX * torch.tanh(th / self.TH_MAX)
        s, c = torch.sin(th_s), torch.cos(th_s)
        den = self.mc + self.mp * s * s
        av = (F + self.mp * s * (self.l * om * om - self.g * c)) / den - self.kd_v * v
        aw = (((self.mc + self.mp) * self.g * s - c * (F + self.mp * self.l * om * om * s))
              / (self.l * den)) - self.kd_w * om
        dF = (self.force_cmd(x, u, th_s) - F) / self.TAU_M
        return torch.stack([v, av, om, aw, dF], 1)

    def hidden_ic(self, x0, rng):
        x = torch.as_tensor(x0, dtype=torch.float64)
        th_s = self.TH_MAX * torch.tanh(x[:, 2] / self.TH_MAX)
        F = self.force_cmd(torch.cat([x, torch.zeros(len(x), 1, dtype=x.dtype)], 1),
                           torch.zeros(len(x), 1, dtype=x.dtype), th_s)
        return F.numpy()[:, None]


class HighwayQueue(Sy.HighwayTraffic):
    NAME_S = "ramp_queue"
    NH = 1
    TAU_R = 20.0                     # s, ramp queue discharge time
    HIDDEN_BOX = [(3.0, 10.0)]       # vehicles; equilibrium at the base demand: 6

    def f(self, x, u):
        rho, n = x[:, :5], x[:, 5]
        q = self.v_free * (rho - rho ** 2 / self.rho_max)
        inflow = torch.cat([torch.full_like(rho[:, :1], self.u_main_base), q[:, :-1]], 1)
        z = torch.zeros_like(rho[:, :1])
        r = (n / self.TAU_R)[:, None]
        ramp = torch.cat([z, z, r, z, z], 1)
        dn = (self.u_ramp_base + u[:, 0]) - n / self.TAU_R
        return torch.cat([(inflow - q + ramp) / self.L, dn[:, None]], 1)


class BioGrowthLag(Sy.Bioreactor):
    NAME_S = "growth_lag"
    NH = 1
    TAU_G = 2.0                      # h, metabolic adaptation of the growth rate
    HIDDEN_BOX = [(0.15, 0.6)]       # 1/h; mu(s) on the IC box spans 0.17..0.67

    def f(self, x, u):
        x1, x2, m = x[:, 0], x[:, 1], x[:, 2]
        D = self.D_nom + u[:, 0]
        sp = _pos(x2)
        mu = self.mu_max * sp / (self.Ks + sp)
        return torch.stack([(m - D) * x1, -m * x1 / self.Y + D * (self.x2f - x2),
                            (mu - m) / self.TAU_G], 1)


class VdpSensorLag(Sy.VanDerPol):
    NAME_S = "sensor_lag"
    NH = 1
    TAU_S = 0.25                     # s, first-order position sensor
    HIDDEN_BOX = None                # s_0 = x1_0 + U[-0.5, 0.5], see hidden_ic

    def f(self, x, u):
        x1, x2, s = x[:, 0], x[:, 1], x[:, 2]
        return torch.stack([x2, self.mu * (1 - x1 ** 2) * x2 - x1 + u[:, 0], (x1 - s) / self.TAU_S], 1)

    def hidden_ic(self, x0, rng):
        return (np.asarray(x0)[:, 0] + rng.uniform(-0.5, 0.5, size=len(x0)))[:, None]


class LorenzRegime(Sy.Lorenz):
    NAME_S = "regime"
    NH = 1
    DELTA, WIDTH = 0.25, 0.5         # rho -> rho (1 + 0.25) around a hidden switch time
    HIDDEN_BOX = [(-40.0, -10.0)]    # clock c_0 = -t_switch, t_switch ~ U[0.2 T, 0.8 T] (T = 50)

    def f(self, x, u):
        x1, x2, x3, c = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        rho = self.rho * (1.0 + self.DELTA * torch.sigmoid(c / self.WIDTH))
        return torch.stack([self.sigma * (x2 - x1), x1 * (rho - x3) - x2 + u[:, 0],
                            x1 * x2 - self.beta * x3, torch.ones_like(c)], 1)


STRUCTURAL = {
    "quad_tank": QuadTank5, "battery": BatteryHyst, "cartpole": CartPoleMotor,
    "highway_traffic": HighwayQueue, "bioreactor": BioGrowthLag, "vdp": VdpSensorLag,
    "lorenz": LorenzRegime,
}
# measured coordinate of the FULL state when the structure is in the sensor (else the config's)
FULL_SENSORS = {"vdp": (2,)}


def default_hidden_ic(cls, x0, rng):
    box = np.asarray(cls.HIDDEN_BOX, float)
    return box[:, 0] + rng.uniform(size=(len(x0), len(box))) * (box[:, 1] - box[:, 0])


class HiddenPlant(Sy.Plant):
    """The true plant seen from outside: nx = the model's (visible) state dimension.

    f_full / h_full act on the full state (visible + hidden). ``f`` on the VISIBLE state is used
    only for the oracle filter's model-error rate: the hidden state is drawn from its declared
    range (fixed generator), so the rate includes the structural part."""

    def __init__(self, full, inner, nv: int, sensors_vis, name: str):
        self.full, self.inner = full, inner
        self.nx, self.nh, self.nu = nv, inner.NH, 1
        self.sensors, self.ny = tuple(sensors_vis), len(sensors_vis)
        self.name = name
        self.PARAMS = {}
        self.structural = inner.NAME_S

    def f_full(self, x, u):
        return self.full.f(x, u)

    def h_full(self, x):
        return self.full.h(x)

    def hidden_ic(self, x0_vis: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if hasattr(type(self.inner), "hidden_ic") and type(self.inner).HIDDEN_BOX is None:
            return np.asarray(self.inner.hidden_ic(x0_vis, rng), float)
        return default_hidden_ic(type(self.inner), x0_vis, rng)

    def f(self, x, u):
        rng = np.random.RandomState(0)
        xn = x.detach().cpu().double().numpy()
        hid = torch.as_tensor(self.hidden_ic(xn, rng), dtype=x.dtype, device=x.device)
        return self.full.f(torch.cat([x, hid], 1), u)[:, :self.nx]

    def h(self, x):
        return x[:, list(self.sensors)]

    def params(self):
        return self.inner.params()


def make_true_plant(scfg: dict, perturb: float, build_plant_orig) -> HiddenPlant:
    """The structural true plant for a system config (same perturbation and actuator design as
    the reference plant build_plant(scfg, perturb))."""
    name = scfg["plant"]
    cls = STRUCTURAL[name]
    ref = build_plant_orig(scfg, perturb=perturb)        # reference true plant (for actuator / params)
    base_ref = ref.base if isinstance(ref, Sy.SaturatingActuator) else ref
    sens_full = FULL_SENSORS.get(name, tuple(scfg.get("sensors") or base_ref.sensors))
    inner = cls(sensors=sens_full, **base_ref.params())   # perturbed parameter values, verbatim
    inner.nx = base_ref.nx + cls.NH
    full = inner
    if isinstance(ref, Sy.SaturatingActuator):
        full = Sy.SaturatingActuator(inner, ref.u_bar, ref.k, ref.j, ref.x_ref)
    return HiddenPlant(full, inner, base_ref.nx, base_ref.sensors, f"{name}/{cls.NAME_S}")


# ---------------------------------------------------------------------------
# Run-time patches of the package (this process only)
# ---------------------------------------------------------------------------

def _strip(cfg: dict) -> dict:
    sc = {k: v for k, v in cfg["system"].items() if k != "structural"}
    return {**cfg, "system": sc}


def install(Dm, Ev=None):
    """Patch data.build_plant / generate / system_scales (and evaluate.build_plant)."""
    if getattr(Dm, "_structural_installed", False):
        return
    bp_orig = Dm.build_plant
    ss_orig = Dm.system_scales

    def build_plant(scfg, perturb=0.0, knowledge="full"):
        if scfg.get("structural") and knowledge == "full":
            if scfg["structural"] != STRUCTURAL[scfg["plant"]].NAME_S:
                raise ValueError(f"structural {scfg['structural']!r} != {STRUCTURAL[scfg['plant']].NAME_S!r}")
            return make_true_plant({k: v for k, v in scfg.items() if k != "structural"}, perturb, bp_orig)
        return bp_orig({k: v for k, v in scfg.items() if k != "structural"}, perturb=perturb, knowledge=knowledge)

    def system_scales(cfg, device="cpu"):
        return ss_orig(_strip(cfg), device)

    gen_orig = Dm.generate

    def generate(plant, des, scales, scfg, dcfg, families, n, regime, seed, device, ic_box=None,
                 sensor=None, process_noise=True):
        if not isinstance(plant, HiddenPlant):
            return gen_orig(plant, des, scales, scfg, dcfg, families, n, regime, seed, device,
                            ic_box=ic_box, sensor=sensor, process_noise=process_noise)
        return generate_hidden(Dm, plant, des, scales, scfg, dcfg, families, n, regime, seed, device,
                               ic_box, sensor, process_noise)

    Dm.build_plant, Dm.system_scales, Dm.generate = build_plant, system_scales, generate
    if Ev is not None:
        Ev.build_plant = build_plant
    Dm._structural_installed = True


class _FullView:
    """integrate() needs plant.f / plant.h on the full state."""

    def __init__(self, hp: HiddenPlant):
        self.f, self.h = hp.f_full, hp.h_full


def generate_hidden(Dm, plant: HiddenPlant, des, scales, scfg, dcfg, families, n, regime, seed, device,
                    ic_box=None, sensor=None, process_noise=True):
    """data.generate for a plant with hidden states: identical design (crossed signals x ICs,
    rejection outside reject_scale x envelope of the VISIBLE state, sensor model), the hidden
    initial state drawn per episode from its declared range (own seed stream), y from the full
    state, labels x = visible state only. Process noise acts on the visible states only."""
    import numpy as np
    rng = np.random.RandomState(seed)
    hrng = np.random.RandomState(seed + 424242)
    k = int(dcfg.get("k_cross", 3))
    n_sig = n_ic = int(math.ceil(n / k))
    sigs = Dm._signals(families, n_sig, regime, rng, des, scfg)
    box = np.asarray(ic_box if ic_box is not None else scfg["ic_box"], dtype=np.float64)
    ics = Dm._sobol(box, n_ic, seed)
    pick = [(i // k, i % n_ic) for i in range(n)]
    X0 = np.stack([ics[c] for _, c in pick])
    H0 = plant.hidden_ic(X0, hrng)
    sg = [sigs[s] for s, _ in pick]
    dom = np.asarray(scales["domain"])
    ctr, half = dom.mean(1), (dom[:, 1] - dom[:, 0]) / 2
    rs = float(dcfg.get("reject_scale", 1.5))
    lo, hi = ctr - rs * half, ctr + rs * half
    nv = plant.nx
    sig_p = None
    if process_noise and float(dcfg.get("noise_proc", 0.0)) > 0:
        sig_p = np.concatenate([float(dcfg["noise_proc"]) * np.asarray(scales["x_std"]), np.zeros(plant.nh)])
    view = _FullView(plant)
    xs, zs, us = [None] * n, [None] * n, [None] * n
    todo, n_rej = list(range(n)), 0
    for attempt in range(6):
        if not todo:
            break
        for c0 in range(0, len(todo), 512):
            ids = todo[c0:c0 + 512]
            x0f = np.concatenate([X0[ids], H0[ids]], 1)
            x, z, u = Dm.integrate(view, des, x0f, [sg[i] for i in ids], sig_p,
                                   seed=seed * 7919 + attempt * 104729 + c0, device=device)
            for j, i in enumerate(ids):
                xs[i], zs[i], us[i] = x[j], z[j], u[j]
        bad = []
        for i in todo:
            xi = xs[i][:, :nv]
            lo_t = torch.as_tensor(lo, dtype=xi.dtype, device=xi.device)
            hi_t = torch.as_tensor(hi, dtype=xi.dtype, device=xi.device)
            if not (torch.isfinite(xs[i]).all() and (xi >= lo_t).all() and (xi <= hi_t).all()):
                bad.append(i)
        n_rej += len(bad)
        for i in bad:
            X0[i] = box[:, 0] + rng.uniform(size=len(box)) * (box[:, 1] - box[:, 0])
            H0[i] = plant.hidden_ic(X0[i:i + 1], hrng)[0]
        todo = bad
    keep = [i for i in range(n) if i not in set(todo)]
    xf = torch.stack([xs[i] for i in keep])
    z = torch.stack([zs[i] for i in keep])
    u = torch.stack([us[i] for i in keep])
    y_clean = torch.stack([plant.h_full(xi) for xi in xf])
    sens = dcfg.get("sensor", {}) if sensor is None else sensor
    y = Dm.measure(y_clean, sens, scales["y_std"], seed, dt=des.dt, t_ref=float(scfg.get("t_ref", 1.0)))
    z_obs = Dm.run_latent(des, y)
    burn = int(math.ceil(scales["t_burn"] / des.dt))
    meta = [{"ic": X0[i].tolist(), "hidden_ic": H0[i].tolist(), **sg[i].describe()} for i in keep]
    f32 = lambda a: a.float().cpu()
    es = Dm.EpisodeSet(f32(xf[..., :nv]), f32(u), f32(y_clean), f32(y), f32(z), f32(z_obs),
                       [sg[i].family for i in keep], burn, des.dt, meta, n_rej)
    es.x_hidden = f32(xf[..., nv:])         # not part of the EpisodeSet contract (diagnostics only)
    return es
