r"""Real-data track ``silverbox``: velocity estimation of the Silverbox electronic Duffing oscillator (Wigren &
Schoukens 2013; nonlinearbenchmark.org). Licence not stated: fetched, never redistributed.

Relevance: a forced nonlinear oscillator (mass-spring-damper with a cubic spring, realised in electronics)
driven by a measured input voltage V1; only the displacement V2 is measured; the velocity is HIDDEN (label
derived offline, mech.py). Input conditions differ between the splits by design of the benchmark: training
and validation are odd random-phase multisines; the test splits are (i) a held-out multisine segment, (ii) the
'arrow' record, filtered Gaussian noise whose amplitude grows linearly, in its first part (amplitudes inside
the training range) and (iii) its last part (amplitudes BEYOND the training range: extrapolation).

Splits on SNLS80mV.mat (indices of the nonlinear_benchmarks package, fixed before any estimator was run):
multisine [40650, 105712) = the package's training part -> train [40650, 96000) (windows of 2048 samples
every 256) and val [96000, 105712) (every 1024); test_multisine [105712, 127400); test_arrow [100, 32100)
(no extrapolation); test_arrow_extrap [32100, 40575). Windows every 1024 samples in the test splits.
Partial model: y'' = -d y' - k1 y - k3 y^3 + g u + b0 (time unit 10 ms): equation-error least squares on
zero-phase derivatives of the TRAINING samples (Butterworth 4 @ 250 Hz), refined by output-error multiple
shooting (200-sample segments) on the same samples.
"""
import numpy as np

from .mech import MechTrack, Segment

FS = 1e7 / 2 ** 14          # 610.35 Hz (Wigren & Schoukens 2013)


def _load(raw):
    import scipy.io as sio
    m = sio.loadmat(str(raw / "SilverboxFiles" / "SNLS80mV.mat"))
    return {"snls": (m["V1"].ravel().astype(float), m["V2"].ravel().astype(float))}


def _sim(th, y0, v0, U, dt):
    """Vectorised RK4 (zero-order hold) of the Duffing model over segments: U (S, L) -> y (S, L+1)."""
    d, k1, k3, g, b0 = th
    f = lambda y, v, u: (v, -d * v - k1 * y - k3 * y ** 3 + g * u + b0)
    Y = [y0]
    y, v = y0, v0
    for k in range(U.shape[1]):
        u = U[:, k]
        a1 = f(y, v, u); a2 = f(y + dt / 2 * a1[0], v + dt / 2 * a1[1], u)
        a3 = f(y + dt / 2 * a2[0], v + dt / 2 * a2[1], u); a4 = f(y + dt * a3[0], v + dt * a3[1], u)
        y = y + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        v = v + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        Y.append(y)
    return np.stack(Y, 1)


def _identify(u, y, v, a, dt, L=200, step=100):
    """1) equation-error least squares on the zero-phase derivatives (initial guess); 2) output-error
    refinement by multiple shooting: segments of L samples every `step`, each started from the measured
    position and the derived velocity, minimising the simulated position error (TRAINING samples only)."""
    from scipy.optimize import least_squares
    Phi = np.stack([-v, -y, -y ** 3, u, np.ones_like(u)], 1)
    th0, *_ = np.linalg.lstsq(Phi, a, rcond=None)
    res0 = a - Phi @ th0
    st = np.arange(0, len(u) - L - 1, step)
    idx = st[:, None] + np.arange(L)[None]
    U, Yt = u[idx], y[st[:, None] + np.arange(L + 1)[None]]
    sc = np.std(y)
    sol = least_squares(lambda th: ((_sim(th, y[st], v[st], U, dt) - Yt) / sc).ravel(), th0, x_scale="jac")
    d, k1, k3, g, b0 = (float(t) for t in sol.x)
    res = a - Phi @ sol.x
    return ({"d": d, "k1": k1, "k3": k3, "g": g, "b0": b0},
            {"q_v": float(np.std(res) * np.sqrt(dt)), "accel_residual_rel": float(np.std(res) / np.std(a)),
             "equation_error_init": dict(zip(("d", "k1", "k3", "g", "b0"), map(float, th0))),
             "equation_error_accel_residual_rel": float(np.std(res0) / np.std(a)),
             "output_error_segment_nrmse": float(np.sqrt(np.mean(sol.fun ** 2))), "oe_segments": int(len(st)),
             "u_scale": float(np.max(np.abs(u)))})


T = MechTrack(
    name="silverbox", N=2048, dt_s=1.0 / FS, time_unit_s=0.01, cutoff_hz=250.0,
    segments=[Segment("train", "snls", 40650, 96000, 256), Segment("val", "snls", 96000, 105712, 1024),
              Segment("test_multisine", "snls", 105712, 127400, 1024), Segment("test_arrow", "snls", 100, 32100, 1024),
              Segment("test_arrow_extrap", "snls", 32100, 40575, 1024)],
    test_splits=["test_multisine", "test_arrow", "test_arrow_extrap"], plant_class="SilverboxDuffing",
    hyperkkl={"nz": 4, "alpha": 0.2}, loader=_load, identify_fn=_identify, prep_version="silverbox-prep-1", edge=100,
    metric_about="settled velocity NRMSE (k >= N/2 of each 2048-sample window; velocity label = zero-phase derivative "
                 "of V2, normalised by its training std); headline = mean of the three test splits")
T.export(globals())
