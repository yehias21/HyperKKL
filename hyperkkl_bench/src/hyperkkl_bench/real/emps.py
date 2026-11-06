r"""Real-data track ``emps``: velocity estimation of the Electro-Mechanical Positioning System (Janot, Gautier,
Brunot 2019; nonlinearbenchmark.org). Licence not stated: fetched, never redistributed.

Relevance: a prismatic axis driven by a measured, time-varying motor force (u = g_tau * vir [N], the
controller output applied to the drive; the loop is closed, but the force is measured and is what drives the
mechanics). Only the motor position q is measured (encoder); the velocity is HIDDEN (no velocity sensor;
label derived offline, see mech.py). Input conditions differ between the splits: train / val are bang-bang
acceleration trajectories; the test record adds force PULSES (a disturbance input absent from training).

Splits (fixed before any estimator was run): record DATA_EMPS samples [200, 20000) -> train (2-s windows
every 0.25 s), [20000, 24641) -> val (every 0.5 s); DATA_EMPS_PULSES -> test_pulses (every 0.5 s).

Task revision (2026-11-01, after the first smoke runs, before any comparison was drawn): with a 50000-count
encoder the velocity is nearly a finite difference of the measured position (a causal second-order
difference, no model, scores 0.0008, better than every estimator), so ``test_pulses`` is a DEGENERATE
estimation task and is kept only as a sanity split. The headline split is ``test_pulses_outage``: the same
windows with the encoder FROZEN (last value held) for four 100-ms bursts in the second half; velocity and
position are scored during the bursts only. Bridging an outage needs the velocity at its start and the
force-to-motion model: a genuine hidden-state task. Filters use the stale-sample rule (hold_missing = True;
at rest the encoder also repeats, where skipping an update loses nothing).
Partial model: rigid axis with viscous + Coulomb friction + offset (the authors' DDM, eq. 3), parameters
by the authors' IDIM-LS on the TRAINING samples only (filtfilt Butterworth 4 @ 100 Hz + central differences;
no decimation), sign(v) smoothed to tanh(v / 1 mm/s).
"""
import numpy as np

from .mech import MechTrack, Segment


def _load(raw):
    import scipy.io as sio
    out = {}
    for rec, f in (("emps", "DATA_EMPS.mat"), ("pulses", "DATA_EMPS_PULSES.mat")):
        m = sio.loadmat(str(raw / "EMPS" / f))
        out[rec] = (float(np.squeeze(m["gtau"])) * m["vir"].ravel(), m["qm"].ravel())
    return out


def _identify(u, y, v, a, dt):
    """IDIM-LS (Janot et al. eq. 2): tau = M a + Fv v + Fc sign(v) + offset, least squares."""
    Phi = np.stack([a, v, np.sign(v), np.ones_like(v)], 1)
    th, *_ = np.linalg.lstsq(Phi, u, rcond=None)
    M, Fv, Fc, off = (float(t) for t in th)
    res = u - Phi @ th
    q_v = float(np.std(res / M) * np.sqrt(dt))
    published = {"M": 95.1089, "Fv": 203.5034, "Fc": 20.3935, "offset": -3.1648}
    return ({"M": M, "Fv": Fv, "Fc": Fc, "offset": off, "v_eps": 1e-3},
            {"q_v": q_v, "force_residual_rel": float(np.linalg.norm(res) / np.linalg.norm(u)),
             "published_IDIM_LS_full_record": published, "u_scale": float(np.max(np.abs(u)))})


T = MechTrack(
    name="emps", N=2000, dt_s=1e-3, time_unit_s=1.0, cutoff_hz=100.0,
    segments=[Segment("train", "emps", 0, 20000, 250), Segment("val", "emps", 20000, 24841, 500),
              Segment("test_pulses", "pulses", 0, 24841, 500), Segment("test_pulses_outage", "pulses", 0, 24841, 500)],
    test_splits=["test_pulses_outage", "test_pulses"], plant_class="EMPSAxis",
    outage={"test_pulses_outage": [(1050, 100), (1300, 100), (1550, 100), (1800, 100)]}, hold_missing=True,
    hyperkkl={"nz": 4, "alpha": 20.0}, loader=_load, identify_fn=_identify, prep_version="emps-prep-1",
    metric_about="velocity NRMSE (velocity label = zero-phase derivative of the encoder, normalised by its training "
                 "std); test_pulses_outage (HEADLINE): scored during four 100-ms encoder outages in the second half of "
                 "each window; test_pulses: k >= N/2 (degenerate: a causal finite difference reaches 0.0008)")
T.headline = lambda results: float(results["test_pulses_outage"]["summary"]["nrmse_velocity"])
T.export(globals())
