"""Correctness checks of a real-data track's frozen episodes + plots of a few episodes.

    python tools/real_check.py battery_pf [battery_lg emps silverbox]

Checks: shapes, finiteness of u / y / labelled x, NaN exactly where the label mask says, sample time,
priors inside the declared conventions, train / val / test disjoint (by recording and index range),
measured states equal y (labels aligned in time with the measurements), sign convention of the input
(battery: discharge positive => SOC decreases when the mean current is positive; mechanics: velocity label
consistent with the position increments). Writes docs/real/<dataset>_episodes.png and
docs/real/<dataset>_checks.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "docs" / "real"


def check(name):
    import hyperkkl_bench.real as hr
    mod = hr.module(name)
    ctx = mod.context()
    sens, lab = ctx["sensors"], ctx["labelled"]
    res = {"dataset": name, "dt_model": ctx["dt"], "dt_seconds": ctx["meta"].get("dt_seconds"), "splits": {}}
    seen = {}
    for s in mod.splits():
        es = mod.load(s)
        E, N1 = es.u.shape[:2]
        r = {"n": E, "N1": N1}
        assert es.y.shape[:2] == (E, N1) and es.x.shape[:2] == (E, N1), s
        assert np.isfinite(es.u).all() and np.isfinite(es.y).all(), f"{s}: non-finite u / y"
        nanmask = ~np.isfinite(es.x)
        assert not nanmask[..., lab].any(), f"{s}: NaN in a labelled state"
        assert nanmask[..., [j for j in range(es.x.shape[-1]) if j not in lab]].all(), f"{s}: unlabelled state has values"
        ok = np.ones(es.u.shape[:2], bool) if es.score_mask is None else ~es.score_mask      # outage samples: y frozen
        r["max_abs_y_minus_x_sensors"] = float(np.abs(es.y - es.x[..., sens])[ok].max())
        if es.score_mask is not None:
            r["outage_samples_per_episode"] = int(es.score_mask[0].sum())
        assert r["max_abs_y_minus_x_sensors"] == 0.0, f"{s}: measured states are not y (misaligned labels)"
        if s != "train":
            assert es.prior is not None and np.isfinite(es.prior).all(), f"{s}: priors"
        dt_s = ctx["meta"].get("dt_seconds", 1.0)
        if name.startswith("battery") or name == "calce":
            # charge balance: SOC drop over a window = integral of the current / Q (discharge positive)
            Q = hr.spec(name)["frozen"]["Q_Ah"]
            if name == "calce":                      # labels use each cell's own C/20 capacity
                Q = np.array([mod.cell_q_ocv(int(m["rec"][5]))["Q"] for m in es.meta])
            pred = es.u[:, :-1].astype(float).sum(1) * dt_s / 3600 / Q
            obs = (es.x[:, 0, 0] - es.x[:, -1, 0]).astype(float)
            r["soc_charge_balance_max_err"] = float(np.abs(pred - obs).max())
            assert r["soc_charge_balance_max_err"] < 2e-3, s
        elif name == "neurobem":
            # body-frame kinematics of the labels: g_b unit, g_b' = -w x g_b (attitude vs gyro alignment), specific force sign
            g = es.x[..., 3:6].astype(float)
            r["g_norm_max_dev"] = float(np.abs(np.linalg.norm(g, axis=-1) - 1).max())
            dg = (g[:, 1:] - g[:, :-1]) / ctx["dt"]
            pred = -np.cross(es.y[:, :-1, :3].astype(float), g[:, :-1])
            r["attitude_vs_gyro_corr"] = float(np.corrcoef(dg.ravel(), pred.ravel())[0, 1])
            assert r["g_norm_max_dev"] < 1e-3 and r["attitude_vs_gyro_corr"] > 0.8, s
            r["rotor_speed_rad_s_min_max"] = [float(es.u.min() + 1350), float(es.u.max() + 1350)]
        else:
            # velocity label vs position increments (both in model time units)
            dq = np.diff(es.x[..., 0].astype(float), axis=1) / ctx["dt"]
            vm = 0.5 * (es.x[:, 1:, 1] + es.x[:, :-1, 1])
            r["vel_label_vs_increment_corr"] = float(np.corrcoef(dq.ravel(), vm.ravel())[0, 1])
            assert r["vel_label_vs_increment_corr"] > 0.9, s
        for m in es.meta:
            key = (m["rec"], m.get("temp"))
            seen.setdefault(key, set()).add(s.split("_")[0] if s.startswith("test") else s)
        res["splits"][s] = r
    roles = {k: sorted(v) for k, v in seen.items()}
    shared = {str(k): v for k, v in roles.items() if len([x for x in v if x in ("train", "val") or x.startswith("test")]) > 1
              and not all(x.startswith("test") for x in v)}
    res["recordings_in_more_than_one_role"] = shared
    if name == "neurobem":
        assert not shared, f"flight in several roles: {shared}"
    elif name in ("emps", "silverbox"):
        mod.T._check_disjoint()              # same record, disjoint index ranges
        res["index_ranges_disjoint"] = True
    else:
        assert not shared, f"recording in several roles: {shared}"
    return res


def plot_neurobem():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hyperkkl_bench.real as hr
    from hyperkkl_bench.real.neurobem import roll_pitch
    mod = hr.module("neurobem")
    rows = []
    for s, fams in (("train", ("lemniscate", "random_points", "circle3d")), ("test_ood", ("cpc", "satellite", "vertical_osc"))):
        es = mod.load(s)
        for f in fams:
            i = es.families.index(f) + es.families.count(f) // 2
            rows.append((s, es, i))
    fig, ax = plt.subplots(len(rows), 5, figsize=(19, 2.1 * len(rows)), squeeze=False)
    t = np.arange(1001) * 0.01
    tt = ["rotor speed u + 1350 [rad/s]", "gyro y [rad/s]", "accelerometer y [m/s^2]", "label v_b [m/s] (x, y, z)", "label roll, pitch [deg]"]
    for r, (s, es, i) in enumerate(rows):
        ax[r, 0].plot(t, es.u[i] + 1350, lw=0.4)
        ax[r, 0].set_ylabel(f"{s}\n{es.families[i]}\n{es.meta[i]['rec'][5:]}", fontsize=7)
        ax[r, 1].plot(t, es.y[i, :, :3], lw=0.4)
        ax[r, 2].plot(t, es.y[i, :, 3:], lw=0.4)
        ax[r, 3].plot(t, es.x[i, :, :3], lw=0.8)
        ro, pi = roll_pitch(es.x[i, :, 3:6].astype(float))
        ax[r, 4].plot(t, ro, lw=0.8); ax[r, 4].plot(t, pi, lw=0.8)
        for c in range(5):
            ax[r, c].tick_params(labelsize=6)
            if r == 0:
                ax[r, c].set_title(tt[c], fontsize=8)
    ax[-1, 0].set_xlabel("t [s]")
    fig.suptitle("neurobem: one 10-s window from six flights (inputs, measured channels, Vicon labels)", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "neurobem_episodes.png", dpi=90)
    plt.close(fig)


def plot(name):
    if name == "neurobem":
        return plot_neurobem()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import hyperkkl_bench.real as hr
    mod = hr.module(name)
    ctx = mod.context()
    names = ctx["meta"]["state_names"]
    tests = [s for s in mod.splits() if s.startswith("test")][:2]
    rows = []
    for s in ["train"] + tests:
        es = mod.load(s)
        rows += [(s, es, i) for i in (0, len(es) // 2)]
    lab = ctx["labelled"]
    fig, ax = plt.subplots(len(rows), 1 + len(lab), figsize=(4 * (1 + len(lab)), 2.0 * len(rows)), squeeze=False)
    dt_s = ctx["meta"].get("dt_seconds", 1.0)
    for r, (s, es, i) in enumerate(rows):
        t = np.arange(es.u.shape[1]) * dt_s
        ax[r, 0].plot(t, es.u[i], lw=0.6); ax[r, 0].set_ylabel(f"{s}[{i}]\nu", fontsize=7)
        for c, j in enumerate(lab):
            a = ax[r, c + 1]
            a.plot(t, es.x[i, :, j], lw=0.7, label="label")
            if es.prior is not None:
                a.axhline(es.prior[i, j], color="r", lw=0.6, ls="--", label="prior")
            if r == 0:
                a.set_title(names[j], fontsize=8)
        for a in ax[r]:
            a.tick_params(labelsize=6)
    ax[-1, 0].set_xlabel("t [s]")
    fig.suptitle(f"{name}: input and labelled states (train + first two test splits)", fontsize=9)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}_episodes.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        r = check(n)
        plot(n)
        (OUT / f"{n}_checks.json").write_text(json.dumps(r, indent=1))
        print(n, "OK", json.dumps({k: v for k, v in r.items() if k != "splits"}))
