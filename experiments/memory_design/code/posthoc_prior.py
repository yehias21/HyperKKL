"""POST-HOC diagnostic (not pre-registered): does the slow latent pay for its memory by remembering
the prior error? Re-runs the trained quad_tank observers (cached models, no training) on the
sequence_ood episodes with (a) the deployed prior x0_hat = centre of the IC box and (b) an oracle
prior x0_hat = true x(0), and records the error profile over time.

    python posthoc_prior.py --system quad_tank [--arms M0,M1,M2,M3,M4,M5]  -> ../posthoc_prior_<system>.json
"""
import argparse
import json

from common_md import Dm, np, torch  # noqa: F401
import run_md as R
from hyperkkl.metrics import episode_metrics
from hyperkkl.observers import run

ap = argparse.ArgumentParser()
ap.add_argument("--system", default="quad_tank"); ap.add_argument("--arms", default="M0,M1,M2,M3,M4,M5")
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--tier", default="sequence_ood")
a = ap.parse_args()
out = {}
for node in ("C3-partial", "C3-mismatch30"):
    for sd in [int(s) for s in a.seeds.split(",")]:
        j = R.Job(node, a.system, sd, "cuda:0")
        es = Dm.eval_data(j.cfg, a.tier).to("cuda:0")
        N1 = es.x.shape[1]
        settle = int(0.5 * (N1 - 1))
        xs = j.scales["x_std"]
        sig = torch.as_tensor(xs, dtype=torch.float32, device="cuda:0")
        box = np.asarray(j.cfg["system"]["ic_box"], float)
        xc = torch.as_tensor(box.mean(1), dtype=torch.float32, device="cuda:0")[None].expand(len(es), -1)
        for arm in a.arms.split(","):
            ob, _ = j.train_arm(arm)
            row = {}
            for nm, x0 in (("center", xc), ("oracle", es.x[:, 0])):
                with torch.no_grad():
                    xh = torch.cat([run(ob, es.y[i:i + 32], es.u[i:i + 32], x0[i:i + 32]) for i in range(0, len(es), 32)])
                em = episode_metrics(es.x, xh, xs, settle)
                e = ((xh - es.x) / sig).pow(2)                          # (E, N1, nx)
                prof = [float(e[:, k0:k0 + N1 // 10].mean().sqrt()) for k0 in range(0, N1 - N1 // 10 + 1, N1 // 10)]
                row[nm] = {"nrmse": float(np.mean(em["nrmse"])),
                           "per_state": [float(np.mean(em[f"nrmse_x{q + 1}"])) for q in range(es.x.shape[-1])],
                           "profile_decile_rms": prof}
            out[f"{node}/s{sd}/{arm}"] = row
            j.log(f"posthoc prior {arm}: center {row['center']['nrmse']:.4f} oracle {row['oracle']['nrmse']:.4f}")
        (R.ROOT / f"posthoc_prior_{a.system}.json").write_text(json.dumps(out, indent=1))
