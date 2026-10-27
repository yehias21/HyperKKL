"""Stage-1 measures on the Stage-2 training sets (seed 0), CPU. Writes data_diag.json."""
import json

from common_qt import *  # noqa
from diag_s1 import load, coverage, knn_coverage, spectra, pe, TAU_SLOW
from run_qt import concat

DD = HERE / "data"
out = {}
for node in ("C3-partial", "C3-mismatch30"):
    cfg, sc, d, ev = load(node, "quad_tank")
    L = lambda part: torch.load(DD / f"{node}_s0_{part}.pt", weights_only=False, map_location="cpu")
    sets = {"X0": d["train"], "X1": L("long_orig"),
            "X2": concat(d["train"].select(range(300)), L("slow_T1")),
            "X3": concat(L("long_orig"), L("slow_T3"))}
    for arm, tr in sets.items():
        dd = dict(d, train=tr)
        cv = coverage(cfg, sc, dd, {})["train"]
        kc, _ = knn_coverage(sc, dd, ev)
        sp = spectra(cfg, dd, {}, TAU_SLOW["quad_tank"])["train"]
        p = pe(cfg, sc, dd, ev, TAU_SLOW["quad_tank"])
        out[f"{node}/{arm}"] = {
            "n_episodes": int(tr.x.shape[0]), "N1": int(tr.x.shape[1]), "n_rejected": int(tr.n_rejected),
            "within_ep_std_cm": {k: v["within_ep_std_window"] for k, v in cv.items()},
            "pooled_std_cm": {k: v["std_window"] for k, v in cv.items()},
            "knn_seq_frac_beyond": kc["sequence_ood"]["frac_beyond_val_p99"],
            "knn_seq_x34_frac_beyond": kc["sequence_ood"]["x34_frac_beyond_val_p99"],
            "frac_fluct_below_wc": sp["frac_fluct_energy_below_wc"], "frac_fluct_below_alpha": sp["frac_fluct_energy_below_alpha"],
            "lp_var_ratio": sp["lp_tau_slow_var_over_fluct_var"],
            "unexplained_x3x4": {k: v["frac_unexplained"] for k, v in p["train"]["unmeasured_independent_var"].items()},
            "cond_zc": p["train"]["cond_zc"], "eig_z": p["train"]["eig_z"]}
        print(node, arm, json.dumps(out[f"{node}/{arm}"])[:600], flush=True)
json.dump(out, open(HERE / "data_diag.json", "w"), indent=1)
