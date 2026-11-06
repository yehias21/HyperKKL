"""reference/LEADERBOARD.md from the stored package reports (reference/runs/<baseline>/*.json).
Per track: track_score of every baseline against ekf_gated on the cells both have; per-cell table of
the clean test_sequence_ood unmeasured NRMSE. Single deterministic reference runs (seed 0 for learned
baselines): seed variability is NOT included (PROTOCOL.md section 5)."""
from pathlib import Path

from hyperkkl_bench import tracks
from hyperkkl_bench.evaluate import Report

RUNS = Path(__file__).resolve().parents[1] / "reference" / "runs"
OUT = RUNS.parent / "LEADERBOARD.md"


def main():
    bases = sorted(d.name for d in RUNS.iterdir() if d.is_dir())
    L = ["# Reference leaderboard (package reports; regenerate with tools/build_leaderboard.py)", "",
         "Headline per cell: mean over the three test tiers of the settled unmeasured NRMSE (clean).",
         "Track score: geometric mean over cells of method / ekf_gated (< 1 is better); wins = cells below ekf_gated.",
         "Learned baselines: one training seed (0); filters are deterministic. Seed variability is not included.", ""]
    for t in ("core", "deployment", "scaling"):
        ref = tracks.reference_reports("ekf_gated", t)
        if not ref:
            continue
        L += [f"## Track `{t}` ({len(ref)} cells with ekf_gated)", "", "| baseline | cells | geo-mean ratio vs ekf_gated | wins |", "|---|---|---|---|"]
        reps = {b: tracks.reference_reports(b, t) for b in bases}
        for b in bases:
            if not reps[b]:
                continue
            s = tracks.track_score(reps[b], ref)
            L.append(f"| {b} | {s['cells']} | {s['geo_mean_ratio']:.3f} | {s['wins']} |")
        cells = sorted(ref, key=lambda c: (c.split("/")[1], c))
        L += ["", "| cell | " + " | ".join(bases) + " |", "|---|" + "---|" * len(bases)]
        for c in cells:
            row = []
            for b in bases:
                r = reps[b].get(c)
                if r is None:
                    row.append("")
                    continue
                v = r.value("test_sequence_ood")
                d = r.results["test_sequence_ood"]["clean"]["summary"].get("div_rate", 0)
                row.append(("n/a" if v is None else f"{v:.4f}") + ("*" if d else ""))
            L.append(f"| {c} | " + " | ".join(row) + " |")
        L += ["", "Per-cell values: test_sequence_ood, clean, unmeasured NRMSE; `*` = some episodes diverged.", ""]
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
