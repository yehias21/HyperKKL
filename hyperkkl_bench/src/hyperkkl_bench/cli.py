"""Command line: ``hyperkkl-bench <command>`` (or ``python -m hyperkkl_bench``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_list(a):
    if a.family:
        from . import families as F
        fams = F.FAMILIES if a.family == "all" else (a.family,)
        for fam in fams:
            ps = F.AUTONOMOUS if fam == "autonomous" else F.DRIVEN
            for p in ps:
                note = " (exact / partial only)" if fam == "autonomous" and p in F.NOMINAL_ONLY else ""
                print(f"{fam}/{p}{note}")
        return
    from . import list_plants
    from .cards import plant_card
    print(f"{'plant':16s} {'role':6s} n_x n_y  group")
    plants = list_plants(include_legacy=a.legacy, role="all" if a.all else None)
    for p in plants:
        c = plant_card(p)
        print(f"{p:16s} {c['status']['role']:6s} {c['n_x']:3d} {c['n_y']:3d}  {c.get('group')}")
    if a.all or a.real:
        from .real import list_real, spec
        for d in list_real():
            print(f"{d:16s} real   (fetch from source: {spec(d)['source']['license']})")


def cmd_card(a):
    if "/" in a.plant:
        from pathlib import Path
        f = Path(__file__).resolve().parents[2] / "docs" / "cards" / "v2" / ("__".join(a.plant.split("/")[:2]) + ".md")
        print(f.read_text() if f.exists() else f"no card {f}")
        return
    from .cards import card_markdown, plant_card
    print(card_markdown(a.plant) if a.md else json.dumps(plant_card(a.plant), indent=1, default=str))


def cmd_check(a):
    from .check import check_plant, spec_from_file
    if a.builtin:
        from .data import metadata
        spec = dict(metadata(a.builtin)["system_config"])
        spec["plant"] = spec.get("plant", a.builtin)
    else:
        spec = spec_from_file(a.spec)
    res = check_plant(spec, fast=a.fast)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    sys.exit(0 if res["pass"] else 1)


def cmd_eval(a):
    from . import baselines, evaluate
    est = baselines.make(a.baseline)
    rep = evaluate(est, a.plant, a.knowledge, conditions=tuple(a.conditions.split(",")),
                   n_episodes=a.n_episodes)
    print(rep.table())
    if a.out:
        rep.save(a.out)
        print("saved", a.out)


def cmd_track(a):
    from . import baselines
    from .tracks import run_track
    from pathlib import Path
    reps = run_track(lambda p, k: baselines.make(a.baseline), a.track,
                     plants=a.plants.split(",") if a.plants else None,
                     knowledge=a.knowledge.split(",") if a.knowledge else None, verbose=True,
                     n_episodes=a.n_episodes)
    out = Path(a.out or f"track_{a.track}_{a.baseline}")
    for cell, rep in reps.items():
        rep.save(out / (cell.replace("/", "__") + ".json"))
        print(rep.table(), "\n")
    print("saved", out)


def cmd_verify(a):
    from .paths import data_root, extra_root
    bad = n = 0
    metas = [(data_root(), m) for m in sorted((data_root() / "metadata").glob("*.json"))]
    metas += [(extra_root(), m) for m in sorted((extra_root() / "metadata").glob("*.json"))]
    for root, m in metas:
        for rel, info in json.loads(m.read_text())["files"].items():
            p = root / rel
            if a.plant and f"/{a.plant}/" not in rel:
                continue
            n += 1
            ok = p.exists() and _sha(p) == info["sha256"]
            bad += not ok
            if not ok:
                print("BAD", rel)
    man = extra_root() / "MANIFEST.json"
    if man.exists():
        for rel, h in json.loads(man.read_text())["files"].items():
            if a.plant and f"/{a.plant}/" not in rel:
                continue
            n += 1
            p = extra_root() / rel
            ok = p.exists() and _sha(p) == h
            bad += not ok
            if not ok:
                print("BAD", rel)
    print(f"{n - bad}/{n} files match their SHA-256")
    sys.exit(1 if bad else 0)


def cmd_score_hidden(a):
    from .hidden import score_hidden
    print(json.dumps(score_hidden(a.submission, a.labels), indent=1))


def cmd_fetch(a):
    from .real import fetch
    r = fetch(a.dataset, from_local=a.from_local, dry_run=a.dry_run)
    print(f"{a.dataset}: {len(r['status']) - len(r['failed'])}/{len(r['status'])} files ok -> {r['raw_dir']}")
    sys.exit(1 if r["failed"] else 0)


def cmd_prepare(a):
    from .real import prepare
    prepare(a.dataset)


def cmd_real_eval(a):
    import hyperkkl_bench.real as hr
    from . import baselines
    est = baselines.make(a.baseline)
    if a.baseline.startswith("ekf") or a.baseline.startswith("ukf"):
        est.refit_for_condition = True
    rep = hr.evaluate(est, a.dataset, splits_=a.splits.split(",") if a.splits else None, n_episodes=a.n_episodes,
                      verbose=True)
    print(rep.table())
    if a.out:
        rep.save(a.out)


def cmd_real_card(a):
    from .real.cards import card_markdown
    print(card_markdown(a.dataset))


def cmd_version(a):
    from .version import BENCH_VERSION, DATA_VERSION, DATA_V2_VERSION, SCORER_VERSION
    print(f"hyperkkl-bench {BENCH_VERSION} | data {DATA_V2_VERSION} (+ legacy {DATA_VERSION}) | scorer {SCORER_VERSION}")


def cmd_eval_v2(a):
    from .baselines import make
    from .families import evaluate_v2
    est = make(a.baseline)
    rep = evaluate_v2(est, a.spec, n_episodes=a.n_episodes, verbose=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def cmd_check_autonomous(a):
    from .families import parse, metadata_v2
    from .autonomy import check_autonomous
    from .data import metadata
    if "/" in a.plant:
        m = metadata_v2(a.plant)
    else:
        m = metadata(a.plant)
    pert = tuple(float(v) for v in a.perturb.split(","))
    r = check_autonomous(m["system_config"], float(m["dt"]), float(m["T"]), perturb=pert, n_ic=a.n_ic)
    print(json.dumps(r, indent=1, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hyperkkl-bench")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("list", help="list plants (core by default)"); p.add_argument("--legacy", action="store_true")
    p.add_argument("--family", choices=["autonomous", "driven", "all"], help="v2: list family/plant cells")
    p.add_argument("--all", action="store_true", help="include deployment, scaling and real-data tracks")
    p.add_argument("--real", action="store_true", help="also list the real-data tracks"); p.set_defaults(f=cmd_list)
    p = sp.add_parser("card", help="plant card"); p.add_argument("plant"); p.add_argument("--md", action="store_true"); p.set_defaults(f=cmd_card)
    p = sp.add_parser("check-plant", help="validate a contributed plant (observable, bounded, design gate)")
    p.add_argument("spec", nargs="?", help="JSON plant spec (see check.py)")
    p.add_argument("--builtin", help="check a registered benchmark plant instead")
    p.add_argument("--fast", action="store_true"); p.add_argument("--out"); p.set_defaults(f=cmd_check)
    p = sp.add_parser("evaluate", help="evaluate a bundled baseline")
    p.add_argument("baseline"); p.add_argument("plant"); p.add_argument("knowledge")
    p.add_argument("--conditions", default="clean,outliers,burst_outage,poor_init")
    p.add_argument("--n-episodes", type=int, default=None); p.add_argument("--out"); p.set_defaults(f=cmd_eval)
    p = sp.add_parser("run-track", help="run a bundled baseline on a track (core, input_ood, deployment, scaling)")
    p.add_argument("track"); p.add_argument("baseline"); p.add_argument("--plants"); p.add_argument("--knowledge")
    p.add_argument("--n-episodes", type=int, default=None); p.add_argument("--out"); p.set_defaults(f=cmd_track)
    p = sp.add_parser("verify-data", help="check SHA-256 of the data files"); p.add_argument("--plant"); p.set_defaults(f=cmd_verify)
    p = sp.add_parser("score-hidden", help="(host) score a hidden-test submission")
    p.add_argument("submission"); p.add_argument("--labels"); p.set_defaults(f=cmd_score_hidden)
    p = sp.add_parser("fetch", help="download a real-data track from its ORIGINAL source (SHA-256 checked)")
    p.add_argument("dataset"); p.add_argument("--from-local", default=None, help="copy from a local copy instead")
    p.add_argument("--dry-run", action="store_true", help="only check that every URL is reachable"); p.set_defaults(f=cmd_fetch)
    p = sp.add_parser("prepare", help="raw files of a real-data track -> frozen episodes"); p.add_argument("dataset")
    p.set_defaults(f=cmd_prepare)
    p = sp.add_parser("real-evaluate", help="evaluate a bundled baseline on a real-data track")
    p.add_argument("baseline"); p.add_argument("dataset"); p.add_argument("--splits"); p.add_argument("--n-episodes", type=int)
    p.add_argument("--out"); p.set_defaults(f=cmd_real_eval)
    p = sp.add_parser("real-card", help="card of a real-data track"); p.add_argument("dataset"); p.set_defaults(f=cmd_real_card)
    p = sp.add_parser("version"); p.set_defaults(f=cmd_version)
    p = sp.add_parser("evaluate-v2", help="v2: evaluate a bundled baseline on family/plant/level")
    p.add_argument("baseline"); p.add_argument("spec"); p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--out"); p.set_defaults(f=cmd_eval_v2)
    p = sp.add_parser("check-autonomous", help="inclusion check 4a: non-trivial unforced dynamics (u = 0)")
    p.add_argument("plant", help="v1 plant name or family/plant/level"); p.add_argument("--n-ic", type=int, default=64)
    p.add_argument("--perturb", default="0,0.3", help="parameter perturbations to check (0 = nominal; 0.3 = mismatch / realistic data plant)")
    p.set_defaults(f=cmd_check_autonomous)
    a = ap.parse_args(argv)
    if a.cmd == "check-plant" and not (a.spec or a.builtin):
        ap.error("check-plant needs a spec file or --builtin NAME")
    a.f(a)


if __name__ == "__main__":
    main()
