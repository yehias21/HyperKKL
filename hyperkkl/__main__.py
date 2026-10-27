"""Command line for the experiment tree.

    python -m hyperkkl tree                       print the tree
    python -m hyperkkl show ID                    node + progress + report path
    python -m hyperkkl new ID --parent P --title "..." [--hypothesis ...] [--rule ...]
                       [--set k=v ...] [--methods a b] [--baselines ekf] [--systems ...]
                       [--seeds 0 1 2] [--contribution C1] [--reference concat]
    python -m hyperkkl queue ID [--force]         create pending jobs
    python -m hyperkkl status                     job counts for every node
    python -m hyperkkl report ID                  aggregate results -> report.md
    python -m hyperkkl verdict ID --outcome supported|refuted|mixed|inconclusive --text "..."
    python -m hyperkkl render                     write experiments/TREE.md
    python -m hyperkkl compare NODE_A:method NODE_B:method [--tier shape_ood --condition clean]
    python -m hyperkkl worker --gpus 0-7 --per-gpu 4
    python -m hyperkkl scorecard ID [ID ...]      cross-node deployment scorecard -> experiments/scorecard/
    python -m hyperkkl reeval ID [ID ...] --baselines koopman_kf greybox_ekf ... [--device cuda:0]
                       [--systems ...] [--seeds ...] [--shard i/n] [--force] [--no-report]
                                                  add baselines to finished runs (no retraining)
"""

from __future__ import annotations

import argparse
import sys

from . import tree as Tr
from .config import parse_value


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hyperkkl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tree")
    sub.add_parser("status")
    sub.add_parser("render")
    s = sub.add_parser("show"); s.add_argument("id")
    s = sub.add_parser("report"); s.add_argument("id")
    s = sub.add_parser("queue"); s.add_argument("id", nargs="+"); s.add_argument("--force", action="store_true")
    s = sub.add_parser("verdict"); s.add_argument("id"); s.add_argument("--outcome", required=True)
    s.add_argument("--text", required=True)
    s = sub.add_parser("new")
    s.add_argument("id"); s.add_argument("--parent", required=True); s.add_argument("--title", required=True)
    s.add_argument("--contribution"); s.add_argument("--question"); s.add_argument("--hypothesis")
    s.add_argument("--rule", dest="decision_rule"); s.add_argument("--set", nargs="*", default=[])
    s.add_argument("--methods", nargs="*"); s.add_argument("--baselines", nargs="*")
    s.add_argument("--systems", nargs="*"); s.add_argument("--seeds", nargs="*", type=int)
    s.add_argument("--reference"); s.add_argument("--tier"); s.add_argument("--condition")
    s.add_argument("--priority", type=int, default=0)
    s = sub.add_parser("compare"); s.add_argument("a"); s.add_argument("b")
    s.add_argument("--tier", default="shape_ood"); s.add_argument("--condition", default="clean")
    s.add_argument("--metric", default="nrmse")
    s = sub.add_parser("scorecard"); s.add_argument("id", nargs="+"); s.add_argument("--out")
    s = sub.add_parser("reeval"); s.add_argument("id", nargs="+"); s.add_argument("--baselines", nargs="+", required=True)
    s.add_argument("--device", default="cuda:0"); s.add_argument("--systems", nargs="*")
    s.add_argument("--seeds", nargs="*", type=int); s.add_argument("--shard", default="0/1")
    s.add_argument("--force", action="store_true"); s.add_argument("--no-report", action="store_true")
    s.add_argument("--allow-code-drift", action="store_true"); s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("worker"); s.add_argument("--gpus", default="0-7")
    s.add_argument("--per-gpu", type=int, default=4)
    a = ap.parse_args(argv)

    if a.cmd == "tree":
        out = Tr.render()
        txt = out.read_text()
        print(txt[txt.index("```") + 4: txt.index("```", txt.index("```") + 3)])
    elif a.cmd == "render":
        print(Tr.render())
    elif a.cmd == "status":
        for nid in Tr.all_nodes():
            st = Tr.refresh_status(nid)
            print(f"{nid:28s} {st:8s} {Tr.progress(nid)}")
    elif a.cmd == "show":
        n = Tr.load_node(a.id)
        import yaml
        print(yaml.safe_dump(n, sort_keys=False, width=92))
        print("progress:", Tr.progress(a.id))
        print("report:  ", Tr.TREE / a.id / "report.md")
    elif a.cmd == "report":
        print(Tr.report(a.id))
        Tr.render()
    elif a.cmd == "queue":
        for nid in a.id:
            print(nid, "queued", Tr.queue(nid, a.force), "jobs")
        Tr.render()
    elif a.cmd == "verdict":
        Tr.set_verdict(a.id, a.outcome, a.text)
        Tr.render()
    elif a.cmd == "new":
        sets = {}
        for kv in a.set:
            k, _, v = kv.partition("=")
            sets[k] = parse_value(v)
        cmp_ = {k: v for k, v in (("reference", a.reference), ("tier", a.tier),
                                  ("condition", a.condition)) if v}
        Tr.new_node(a.id, None if a.parent in ("none", "null") else a.parent, a.title,
                    contribution=a.contribution, question=a.question, hypothesis=a.hypothesis,
                    decision_rule=a.decision_rule, set=sets or None, methods=a.methods,
                    baselines=a.baselines, systems=a.systems, seeds=a.seeds,
                    compare=cmp_ or None, priority=a.priority)
        Tr.render()
        print(Tr.node_path(a.id))
    elif a.cmd == "compare":
        for sysn, d in Tr.cross_compare(a.a, a.b, a.tier, a.condition, a.metric).items():
            if "rel" not in d:
                print(f"{sysn:16s} no paired episodes (seeds {d['seeds']})")
                continue
            if not d["same_code"]:
                print(f"{sysn:16s} WARNING: the two nodes ran different code versions")
            flag = "better" if d.get("better") else "worse" if d.get("worse") else "n.s."
            print(f"{sysn:16s} {a.a}={d['a']:.4f}  {a.b}={d['b']:.4f}  change {100 * d['rel']:+.1f}% "
                  f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]  {flag}  seeds {d['seeds']}")
    elif a.cmd == "scorecard":
        pm, pt = Tr.scorecard(a.id, a.out)
        print(pm.read_text())
        print(pm)
        print(pt)
    elif a.cmd == "reeval":
        from .reeval import reeval_node
        i, n = (int(v) for v in a.shard.split("/"))
        for nid in a.id:
            reeval_node(nid, a.baselines, a.device, systems=a.systems, seeds=a.seeds, shard=(i, n),
                        force=a.force, allow_code_drift=a.allow_code_drift,
                        report=not a.no_report and n == 1, dry_run=a.dry_run)
        if not a.no_report and n == 1:
            Tr.render()
    elif a.cmd == "worker":
        from .sched import parse_gpus, worker
        worker(parse_gpus(a.gpus), a.per_gpu)


if __name__ == "__main__":
    sys.exit(main())
