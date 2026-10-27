"""A browsable, generated HISTORY of every experiment in both trees.

    .venv/bin/python -m analysis.make_history     # -> experiments/HISTORY.md, experiments/history.html

Covers experiments/tree_v1/tree (protocol v1, archived: amplitude OOD tier) and
experiments/tree (protocol v2 + post-review nodes). Read-only on both trees; it writes only
the two output files. Verdicts are MECHANICAL:
  * v2 nodes: the decision-rule engine (analysis/evidence.py) via paper/generated/outcomes.json
    (written by analysis.paper_results; recomputed here if missing);
  * v1 nodes: the same engine on the v1 tiers (ood, waveform_ood; 5 dev systems, need 4/5) for
    the C1 / C3 rule families, plus the verdict recorded in node.yaml where one exists.
Section (a): by contribution; (b): chronological timeline from JOURNAL.md and node 'created'
dates, with dead ends and refuted hypotheses.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments"
V1 = EXP / "tree_v1" / "tree"
V2 = EXP / "tree"
GEN = ROOT / "paper" / "generated"

FIGS = {"C1": ["fig1_c1_where", "tab1_c1", "tab6_capacity"], "C2-floor": ["fig2a_c2_floor", "fig2b_c2_probe"],
        "C2-window": ["fig2c_c2_window"], "C3": ["fig3_c3_crossover", "tab2_c3", "fig4_scorecard"],
        "TEST-C1": ["fig1_c1_where", "tab1_c1"], "TEST-C3": ["fig3_c3_crossover_test"],
        "C3-realistic": ["fig3_c3_crossover", "tab3_scorecard_realistic", "fig4_scorecard"]}

# why v1 nodes / runs were superseded (JOURNAL.md 2026-10-25 10:45, TASKS.md, systems.py NOT_PERTURBED)
V1_REASON = ("Protocol v1 archived 2026-10-25 10:45 (JOURNAL.md): its amplitude OOD tier drove inputs beyond the "
             "plants' physical input bounds (TASKS.md: OOD amplitudes compressed on bounded plants such as "
             "bioreactor and traffic). Superseded by protocol v2: shape/sequence OOD inside one amplitude box, per-plant "
             "t_ref, partial-knowledge models, colored_t noise, robust/oracle filters, deployment scorecard; 6 dev "
             "systems (battery added).")
V1_EXTRA = {
    "C1-latent-K1": "Dead end: failed 6/15 jobs twice in v1; not carried to v2 (latent objective control left to the "
                    "reviews' R3 request).",
    "C3-mismatch05": "Dead end: failed 14/15 jobs in v1 (twice); the 5% level was dropped in v2.",
    "C2-window-a05": "Not carried to v2 (v2 tests the 1/alpha scaling with C2-window-a2 only).",
    "C3-faults": "bioreactor runs with the old perturbation rule moved to tree_v1/superseded/ (perturbed() also "
                 "scaled the operating point x2f, D_nom; fix: NOT_PERTURBED operating-point rule). quad_tank: perturbed() "
                 "scaled g, v1n, v2n -> every episode rejected (TASKS.md); fixed with NOT_PERTURBED = (g, v1n, v2n).",
}
for k in ("C3-mismatch10", "C3-mismatch20", "C3-mismatch30", "C3-mismatch50"):
    V1_EXTRA[k] = V1_EXTRA["C3-faults"]


def _contrib(n: dict) -> str:
    c = str(n.get("contribution") or "")
    if c in ("C1", "C2", "C3", "C4"):
        return c
    nid = n["id"]
    m = re.search(r"C(\d)", nid)
    return f"C{m.group(1)}" if m else "infra"


def _figs(nid: str) -> list:
    for key in (nid, "TEST-C1" if nid == "TEST-C1" else None, "TEST-C3" if nid == "TEST-C3" else None):
        if key and key in FIGS:
            return FIGS[key]
    for key in ("C2-floor", "C2-window", "C1", "C3"):
        if nid.startswith(key):
            return FIGS[key]
    return []


def _load_nodes(tree: Path) -> dict:
    return {p.parent.name: yaml.safe_load(p.read_text()) for p in sorted(tree.glob("*/node.yaml"))}


def _inherit(nodes, nid, key, default=None):
    cur = nid
    while cur:
        n = nodes.get(cur) or {}
        if n.get(key) is not None:
            return n[key]
        cur = n.get("parent")
    return default


def _jobs(tree: Path, nodes: dict, nid: str) -> tuple:
    systems = _inherit(nodes, nid, "systems", []) or []
    seeds = _inherit(nodes, nid, "seeds", [0]) or [0]
    total = len(systems) * len(seeds)
    done = sum((tree / nid / "runs" / s / f"seed{k}" / "results.json").exists() for s in systems for k in seeds)
    return done, total


# ---------------------------------------------------------------------------
# v2 outcomes
# ---------------------------------------------------------------------------

def v2_outcomes() -> dict:
    p = GEN / "outcomes.json"
    if p.exists():
        return json.loads(p.read_text())["outcomes"]
    from analysis.evidence import analyse_all
    from analysis.store import Store
    res = analyse_all(Store())
    return {k: {kk: v.get(kk) for kk in ("label", "verdict", "evidence", "progress")} for k, v in res.items()}


# ---------------------------------------------------------------------------
# v1 mechanical verdicts (same engine, v1 tiers)
# ---------------------------------------------------------------------------

def v1_outcomes(nodes: dict) -> dict:
    from analysis import rules as R
    from analysis.evidence import LOCATIONS, c1_rules, c1_tables, c3_votes
    from analysis.store import Store, use_tree
    from analysis import evidence as _E
    _E.SINGLE["on"] = True                       # v1 read with the same single-method convention
    out = {}
    tiers = ("id", "ood", "waveform_ood")
    from analysis import store as _St
    _ex = _St.EXCLUDE["on"]
    _St.EXCLUDE["on"] = False                    # the archived v1 record is shown complete (no post-hoc exclusion)
    try:
        return _v1_outcomes(nodes, out, tiers, R, LOCATIONS, c1_rules, c1_tables, c3_votes, Store, use_tree)
    finally:
        _St.EXCLUDE["on"] = _ex


def _v1_outcomes(nodes, out, tiers, R, LOCATIONS, c1_rules, c1_tables, c3_votes, Store, use_tree):
    with use_tree(V1, EXP / "tree_v1" / "queue"):
        st = Store()
        for nid, n in nodes.items():
            systems = _inherit(nodes, nid, "systems", []) or []
            try:
                runs = st.runs(nid)
            except Exception as e:                       # never fail the history on one node
                out[nid] = {"label": "n/a", "verdict": f"engine error: {e!r}", "evidence": []}
                continue
            if not runs:
                out[nid] = {"label": R.PENDING, "verdict": "no finished runs", "evidence": []}
                continue
            need = max(1, len(systems) - 1)              # v1 rules: >= 4/5
            methods = st.methods(next(iter(runs.values())), "ood")
            if nid.startswith("C1") and any(m in methods for m in LOCATIONS):
                T = c1_tables(st, nid, systems, tiers=tiers)
                RR = c1_rules(T, systems, need=need, tiers=tiers)
                ev = []
                for t in ("ood", "waveform_ood"):
                    if t not in T:
                        continue
                    ws = RR["weight_space"][t]["vote"]
                    hv = RR["counts"][t]["hyper"]
                    locs = ", ".join(f"{'HyperKKL' if l == 'lora' else l} {v.frac}" for l, v in RR['location'][t].items())
                    ev.append(f"{t}: HyperKKL (r=4) beats latent & concat {ws.frac}; dense ablation better/worse than "
                              f"latent {len(hv['better'])}/{len(hv['worse'])} of {hv['n']}; beats every other location: {locs}")
                lab = R.REPORTED
                if nid == "C1-where":
                    pref = [l for l in LOCATIONS if RR["location"].get("ood", {}).get(l) and
                            RR["location"]["ood"][l].label == R.PASS]
                    lab = R.PASS if pref else R.FAIL
                    ev.insert(0, f"v1 rule (ood tier, >= {need}/{len(systems)}): preferred location: "
                                 f"{', '.join(pref) or 'none'}")
                out[nid] = {"label": lab, "verdict": ev[0] if ev else "", "evidence": ev}
            elif nid.startswith("C3"):
                ev, labs = [], []
                for t in ("id", "ood"):
                    blocks, votes = c3_votes(st, nid, systems, t, need=need)
                    if not any(blocks.values()):
                        continue
                    labs.append(votes["any"].label)
                    nf = [len(b["failed"]) for b in blocks.values() if b]
                    ev.append(f"{t}/clean: best learned beats best filter {votes['any'].frac}, best WORKING filter "
                              f"{votes['working'].frac}; failed filters per system {min(nf)}-{max(nf)}")
                out[nid] = {"label": R.combine(labs) if labs else R.PENDING,
                            "verdict": "; ".join(ev), "evidence": ev}
            elif nid.startswith("C2-window"):
                from analysis.evidence import window_curves
                W = window_curves(st, nid, tiers=("ood", "id"))
                ev = [f"{s}: knee (ood) {d.get('ood', {}).get('knee')} samples, L* {d['Lstar']:.0f}, refined "
                      f"{d['Lref']:.0f}" for s, d in W.items()]
                out[nid] = {"label": R.REPORTED, "verdict": "; ".join(ev[:3]) + (" ..." if len(ev) > 3 else ""),
                            "evidence": ev}
            else:
                out[nid] = {"label": R.REPORTED, "verdict": "", "evidence": []}
            v = n.get("verdict") or {}
            if v:
                out[nid]["recorded"] = f"{v.get('outcome')} ({v.get('date')}): {v.get('text')}"
            out[nid]["problems"] = [m for nn, m in st.problems if nn == nid]
    return out


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def journal() -> list:
    ev = []
    for line in (EXP / "JOURNAL.md").read_text().splitlines():
        m = re.match(r"^- (\d{4}-\d\d-\d\d \d\d:\d\d) \S+ (.*)$", line)
        if m:
            ev.append((m.group(1), m.group(2).strip()))
    return ev


def _kind(text: str) -> str:
    t = text.lower()
    if "archived" in t or "supersed" in t:
        return "milestone"
    if "verdict" in t:
        return "verdict"
    if " failed (" in t:
        return "failure"
    if " created under" in t:
        return "created"
    if " done (" in t:
        return "done"
    return "queued"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build() -> dict:
    n1, n2 = _load_nodes(V1), _load_nodes(V2)
    o1, o2 = v1_outcomes(n1), v2_outcomes()
    entries = []
    for ver, nodes, outc, tree in (("v1", n1, o1, V1), ("v2", n2, o2, V2)):
        kids = {}
        for nid, n in nodes.items():
            kids.setdefault(n.get("parent"), []).append(nid)
        for nid, n in nodes.items():
            o = outc.get(nid) or {}
            if not o and nid.endswith("-s2") and n.get("parent") in outc:
                po = outc[n["parent"]]
                o = {"label": "POOLED", "verdict": f"seed 2 of {n['parent']}; decided jointly there: "
                     f"{po.get('shown') or po.get('label')} - {po.get('verdict', '')}"}
            done, total = _jobs(tree, nodes, nid)
            e = {"id": nid, "version": ver, "contribution": _contrib(n), "parent": n.get("parent"),
                 "children": sorted(kids.get(nid, [])), "title": n.get("title", ""),
                 "question": n.get("question", ""), "hypothesis": n.get("hypothesis", ""),
                 "rule": n.get("decision_rule", ""), "created": str(n.get("created", "")), "status": n.get("status"),
                 "label": o.get("shown") or o.get("label", "n/a"), "verdict": o.get("verdict", ""),
                 "key": [x.strip() for x in (o.get("evidence") or []) if x.strip()][:4],
                 "recorded": o.get("recorded") or ((n.get("verdict") or {}).get("text") and
                                                   f"{n['verdict'].get('outcome')}: {n['verdict'].get('text')}"),
                 "jobs": f"{done}/{total}", "report": f"{'tree_v1/tree' if ver == 'v1' else 'tree'}/{nid}/report.md",
                 "report_exists": (tree / nid / "report.md").exists(),
                 "figs": _figs(nid) if ver == "v2" else [], "problems": o.get("problems", [])}
            if ver == "v1":
                e["superseded"] = (f"Superseded by v2 node {nid}." if nid in n2 else "Not carried to v2.") + " " + \
                    V1_REASON + (" " + V1_EXTRA[nid] if nid in V1_EXTRA else "")
            entries.append(e)
    # composite v2 rules (not nodes): keep them for the contribution view
    for k in ("C1-budget", "C3-mismatch (crossover)"):
        if k in o2:
            o = o2[k]
            entries.append({"id": k, "version": "v2", "contribution": "C1" if k.startswith("C1") else "C3",
                            "parent": None, "children": [], "title": "joint rule over several nodes",
                            "question": o.get("question", ""), "hypothesis": "", "rule": o.get("rule", ""),
                            "created": "", "status": "", "label": o.get("label"), "verdict": o.get("verdict", ""),
                            "key": [x.strip() for x in (o.get("evidence") or [])][:4], "recorded": None,
                            "jobs": "-", "report": "", "report_exists": False, "figs": _figs(k), "problems": []})
    # timeline
    tl = [{"t": t, "kind": _kind(x), "text": re.sub(r"\*\*", "", x), "version": "v1" if t < "2026-10-25 10:45" else "v2"}
          for t, x in journal()]
    seen = {re.sub(r"\*\*", "", x).split(" ")[0] for _, x in journal() if " created under" in x}
    for e in entries:
        if e["created"] and e["version"] == "v2" and e["id"] not in seen:
            tl.append({"t": e["created"] + " --:--", "kind": "created", "version": "v2",
                       "text": f"{e['id']} created under `{e['parent']}` (node.yaml date): {e['title']}"})
    now = f"{_dt.datetime.now():%Y-%m-%d %H:%M}"
    for e in entries:
        if e["label"].rstrip("*") == "FAIL":
            tl.append({"t": now, "kind": "refuted", "version": e["version"],
                       "text": f"{e['id']} ({e['version']}): pre-registered rule FAILS mechanically - {e['verdict'][:300]}"})
        elif e["label"].rstrip("*") == "PASS":
            tl.append({"t": now, "kind": "supported", "version": e["version"],
                       "text": f"{e['id']} ({e['version']}): rule PASSES - {e['verdict'][:300]}"})
    tl.sort(key=lambda x: x["t"])
    return {"generated": now, "entries": entries, "timeline": tl}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

CONTRIB = {"C1": "C1 - where the input enters", "C2": "C2 - how much input history (memory)",
           "C3": "C3 - when learned beats filters", "C4": "C4 - testbed / benchmark",
           "infra": "Pipeline sanity / all contributions"}


def write_md(H: dict, path: Path):
    L = [f"# Experiment history (generated {H['generated']})", "",
         "Generated by `.venv/bin/python -m analysis.make_history` (also run by `analysis.paper_results`). Covers the "
         "archived protocol-v1 tree (`tree_v1/tree`, amplitude OOD tier) and the protocol-v2 tree (`tree`). Verdicts are "
         "the pre-registered decision rules applied mechanically (`analysis/evidence.py`; v1: same engine on the v1 "
         "tiers). Browsable version: `history.html`.", "",
         "Outcome legend: PASS / FAIL (rule decided), INCONCLUSIVE (pending jobs decide), PENDING (no data), REPORTED "
         "(descriptive rule).", "", "## (a) By contribution", ""]
    for c in ("C1", "C2", "C3", "C4", "infra"):
        es = [e for e in H["entries"] if e["contribution"] == c]
        if not es:
            continue
        L += [f"### {CONTRIB[c]}", ""]
        for ver in ("v2", "v1"):
            for e in [x for x in es if x["version"] == ver]:
                L += [f"#### `{e['id']}` ({ver}) - {e['label']}", "",
                      f"- title: {e['title']}", f"- parent: `{e['parent']}`; children: "
                      + (", ".join(f"`{k}`" for k in e["children"]) or "none") + f"; jobs {e['jobs']}; status "
                      f"{e['status']}; created {e['created']}",
                      f"- question: {e['question']}", f"- hypothesis: {e['hypothesis']}",
                      f"- pre-registered rule: {e['rule']}", f"- **mechanical verdict: {e['label']}** - {e['verdict']}"]
                L += [f"  - {k}" for k in e["key"]]
                if e.get("recorded"):
                    L.append(f"- verdict recorded in node.yaml: {e['recorded']}")
                if e.get("superseded"):
                    L.append(f"- superseded: {e['superseded']}")
                if e["problems"]:
                    L.append("- data problems: " + "; ".join(e["problems"]))
                links = ([f"[report]({e['report']})"] if e["report_exists"] else []) + \
                    [f"[{f}](../paper/generated/{f}.pdf)" for f in e["figs"]]
                if links:
                    L.append("- links: " + ", ".join(links))
                L.append("")
    L += ["## (b) Timeline", "", "From `JOURNAL.md` (events), node.yaml `created` dates, and the mechanical outcomes at "
          "generation time (refuted / supported).", "", "| time | version | kind | event |", "|---|---|---|---|"]
    for t in H["timeline"]:
        L.append(f"| {t['t']} | {t['version']} | {t['kind']} | {t['text'].replace('|', '/')} |")
    path.write_text("\n".join(L) + "\n")


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--card:#fafafa;--pass:#1a7f37;--fail:#b42318;--inc:#9a6700;--pen:#6e7781}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#e8e8e8;--mut:#9a9a9a;--line:#333;--card:#1a1a1a;--pass:#3fb950;--fail:#f85149;--inc:#d29922;--pen:#8b949e}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:22px}h2{font-size:18px;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:28px}h3{font-size:16px}
details{border:1px solid var(--line);border-radius:6px;margin:6px 0;background:var(--card)}
summary{cursor:pointer;padding:6px 10px;list-style:none}summary::-webkit-details-marker{display:none}
summary::before{content:'\\25B8';display:inline-block;width:1em;color:var(--mut)}details[open]>summary::before{content:'\\25BE'}
.body{padding:4px 14px 10px 26px}.mut{color:var(--mut)}code{font-size:12.5px}
.lab{font-weight:600;font-size:12px;padding:1px 6px;border-radius:4px;border:1px solid currentColor;margin-left:6px}
.PASS,.PASS\\*{color:var(--pass)}.FAIL\\*{color:var(--fail)}.PASS.FAIL{color:var(--fail)}.INCONCLUSIVE{color:var(--inc)}.PENDING,.REPORTED,.n\\/a{color:var(--pen)}
.v1{opacity:.85}.tag{font-size:11px;color:var(--mut);margin-left:6px}
table{border-collapse:collapse;width:100%;font-size:12.5px}td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}
.kind-refuted td{color:var(--fail)}.kind-supported td{color:var(--pass)}.kind-failure td{color:var(--inc)}.kind-milestone td{font-weight:600}
#filter{margin:8px 0;padding:4px 8px;width:300px;max-width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}
ul{margin:4px 0;padding-left:18px}
"""

JS = """
function f(){const q=document.getElementById('filter').value.toLowerCase();
document.querySelectorAll('[data-s]').forEach(e=>{e.style.display=e.dataset.s.includes(q)?'':'none'});}
function all(o){document.querySelectorAll('details').forEach(d=>d.open=o);}
"""


def write_html(H: dict, path: Path):
    e_ = html.escape
    P = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>Experiment History</title><style>{CSS}</style><script>{JS}</script></head><body>",
         f"<h1>HyperKKL experiment history</h1><p class=mut>Generated {e_(H['generated'])} by "
         "<code>analysis/make_history.py</code>. Protocol v1 (archived, amplitude OOD) and v2. Verdicts: pre-registered "
         "rules applied mechanically.</p>",
         "<input id=filter placeholder='filter (id, text, PASS, v1...)' oninput=f()> "
         "<button onclick=all(true)>expand all</button> <button onclick=all(false)>collapse all</button>",
         "<h2>(a) By contribution</h2>"]
    for c in ("C1", "C2", "C3", "C4", "infra"):
        es = [x for x in H["entries"] if x["contribution"] == c]
        if not es:
            continue
        P.append(f"<details open><summary><b>{e_(CONTRIB[c])}</b> <span class=mut>({len(es)} nodes)</span></summary><div class=body>")
        for ver in ("v2", "v1"):
            for e in [x for x in es if x["version"] == ver]:
                s = " ".join([e["id"], ver, e["label"], e["title"], e["verdict"], e["question"]]).lower()
                links = ([f"<a href='{e_(e['report'])}'>report.md</a>"] if e["report_exists"] else []) + \
                    [f"<a href='../paper/generated/{f}.pdf'>{f}</a>" for f in e["figs"]]
                P.append(f"<details class={ver} data-s='{e_(s)}'><summary><code>{e_(e['id'])}</code>"
                         f"<span class='lab {e_(e['label'])}'>{e_(e['label'])}</span><span class=tag>{ver} &middot; jobs "
                         f"{e_(e['jobs'])}</span> <span class=mut>{e_(e['title'])}</span></summary><div class=body><ul>"
                         f"<li><b>parent</b> <code>{e_(str(e['parent']))}</code>; <b>children</b> "
                         f"{', '.join('<code>' + e_(k) + '</code>' for k in e['children']) or 'none'}; status "
                         f"{e_(str(e['status']))}; created {e_(e['created'])}</li>"
                         f"<li><b>question</b> {e_(e['question'])}</li><li><b>hypothesis</b> {e_(e['hypothesis'])}</li>"
                         f"<li><b>pre-registered rule</b> {e_(e['rule'])}</li>"
                         f"<li><b>mechanical verdict</b> <span class={e_(e['label'])}>{e_(e['label'])}</span>: "
                         f"{e_(e['verdict'])}<ul>" + "".join(f"<li class=mut>{e_(k)}</li>" for k in e["key"]) + "</ul></li>"
                         + (f"<li><b>recorded verdict</b> {e_(e['recorded'])}</li>" if e.get("recorded") else "")
                         + (f"<li><b>superseded</b> {e_(e['superseded'])}</li>" if e.get("superseded") else "")
                         + (f"<li><b>data problems</b> {e_('; '.join(e['problems']))}</li>" if e["problems"] else "")
                         + (f"<li>{' &middot; '.join(links)}</li>" if links else "") + "</ul></div></details>")
        P.append("</div></details>")
    P.append("<h2>(b) Timeline</h2><p class=mut>JOURNAL.md events, node creation dates, and mechanical outcomes at "
             "generation time (red: pre-registered rule fails; green: passes; amber: job failures).</p>"
             "<table><tr><th>time</th><th>ver.</th><th>kind</th><th>event</th></tr>")
    for t in H["timeline"]:
        P.append(f"<tr class='kind-{t['kind']}' data-s='{e_(t['text'].lower() + ' ' + t['version'])}'><td>{e_(t['t'])}</td>"
                 f"<td>{t['version']}</td><td>{t['kind']}</td><td>{e_(t['text'])}</td></tr>")
    P.append("</table></body></html>")
    path.write_text("\n".join(P))


def main(argv=None):
    H = build()
    write_md(H, EXP / "HISTORY.md")
    write_html(H, EXP / "history.html")
    print(f"wrote {EXP / 'HISTORY.md'} and {EXP / 'history.html'} ({len(H['entries'])} entries, "
          f"{len(H['timeline'])} timeline events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
