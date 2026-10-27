"""Per-node evidence: apply each node's PRE-REGISTERED decision rule (node.yaml) mechanically.

Every analyse_* function returns a dict
    node, contribution, split ('dev' | 'test'), question, rule (verbatim), label
    (PASS / FAIL / INCONCLUSIVE / PENDING / REPORTED), verdict (one line), evidence
    (list of lines: k/n, which systems, CIs), progress, data (for figures / tables)
and never raises on missing data: absent systems are listed as pending.

Interpretation choices (stated once, applied everywhere; printed in RESULTS.md):
  I1  'beats' = paired 95% CI of the relative change excludes 0 in the better direction
      (hyperkkl.metrics.paired_relative: Student-t over seeds; 1 seed: episode bootstrap).
  I2  Where a rule names no tier, it is applied on shape_ood AND sequence_ood separately
      and passes only if it passes on both (the C1-where convention); id is reported.
  I3  'best learned' = lowest mean NRMSE among latent / lora / hyper in that block (C3);
      'best filter' = lowest among ekf, ukf, pf, kkl_mb (nominal) and ekf_huber, ukf_huber,
      ekf_oracle, ukf_oracle (robust/oracle); NLOX is a learned baseline, reported apart.
      The selection is made on the same episodes it is tested on (as in the node reports);
      the selection bias favours whichever side has more candidates.
  I4  'within CI' = the 95% CI includes 0; 'no loss' = not significantly worse. For control
      nodes the equivalence version (TOST, +-10%, 90% CI) is reported next to it, with an
      'underpowered' flag (fewer than 3 seeds or CI half-width >= 10%).
  I5  Climatology (store.Store.climatology) is the failure line for baselines: a filter
      whose mean NRMSE exceeds it is FAILED, and relative changes are quoted against the
      best WORKING filter (with the count of failed filters).
  I6  Clean-condition climatology is used for every stress condition of C3-faults (sensor
      and input faults do not change the true state; plant_drift does slightly: flagged).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from hyperkkl.config import get as cfg_get
from hyperkkl.metrics import hierarchical, paired_relative
from hyperkkl.tree import EPISODE_KEYS

from . import rules as R
from . import store as _S
from .store import dev
from .store import (DATA_DRIVEN, DATA_DRIVEN_NEW, DEV, FILTERS, FILTERS_NOMINAL, FILTERS_ROBUST, LEARNED_C3,
                    LOCATIONS, OOD_TIERS,
                    ROLES, TEST, TIERS, Store, log_rejections, rel_model_error)

TIER_SHORT = {"id": "id", "shape_ood": "shape", "sequence_ood": "seq"}

# Author decision 2026-10-26: the paper presents ONE learned method, HyperKKL := the 'lora' runs (rank r = 4;
# a hypernetwork that generates rank-r decoder weight updates from the input-history context). The 'hyper'
# variant is a dense-subspace generator ABLATION (appendix only). In SINGLE mode no rule selects the better of
# several learned variants on the evaluation data; analyse_all_both() runs both readings and keeps the
# original (pre-registered, as written) outcome next to the single-method re-reading.
HEADLINE = "lora"
ABLATION = "hyper"
SINGLE = {"on": False}
REREAD_NOTE = "single-method re-reading (author decision, 2026-10-26)"


def kkl_candidates():
    return (HEADLINE,) if SINGLE["on"] else LEARNED_C3


def loc_set():
    return tuple(l for l in LOCATIONS if l != ABLATION) if SINGLE["on"] else LOCATIONS


def kkl_variants():
    """The learned variants a rule iterates over ('for lora and hyper' rules)."""
    return (HEADLINE,) if SINGLE["on"] else (HEADLINE, ABLATION)


def pr(a, b, alpha=0.05):
    return paired_relative(a, b, n_boot=1000, alpha=alpha)


def fmt_p(p, abs_a=None, abs_b=None) -> str:
    if not p:
        return "n/a"
    s = f"{100 * p['rel']:+.0f}% [{100 * p['ci'][0]:+.0f}, {100 * p['ci'][1]:+.0f}]"
    if p.get("better") or p.get("worse"):
        s += "*"
    if abs_a is not None and abs_b is not None:
        s += f" ({abs_a:.3g} vs {abs_b:.3g})"
    return s


def block_stats(store: Store, seeds: dict, tier, cond="clean", methods=None, seed_list=None):
    ms = methods or store.methods(seeds, tier, cond)
    vals, means, ci = {}, {}, {}
    for m in ms:
        v = store.per_seed(seeds, tier, cond, m, seed_list=seed_list)
        if v is None:
            continue
        vals[m] = v
        mu, lo, hi = hierarchical(v, n_boot=1000)
        means[m], ci[m] = mu, (lo, hi)
    return vals, means, ci


def cross_pair(store: Store, na, ma, nb, mb, system, tier, cond="clean", alpha=0.05):
    """Paired relative change of na:ma against nb:mb on their shared seeds (same episodes,
    checked as in tree.cross_compare). -> (paired dict or None, seeds)."""
    ra, rb = store.runs(na).get(system, {}), store.runs(nb).get(system, {})
    seeds = sorted(set(ra) & set(rb))
    av, bv = [], []
    for k in seeds:
        A, B = store.block(ra[k], tier, cond), store.block(rb[k], tier, cond)
        if not A or not B or ma not in A["methods"] or mb not in B["methods"]:
            return None, []
        if A["families"] != B["families"]:
            store.problem(na, f"{system} seed {k}: episodes differ from {nb} ({tier}); pairing refused")
            return None, []
        ca, cb = ra[k]["config"], rb[k]["config"]
        diff = [key for key in EPISODE_KEYS + (f"conditions.{cond}",) if cfg_get(ca, key) != cfg_get(cb, key)]
        if diff:
            store.problem(na, f"{system} seed {k}: {', '.join(diff)} differ from {nb}; pairing refused")
            return None, []
        av.append(np.asarray(A["methods"][ma]["nrmse"], float))
        bv.append(np.asarray(B["methods"][mb]["nrmse"], float))
    if not seeds:
        return None, []
    return pr(av, bv, alpha), seeds


def _base(store: Store, nid: str, split=None) -> dict:
    n = store.node(nid)
    systems = sorted({s for s, _ in store.expected(nid)})
    if split is None:
        split = "test" if systems and all(ROLES[s]["role"] == "test" for s in systems) else "dev"
    return {"node": nid, "contribution": n.get("contribution"), "split": split, "title": n.get("title"),
            "question": n.get("question", ""), "hypothesis": n.get("hypothesis", ""),
            "rule": n.get("decision_rule", ""), "label": R.PENDING, "verdict": "", "evidence": [],
            "progress": store.progress(nid), "systems": systems, "data": {}, "status": n.get("status")}


def _prog(p) -> str:
    return f"{p['done']}/{p['total']} jobs done" + (f", {p['running']} running" if p.get("running") else "") \
        + (f", {p['pending']} queued" if p.get("pending") else "") + (f", {p['failed']} failed" if p.get("failed") else "")


# ---------------------------------------------------------------------------
# R0
# ---------------------------------------------------------------------------

def analyse_R0(store: Store) -> dict:
    out = _base(store, "R0")
    runs = store.runs("R0")
    p = out["progress"]
    ok_jobs = p["failed"] == 0
    rej, beats, ev = {}, {}, []
    for s in out["systems"]:
        if s not in runs:
            continue
        k = sorted(runs[s])[0]
        r = log_rejections("R0", s, k)
        n_ep = int(cfg_get(runs[s][k]["config"], "eval.n_episodes") or 64)
        rej[s] = max((v / n_ep for v in r.values()), default=float("nan"))
        vals, means, _ = block_stats(store, runs[s], "id")
        if "autonomous" not in means:
            continue
        losers = [m for m in LOCATIONS if m in means and not means[m] < means["autonomous"]]
        beats[s] = not losers
        ev.append(f"{s}: autonomous {means['autonomous']:.3g}; " + ", ".join(
            f"{m} {fmt_p(pr(vals[m], vals['autonomous']))}" for m in LOCATIONS if m in vals)
            + (f"; NOT beaten by {', '.join(losers)}" if losers else ""))
    v = R.vote(beats, out["systems"], len(out["systems"]))
    rej_ok = all(np.isfinite(x) and x < 0.05 for x in rej.values()) and len(rej) == len(out["systems"])
    out["label"] = R.combine([R.PASS if ok_jobs else R.FAIL, R.PASS if rej_ok else
                              (R.INCONCLUSIVE if len(rej) < len(out["systems"]) else R.FAIL), v.label])
    out["verdict"] = (f"no failed jobs: {ok_jobs}; max rejection {100 * max(rej.values(), default=float('nan')):.1f}% "
                      f"(< 5%: {rej_ok}); every conditioned variant beats autonomous on id: {v.frac} systems")
    out["evidence"] = ["Single seed: CIs are episode-bootstrap (training variability not measured). "
                       "'beats' = lower mean NRMSE (the rule asks for no CI)."] + ev
    return out


# ---------------------------------------------------------------------------
# C1: where the input enters
# ---------------------------------------------------------------------------

def c1_tables(store: Store, nid: str, systems, runs=None, tiers=TIERS) -> dict:
    """{tier: {system: {'means', 'ci', 'pairs' {(a, b): paired}, 'n_seeds', 'ekf'}}}"""
    runs = store.runs(nid) if runs is None else runs
    out = {}
    for tier in tiers:
        for s in systems:
            if s not in runs:
                continue
            vals, means, ci = block_stats(store, runs[s], tier)
            locs = [m for m in LOCATIONS if m in vals]
            if not locs:
                continue
            pairs = {(a, b): pr(vals[a], vals[b]) for a in locs for b in locs if a != b}
            for m in ("autonomous", "ekf"):
                if m in vals and "latent" in vals:
                    pairs[(m, "latent")] = pr(vals[m], vals["latent"])
            out.setdefault(tier, {})[s] = {"means": means, "ci": ci, "pairs": pairs, "vals": vals,
                                           "n_seeds": len(runs[s])}
    return out


def c1_rules(T: dict, systems, need=4, tiers=TIERS) -> dict:
    """Location rule ('preferred') and the weaker weight-space rule, per OOD tier."""
    res = {"location": {}, "weight_space": {}, "counts": {}}
    for tier in tiers:
        Tt = T.get(tier, {})
        for loc in loc_set():
            win, veto = {}, {}
            for s in systems:
                if s not in Tt or loc not in Tt[s]["means"]:
                    continue
                others = [o for o in loc_set() if o != loc and o in Tt[s]["means"]]
                win[s] = all(R.better(Tt[s]["pairs"][(loc, o)]) for o in others)
                veto[s] = any(R.worse(Tt[s]["pairs"][(loc, o)]) for o in others)
            res["location"].setdefault(tier, {})[loc] = R.vote(win, systems, need, veto=veto)
        ws, ws_pick, both = {}, {}, {}
        for s in systems:
            cand = (HEADLINE,) if SINGLE["on"] else tuple(m for m in ("lora", "hyper") if m in Tt.get(s, {}).get("means", {}))
            if s not in Tt or not cand or not all(m in Tt[s]["means"] for m in cand + ("latent", "concat")):
                continue
            pick = R.argmin(Tt[s]["means"], cand)
            ws_pick[s] = pick
            ws[s] = R.better(Tt[s]["pairs"][(pick, "latent")]) and R.better(Tt[s]["pairs"][(pick, "concat")])
            both[s] = all(R.better(Tt[s]["pairs"][(m, r)]) for m in cand for r in ("latent", "concat"))
        res["weight_space"][tier] = {"vote": R.vote(ws, systems, need), "pick": ws_pick,
                                     "both": R.vote(both, systems, need)}
        # plain counts vs latent, and hyper vs lora
        cnt = {}
        for m in ("concat", "film", "lora", "hyper"):
            b = {s: R.better(Tt[s]["pairs"][(m, "latent")]) for s in systems if s in Tt and (m, "latent") in Tt[s]["pairs"]}
            w = {s: R.worse(Tt[s]["pairs"][(m, "latent")]) for s in systems if s in Tt and (m, "latent") in Tt[s]["pairs"]}
            cnt[m] = {"better": [s for s, x in b.items() if x], "worse": [s for s, x in w.items() if x], "n": len(b)}
        hv = {s: Tt[s]["pairs"].get(("hyper", "lora")) for s in systems if s in Tt}
        cnt["hyper_vs_lora"] = {"hyper_better": [s for s, p in hv.items() if R.better(p)],
                                "lora_better": [s for s, p in hv.items() if R.worse(p)], "n": len(hv)}
        res["counts"][tier] = cnt
    return res


def _c1_evidence(T, RR, systems, tiers=OOD_TIERS) -> list:
    ev = []
    for tier in tiers:
        Tt = T.get(tier, {})
        if not Tt:
            ev.append(f"{tier}: pending")
            continue
        locs = "; ".join(f"{loc} {RR['location'][tier][loc].frac}" + (
            f" (veto: {', '.join(RR['location'][tier][loc].vetoed)})" if RR['location'][tier][loc].vetoed else "")
            for loc in RR["location"][tier])
        ev.append(f"{tier} - beats every other location (CI): {locs}")
        wv = RR["weight_space"][tier]
        if SINGLE["on"]:
            ev.append(f"{tier} - HyperKKL (r=4) beats latent AND concat: {wv['vote'].text()}")
        else:
            ev.append(f"{tier} - weight-space (better of lora/hyper) beats latent AND concat: {wv['vote'].text()}; "
                      f"both lora and hyper do: {wv['both'].frac}")
        for s in systems:
            if s not in Tt:
                continue
            m = Tt[s]["means"]
            cells = ", ".join(f"{'HyperKKL' if (k == 'lora' and SINGLE['on']) else k} {fmt_p(Tt[s]['pairs'][(k, 'latent')])}"
                              for k in (("concat", "film", "lora") if SINGLE["on"] else ("concat", "film", "lora", "hyper"))
                              if (k, "latent") in Tt[s]["pairs"])
            ev.append(f"  {s} (latent NRMSE {m.get('latent', float('nan')):.3g}, n={Tt[s]['n_seeds']} seeds): {cells}")
    return ev


def analyse_C1_where(store: Store, nid="C1-where", split=None) -> dict:
    out = _base(store, nid, split)
    systems = out["systems"]
    T = c1_tables(store, nid, systems)
    RR = c1_rules(T, systems)
    labels = {loc: R.combine([RR["location"][t][loc].label for t in OOD_TIERS]) for loc in loc_set()}
    passing = [l for l, lab in labels.items() if lab == R.PASS]
    if SINGLE["on"]:
        hk = "; ".join(f"{t} {RR['weight_space'][t]['vote'].frac}" for t in OOD_TIERS)
        if nid.startswith("TEST"):
            out["label"] = R.combine([RR["weight_space"][t]["vote"].label for t in OOD_TIERS])
            out["verdict"] = f"HyperKKL (r=4) beats latent and concat: {hk} (need 4/6 on each)"
        else:
            out["label"] = R.PASS if passing else (R.FAIL if all(l == R.FAIL for l in labels.values())
                                                   else R.INCONCLUSIVE)
            out["verdict"] = ("location rule over latent / concat / film / HyperKKL (dense-subspace ablation "
                              "excluded): preferred " + (", ".join("HyperKKL" if l == HEADLINE else l for l in passing)
                                                         or "NONE") + f". HyperKKL (r=4) beats latent and concat: {hk}")
    elif nid.startswith("TEST"):
        # the pre-registered TEST rule is the weaker weight-space claim
        lab = R.combine([RR["weight_space"][t]["vote"].label for t in OOD_TIERS])
        out["label"] = lab
        out["verdict"] = ("weight-space beats latent and concat: " + "; ".join(
            f"{t} {RR['weight_space'][t]['vote'].frac}" for t in OOD_TIERS) + " (need 4/6 on each)")
    else:
        out["label"] = R.PASS if passing else (R.FAIL if all(l == R.FAIL for l in labels.values()) else R.INCONCLUSIVE)
        out["verdict"] = ("preferred location: " + (", ".join(passing) if passing else "NONE") +
                          ". Weaker claim (not the pre-registered rule), weight-space beats latent and concat: " +
                          "; ".join(f"{t} {RR['weight_space'][t]['vote'].frac}" for t in OOD_TIERS))
    out["evidence"] = _c1_evidence(T, RR, systems, OOD_TIERS + ("id",))
    out["data"] = {"T": T, "rules": RR, "location_labels": labels}
    return out


def _winner(T, tier, s):
    means = T.get(tier, {}).get(s, {}).get("means", {})
    if SINGLE["on"]:
        return HEADLINE if HEADLINE in means else None
    return R.argmin(means, LOCATIONS)


def analyse_control(store: Store, group: list, kind: str, ref="C1-where") -> dict:
    """C1 control nodes. kind: 'stability' (budget, ctx), 'basewidth'."""
    nid = group[0]
    out = _base(store, nid)
    out["node"] = " + ".join(group) if len(group) > 1 else nid
    prog = [store.progress(n) for n in group]
    out["progress"] = {k: sum(p[k] for p in prog) for k in prog[0]}
    systems = sorted({s for s, _ in store.expected_pooled(group)})
    runs = store.pooled(group)
    Tref = c1_tables(store, ref, systems)
    Tc = c1_tables(store, nid, systems, runs=runs)
    per = {}
    for tier in OOD_TIERS:
        ok, ok_tost, changed, under, rows = {}, {}, {}, [], []
        for s in systems:
            W, B = _winner(Tref, tier, s), _winner(Tc, tier, s)
            if W is None or B is None:
                continue
            vals = Tc[tier][s]["vals"]
            p = None if W == B else pr(vals[W], vals[B])
            p90 = None if W == B else pr(vals[W], vals[B], alpha=0.10)
            t = R.tost(p90) if p90 else {"noninferior": True, "underpowered": Tc[tier][s]["n_seeds"] < 3}
            ok[s] = W == B or R.no_loss(p)
            ok_tost[s] = W == B or bool(t["noninferior"])
            changed[s] = W != B
            if t["underpowered"]:
                under.append(s)
            rows.append(f"  {s}: C1-where winner {W}, best here {B}" + ("" if W == B else
                        f"; {W} vs {B} {fmt_p(p)}; TOST non-inferior(10%): {t['noninferior']}") +
                        f" (n={Tc[tier][s]['n_seeds']} seeds)")
        if kind == "basewidth":
            v = R.vote({s: not c for s, c in changed.items()}, systems, len(systems) - 2)   # changes on <= 2
        else:
            v = R.vote(ok, systems, 4)
        vt = R.vote(ok_tost, systems, 4)
        per[tier] = {"vote": v, "tost": vt, "underpowered": under, "rows": rows, "changed": changed}
    lab = R.combine([per[t]["vote"].label for t in OOD_TIERS])
    out["label"] = lab
    if kind == "basewidth":
        out["verdict"] = "winner changes on " + "; ".join(
            f"{t} {sum(per[t]['changed'].values())}/{len(per[t]['changed'])}" for t in OOD_TIERS) + \
            " ('base-dependent' if >= 3/6; PASS = not base-dependent)"
    else:
        out["verdict"] = "C1-where winner best-or-within-CI: " + "; ".join(
            f"{t} {per[t]['vote'].frac}" for t in OOD_TIERS) + " (need 4/6); TOST(+-10%) version: " + "; ".join(
            f"{t} {per[t]['tost'].frac}" for t in OOD_TIERS)
    ev = []
    for t in OOD_TIERS:
        ev.append(f"{t}: {per[t]['vote'].text()}; underpowered (TOST) on {len(per[t]['underpowered'])}/"
                  f"{len(per[t]['changed'])}: {', '.join(per[t]['underpowered']) or '-'}")
        ev += per[t]["rows"]
    seeds_n = sorted({len(ss) for ss in runs.values()})
    ev.insert(0, f"seeds per system: {seeds_n or 'none'}; 'within CI' at df = n-1 is a weak test "
                 "(t_0.975 = 12.7 at 2 seeds, 4.3 at 3), hence the TOST column.")
    out["evidence"] = ev
    out["data"] = {"per": per, "Tc": Tc}
    return out


def analyse_cross_same_method(store: Store, nid: str, ref: str, method_of, tiers, criterion, need=4,
                              alpha90=False) -> dict:
    """Generic cross-node rule: per system compare nid:m against ref:m (m = method_of(s, tier))."""
    out = _base(store, nid)
    systems = out["systems"]
    per = {}
    for tier in tiers:
        ok, rows = {}, []
        for s in systems:
            m = method_of(s, tier)
            if m is None:
                continue
            p, seeds = cross_pair(store, nid, m, ref, m, s, tier)
            if p is None:
                continue
            p90, _ = cross_pair(store, nid, m, ref, m, s, tier, alpha=0.10)
            ok[s] = criterion(p, p90)
            t = R.tost(p90)
            rows.append(f"  {s}: {nid}:{m} vs {ref}:{m} {fmt_p(p)} seeds {seeds}; 90% CI "
                        f"[{100 * p90['ci'][0]:+.0f}, {100 * p90['ci'][1]:+.0f}]%"
                        + (" (underpowered)" if t["underpowered"] else ""))
        per[tier] = {"vote": R.vote(ok, systems, need), "rows": rows}
    out["label"] = R.combine([per[t]["vote"].label for t in tiers])
    out["evidence"] = sum(([f"{t}: {per[t]['vote'].text()}"] + per[t]["rows"] for t in tiers), [])
    out["data"] = {"per": per}
    return out


# ---------------------------------------------------------------------------
# C2: memory
# ---------------------------------------------------------------------------

FLOOR_NODES = {"C2-floor-025": 0.25, "C2-floor-05": 0.5, "C2-floor-1": 1.0}


def analyse_floor(store: Store, nid: str, ref="C1-where") -> dict:
    out = _base(store, nid)
    c = FLOOR_NODES[nid]
    systems = out["systems"]
    ok, ok_tost, under, rows, probe_ok, data = {}, {}, [], [], {}, {}
    for s in systems:
        cells, good, good_t, have = [], True, True, False
        for m in kkl_variants():
            for tier in ("id", "sequence_ood"):
                p, seeds = cross_pair(store, nid, m, ref, m, s, tier)
                if p is None:
                    continue
                have = True
                p90, _ = cross_pair(store, nid, m, ref, m, s, tier, alpha=0.10)
                t = R.tost(p90)
                good &= R.no_loss(p)
                good_t &= bool(t["noninferior"])
                if t["underpowered"]:
                    under.append(f"{s}/{m}/{TIER_SHORT[tier]}")
                data.setdefault(s, {})[(m, tier)] = {"p": p, "p90": p90}
                cells.append(f"{m}/{TIER_SHORT[tier]} {fmt_p(p)}")
        if have:
            ok[s], ok_tost[s] = good, good_t
            rows.append(f"  {s}: " + "; ".join(cells))
        # Lemma 2 check: probe rate >= floor rate
        for k, r in store.runs(nid).get(s, {}).items():
            for m in kkl_variants():
                mp = (r.get("memory_probe") or {}).get(m) or {}
                if "rate_other" in mp:
                    floor = c * float(mp.get("alpha_design", float("nan")))
                    probe_ok.setdefault(s, []).append((m, k, float(mp["rate_other"]), floor))
    v = R.vote(ok, systems, 5)
    vt = R.vote(ok_tost, systems, 5)
    pchk = {s: all(rate >= fl for _, _, rate, fl in L) for s, L in probe_ok.items()}
    pv = R.vote(pchk, systems, len(systems))
    ratio = {s: float(np.median([rate / fl for _, _, rate, fl in L])) for s, L in probe_ok.items()}
    out["label"] = R.combine([v.label, pv.label])
    mlab = "HyperKKL" if SINGLE["on"] else "lora & hyper"
    out["verdict"] = (f"no loss vs no floor ({mlab}, id & seq): {v.frac} (need 5/6); "
                      f"TOST(10%) non-inferior: {vt.frac}; probe decays at >= floor rate: {pv.frac} systems")
    out["evidence"] = [f"no-loss vote: {v.text()}", f"TOST non-inferiority vote: {vt.text()}; underpowered cells: "
                       f"{len(under)}", f"probe rate / floor rate (median over seeds and lora/hyper): " +
                       ", ".join(f"{s} {x:.2f}" for s, x in ratio.items()) + f"; check: {pv.text()}"] + rows
    if nid != "C2-floor-05":
        out["evidence"].insert(0, "The pre-registered decision is on C2-floor-05; this node applies the same rule "
                                  "for the dose-response.")
    out["data"] = {"c": c, "pairs": data, "probe": probe_ok, "probe_ratio": ratio}
    return out


def window_curves(store: Store, nid: str, tiers=TIERS) -> dict:
    """{system: {tier: {'L': [...], 'mean': {L: m}, 'ci', 'rel': {L: paired vs 512}}, 'Lstar', ...}}"""
    out = {}
    for s, seeds in store.runs(nid).items():
        any_r = next(iter(seeds.values()))
        dt, alpha = float(any_r["scales"]["dt"]), float(any_r["scales"]["alpha"] or any_r["config"]["system"]["alpha"])
        l_q95 = float(np.mean([r["phase1"].get("l_dec_q95", float("nan")) for r in seeds.values()]))
        eta = 0.05
        Ls_s = math.log(1 / eta) / alpha
        Lr_s = (math.log(1 / eta) + math.log(max(l_q95, 1.0))) / alpha
        d = {"dt": dt, "alpha": alpha, "l_dec_q95": l_q95, "Lstar_s": Ls_s, "Lstar": Ls_s / dt,
             "Lref_s": Lr_s, "Lref": Lr_s / dt, "n_seeds": len(seeds)}
        for tier in tiers:
            vals, means, ci = block_stats(store, seeds, tier)
            Ls = sorted(int(m.split("window=")[1]) for m in vals if "window=" in m)
            if not Ls:
                continue
            key = lambda L: f"lora@window:window={L}"
            ref = key(max(Ls))
            rel = {L: pr(vals[key(L)], vals[ref]) for L in Ls if L != max(Ls)}
            d[tier] = {"L": Ls, "mean": {L: means[key(L)] for L in Ls}, "ci": {L: ci[key(L)] for L in Ls},
                       "rel": rel, "Lmax": max(Ls),
                       "knee": R.knee([L for L in Ls if L != max(Ls)], {L: p["rel"] for L, p in rel.items()})}
        out[s] = d
    return out


def analyse_window_a1(store: Store, nid="C2-window-a1") -> dict:
    out = _base(store, nid)
    systems = out["systems"]
    W = window_curves(store, nid)
    res, rows = {}, []
    for which, key in (("theory", "Lstar"), ("refined", "Lref")):
        for tier in ("id", "sequence_ood"):
            ok, untestable = {}, []
            for s in systems:
                if s not in W or tier not in W[s]:
                    continue
                d = W[s][tier]
                Ltest = [L for L in d["L"] if L >= W[s][key] and L != d["Lmax"]]
                if not Ltest:
                    untestable.append(s)          # L* beyond the largest window below 512
                    continue
                ok[s] = all(d["rel"][L]["rel"] <= 0.05 for L in Ltest)
            testable = [s for s in systems if s not in untestable]
            # theory: the pre-registered bar (5); refined (post hoc): the same 5/6 fraction of the testable systems
            v = R.vote(ok, testable, 5 if which == "theory" else max(1, math.ceil(5 / 6 * len(testable))))
            v.untestable = untestable
            res[(which, tier)] = v
    for s in systems:
        if s not in W:
            continue
        w = W[s]
        rows.append(f"  {s}: L* = {w['Lstar']:.0f} samples ({w['Lstar_s']:.2f} s), refined L* = {w['Lref']:.0f} "
                    f"samples (l_dec_q95 = {w['l_dec_q95']:.3g}); knee id {w.get('id', {}).get('knee')}, "
                    f"seq {w.get('sequence_ood', {}).get('knee')}; e(256)/e(512)-1 seq "
                    + (fmt_p(w['sequence_ood']['rel'].get(256)) if 'sequence_ood' in w else 'n/a'))
    out["label"] = R.combine([res[("theory", t)].label for t in ("id", "sequence_ood")])
    out["verdict"] = ("e(L >= L*) within 5% of e(512): " + "; ".join(
        f"{TIER_SHORT[t]} {res[('theory', t)].frac}" for t in ("id", "sequence_ood")) + " (need 5/6). Refined "
        "L* = (ln(1/eta)+ln l)/alpha (post hoc, not pre-registered): " + "; ".join(
        f"{TIER_SHORT[t]} {res[('refined', t)].frac}" for t in ("id", "sequence_ood")))
    out["evidence"] = [f"{w} L*, {TIER_SHORT[t]}: {v.text()}" + (f"; NOT TESTABLE (L* >= largest window): "
                       f"{', '.join(v.untestable)}" if v.untestable else "") for (w, t), v in res.items()] + rows + [
        "Note: 'within 5%' is read as e(L)/e(512) - 1 <= +5% for EVERY grid L >= L* (point estimates); "
        "an L* with no grid window in [L*, 512) is 'not testable' and left out (the need stays 5)."]
    out["data"] = {"W": W, "votes": res}
    return out


def analyse_window_a2(store: Store, nid="C2-window-a2", ref="C2-window-a1") -> dict:
    out = _base(store, nid)
    systems = out["systems"]
    W1, W2 = window_curves(store, ref), window_curves(store, nid)
    ok, rows, untestable = {}, [], []
    for s in systems:
        if s not in W2 or s not in W1:
            continue
        k1 = W1[s].get("sequence_ood", {}).get("knee")
        k2 = W2[s].get("sequence_ood", {}).get("knee")
        if k1 is None and k2 is None:
            untestable.append(s)                 # no knee inside the grid at either bandwidth
        elif k1 is None or k2 is None:
            ok[s] = False
        else:
            ok[s] = max(k1, k2) / min(k1, k2) <= 2
        rows.append(f"  {s}: knee (seq, samples; smallest L within 5% of e(512)) alpha x1 {k1}, alpha x2 {k2}")
    v = R.vote(ok, [s for s in systems if s not in untestable], 2)
    if untestable:
        rows.append(f"  not testable (no knee below 512 at either bandwidth): {', '.join(untestable)}")
    out["label"] = v.label
    out["verdict"] = f"knee in samples changes by <= 2x: {v.frac} (need 2/3)"
    out["evidence"] = [v.text()] + rows
    out["data"] = {"W": W2}
    return out


# C2-trunc: fixed-architecture truncated-history SFM context (experiments/memory/PREREG.md, 2026-10-25 23:50 CET).
TRUNC_NODES = ("C2-trunc-test", "C2-trunc-dev")
TRUNC_ETA = 0.05
# samples, pre-registered: L* = (ln(1/eta) + ln l)/(alpha dt) with l = median phase-1 l_dec_q95 of TEST-C1 / C1-where,
# and L*_full = L* + ln(|B| ybar/alpha)/(alpha dt) (the design constant of Theorem m2-window(a)).
TRUNC_LSTAR = {"fhn": (163, 241), "rossler": (356, 540), "cartpole": (283, 539), "two_link_arm": (325, 485),
               "highway_traffic": (1058, 1115), "network": (512, 849),
               "battery": (436, 542), "bioreactor": (219, 289), "duffing": (281, 276), "lorenz": (234, 412),
               "quad_tank": (1123, 2483), "vdp": (199, 249)}
TRUNC_L_PREREG = {"fhn": 8, "rossler": 138, "cartpole": 76, "two_link_arm": 224, "highway_traffic": 49719, "network": 40,
                  "battery": 815, "bioreactor": 43, "duffing": 299, "lorenz": 9, "quad_tank": 14, "vdp": 23}
TRUNC_NEED = 4


def _hist_arms(methods, m) -> dict:
    """{H: method name} of the truncated arms of method m ('m@sfm:history=H')."""
    return {int(n.split("history=")[1]): n for n in methods if n.startswith(m + "@sfm:history=")}


def analyse_trunc(store: Store, nid: str) -> dict:
    """(S) every arm H with L* <= H <= settle within +5% of the inf arm (point estimate; TOST at 5% reported) and
    (T) the largest arm H <= L*/4 significantly worse, both for lora and hyper on id and sequence_ood; each on
    >= 4/6 systems. Secondary counts with L*_full. Never raises on missing data."""
    out = _base(store, nid)
    systems = out["systems"]
    runs = store.runs(nid)
    votes, curves, rows = {}, {}, []
    for which in ("Lstar", "Lfull"):
        S_ok, T_ok, S_tost, nt_S, nt_T = {}, {}, {}, [], []
        for s in systems:
            seeds = runs.get(s)
            if not seeds:
                continue
            any_r = next(iter(seeds.values()))
            N = int(any_r["scales"]["N"])
            settle = N // 2
            Lref = TRUNC_LSTAR[s][0 if which == "Lstar" else 1]
            s_ok, t_ok, s_tost, s_have, t_have, cells = True, True, True, False, False, {}
            for m in kkl_variants():
                for tier in ("id", "sequence_ood"):
                    vals, means, ci = block_stats(store, seeds, tier)
                    arms = _hist_arms(vals, m)
                    if m not in vals or not arms:
                        continue
                    rel = {H: pr(vals[n], vals[m]) for H, n in sorted(arms.items())}
                    rel90 = {H: pr(vals[n], vals[m], alpha=0.10) for H, n in sorted(arms.items())}
                    cells[(m, tier)] = {"H": sorted(arms), "rel": rel, "rel90": rel90,
                                        "mean": {H: means[n] for H, n in arms.items()}, "mean_inf": means[m],
                                        "knee": R.knee(sorted(arms), {H: p["rel"] for H, p in rel.items()}),
                                        "monotone": all(means[arms[b]] <= 1.10 * means[arms[a]] for a, b in
                                                        zip(sorted(arms)[:-1], sorted(arms)[1:]))}
                    suff = [H for H in arms if Lref <= H <= settle]
                    if suff:
                        s_have = True
                        s_ok &= all(rel[H]["rel"] <= 0.05 for H in suff)
                        s_tost &= all(bool(R.tost(rel90[H], 0.05)["noninferior"]) for H in suff)
                    low = [H for H in arms if H <= Lref / 4]
                    if low:
                        t_have = True
                        t_ok &= R.worse(rel[max(low)])
            if not cells:
                continue
            if s_have:
                S_ok[s], S_tost[s] = s_ok, s_tost
            else:
                nt_S.append(s)
            if t_have:
                T_ok[s] = t_ok
            else:
                nt_T.append(s)
            if which == "Lstar":
                l_new = float(np.mean([float(r["phase1"].get("l_dec_q95", float("nan"))) for r in seeds.values()]))
                curves[s] = {"settle": settle, "Lstar": TRUNC_LSTAR[s][0], "Lfull": TRUNC_LSTAR[s][1],
                             "l_dec_new": l_new, "l_prereg": TRUNC_L_PREREG[s], "cells": cells}
                suff = [H for H in cells[next(iter(cells))]["H"] if TRUNC_LSTAR[s][0] <= H <= settle]
                low = [H for H in cells[next(iter(cells))]["H"] if H <= TRUNC_LSTAR[s][0] / 4]
                rows.append(f"  {s}: L* = {TRUNC_LSTAR[s][0]} (full {TRUNC_LSTAR[s][1]}), settle {settle}; "
                            f"(S) arms {suff}: " + "; ".join(f"{m}/{TIER_SHORT[t]} " + ", ".join(
                                f"H{H} {fmt_p(c['rel'][H])}" for H in suff) for (m, t), c in cells.items())
                            + (f"; (T) arm {max(low)}: " + "; ".join(f"{m}/{TIER_SHORT[t]} {fmt_p(c['rel'][max(low)])}"
                                                                   for (m, t), c in cells.items()) if low else "; (T) no arm")
                            + "; knee " + ", ".join(f"{m}/{TIER_SHORT[t]} {c['knee']}" for (m, t), c in cells.items())
                            + "; monotone " + ", ".join(str(c["monotone"]) for c in cells.values()))
        vS = R.vote(S_ok, [s for s in systems if s not in nt_S], TRUNC_NEED)
        vT = R.vote(T_ok, [s for s in systems if s not in nt_T], TRUNC_NEED)
        vTost = R.vote(S_tost, [s for s in systems if s not in nt_S], TRUNC_NEED)
        vS.untestable, vT.untestable = nt_S, nt_T
        votes[which] = (vS, vT, vTost)
    vS, vT, vTost = votes["Lstar"]
    out["label"] = R.combine([vS.label, vT.label])
    out["verdict"] = (f"(S) within +5% of inf at every arm in [L*, settle]: {vS.frac}; (T) significantly worse at the "
                      f"largest arm <= L*/4: {vT.frac} (need {TRUNC_NEED}/6 each); TOST(5%) non-inferior: {vTost.frac}"
                      + ("; sufficient bound supported, tightness not" if vS.label == R.PASS and vT.label == R.FAIL else ""))
    fS, fT, fTost = votes["Lfull"]
    out["evidence"] = [f"(S) sufficiency: {vS.text()}" + (f"; NOT TESTABLE (no arm in [L*, settle]): {', '.join(vS.untestable)}" if vS.untestable else ""),
                       f"(T) tightness: {vT.text()}" + (f"; NOT TESTABLE: {', '.join(vT.untestable)}" if vT.untestable else ""),
                       f"(S) TOST non-inferiority at 5% (90% CI): {vTost.text()}",
                       f"secondary, with the design constant (L*_full): (S) {fS.frac}, (T) {fT.frac}, TOST {fTost.frac}"
                       + (f"; (S) not testable: {', '.join(fS.untestable)}" if fS.untestable else "")] + rows
    out["data"] = {"curves": curves, "votes": {k: tuple(v.__dict__ for v in vs) for k, vs in votes.items()}}
    return out


# ---------------------------------------------------------------------------
# C3: learned vs filters
# ---------------------------------------------------------------------------

def c3_block(store: Store, nid, s, tier, cond="clean") -> dict:
    seeds = store.runs(nid).get(s)
    if not seeds:
        return None
    vals, means, ci = block_stats(store, seeds, tier, cond)
    if not any(m in means for m in LEARNED_C3):
        return None
    clim, how = store.climatology(nid, s, tier)
    BL = R.argmin(means, kkl_candidates())
    BF, BN, BR = R.argmin(means, FILTERS), R.argmin(means, FILTERS_NOMINAL), R.argmin(means, FILTERS_ROBUST)
    BW, bad, present = R.best_working(means, FILTERS, clim)
    # (b) data-driven same-data baselines: decided only once the NEW baselines are merged in every seed
    # (nlox alone is always there; it is reported separately as before)
    dd_new = [m for m in DATA_DRIVEN_NEW if m in means]
    BD = R.argmin(means, DATA_DRIVEN) if dd_new else None
    comps = {}
    for tag, F in (("any", BF), ("nominal", BN), ("robust", BR), ("working", BW),
                   ("nlox", "nlox" if "nlox" in vals else None), ("data_driven", BD)):
        if BL and F:
            comps[tag] = {"filter": F, "p": pr(vals[BL], vals[F]), "failed": R.failed(means[F], clim)}
    merged = sum(1 for r in seeds.values()
                 if any(m in ((store.block(r, tier, cond) or {}).get("methods") or {}) for m in DATA_DRIVEN_NEW))
    return {"means": means, "ci": ci, "clim": clim, "clim_how": how, "BL": BL, "BF": BF, "BN": BN, "BR": BR,
            "BW": BW, "BD": BD, "failed": bad, "present": present, "comps": comps, "n_seeds": len(seeds),
            "vals": vals, "dd_merged": merged, "dd_present": dd_new}


def _c3_line(s, b) -> str:
    if b is None:
        return f"  {s}: pending"
    bl = f"{b['BL']} {b['means'][b['BL']]:.3g}"
    any_ = b["comps"].get("any")
    w = b["comps"].get("working")
    t = f"  {s}: {'HyperKKL' if SINGLE['on'] else 'best learned'} {bl}; climatology {b['clim']:.3g}; failed filters {len(b['failed'])}/{len(b['present'])}"
    if any_:
        t += f"; best filter {any_['filter']} {b['means'][any_['filter']]:.3g}" + (" FAILED" if any_["failed"] else "")
        if not any_["failed"]:
            t += f" -> {fmt_p(any_['p'])}"
    if w and (not any_ or any_["failed"]):
        t += f"; best WORKING filter {w['filter']} {b['means'][w['filter']]:.3g} -> {fmt_p(w['p'])}"
    elif not w:
        t += "; NO working filter"
    if "nlox" in b["comps"]:
        t += f"; vs NLOX {fmt_p(b['comps']['nlox']['p'])}"
    d = b["comps"].get("data_driven")
    if d:
        t += (f"; vs best DATA-DRIVEN {d['filter']} {b['means'][d['filter']]:.3g}" + (" FAILED" if d["failed"] else "")
              + f" -> {fmt_p(d['p'])}")
    else:
        t += f"; data-driven baselines merged in {b['dd_merged']}/{b['n_seeds']} seeds (PENDING)"
    return t + f" (n={b['n_seeds']})"


def c3_votes(store, nid, systems, tier, cond="clean", tags=("any", "nominal", "robust", "working", "data_driven"),
             need=4):
    blocks = {s: c3_block(store, nid, s, tier, cond) for s in systems}
    votes = {}
    for tag in tags:
        ok = {}
        for s, b in blocks.items():
            if b is None:
                continue
            c = b["comps"].get(tag)
            if tag == "data_driven" and not c:
                continue                          # baselines not merged yet: the system is pending for (b)
            if c:
                ok[s] = R.better(c["p"])
            elif tag == "working":
                # every filter FAILED: the learned observer wins iff it is itself below climatology
                ok[s] = b["BL"] is not None and not R.failed(b["means"][b["BL"]], b["clim"])
            else:
                ok[s] = False
        votes[tag] = R.vote(ok, systems, need)
    return blocks, votes


def analyse_c3_two_tier(store: Store, nid: str, primary="any", need=4,
                        tags=("any", "nominal", "robust", "working", "data_driven")) -> dict:
    """Rules of the form 'best learned beats best filter [of each kind], paired CI, on id and
    sequence_ood, >= 4/6'. primary: which comparison decides ('any', 'working', 'each')."""
    out = _base(store, nid)
    systems = out["systems"]
    ev, data, labels = [], {}, []
    for tier in ("id", "sequence_ood"):
        blocks, votes = c3_votes(store, nid, systems, tier, tags=tags)
        data[tier] = {"blocks": blocks, "votes": votes}
        if primary == "each":
            labels += [votes["nominal"].label, votes["robust"].label]
        else:
            labels.append(votes[primary].label)
        ev.append(f"{tier}: learned beats best filter (any) {votes['any'].frac}; nominal {votes['nominal'].frac}; "
                  f"robust/oracle {votes['robust'].frac}; best WORKING {votes['working'].frac}; "
                  f"(b) best DATA-DRIVEN {votes['data_driven'].frac}"
                  + (f"; pending: {', '.join(votes['any'].missing)}" if votes['any'].missing else ""))
        ev += [_c3_line(s, blocks[s]) for s in systems]
    out["label"] = R.combine(labels)
    nf = [len(b["failed"]) for t in data for b in data[t]["blocks"].values() if b]
    out["verdict"] = "; ".join(f"{TIER_SHORT[t]}: vs best filter {data[t]['votes']['any'].frac}, vs best working "
                               f"{data[t]['votes']['working'].frac}" for t in data) + \
        (f"; failed filters per system {min(nf)}-{max(nf)}" if nf else "")
    out["evidence"] = ev
    out["data"] = data
    return out


def analyse_faults(store: Store, nid="C3-faults") -> dict:
    out = _base(store, nid)
    systems = out["systems"]
    runs = store.runs(nid)
    conds = []
    for ss in runs.values():
        for r in ss.values():
            for c in (r["results"].get("sequence_ood") or {}):
                if c not in conds:
                    conds.append(c)
    data, ev, regimes, rows = {}, [], [], []
    for tier in OOD_TIERS + ("id",):
        for c in conds:
            blocks, votes = c3_votes(store, nid, systems, tier, c)
            if not any(blocks.values()):
                continue
            data.setdefault(tier, {})[c] = {"blocks": blocks, "votes": votes}
    for c in conds:
        labs = [data[t][c]["votes"]["any"].label for t in OOD_TIERS if c in data.get(t, {})]
        lab = R.combine(labs)
        if lab == R.PASS:
            regimes.append(c)
        cells = "; ".join(f"{TIER_SHORT[t]} any {data[t][c]['votes']['any'].frac} nominal "
                          f"{data[t][c]['votes']['nominal'].frac} robust {data[t][c]['votes']['robust'].frac} "
                          f"working {data[t][c]['votes']['working'].frac}" for t in OOD_TIERS if c in data.get(t, {}))
        rows.append({"cond": c, "label": lab, "cells": cells})
        ev.append(f"{c}: learned-wins regime {lab} - {cells}")
    out["label"] = R.REPORTED
    out["verdict"] = ("learned-wins regime (>= 4/6 vs the best filter of either kind, shape and seq): "
                      + (", ".join(regimes) if regimes else "none") + f" of {len(conds)} conditions")
    out["evidence"] = ev + ["Clean-condition climatology used for every condition (I6)."] + \
        [_c3_line(s, data.get("sequence_ood", {}).get("clean", {}).get("blocks", {}).get(s)) for s in systems]
    out["data"] = {"by": data, "conds": conds, "rows": rows}
    return out


def analyse_partial(store: Store, nid="C3-partial") -> dict:
    out = analyse_c3_two_tier(store, nid)
    systems = out["systems"]
    me = {s: rel_model_error(s, "partial") for s in systems}
    pred_learned = {"bioreactor", "vdp", "lorenz"}
    pred_filter = {"duffing", "quad_tank"}
    match, rows = {}, []
    for s in systems:
        b = out["data"]["sequence_ood"]["blocks"].get(s)
        if b is None:
            continue
        c = b["comps"].get("working") or b["comps"].get("any")
        lw = R.better(c["p"]) if c else True
        if s in pred_learned:
            match[s] = lw
        elif s in pred_filter:
            match[s] = not lw
        rows.append(f"  {s}: model error {me[s]:.2f}; learned beats best working filter: {lw}")
    named = sorted(x for x in pred_learned | pred_filter if x in dev())     # 5 named plants; 4 after exclusion
    v = R.vote(match, named, len(named))
    out["pattern"] = v
    out["label"] = R.REPORTED
    out["verdict"] += f"; hypothesis pattern (learned wins on bioreactor/vdp/lorenz, filters on duffing/quad_tank) " \
                      f"holds on {v.frac} named systems"
    out["evidence"] += [f"model error vs outcome (sequence_ood, vs best working filter):"] + rows
    out["data"]["model_error"] = me
    return out


def analyse_partial_modelfree(store: Store, nid="C3-partial-modelfree", ref="C3-partial") -> dict:
    blocks = {}

    def mof(s, tier):
        b = blocks.setdefault((s, tier), c3_block(store, ref, s, tier))
        return b["BL"] if b else None
    out = analyse_cross_same_method(store, nid, ref, mof, ("id", "sequence_ood"),
                                    criterion=lambda p, p90: R.worse(p))   # modelfree worse = model helps
    out["label"] = out["data"]["per"]["sequence_ood"]["vote"].label
    out["verdict"] = ("C3-partial (partial model in the PDE term) beats model-free phase 1: " + "; ".join(
        f"{TIER_SHORT[t]} {out['data']['per'][t]['vote'].frac}" for t in ("id", "sequence_ood")) +
        " (need 4/6; decided on the compare tier sequence_ood)")
    return out


def analyse_modelfree(store: Store, nid="C1-modelfree", ref="C1-where") -> dict:
    Tref = c1_tables(store, ref, dev())
    out = analyse_cross_same_method(store, nid, ref, lambda s, t: _winner(Tref, t, s), OOD_TIERS,
                                    criterion=lambda p, p90: R.worse(p))
    out["verdict"] = "C1-where beats model-free phase 1 (same method, CI): " + "; ".join(
        f"{TIER_SHORT[t]} {out['data']['per'][t]['vote'].frac}" for t in OOD_TIERS) + " (need 4/6)"
    return out


def analyse_rank(store: Store, nid: str, ref="C1-where") -> dict:
    out = analyse_cross_same_method(store, nid, ref, lambda s, t: "lora", ("sequence_ood",),
                                    criterion=lambda p, p90: bool(R.tost(p90)["noninferior"]))
    out["verdict"] = (f"lora at this rank within +-10% of rank 4 or better (TOST non-inferiority, 90% CI): "
                      f"{out['data']['per']['sequence_ood']['vote'].frac} (need 4/6)")
    return out


# ---------------------------------------------------------------------------
# C1 per-variant tuning (experiments/tuning/PROTOCOL.md)
# ---------------------------------------------------------------------------

TUNE_SELECTION = Path(__file__).resolve().parent.parent / "experiments" / "tuning" / "selection.json"


def _selection():
    import json
    return json.loads(TUNE_SELECTION.read_text()) if TUNE_SELECTION.exists() else None


def _stub(nid, split, question, rule) -> dict:
    """Entry for a node that does not exist yet (created later from selection.json)."""
    return {"node": nid, "contribution": "C1", "split": split, "title": nid, "question": question,
            "hypothesis": "", "rule": rule, "label": R.PENDING, "verdict": "node not created yet", "evidence": [],
            "progress": {}, "systems": [], "data": {}, "status": "planned"}


TUNE_RULE = ("Tuned HyperKKL (the low-rank hypernetwork, 'lora' code path) beats tuned latent AND tuned concat with a "
             "paired 95% CI on >= 4/6 dev and >= 4/6 test systems, shape_ood and sequence_ood separately "
             "(experiments/tuning/PROTOCOL.md, section 8, 2026-10-26 00:52; dense 'hyper' = appendix ablation)")
HEADLINE = "lora"                      # HyperKKL in the paper


def _rule_systems(systems) -> tuple:
    """(systems, need) of the current reading: the store's exclusion switch removes EXCLUDED_SYSTEMS
    (post-hoc, headline reading) and scales the pre-registered 4/6 to ceil(4 n / 6)."""
    from . import store as _st
    keep = [s_ for s_ in systems if not _st.excluded(s_)]
    return keep, (4 if len(keep) == len(systems) else math.ceil(4 * len(keep) / 6))


def _headline_votes(T, systems, m=HEADLINE, need=None) -> dict:
    """{tier: Vote}: m beats latent AND concat (paired 95% CI) per system (current reading)."""
    systems, need0 = _rule_systems(systems)
    need = need0 if need is None else need
    out = {}
    for tier in OOD_TIERS:
        Tt = T.get(tier, {})
        ok = {s: R.better(Tt[s]["pairs"][(m, "latent")]) and R.better(Tt[s]["pairs"][(m, "concat")])
              for s in systems if s in Tt and all(k in Tt[s]["means"] for k in (m, "latent", "concat"))}
        out[tier] = R.vote(ok, systems, need)
    return out


def analyse_tune_select(store: Store, nid="C1-tune-select") -> dict:
    """REPORTED: the mechanical selection (analysis/select_tuned.py -> selection.json)."""
    from .select_tuned import select_nodes
    if nid not in store.nodes():
        return _stub(nid, "dev", "Per-variant tuning: which configuration of each variant is selected on validation?",
                     "argmin of the mean dev validation NRMSE over 8 configurations (PROTOCOL.md)")
    out = _base(store, nid, "dev")
    kids = [k for k in select_nodes() if k in store.nodes()]
    prog = [store.progress(k) for k in kids]
    out["progress"] = {k: sum(p[k] for p in prog) for k in prog[0]} if prog else store.progress(nid)
    sel = _selection()
    out["label"] = R.REPORTED if sel else R.PENDING
    if not sel:
        out["verdict"] = "selection.json not written yet (analysis/select_tuned.py)"
        return out
    out["verdict"] = "selected: " + "; ".join(
        f"{k} {v['selected']} ({100 * (v.get('rel_vs_default') or 0):+.1f}% val vs default)" for k, v in sel["variants"].items())
    ev = [f"rule: {sel['rule']['score']}; tie {100 * sel['rule']['tie']:.1f}% -> default; written {sel['generated']}"]
    for k, v in sel["variants"].items():
        rows = sorted(v["configs"], key=lambda r: r["score_mean"] if r["score_mean"] is not None else 1e9)
        ev.append(f"{k}: selected {v['selected']}" + (" (tie -> default)" if v["tie_to_default"] else "") +
                  "; robustness (gmean / mean rank / val loss agree): " +
                  ", ".join(f"{a} {'yes' if b['agrees'] else 'NO (' + b['choice'] + ')'}" for a, b in v["robustness"].items()))
        ev.append("  " + "; ".join(f"{r['method']} {r['score_mean']:.4g}" for r in rows))
        ws = sorted({(s_, d.get("width"), d.get("n_params")) for r in rows for s_, d in r["per_system"].items()
                     if r["method"] == v["selected"]})
        ev.append("  selected widths/params: " + ", ".join(f"{s_} w={w} p={p}" for s_, w, p in ws))
    out["evidence"] = ev
    out["data"] = {"selection": sel}
    return out


def _renamed(runs: dict, mapping: dict) -> dict:
    """runs with method names mapped (tuned method string -> its kind) in every block."""
    out = {}
    for s, ss in runs.items():
        for k, r in ss.items():
            res = {t: {c: dict(b, methods={mapping.get(m, m): v for m, v in b["methods"].items()})
                       for c, b in conds.items()} for t, conds in (r.get("results") or {}).items()}
            out.setdefault(s, {})[k] = dict(r, results=res)
    return out


def analyse_C1_tuned(store: Store, dev_node="C1-tuned", test_node="TEST-C1-tuned") -> tuple:
    """Rule (PROTOCOL.md section 8, 2026-10-26 00:52, replacing section 5): tuned HyperKKL ('lora')
    beats tuned latent AND tuned concat (paired 95% CI) on >= 4/6 dev and >= 4/6 test systems, on
    shape_ood and sequence_ood separately. The original 'better of lora/hyper' vote is reported as
    a secondary line. Returns (dev-split entry, test-split entry)."""
    from hyperkkl.conditioning import method_kind
    outs = []
    parts = {}
    nodes = store.nodes()
    for nid, ref, split in ((dev_node, "C1-where", "dev"), (test_node, "TEST-C1", "test")):
        if nid not in nodes:
            outs.append(_stub(nid, split, "With an equal tuning budget per variant, does weight-space conditioning "
                                          "still beat latent injection and the additive residual?", TUNE_RULE))
            parts[nid] = None
            continue
        o = _base(store, nid)
        methods = nodes[nid].get("methods") or []
        mapping = {m: method_kind(m) for m in methods}
        systems = o["systems"]
        T = c1_tables(store, nid, systems, runs=_renamed(store.runs(nid), mapping))
        RR = c1_rules(T, systems)
        votes = _headline_votes(T, systems)
        o["label"] = R.combine([v.label for v in votes.values()])
        ev = [f"{t} - HEADLINE: tuned HyperKKL (lora) beats tuned latent AND concat: {v.text()}" for t, v in votes.items()]
        ev += [f"{t} - secondary (original section 5): better of tuned lora/hyper beats latent AND concat: "
               f"{RR['weight_space'][t]['vote'].frac}" for t in OOD_TIERS]
        ev += ["per system, by plant size:"] + _hk_per_system(T, systems)
        ev += _c1_evidence(T, RR, systems, OOD_TIERS + ("id",))
        # tuned vs default (the parent node: C1-where seeds 0-2 / TEST-C1), same episodes
        for tier in OOD_TIERS:
            for m, kind in mapping.items():
                cells, same_base = [], 0
                for s in systems:
                    p, seeds = cross_pair(store, nid, m, ref, kind, s, tier)
                    cells.append(f"{s} {fmt_p(p)}" if p else f"{s} n/a")
                ev.append(f"{tier} - tuned {m} vs {ref} default {kind}: " + "; ".join(cells))
        # are the base maps the same as the parent node's (phase-1 diagnostics equal)?
        eq, tot = 0, 0
        ra, rb = store.runs(nid), store.runs(ref)
        for s in ra:
            for k in ra[s]:
                if k in rb.get(s, {}):
                    tot += 1
                    eq += ra[s][k]["phase1"] == rb[s][k]["phase1"]
        ev.insert(0, f"base maps identical to {ref} (phase-1 diagnostics equal): {eq}/{tot} (system, seed) pairs")
        o["evidence"] = ev
        o["data"] = {"T": T, "rules": RR, "mapping": mapping}
        parts[nid] = votes
        outs.append(o)
    dev, test = outs
    labels = [v.label for vs in parts.values() if vs for v in vs.values()]
    if all(parts.values()):
        joint = R.combine(labels)
    else:                                   # the missing half can still fail the AND rule, never pass it
        joint = R.FAIL if R.FAIL in labels else (R.INCONCLUSIVE if labels else R.PENDING)
    txt = lambda vs: "; ".join(f"{t} {v.frac} (need {v.need})" for t, v in vs.items()) if vs else "pending"
    dev["label"] = joint
    dev["verdict"] = (f"tuned HyperKKL (lora) beats tuned latent and concat: dev {txt(parts[dev_node])}; "
                      f"test {txt(parts[test_node])}; all four must pass")
    if dev["progress"]:
        dev["progress"] = {k: dev["progress"].get(k, 0) + test["progress"].get(k, 0) for k in dev["progress"]}
    if parts[test_node] is not None:
        test["label"] = R.combine([v.label for v in parts[test_node].values()])
        test["verdict"] = f"test half of the C1-tuned rule: {txt(parts[test_node])} (need 4/6 on each)"
    return dev, test


def _merged_runs(store: Store, sources: dict) -> dict:
    """{system: {seed: run}} whose blocks hold, under each KIND, the method of the node given by
    sources {kind: (node, method)}; only (system, seed) pairs present in every source node, and
    only when every source saw the same episodes (families) in that block."""
    runs = {k: store.runs(n) for k, (n, _) in sources.items()}
    out = {}
    for s in set.intersection(*[set(r) for r in runs.values()]):
        for k in set.intersection(*[set(r[s]) for r in runs.values()]):
            base = runs[next(iter(sources))][s][k]
            res = {}
            for t, conds in (base.get("results") or {}).items():
                for c, b in conds.items():
                    ms, ok = {}, True
                    for kind, (n, m) in sources.items():
                        bb = store.block(runs[kind][s][k], t, c)
                        if not bb or m not in bb["methods"] or bb["families"] != b["families"]:
                            ok = False
                            break
                        ms[kind] = bb["methods"][m]
                    if ok:
                        res.setdefault(t, {})[c] = dict(b, methods=ms)
            out.setdefault(s, {})[k] = dict(base, results=res)
    return out


def _hk_per_system(T, systems, tiers=OOD_TIERS) -> list:
    """Per system: tuned HyperKKL (lora) vs tuned concat and vs tuned latent, with (n_x, n_y)."""
    from .store import ROLES
    out = []
    for tier in tiers:
        Tt = T.get(tier, {})
        for s_ in sorted(systems, key=lambda x: (ROLES[x]["nx"], ROLES[x]["ny"], x)):
            if s_ not in Tt or HEADLINE not in Tt[s_]["means"]:
                continue
            pp = Tt[s_]["pairs"]
            out.append(f"  {TIER_SHORT[tier]} {s_} (n_x={ROLES[s_]['nx']}, n_y={ROLES[s_]['ny']}): HyperKKL vs concat "
                       f"{fmt_p(pp.get((HEADLINE, 'concat')))}, vs latent {fmt_p(pp.get((HEADLINE, 'latent')))}")
    return out


STAGE_FILES = {"ext1": ("selection_ext1b.json", "selection_ext1.json"), "ext2": ("selection_ext2.json",)}
DEV_CONFIRM = ("C1-tuned", "C1-tuned-ext1", "C1-tuned-ext2")
TEST_CONFIRM = ("TEST-C1-tuned", "TEST-C1-tuned-ext1", "TEST-C1-tuned-ext2")


def analyse_C1_tuned_stage(store: Store, label: str) -> tuple:
    """Verdict after an equal extension (PROTOCOL.md sections 9-10; NOT pre-registered): the
    section-8 headline rule on that stage's selection. Each variant's selected method is taken from
    whichever confirmation node ran it (same episodes and base maps, paired per seed)."""
    import json
    sel = None
    for f in STAGE_FILES[label]:
        p = TUNE_SELECTION.with_name(f)
        if p.exists():
            sel, fname = json.loads(p.read_text()), f
            break
    q = f"After the equal '{label}' extension: does tuned HyperKKL still beat tuned latent and concat?"
    rule = f"EXTENDED '{label}' (not pre-registered; PROTOCOL.md sections 9-10): " + TUNE_RULE
    ids = (f"C1-tuned-{label}", f"TEST-C1-tuned-{label}")
    if not sel:
        return _stub(ids[0], "dev", q, rule), _stub(ids[1], "test", q, rule)
    nodes = store.nodes()
    outs, parts = [], {}
    for confirm, nid, split in ((DEV_CONFIRM, ids[0], "dev"), (TEST_CONFIRM, ids[1], "test")):
        src, missing = {}, []
        for k, m in sel["selected"].items():
            n = next((n for n in confirm if n in nodes and m in (nodes[n].get("methods") or [])), None)
            if n is None:
                missing.append(m)
            else:
                src[k] = (n, m)
        if missing:
            o = _stub(nid, split, q, rule)
            o["verdict"] = f"{fname}: not yet confirmed: {missing}"
            outs.append(o)
            parts[split] = None
            continue
        o = _base(store, confirm[0], split)
        o.update(node=nid, question=q, rule=rule)
        used = sorted({n for n, _ in src.values()})
        ps = [store.progress(n) for n in used]
        o["progress"] = {k: sum(p_[k] for p_ in ps) for k in ps[0]}
        runs = _merged_runs(store, src)
        systems = o["systems"]
        T = c1_tables(store, confirm[0], systems, runs=runs)
        RR = c1_rules(T, systems)
        votes = _headline_votes(T, systems)
        o["label"] = R.combine([v.label for v in votes.values()])
        o["evidence"] = [f"{fname}: selected {sel['selected']}; runs from {used}"] + \
            [f"{t} - '{label}' HEADLINE: tuned HyperKKL (lora) beats tuned latent AND concat: {v.text()}"
             for t, v in votes.items()] + ["per system, by plant size:"] + _hk_per_system(T, systems) + \
            _c1_evidence(T, RR, systems, OOD_TIERS + ("id",))
        o["data"] = {"T": T, "rules": RR, "sources": src}
        parts[split] = votes
        outs.append(o)
    dev, test = outs
    labels = [v.label for vs in parts.values() if vs for v in vs.values()]
    joint = R.combine(labels) if all(parts.values()) else (R.FAIL if R.FAIL in labels else (R.INCONCLUSIVE if labels else R.PENDING))
    txt = lambda vs: "; ".join(f"{t} {v.frac} (need {v.need})" for t, v in vs.items()) if vs else "pending"
    dev["label"] = joint
    dev["verdict"] = (f"EXTENDED '{label}': tuned HyperKKL beats tuned latent and concat: dev {txt(parts['dev'])}; "
                      f"test {txt(parts['test'])}; all four must pass")
    if parts["test"] is not None:
        test["label"] = R.combine([v.label for v in parts["test"].values()])
        test["verdict"] = f"test half of the EXTENDED '{label}' rule: {txt(parts['test'])}"
    return dev, test


def analyse_data_curve(store: Store, nid: str) -> dict:
    out = _base(store, nid)
    runs = store.runs(nid)
    ev = []
    for s in out["systems"]:
        if s not in runs:
            continue
        _, means, _ = block_stats(store, runs[s], "shape_ood")
        ev.append(f"  {s}: " + ", ".join(f"{m} {means[m]:.3g}" for m in LEARNED_C3 if m in means))
    out["label"] = R.REPORTED if runs else R.PENDING
    out["verdict"] = "learning curve reported (Figure F3 companion); no pass/fail rule"
    out["evidence"] = ev
    return out


GREYBOX = ("greybox_ekf", "greybox_ukf", "greybox_ekf_oracle")
REGIME_KKL = ("lora", "hyper")


def analyse_regimes(store: Store, nid="C3-regimes") -> dict:
    """node.yaml rule, as written: 'Per condition, best KKL (lora/hyper) vs best grey-box filter
    (greybox_ekf/ukf/ekf_oracle): paired 95% CI excluding 0 on >= 4/6 dev systems on sequence_ood =
    KKL-wins regime; report every condition with k/6 and the absolute NRMSE; also report divergence rate
    and forget-prior time.' Selection of the best of each side as elsewhere (I3)."""
    out = _base(store, nid)
    systems = out["systems"]
    runs = store.runs(nid)
    conds = list(cfg_get(next(iter(next(iter(runs.values())).values()))["config"], "eval.conditions") or []) if runs \
        else list((store.node(nid).get("set") or {}).get("eval.conditions") or [])
    tier = "sequence_ood"
    rows, ev, regimes = {}, [], []
    for c in conds:
        per, kl, gl = {}, {}, {}
        for s in systems:
            seeds = runs.get(s)
            if not seeds:
                continue
            vals, means, _ = block_stats(store, seeds, tier, c)
            K, G = R.argmin(means, tuple(m for m in REGIME_KKL if m in kkl_candidates())), R.argmin(means, GREYBOX)
            if not K or not G:
                continue
            p = pr(vals[K], vals[G])

            def div(m):
                v = [store.episodes(r, tier, c, m, "diverged") for r in seeds.values()]
                v = [x for x in v if x is not None]
                return float(np.mean([np.mean(x) for x in v])) if v else float("nan")

            def forget(m):
                v = [((store.block(r, tier, c) or {}).get("scorecard") or {}).get(m, {}).get("t_forget_median_s")
                     for r in seeds.values()]
                v = [float(x) for x in v if x is not None and np.isfinite(x)]
                return float(np.mean(v)) if v else float("nan")
            per[s] = {"K": K, "G": G, "kkl": means[K], "gb": means[G], "p": p, "div_k": div(K), "div_g": div(G),
                      "forget_k": forget(K), "forget_g": forget(G), "n": len(seeds)}
            kl[s] = R.better(p)
            gl[s] = R.worse(p)
        v = R.vote(kl, systems, 4)
        lose = sum(gl.values())
        rows[c] = {"vote": v, "gb_wins": lose, "per": per}
        if v.label == R.PASS:
            regimes.append(c)
        ev.append(f"{c}: KKL-wins regime {v.label} - KKL better {v.frac}, grey-box better {lose}/{len(gl)}")
        for s, d in per.items():
            ev.append(f"  {s}: {d['K']} {d['kkl']:.3g} vs {d['G']} {d['gb']:.3g} -> {fmt_p(d['p'])}; div "
                      f"{d['div_k']:.2f}/{d['div_g']:.2f}" + (f"; forget {d['forget_k']:.3g}/{d['forget_g']:.3g} s"
                                                            if np.isfinite(d['forget_k']) else "") + f" (n={d['n']})")
    out["label"] = R.REPORTED if runs else R.PENDING
    out["verdict"] = ("KKL-wins regimes (>= 4/6, sequence_ood, vs best grey-box): " +
                      (", ".join(regimes) or "none") + f" of {len(conds)} conditions; " +
                      "; ".join(f"{c} {r['vote'].frac}" for c, r in rows.items()))
    out["evidence"] = ev
    out["data"] = {"rows": rows, "conds": conds}
    return out


DD_NODES = ("C3-mismatch10", "C3-mismatch20", "C3-mismatch30", "C3-mismatch50", "C3-noise-ct", "C3-partial",
            "C3-realistic", "TEST-C3")


def analyse_data_driven(store: Store, res: dict, nid: str) -> dict:
    """Comparison (b): best KKL observer vs the best DATA-DRIVEN same-data baseline (nlox, koopman_kf[_bilinear],
    greybox_ekf/ukf/ekf_oracle), paired CI, >= 4/6 on id and sequence_ood (the bar of the node's own rule). Not
    pre-registered: added after the reviews (Reviewer-C E1/E2). PENDING until the new baselines are merged."""
    base = res[nid]
    out = {k: base.get(k) for k in ("contribution", "split", "systems")}
    out.update({"node": nid + " (b)", "title": f"{nid}: best KKL vs best data-driven same-data baseline",
                "question": "Does the KKL structure help beyond learning from real-plant data? (best KKL vs best "
                            "data-driven baseline trained on the same data)",
                "rule": "Not pre-registered (added after review E1/E2): best KKL beats the best data-driven baseline "
                        "(nlox, koopman_kf[_bilinear], greybox_ekf/ukf/ekf_oracle) with a paired 95% CI excluding 0 on "
                        ">= 4/6 systems, on id and sequence_ood.", "evidence": [], "data": {}})
    runs = store.runs(nid)
    tot = sum(len(v) for v in runs.values())
    merged = 0
    for ss in runs.values():
        for r in ss.values():
            b = store.block(r, "sequence_ood") or {}
            merged += any(m in (b.get("methods") or {}) for m in DATA_DRIVEN_NEW)
    p = dict(base.get("progress") or {})
    out["progress"] = {"done": merged, "total": len(store.expected(nid)), "running": 0, "pending": 0, "failed": 0}
    out["merged"] = f"{merged}/{len(store.expected(nid))}"
    labs, ev = [], [f"new data-driven baselines merged into {merged}/{len(store.expected(nid))} runs"]
    for tier in ("id", "sequence_ood"):
        d = (base.get("data") or {}).get(tier)
        if not d or "votes" not in d:
            continue
        v = d["votes"]["data_driven"]
        labs.append(v.label)
        out["data"][tier] = {"vote": v}
        ev.append(f"{tier}: best KKL beats best data-driven {v.text()}")
        for s, b in d["blocks"].items():
            if b and b["comps"].get("data_driven"):
                c = b["comps"]["data_driven"]
                ev.append(f"  {s}: {b['BL']} {b['means'][b['BL']]:.3g} vs {c['filter']} {b['means'][c['filter']]:.3g}"
                          + (" (FAILED)" if c["failed"] else "") + f" -> {fmt_p(c['p'])}")
    out["label"] = R.combine(labs) if merged else R.PENDING
    if merged and out["label"] == R.PENDING:
        out["verdict"] = f"pending: merged in {out['merged']} runs, but no system has the new baselines in all its seeds yet"
        out["evidence"] = ev
        return out
    out["verdict"] = ("best KKL vs best data-driven: " + "; ".join(
        f"{TIER_SHORT[t]} {out['data'][t]['vote'].frac}" for t in out["data"]) + f" (merged runs {out['merged']})"
        if merged else f"pending: new data-driven baselines merged in {out['merged']} runs")
    out["evidence"] = ev
    return out


def model_wrong_summary(store: Store, scen) -> dict:
    """Model-wrong cells (mismatch, partial, realistic) x dev systems, sequence_ood: does the best KKL observer beat
    the best WORKING model-based filter? Returns counts and the model-error band (used by RESULTS.md and macros)."""
    pts = []
    for lab, nid, cond, k_, m_ in scen:
        if (k_ == "full" and m_ == 0) or lab.startswith("drift"):
            continue
        for s in dev():
            b = c3_block(store, nid, s, "sequence_ood", cond)
            if not b:
                continue
            c = b["comps"].get("working")
            win = R.better(c["p"]) if c else not R.failed(b["means"][b["BL"]], b["clim"])
            pts.append((rel_model_error(s, k_, m_), win, f"{lab}/{s}"))
    wins = [p for p in pts if p[1]]
    loss = [p for p in pts if not p[1]]
    return {"n": len(pts), "wins": len(wins), "min_err_win": min(wins)[0] if wins else float("nan"),
            "min_err_win_cell": min(wins)[2] if wins else "", "max_err_nowin": max(loss)[0] if loss else float("nan"),
            "max_err_nowin_cell": max(loss)[2] if loss else "", "nonwins": [p[2] for p in sorted(loss)]}


MISMATCH = {"C3-mismatch10": 0.1, "C3-mismatch20": 0.2, "C3-mismatch30": 0.3, "C3-mismatch50": 0.5}


def analyse_crossover(store: Store, results: dict) -> dict:
    """Composite (hypothesis of the mismatch nodes): crossover between 10% and 30%."""
    lv = {MISMATCH[n]: results[n]["label"] for n in MISMATCH if n in results}
    out = {"node": "C3-mismatch (crossover)", "contribution": "C3", "split": "dev",
           "question": "Where does the filter-vs-learned crossover sit?",
           "rule": "Hypothesis of C3-mismatch*: crossover between 10% and 30% mismatch (learned wins at a level "
                   "per that node's rule).", "progress": {}, "data": {"levels": lv}, "systems": list(dev())}
    first = min((m for m, l in lv.items() if l == R.PASS), default=None)
    if lv.get(0.1) == R.FAIL and lv.get(0.3) == R.PASS:
        out["label"] = R.PASS
    elif lv.get(0.1) == R.PASS or lv.get(0.3) == R.FAIL:
        out["label"] = R.FAIL
    else:
        out["label"] = R.INCONCLUSIVE
    out["verdict"] = "level outcomes: " + ", ".join(f"{int(100 * m)}% {l}" for m, l in sorted(lv.items())) + \
        (f"; first winning level {int(100 * first)}%" if first is not None else "")
    out["evidence"] = [out["verdict"]]
    return out


# ---------------------------------------------------------------------------
# Everything
# ---------------------------------------------------------------------------

def analyse_all(store: Store) -> dict:
    res = {}
    res["R0"] = analyse_R0(store)
    res["C1-where"] = analyse_C1_where(store)
    res["TEST-C1"] = analyse_C1_where(store, "TEST-C1", "test")
    res["C1-budget-s"] = analyse_control(store, ["C1-budget-s", "C1-budget-s-s2"], "stability")
    res["C1-budget-l"] = analyse_control(store, ["C1-budget-l", "C1-budget-l-s2"], "stability")
    # the budget rule is joint: 'at both budgets'
    vs = [res[n]["data"]["per"][t]["vote"] for n in ("C1-budget-s", "C1-budget-l") for t in OOD_TIERS]
    both = {}
    for t in OOD_TIERS:
        a, b = res["C1-budget-s"]["data"]["per"][t]["vote"], res["C1-budget-l"]["data"]["per"][t]["vote"]
        ok = {s: True for s in set(a.met) & set(b.met)}
        ok.update({s: False for s in set(a.not_met) | set(b.not_met)})
        both[t] = R.vote(ok, list(dev()), 4)
    res["C1-budget"] = {"node": "C1-budget-s + C1-budget-l (joint rule)", "contribution": "C1", "split": "dev",
                        "question": store.node("C1-budget-s")["question"],
                        "rule": store.node("C1-budget-s")["decision_rule"],
                        "label": R.combine([both[t].label for t in OOD_TIERS]),
                        "verdict": "stable at BOTH budgets: " + "; ".join(f"{t} {both[t].frac}" for t in OOD_TIERS),
                        "evidence": [f"{t}: {both[t].text()}" for t in OOD_TIERS], "progress": {},
                        "data": {"both": both}, "systems": list(dev())}
    del vs
    res["C1-ctx"] = analyse_control(store, ["C1-ctx"], "stability")
    res["C1-basewidth-64"] = analyse_control(store, ["C1-basewidth-64", "C1-basewidth-64-s2"], "basewidth")
    res["C1-basewidth-256"] = analyse_control(store, ["C1-basewidth-256", "C1-basewidth-256-s2"], "basewidth")
    res["C1-modelfree"] = analyse_modelfree(store)
    res["C1-rank1"] = analyse_rank(store, "C1-rank1")
    res["C1-rank16"] = analyse_rank(store, "C1-rank16")
    res["C1-tune-select"] = analyse_tune_select(store)
    res["C1-tuned"], res["TEST-C1-tuned"] = analyse_C1_tuned(store)
    for lab in ("ext1", "ext2"):
        res[f"C1-tuned-{lab}"], res[f"TEST-C1-tuned-{lab}"] = analyse_C1_tuned_stage(store, lab)
    for n in FLOOR_NODES:
        res[n] = analyse_floor(store, n)
    res["C2-window-a1"] = analyse_window_a1(store)
    res["C2-window-a2"] = analyse_window_a2(store)
    for n in TRUNC_NODES:
        res[n] = analyse_trunc(store, n)
    res["C3-faults"] = analyse_faults(store)
    for n in MISMATCH:
        res[n] = analyse_c3_two_tier(store, n, primary="any")
    res["C3-mismatch (crossover)"] = analyse_crossover(store, res)
    res["C3-noise-ct"] = analyse_c3_two_tier(store, "C3-noise-ct", primary="each")
    res["C3-partial"] = analyse_partial(store)
    res["C3-partial-modelfree"] = analyse_partial_modelfree(store)
    res["C3-realistic"] = analyse_c3_two_tier(store, "C3-realistic", primary="any")
    res["TEST-C3"] = analyse_c3_two_tier(store, "TEST-C3", primary="working")
    for n in DD_NODES:
        res[n + " (b)"] = analyse_data_driven(store, res, n)
    res["C3-regimes"] = analyse_regimes(store)
    res["C3-data50"] = analyse_data_curve(store, "C3-data50")
    res["C3-data150"] = analyse_data_curve(store, "C3-data150")
    for k, v in res.items():
        p = v.get("progress") or {}
        v["provisional"] = False
        if k.endswith(" (b)"):                   # progress = merged runs; label/verdict set by analyse_data_driven
            v["provisional"] = v["label"] in (R.PASS, R.FAIL) and p.get("done", 0) < p.get("total", 0)
            continue
        if p and p.get("done", 0) == 0:
            v["label"] = R.PENDING
            v["verdict"] = f"pending ({_prog(p)})"
        elif p and p.get("done", 0) < p.get("total", 0):
            v["verdict"] = f"[PARTIAL DATA: {_prog(p)}] " + v["verdict"]
            # pending SEEDS of systems already counted can move their CIs: a PASS/FAIL is not final
            v["provisional"] = v["label"] in (R.PASS, R.FAIL)
    members = {"C1-budget": ["C1-budget-s", "C1-budget-l"],
               "C3-mismatch (crossover)": list(MISMATCH)}
    for k, ms in members.items():
        res[k]["provisional"] = any(res[m].get("provisional") or res[m]["label"] in (R.PENDING, R.INCONCLUSIVE)
                                    for m in ms) and res[k]["label"] in (R.PASS, R.FAIL)
    for v in res.values():
        v["shown"] = v["label"] + ("*" if v.get("provisional") else "")
    return res


# nodes whose rule (or its 'best learned' / 'winner' selection) depends on which learned variant is used
REREAD = ("C1-where", "TEST-C1", "C1-budget-s", "C1-budget-l", "C1-budget", "C1-ctx", "C1-basewidth-64",
          "C1-basewidth-256", "C1-modelfree", "C2-floor-025", "C2-floor-05", "C2-floor-1", "C2-trunc-test",
          "C2-trunc-dev", "C3-faults", "C3-mismatch10", "C3-mismatch20", "C3-mismatch30", "C3-mismatch50",
          "C3-mismatch (crossover)", "C3-noise-ct", "C3-partial", "C3-partial-modelfree", "C3-realistic", "TEST-C3",
          "C3-regimes") + tuple(n + " (b)" for n in DD_NODES)


def _scale_thresholds(text: str, n_old: int = 6) -> str:
    """'need 4/6' / '>= 4/6' in a k/5 headline verdict -> 'need 4/5 (pre-reg. 4/6)' (ceil(k * n_new / n_old))."""
    import re as _re
    n_new = len(dev())

    def sub(m):
        k = int(m.group(2))
        return f"{m.group(1)}{math.ceil(k * n_new / n_old)}/{n_new} (pre-reg. {k}/{n_old})"
    return _re.sub(r"(need |>= )(\d)/" + str(n_old) + r"\b", sub, text)


def analyse_all_both(store: Store) -> dict:
    """Three readings, headline last:
         original  as pre-registered: best-of-variants selection, all 6 dev systems (k/6)
         single    single method (HyperKKL := lora r=4), all 6 dev systems
         headline  single method, EXCLUDED_SYSTEMS removed (k/5; post-hoc exclusion)
    Each headline entry carries original_label/_verdict, single_label/_verdict and excluded_reading."""
    _S.EXCLUDE["on"] = False
    SINGLE["on"] = False
    orig = analyse_all(store)
    SINGLE["on"] = True
    single = analyse_all(store)
    _S.EXCLUDE["on"] = True
    res = analyse_all(store)            # SINGLE and EXCLUDE stay on: figures / tables / report use the headline
    for k, r in res.items():
        o, si = orig.get(k) or {}, single.get(k) or {}
        r["reread"] = k in REREAD
        r["original_label"] = o.get("shown", o.get("label"))
        r["original_verdict"] = o.get("verdict", "")
        r["single_label"] = si.get("shown", si.get("label"))
        r["single_verdict"] = si.get("verdict", "")
        dev_rule = r.get("split") == "dev" or k in ("C1-budget", "C3-mismatch (crossover)")
        r["excluded_reading"] = bool(dev_rule)
        if dev_rule:
            r["exclusion_note"] = _S.EXCLUSION_NOTE
            if "test" not in (r.get("verdict") or "").lower():      # mixed dev+test verdicts keep their text
                r["verdict"] = _scale_thresholds(r.get("verdict") or "")
        if r["reread"]:
            r["reread_note"] = REREAD_NOTE
    return res

