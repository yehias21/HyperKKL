"""Paired arm-vs-control on archived runs. Metric per run & tier: median over families of nrmse_steady (method dyn_gru).
Tiers: ood (graded OOD: amplitude/rate beyond training), waveform_ood (chirp, filtered_noise), id."""
import json, sys, math, statistics as st, collections
from scipy import stats
R = json.load(open('runs_HyperKKL.json'))
IDX = {(r['tag'], r['system'], r['seed']): r for r in R}
IGN = {'evaluation.eval_rep'}  # reported separately
def metric(r, tier, m='dyn_gru'):
    x = (r['res'].get(tier) or {}).get(m)
    if not x: return None
    v = [a for a in x['fam'].values() if a is not None and math.isfinite(a)]
    return st.median(v) if v else None
def tci(d):
    n = len(d); m = st.mean(d)
    if n < 2: return m, None, n
    h = stats.t.ppf(.975, n - 1) * st.stdev(d) / math.sqrt(n); return m, h, n
def cdiff(a, b):
    ks = set(a['cfg']) | set(b['cfg'])
    return {k: (a['cfg'].get(k), b['cfg'].get(k)) for k in sorted(ks) if a['cfg'].get(k) != b['cfg'].get(k)}
PAIRS = [l.split('|') for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
out = []
for name, arm, ctl, note in [[x.strip() for x in p] for p in PAIRS]:
    keys = sorted({(s, sd) for (t, s, sd) in IDX if t == arm} & {(s, sd) for (t, s, sd) in IDX if t == ctl})
    diffs = collections.Counter()
    for s, sd in keys:
        for k, v in cdiff(IDX[(arm, s, sd)], IDX[(ctl, s, sd)]).items():
            if not k.startswith('system.design') and k not in ('phase2.window_size',): diffs[(k, repr(v))] += 1
    print('\n### %s  [%s vs %s]  %s  n_pairs=%d' % (name, arm, ctl, note, len(keys)))
    print('config differences (arm, ctl) [count]:', '; '.join('%s=%s[%d]' % (k, v, c) for (k, v), c in diffs.items()))
    rec = {'name': name, 'arm': arm, 'ctl': ctl, 'note': note, 'diffs': {k + '=' + v: c for (k, v), c in diffs.items()}, 'tiers': {}}
    for tier in ('ood', 'waveform_ood', 'id'):
        rows = []; per = collections.defaultdict(list)
        for s, sd in keys:
            a, c = metric(IDX[(arm, s, sd)], tier), metric(IDX[(ctl, s, sd)], tier)
            if a is None or c is None: continue
            rows.append((s, sd, a, c)); per[s].append(100 * (a / c - 1))
        if not rows: continue
        rel = [100 * (a / c - 1) for _, _, a, c in rows]
        lr = [100 * (math.exp(math.log(a / c)) - 1) for _, _, a, c in rows]
        dl = [math.log(a / c) for _, _, a, c in rows]
        m, h, n = tci(dl)
        gm = 100 * (math.exp(m) - 1); lo = hi = None
        if h is not None: lo, hi = 100 * (math.exp(m - h) - 1), 100 * (math.exp(m + h) - 1)
        ma, ha, _ = tci([a - c for _, _, a, c in rows])
        persys = {s: tci(v) for s, v in per.items()}
        print(' tier %-12s n=%d  geo-mean rel %+.1f%% CI[%s, %s]  mean abs diff %+.4f ±%s  median rel %+.1f%%' % (
            tier, n, gm, 'nan' if lo is None else '%+.1f' % lo, 'nan' if hi is None else '%+.1f' % hi, ma,
            'nan' if ha is None else '%.4f' % ha, st.median(rel)))
        for s, (pm, ph, pn) in persys.items():
            print('     %-14s n=%d mean rel %+.1f%% %s  raw: %s' % (s, pn, pm, '' if ph is None else '±%.1f' % ph,
                  ', '.join('s%d %.4f->%.4f' % (sd, c, a) for ss, sd, a, c in rows if ss == s)))
        rec['tiers'][tier] = {'n': n, 'geo_rel': gm, 'ci': [lo, hi], 'abs': [ma, ha], 'median_rel': st.median(rel),
                              'persys': {s: [pm, ph, pn] for s, (pm, ph, pn) in persys.items()},
                              'raw': rows}
    out.append(rec)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
