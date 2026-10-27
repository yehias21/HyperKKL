"""Paired arm-vs-control comparisons over archived runs.
usage: python -I compare.py runs.json pairs.txt out.json
pairs.txt lines: label | arm_tag:method | ctrl_tag:method | note"""
import json, sys, math, statistics as st, collections
rows = json.load(open(sys.argv[1]))
G5 = ['constant','ramp','sinusoid','am','prbs']
TQ = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,15:2.131,20:2.086,25:2.06,30:2.042}
def tq(df):
    ks = sorted(TQ); return TQ[max(k for k in ks if k <= df)] if df >= 1 else float('nan')
idx = collections.defaultdict(list)
for r in rows: idx[(r['tag'], r['system'], r['seed'])].append(r)
def get(tag, s, sd):
    L = idx.get((tag, s, sd)); return L[0] if L else None
def med(r, m, tier):
    x = (r['res'].get(tier) or {}).get(m)
    if not x: return None
    fams = G5 if tier == 'ood' else sorted(x['fam'])
    v = [x['fam'].get(f) for f in fams]; v = [a for a in v if a is not None and a == a]
    return st.median(v) if len(v) == len(fams) else None
def cfgdiff(a, b):
    return sorted(k for k in set(a['cfg']) | set(b['cfg']) if json.dumps(a['cfg'].get(k)) != json.dumps(b['cfg'].get(k)) and k not in ('evrep',))
out = []
for line in open(sys.argv[2]):
    line = line.strip()
    if not line or line.startswith('#'): continue
    lab, arm, ctl, note = [x.strip() for x in line.split('|')]
    at, am = arm.split(':'); ct, cm = ctl.split(':')
    systems = sorted(set(s for (t, s, sd) in idx if t == at))
    persys = {}; alld = []; diffs = set(); wf = []
    for s in systems:
        ds = []; dw = []
        for (t, s2, sd) in sorted(idx):
            if t != at or s2 != s: continue
            a = get(at, s, sd); c = get(ct, s, sd)
            if not a or not c: continue
            ma, mc = med(a, am, 'ood'), med(c, cm, 'ood')
            if ma is None or mc is None: continue
            ev_ok = a['cfg'].get('evrep') == c['cfg'].get('evrep')
            diffs.update(cfgdiff(a, c))
            ds.append((sd, 100 * (ma - mc) / mc, ma, mc, ev_ok))
            wa, wc = med(a, am, 'waveform_ood'), med(c, cm, 'waveform_ood')
            if wa is not None and wc is not None: dw.append(100 * (wa - wc) / wc)
        if ds:
            d = [x[1] for x in ds]
            persys[s] = {'n': len(d), 'mean': st.mean(d), 'ci': (tq(len(d)-1) * st.stdev(d) / math.sqrt(len(d))) if len(d) > 1 else None,
                         'arm': st.mean(x[2] for x in ds), 'ctl': st.mean(x[3] for x in ds), 'seeds': [x[0] for x in ds],
                         'evrep_matched': all(x[4] for x in ds), 'wave': st.mean(dw) if dw else None}
            alld += d; wf += dw
    sysmeans = [v['mean'] for v in persys.values()]
    out.append({'label': lab, 'arm': arm, 'ctl': ctl, 'note': note, 'persys': persys,
                'median_sys': st.median(sysmeans) if sysmeans else None,
                'n_sys': len(persys), 'n_pairs': len(alld), 'cfgdiff': sorted(diffs),
                'n_better': sum(1 for v in sysmeans if v < 0),
                'n_sig_better': sum(1 for v in persys.values() if v['ci'] is not None and v['mean'] + v['ci'] < 0),
                'n_sig_worse': sum(1 for v in persys.values() if v['ci'] is not None and v['mean'] - v['ci'] > 0),
                'wave_median': st.median(wf) if wf else None})
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for o in out:
    print('== %s  [%s vs %s]  median_sys=%s  n_sys=%d pairs=%d better=%d sig+=%d sig-=%d wave_med=%s  diff=%s' % (
        o['label'], o['arm'], o['ctl'], None if o['median_sys'] is None else '%+.1f%%' % o['median_sys'], o['n_sys'], o['n_pairs'],
        o['n_better'], o['n_sig_better'], o['n_sig_worse'], None if o['wave_median'] is None else '%+.1f%%' % o['wave_median'], ','.join(o['cfgdiff'])))
    for s, v in o['persys'].items():
        print('     %-13s n=%d %+7.1f%% %s  arm=%.4g ctl=%.4g wave=%s ev=%s' % (s, v['n'], v['mean'], ('+-%.1f' % v['ci']) if v['ci'] is not None else '     ', v['arm'], v['ctl'],
              None if v['wave'] is None else '%+.0f%%' % v['wave'], v['evrep_matched']))
