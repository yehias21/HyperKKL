"""PDE-on runs whose config equals the pdefalse_<sys> run except use_pde, up to keys that are absent (None) on one side
and set to the code default on the other. Gives the replicate spread of PDE-on at the same config (seed 0)."""
import json, math, statistics as st, datetime
R = json.load(open('runs_HyperKKL.json'))
DEF = {'system.noise_kind': 'gaussian', 'system.outlier_rate': 0.0, 'system.param_bias': 0.0, 'system.eval_param_bias': 0.0,
       'system.baseline_knows_change': False, 'evaluation.eval_rep': 0, 'system.param_jitter': 0.0, 'system.spectrum_override': '',
       'system.alpha_override': 0.0, 'evaluation.obs_init_radius': 0.0}
INERT = {'system.noise_rho', 'system.outlier_scale', 'system.quant_levels', 'system.dropout_rate', 'system.drift_amp', 'system.param_jitter_keys'}
def med(r, t):
    x = (r['res'].get(t) or {}).get('dyn_gru')
    if not x: return None
    v = [a for a in x['fam'].values() if a is not None and math.isfinite(a)]; return st.median(v) if v else None
out = {}
for sysn in ('bioreactor', 'fhn'):
    off = [r for r in R if r['tag'] == 'pdefalse_' + sysn][0]
    print('\n==', sysn, 'PDE OFF', off['tag'], 'ood %.4f wave %.4f id %.4f' % (med(off, 'ood'), med(off, 'waveform_ood'), med(off, 'id')),
          'phase1 extra use_pde=', off['extra'].get('phase1', {}).get('use_pde'))
    ons = []
    for r in R:
        if r['system'] != sysn or r['seed'] != 0 or r is off or 'dyn_gru' not in (r['res'].get('ood') or {}): continue
        ok = True
        for k in set(r['cfg']) | set(off['cfg']):
            a, b = off['cfg'].get(k), r['cfg'].get(k)
            if k == 'phase1.use_pde' or a == b or k in INERT or k.startswith('system.design'): continue
            if a is None and b == DEF.get(k, '__x') or b is None and a == DEF.get(k, '__x'): continue
            ok = False; break
        if ok and r['cfg'].get('phase1.use_pde') is True:
            ons.append(r)
    for r in sorted(ons, key=lambda r: r['mtime']):
        print('  ON %-14s %s ood %.4f wave %.4f id %.4f' % (r['tag'], datetime.datetime.fromtimestamp(r['mtime']).strftime('%m-%d %H:%M'), med(r, 'ood'), med(r, 'waveform_ood'), med(r, 'id')))
    rec = {}
    for t in ('ood', 'waveform_ood', 'id'):
        v = [med(r, t) for r in ons]; o = med(off, t)
        rank = sum(x < o for x in v)
        print('  %s: ON n=%d mean %.4f sd %.4f min %.4f max %.4f ; OFF %.4f -> OFF/meanON-1 = %+.1f%%, OFF beats %d/%d ON replicates' % (
            t, len(v), st.mean(v), st.stdev(v), min(v), max(v), o, 100 * (o / st.mean(v) - 1), len(v) - rank, len(v)))
        rec[t] = dict(n=len(v), mean=st.mean(v), sd=st.stdev(v), mn=min(v), mx=max(v), off=o, tags=[r['tag'] for r in ons])
    out[sysn] = rec
json.dump(out, open('pde_replicates.json', 'w'), indent=1)
