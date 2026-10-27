import json, sys, statistics as st, time, collections
rows = json.load(open(sys.argv[1]))
G5 = ['constant','ramp','sinusoid','am','prbs']
def med(r, m='dyn_gru', tier='ood'):
    x = r['res'].get(tier, {}).get(m)
    if not x: return None
    v = [x['fam'].get(f) for f in (G5 if tier=='ood' else x['fam'])]
    v = [a for a in v if a is not None]
    return st.median(v) if v else None
tags = collections.defaultdict(list)
for r in rows: tags[r['tag']].append(r)
# reference config: legacy default
for t in sorted(tags, key=lambda t: min(r['mtime'] for r in tags[t])):
    rs = tags[t]
    sy = sorted(set(r['system'] for r in rs)); se = sorted(set(r['seed'] for r in rs))
    ms = sorted(set(m for r in rs for m in (r['res'].get('ood') or {})))
    # varying cfg keys within tag and constant ones
    keys = set(k for r in rs for k in r['cfg'])
    const = {k: rs[0]['cfg'].get(k) for k in keys if all(json.dumps(r['cfg'].get(k)) == json.dumps(rs[0]['cfg'].get(k)) for r in rs)}
    var = sorted(k for k in keys if k not in const)
    date = time.strftime('%m-%d %H:%M', time.gmtime(min(r['mtime'] for r in rs)))
    c = const
    sig = 'p1 %s/%s/%sep/%sn p2 %s/%s/%sep/%sn frz=%s res=%s curv=%s rank=%s nm=%s np=%s im=%s' % tuple(c.get(k) for k in ['p1.opt','p1.lr','p1.ep','p1.nep','p2.opt','p2.lr','p2.ep','p2.nep','p2.frz','p2.res','p2.curv','p2.rank','nm','np','imode'])
    extra = {k: c[k] for k in ['p1.gm','p2.gm','p2.fam','p2.cur','p2.W','nk','pk','outl','alpha_ov','nz_ov','spec_ov','pbias','epbias','pjit','latinit','rad','p1.syl','p1.ncoll','p2.dreg','p2.smooth','p2.sfmb','p1.pde','evrep'] if k in c and c[k] not in (None, 0, 0.0, '', [], 'sum', False, 'gaussian', 'encoder', 16384, 0.001, 8)}
    print('%-16s %s n=%3d sys=%s seeds=%s methods=%s' % (t, date, len(rs), ','.join(sy), se, ','.join(ms)))
    print('      ', sig, extra, ('VAR:' + ','.join(var)) if var else '')
