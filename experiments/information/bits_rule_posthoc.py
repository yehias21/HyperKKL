import json,sys,math
sys.path.insert(0,'information')
import score as S
R=json.load(open('p2_rescore/results.json'))['plants']
for tier in ('id','sequence_ood'):
  print(f"\n== tier {tier} (budget: eval_id window 3/alpha; eps = best grey-box unmeasured error)")
  print(f"{'plant':16s} {'A_win':>7s} {'full':>7s} {'eps_gb':>7s} {'bits_need-avail':>15s} {'predict':>12s} {'HyperKKL/gb':>12s} {'src':>5s} outcome")
  for s in S.L.ALL:
    if s=='duffing': continue
    bq=S.budget(s); gbm,eps=S.best_greybox(s,tier)
    if gbm is None or eps!=eps: continue
    ms=S.A[s]['partial']['methods']
    fold=None;src='old'
    p=R.get(s,{}).get(f"{tier}/unm")
    if p:
      eps=p['greybox_mean']
      cands=[(p[a]['fold'],a,p[a]['fold_ci']) for a in ('A2','A4') if a in p and p[a]]
      if cands: f,src,ci=min(cands); fold=f
    if fold is None and 'lora' in ms and tier in ms['lora']:
      fold=ms['lora'][tier]['unmeasured']/eps; ci=None
    need_minus_avail=math.log2(bq['A']/eps)  # >0: window short of bits needed to reach eps
    pred='CANNOT win' if bq['A']>eps else 'can win'
    if ci: out='win' if ci[1]<1 else ('lose' if ci[0]>1 else 'tie')
    else: out='win' if fold<1 else 'lose'
    ok = (pred=='CANNOT win' and out!='win') or (pred=='can win')
    print(f"{s:16s} {bq['A']:7.4f} {bq['full_unm']:7.4f} {eps:7.4f} {need_minus_avail:+15.2f} {pred:>12s} {fold:12.2f} {src:>5s} {out}{'' if ok else '  <-- VIOLATES'}")
