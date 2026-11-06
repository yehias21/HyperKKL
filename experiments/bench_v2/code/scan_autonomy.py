"""Inclusion check 'non-trivial unforced dynamics': simulate each plant at u = 0, from the IC box, nominal and
perturbed (+0.3), T x 5 on the plant's grid, 64 ICs. Reports, for the last 20 % of the run:
  late_std   median over ICs of the time std of each state / x_std (reference std of the forced data)
  ratio      late_std / early_std (early = 20-40 % of the run)
  n_eq       distinct end states are not used; this is a per-trajectory test
Rule evaluated later in DESIGN.md."""
import json, sys
import env
import numpy as np, torch, yaml
from hyperkkl import data as Dm, systems as SY
from hyperkkl import systems_extra, systems_bounded, systems_chaos, systems_casestudy  # noqa
from hyperkkl.config import load
torch.set_num_threads(4)
names = sys.argv[1:] or ["vdp","duffing","fhn","lorenz","rossler","lorenz96","lorenz96_n10","network","battery","bioreactor",
  "quad_tank","cartpole_b","two_link_arm_b","highway_traffic"]
out = {}
for n in names:
    cfg = load(n); sc = dict(cfg["system"]); des = Dm.make_design(sc)
    box = np.asarray(sc["ic_box"], float); nx = len(box)
    rng = np.random.RandomState(0); x0 = box[:,0] + (box[:,1]-box[:,0])*rng.rand(64, nx)
    for pert in (0.0, 0.3):
        pl = SY.build_plant(sc, perturb=pert)
        N = min(5*des.N, 40000)
        x = torch.as_tensor(x0, dtype=torch.float64); u = torch.zeros(64,1,dtype=torch.float64); h=des.dt
        xs=[]
        ev = max(1, N//2000)
        for k in range(N):
            a1=pl.f(x,u);a2=pl.f(x+h/2*a1,u);a3=pl.f(x+h/2*a2,u);a4=pl.f(x+h*a3,u); x=x+h/6*(a1+2*a2+2*a3+a4)
            if k%ev==0: xs.append(x)
        X=torch.stack(xs,1).numpy(); fin=np.isfinite(X).all((1,2)); X=X[fin]
        M=X.shape[1]; early=X[:,int(.2*M):int(.4*M)]; late=X[:,int(.8*M):]
        sd=np.maximum(np.abs(X[:,int(.2*M):]).std((0,1)),1e-9)
        ls=np.median(late.std(1)/sd,0); es=np.median(early.std(1)/sd,0)
        out[f"{n}|{pert}"]={"finite":int(fin.sum()),"late_std_max":float(ls.max()),"ratio":float(ls.max()/max(es.max(),1e-12)),
           "frac_active":float((late.std(1).max(1)/sd.max()>0.05).mean()),"T":float(N*h),"sensors":list(sc["sensors"]),"nx":nx}
        print(n,pert,{k:(round(v,4) if isinstance(v,float) else v) for k,v in out[f'{n}|{pert}'].items() if k!='sensors'},flush=True)
import os
os.makedirs("../design/autonomy_scan",exist_ok=True)
json.dump(out,open(f"../design/autonomy_scan/{'_'.join(names)}.json","w"),indent=1)
