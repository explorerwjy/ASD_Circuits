import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
f='results/CircuitSearch/PD_HighConf_DA/pareto_fronts/PD_HighConf_DA_size_40_pareto_front.csv'
p=pd.read_csv(f)
base=p[p.circuit_type=='baseline'].iloc[0]
opt=p[p.circuit_type=='optimized'].copy().sort_values('mean_bias',ascending=False).reset_index(drop=True)
opt['dBias_pct']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
opt['dCCS_pct']=100*(opt.circuit_score-base.circuit_score)/base.circuit_score
print("PD_HighConf_DA size 40 — PARETO FRONT (%d optimized points)" % len(opt))
print("BASELINE (top-40 by bias): mean_bias %.4f  CCS %.4f\n" % (base.mean_bias,base.circuit_score))
print("%4s %10s %10s %9s %9s %8s  %s" % ("idx","bias_lim","mean_bias","CCS","dBias%","dCCS%","note"))
sel_i=int((opt.dBias_pct-(-20.0)).abs().argmin())
for i,r in opt.iterrows():
    note=""
    if i==0: note="LEFT-MOST (highest bias, lowest CCS)"
    if i==len(opt)-1: note="RIGHT-MOST (lowest bias, highest CCS)"
    if i==sel_i: note=(note+"  <<< SELECTED (ASD-matched -20% bias)").strip()
    print("%4d %10.3f %10.4f %9.4f %9.1f %8.1f  %s" % (i,r.bias_limit,r.mean_bias,r.circuit_score,r.dBias_pct,r.dCCS_pct,note))
print("\nSELECTED point index %d of %d (0=left/highest-bias, %d=right/highest-CCS)" % (sel_i,len(opt)-1,len(opt)-1))
