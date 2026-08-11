import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import ScoreCircuit_SI_Joint, STR2Region
anno=STR2Region()
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
rank=pd.Series(np.arange(1,len(d)+1),index=d.index)
rows=[];lim=None
for l in open("results/CircuitSearch/PD_HighConf_DA/best_circuits/size_13_best_circuits.txt"):
    if l.startswith('# Bias limit:'): lim=float(l.split(':')[1])
    elif not l.startswith('#') and l.strip():
        sc,mb,st=l.rstrip().split('\t'); rows.append((lim,float(sc),float(mb),st.split(',')))
p=pd.DataFrame(rows,columns=['bias_limit','CCS','mean_bias','structures']).drop_duplicates(subset=['CCS','mean_bias'])
p=p.sort_values('mean_bias',ascending=False).reset_index(drop=True)
base=list(d.index[:13]); bC=ScoreCircuit_SI_Joint(base,Info); bB=d.loc[base,'EFFECT'].mean()
print("BASELINE (top-13 by bias):        CCS %.3f  mean_bias %.3f" % (bC,bB))
print("\nASD reference (published, size 46): baseline CCS 0.505 bias 0.382 -> selected CCS 0.939 bias 0.306")
print("   i.e. ASD accepted  bias -20%%  to gain  CCS +86%%\n")
p['dCCS_pct']=100*(p.CCS-bC)/bC; p['dBias_pct']=100*(p.mean_bias-bB)/bB
p['ratio']=p.dCCS_pct/(-p.dBias_pct).replace(0,np.nan)
print("%9s %9s %8s %8s %8s  %s" % ("mean_bias","CCS","dBias%","dCCS%","gain/loss","dopaminergic core retained?"))
for _,r in p.iterrows():
    coreset={'Ventral_tegmental_area','Substantia_nigra_compact_part','Dorsal_nucleus_raphe'}
    keep=sorted(coreset & set(r.structures))
    print("%9.3f %9.3f %8.1f %8.1f %8s  %s" % (r.mean_bias,r.CCS,r.dBias_pct,r.dCCS_pct,
          "%.1f"%r.ratio if r.ratio==r.ratio else "-", ("YES "+",".join(x[:12] for x in keep)) if len(keep)==3 else ("partial: "+",".join(x[:12] for x in keep) if keep else "NO")))
sel=p[(p.dBias_pct>-25)].sort_values('dCCS_pct',ascending=False).iloc[0]
print("\n>>> ASD-analogous selection (largest CCS gain within a 25%% bias sacrifice):")
print("    mean_bias %.3f (%.1f%%)   CCS %.3f (+%.1f%%)" % (sel.mean_bias,sel.dBias_pct,sel.CCS,sel.dCCS_pct))
for s in sorted(sel.structures,key=lambda x:rank[x]):
    print("      rank %3d  bias %6.3f  %-44s %s" % (rank[s],d.loc[s,'EFFECT'],s[:44],anno.get(s,'?')))
print("    dropped from baseline top-13:", sorted(set(base)-set(sel.structures),key=lambda x:rank[x]))
