import sys, os, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import STR2Region, ScoreCircuit_SI_Joint
from disease_validation import load_ground_truth
anno=STR2Region(); GT=load_ground_truth('config/disease_validation_ground_truth.yaml')
core=set(GT['structures']['parkinson']['core']); braak=set(GT['structures']['parkinson']['braak_early'])
mark=lambda s: "  <== PD CORE" if s in core else ("  <== Braak-early" if s in braak else "")
for name in ["PD_HighConf_DA","PD_Sens_DA"]:
    d=pd.read_csv(f"results/STR_ISH/{name}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
    ds=pd.read_csv(f"results/STR_ISH/{name}_SibMut_bias_addP_sibling.csv",index_col=0)
    n=len(pd.read_csv(f"dat/Genetics/GeneWeights/{name}.gw",header=None))
    print(f"\n=== {name} (n={n}) top 20 of 213 | q_rand={int((d['q-value']<0.10).sum())}, q_sib={int((ds['q-value']<0.10).sum())} at q<0.10 ===")
    for i,(k,r) in enumerate(d.head(20).iterrows(),1):
        print("%2d. %7.3f  q_rand=%.3f q_sib=%.3f  %-46s %-13s%s" % (i,r['EFFECT'],r['q-value'],ds.loc[k,'q-value'],k[:46],anno.get(k,'?'),mark(k)))
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
SIZES=[100,60,46,30,20,10]
prof=lambda ss:[ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
print("\n\n=== CCS, size-matched nulls (empirical p) ===")
print("%-16s %-26s %s" % ("set","null","  ".join("N=%-3d"%N for N in SIZES)))
for name in ["PD_HighConf_DA","PD_Sens_DA","PD_Sens_Atypical","PD_Primary"]:
    o=np.array(prof(pd.read_csv(f"results/STR_ISH/{name}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False).index.values))
    for lbl,pq in [("sibling-mutability (=ASD)",f"results/STR_ISH/null_bias/{name}_SibMut_null_bias_sibling.parquet"),
                   ("random uniform (matched)", f"results/STR_ISH/null_bias/{name}_null_bias_random.parquet")]:
        if not os.path.exists(pq): print("%-16s %-26s n/a"%(name,lbl)); continue
        nb=pd.read_parquet(pq)
        null=np.array(Parallel(n_jobs=10)(delayed(prof)(nb[c].sort_values(ascending=False).index.values) for c in nb.columns))
        p=[(np.sum(null[:,j]>=o[j])+1)/(null.shape[0]+1) for j in range(len(SIZES))]
        print("%-16s %-26s %s" % (name if lbl.startswith('sib') else "",lbl,"  ".join("%.3f"%x for x in p)))
