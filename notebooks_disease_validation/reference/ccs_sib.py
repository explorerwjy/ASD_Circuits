import sys, os, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import ScoreCircuit_SI_Joint
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
ASD=np.load('dat/allen-mouse-conn/RankScores/RankScore.Ipsi.Cont.npy')
topNs=np.arange(200,5,-1); SIZES=[100,60,46,30,20,10]
ai={N:int(np.where(topNs==N)[0][0]) for N in SIZES}
def prof(ss): return [ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
print("%-22s %4s | %-28s | %s" % ("gene set","n","null used","  ".join("N=%-3d"%N for N in SIZES)))
print("-"*104)
for s in ["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G","StriatalDegeneration"]:
    obs=pd.read_csv(f"results/STR_ISH/{s}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False).index.values
    o=np.array(prof(obs))
    n=len(pd.read_csv(f"dat/Genetics/GeneWeights/{s}.gw",header=None))
    rows=[]
    for lbl,pq in [("sibling-mutability (=ASD)",f"results/STR_ISH/null_bias/{s}_SibMut_null_bias_sibling.parquet"),
                   ("random uniform (matched)",  f"results/STR_ISH/null_bias/{s}_null_bias_random.parquet")]:
        if not os.path.exists(pq): rows.append((lbl,None)); continue
        nb=pd.read_parquet(pq)
        null=np.array(Parallel(n_jobs=10)(delayed(prof)(nb[c].sort_values(ascending=False).index.values) for c in nb.columns))
        rows.append((lbl,[(np.sum(null[:,j]>=o[j])+1)/(null.shape[0]+1) for j in range(len(SIZES))]))
    rows.append(("ASD 61-gene band (WRONG)",[(np.sum(ASD[:,ai[N]]>=o[j])+1)/(ASD.shape[0]+1) for j,N in enumerate(SIZES)]))
    for i,(lbl,p) in enumerate(rows):
        head="%-22s %4d" % (s,n) if i==0 else "%-22s %4s" % ("","")
        print("%s | %-28s | %s" % (head,lbl,"  ".join("%.3f"%x for x in p) if p else "n/a"))
