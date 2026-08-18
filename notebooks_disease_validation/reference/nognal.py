import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, Fil2Dict, STR2Region, ScoreCircuit_SI_Joint
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
base=[int(g) for g in Fil2Dict('dat/Genetics/GeneWeights/PD_HighConf_DA.gw')]
drop=[int(S2E[s]) for s in ['DNAJC6','ATP13A2','GBA','SYNJ1','PLA2G6']]
variants={
 'curated 19 (unchanged)': base,
 'fake 17 (+PDE8B,ADCY5,GNAL, -5)': [g for g in base if g not in drop]+[int(S2E[s]) for s in ['PDE8B','ADCY5','GNAL']],
 'fake 16 (+PDE8B,ADCY5, -5, no GNAL)': [g for g in base if g not in drop]+[int(S2E[s]) for s in ['PDE8B','ADCY5']],
 'curated 19 + PDE8B,ADCY5 (drop nothing)': base+[int(S2E[s]) for s in ['PDE8B','ADCY5']],
}
sw=pd.read_csv('dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',header=None)
idx={g:i for i,g in enumerate(Z2.index)}; SIB=np.array([idx[g] for g in sw[0].astype(int) if g in idx])
M=Z2.values
print("%-42s %4s %6s %8s %8s %8s | %8s" % ("gene set","n","CPrank","q<0.10","SNc r","VTA r","CCS@13"))
print("-"*104)
for lbl,ent in variants.items():
    obs=Z2.loc[ent].mean(axis=0)
    rng=np.random.default_rng(42); N=10000
    nl=np.array([np.nanmean(M[rng.choice(SIB,size=len(ent),replace=False)],axis=0) for _ in range(N)])
    p=((nl>=obs.values).sum(axis=0)+1)/(N+1)
    o=np.argsort(p); q=np.empty_like(p); r_=1.0; m=len(p)
    for k in range(m-1,-1,-1): r_=min(r_,p[o[k]]*m/(k+1)); q[o[k]]=r_
    q=pd.Series(q,index=Z2.columns); rk=obs.rank(ascending=False)
    order=obs.sort_values(ascending=False).index.values
    print("%-42s %4d %6d %8d %8d %8d | %8.3f" % (lbl,len(ent),rk['Caudoputamen'],(q<0.10).sum(),
          rk['Substantia_nigra_compact_part'],rk['Ventral_tegmental_area'],ScoreCircuit_SI_Joint(order[:13],Info)))
print("\ntop-10 structures, 'fake 16 (no GNAL)':")
ent=variants['fake 16 (+PDE8B,ADCY5, -5, no GNAL)']
o2=Z2.loc[ent].mean(axis=0).sort_values(ascending=False)
for i,(k,v) in enumerate(o2.head(10).items(),1): print("   %2d. %6.3f  %-42s %s"%(i,v,k[:42],anno.get(k,'?')))
