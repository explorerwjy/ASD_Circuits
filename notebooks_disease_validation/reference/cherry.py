import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import LoadGeneINFO, Fil2Dict, STR2Region, ScoreCircuit_SI_Joint
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
base=[int(g) for g in Fil2Dict('dat/Genetics/GeneWeights/PD_HighConf_DA.gw')]
drop=[int(S2E[s]) for s in ['DNAJC6','ATP13A2','GBA','SYNJ1','PLA2G6']]
add =[int(S2E[s]) for s in ['PDE8B','ADCY5','GNAL']]
fake=[g for g in base if g not in drop]+add
print("FAKE SET (%d genes) = 19 - 5 CP-negative + 3 striatal markers" % len(fake))
print("  ",", ".join(sorted(E2S.get(g,str(g)) for g in fake)))
obs=Z2.loc[fake].mean(axis=0)
M=Z2.values; idx={g:i for i,g in enumerate(Z2.index)}
sw=pd.read_csv('dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',header=None)
pools={'random':np.arange(len(Z2.index)),'sibling':np.array([idx[g] for g in sw[0].astype(int) if g in idx])}
res={}
for lbl,P in pools.items():
    rng=np.random.default_rng(42); N=10000
    nl=np.array([np.nanmean(M[rng.choice(P,size=len(fake),replace=False)],axis=0) for _ in range(N)])
    p=((nl>=obs.values).sum(axis=0)+1)/(N+1)
    o=np.argsort(p); q=np.empty_like(p); run=1.0; m=len(p)
    for k in range(m-1,-1,-1): run=min(run,p[o[k]]*m/(k+1)); q[o[k]]=run
    res[lbl]=(p,q)
d=pd.DataFrame({'structure':Z2.columns,'region':[anno.get(c,'?') for c in Z2.columns],'EFFECT':obs.values,
  'q_random':res['random'][1],'q_sibling':res['sibling'][1]})
d['rank']=d.EFFECT.rank(ascending=False).astype(int); d=d.sort_values('rank')
print("\n  q<0.10: %d random / %d sibling" % ((d.q_random<0.10).sum(),(d.q_sibling<0.10).sum()))
print("\n  TOP 15:")
for _,r in d.head(15).iterrows():
    f="  <== STRIATUM" if r.region in ('Striatum','Pallidum') else ""
    print("   %2d. %6.3f q_sib=%.3f  %-40s %-14s%s" % (r['rank'],r.EFFECT,r.q_sibling,r.structure[:40],r.region,f))
print("\n  key structures:")
for s in ['Caudoputamen','Nucleus_accumbens','Substantia_nigra_compact_part','Ventral_tegmental_area','Dorsal_nucleus_raphe']:
    x=d[d.structure==s].iloc[0]; print("   rank %3d  EFFECT %6.3f  q_sib=%.3f  %s" % (x['rank'],x.EFFECT,x.q_sibling,s))
# CCS
SIZES=[10,13,20,30,46,60]
order=obs.sort_values(ascending=False).index.values
prof=lambda ss:[ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
o_ccs=np.array(prof(order))
rng=np.random.default_rng(7); P=pools['sibling']; cols=np.array(Z2.columns)
null=np.array(Parallel(n_jobs=12)(delayed(lambda dr: prof(cols[np.argsort(-np.nanmean(M[dr],axis=0))]))(
    rng.choice(P,size=len(fake),replace=False)) for _ in range(3000)))
print("\n  CCS (sibling null, 3000 sims):")
for j,N in enumerate(SIZES):
    p=(np.sum(null[:,j]>=o_ccs[j])+1)/3001
    print("   N=%3d  CCS %6.3f  p=%.4f%s" % (N,o_ccs[j],p," <== sig" if p<0.05 else ""))
