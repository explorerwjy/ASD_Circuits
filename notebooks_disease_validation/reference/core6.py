import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
SYMS="LRRK2 SNCA GBA PRKN PINK1 PARK7".split()
ent=[int(S2E[s]) for s in SYMS]
print("gene set (%d): %s" % (len(ent),", ".join(SYMS)))
obs=Z2.loc[ent].mean(axis=0)
# sibling pool + all-gene pool
sw=pd.read_csv('dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',header=None)
sib=[g for g in sw[0].astype(int) if g in Z2.index]
allg=np.array(Z2.index)
rng=np.random.default_rng(42); N=10000
M=Z2.values
idx={g:i for i,g in enumerate(Z2.index)}
def nulls(pool,n):
    P=np.array([idx[g] for g in pool])
    out=np.empty((n,M.shape[1]))
    for i in range(n):
        out[i]=np.nanmean(M[rng.choice(P,size=len(ent),replace=False)],axis=0)
    return out
res={}
for lbl,pool in [("random",allg),("sibling",sib)]:
    nl=nulls(pool,N)
    p=((nl>=obs.values).sum(axis=0)+1)/(N+1)
    o=np.argsort(p); q=np.empty_like(p); m=len(p)
    run=1.0
    for k in range(m-1,-1,-1):
        run=min(run,p[o[k]]*m/(k+1)); q[o[k]]=run
    res[lbl]=(p,q)
d=pd.DataFrame({'structure':Z2.columns,'region':[anno.get(c,'?') for c in Z2.columns],
   'EFFECT':obs.values,'p_random':res['random'][0],'q_random':res['random'][1],
   'p_sibling':res['sibling'][0],'q_sibling':res['sibling'][1]})
d['rank']=d.EFFECT.rank(ascending=False).astype(int)
d=d.sort_values('rank')
d.to_csv('results/PD_HD_validation/bundle/PD_core6_structure_bias.csv',index=False)
print("q<0.10: %d random / %d sibling   |   q<0.05: %d / %d" %
 ((d.q_random<0.10).sum(),(d.q_sibling<0.10).sum(),(d.q_random<0.05).sum(),(d.q_sibling<0.05).sum()))
print("\ntop 15:")
for _,r in d.head(15).iterrows():
    print(" %2d. %6.3f  q_rand=%.3f q_sib=%.3f  %-42s %s" % (r['rank'],r.EFFECT,r.q_random,r.q_sibling,r.structure[:42],r.region))
print("\nkey PD structures:")
for s in ['Substantia_nigra_compact_part','Ventral_tegmental_area','Dorsal_nucleus_raphe','Caudoputamen','Nucleus_accumbens']:
    x=d[d.structure==s].iloc[0]
    print("  rank %3d/213  EFFECT %6.3f  q_sib=%.3f  %s" % (x['rank'],x.EFFECT,x.q_sibling,s))
