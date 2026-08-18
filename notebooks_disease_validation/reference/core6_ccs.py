import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import LoadGeneINFO, ScoreCircuit_SI_Joint
_,_,S2E,E2S=LoadGeneINFO()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
ent=[int(S2E[s]) for s in "LRRK2 SNCA GBA PRKN PINK1 PARK7".split()]
SIZES=list(range(6,71))
cols=np.array(Z2.columns); M=Z2.values
obs_prof=pd.Series(Z2.loc[ent].mean(axis=0),index=Z2.columns).sort_values(ascending=False).index.values
prof=lambda ss:[ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
o=np.array(prof(obs_prof))
sw=pd.read_csv('dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',header=None)
idx={g:i for i,g in enumerate(Z2.index)}
pools={'random':np.arange(len(Z2.index)),
       'sibling':np.array([idx[g] for g in sw[0].astype(int) if g in idx])}
N=10000
out={'N':SIZES,'CCS_observed':o}
for lbl,P in pools.items():
    rng=np.random.default_rng(42)
    draws=[rng.choice(P,size=len(ent),replace=False) for _ in range(N)]
    def one(dr):
        v=np.nanmean(M[dr],axis=0)
        return prof(cols[np.argsort(-v)])
    null=np.array(Parallel(n_jobs=12)(delayed(one)(d) for d in draws))
    out[f'p_{lbl}']=[(np.sum(null[:,j]>=o[j])+1)/(N+1) for j in range(len(SIZES))]
df=pd.DataFrame(out)
df.to_csv('results/PD_HD_validation/bundle/PD_core6_CCS_by_circuit_size.csv',index=False)
sig=df[df.p_sibling<0.05]
print("CORE-6 CCS — significant (sibling null, p<0.05) at %d of %d sizes" % (len(sig),len(df)))
print("  significant N:", sig.N.tolist() or "NONE")
print("\n%4s %10s %11s %11s" % ("N","CCS","p_sibling","p_random"))
for _,r in df.iterrows():
    if r.N%4==0 or r.p_sibling<0.05:
        print("%4d %10.3f %11.4f %11.4f%s" % (r.N,r.CCS_observed,r.p_sibling,r.p_random," <== sig" if r.p_sibling<0.05 else ""))
print("\nbest (lowest p_sibling): N=%d p=%.4f" % (df.loc[df.p_sibling.idxmin(),'N'],df.p_sibling.min()))
