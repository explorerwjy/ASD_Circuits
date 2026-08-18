"""EXPLORATORY: circuit search on the curated 19-gene PD set using a WEIGHT-based
objective (mean log1p connection weight) instead of the distance-conditioned
information score. Standalone -- does not touch src/ or the Snakemake pipeline."""
import sys,os,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from SA import Annealer
from joblib import Parallel, delayed
from ASD_Circuits import BiasLim, ScoreCircuit_SI_Joint, STR2Region
anno=STR2Region()
W=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv',index_col=0)
I=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
B=pd.read_csv('results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv',index_col=0).sort_values('EFFECT',ascending=False)
LW=np.log1p(W)                                   # compress the orders-of-magnitude spread
def score_w(S):
    sub=LW.loc[S,S].values
    n=len(S)*(len(S)-1)
    return (sub.sum()-np.trace(sub))/n
class SA_W(Annealer):
    def __init__(self,BiasDF,state,cand,minbias):
        self.BiasDF=BiasDF; self.cand=cand; self.minbias=minbias
        super().__init__(state)
    def move(self):
        e0=self.energy()
        i=np.random.choice(np.where(self.state==1)[0],1); j=np.random.choice(np.where(self.state==0)[0],1)
        self.state[i]=0; self.state[j]=1
        if self.BiasDF.loc[self.cand[np.where(self.state==1)],'EFFECT'].mean()<self.minbias:
            self.state[i]=1; self.state[j]=0
        return self.energy()-e0
    def energy(self): return -score_w(self.cand[np.where(self.state==1)[0]])
def one(minbias,size=40,topN=213,steps=50000,seed=0):
    np.random.seed(seed)
    cand=B.head(topN).index.values
    st=np.zeros(topN); st[:size]=1          # seed with top-`size` by bias (max possible mean)
    if B.loc[cand[:size],'EFFECT'].mean()<minbias: return None
    ins=SA_W(B.head(topN),st,cand,minbias); ins.copy_strategy="method"
    ins.Tmax=1e-2; ins.Tmin=5e-5; ins.steps=steps; ins.updates=0
    _,_,state,e=ins.anneal()
    return (minbias, -e, list(cand[np.where(state==1)[0]]))
lims=sorted({b for _,b in BiasLim(B,40)})
lims=[b for b in lims if b<=B.head(40)['EFFECT'].mean()][::3]
print("running %d bias limits x 12 restarts, size 40 ..."%len(lims))
res=Parallel(n_jobs=12)(delayed(one)(b,40,213,50000,r) for b in lims for r in range(12))
best={}
for r in res:
    if r and (r[0] not in best or r[1]>best[r[0]][0]): best[r[0]]=(r[1],r[2])
rows=[]
for b,(sc,mem) in sorted(best.items()):
    rows.append(dict(bias_limit=b,mean_bias=B.loc[mem,'EFFECT'].mean(),W_score=sc,
                     CCS_SI=ScoreCircuit_SI_Joint(mem,I),
                     has_CP='Caudoputamen' in mem,has_NAcc='Nucleus_accumbens' in mem,
                     has_SNc='Substantia_nigra_compact_part' in mem,structures=",".join(mem)))
df=pd.DataFrame(rows).sort_values('mean_bias',ascending=False)
df.to_csv('results/PD_HD_validation/exploratory/PD_weightbased_pareto_size40_fullrange.csv',index=False)
print("\n%9s %10s %9s %9s %6s %6s %6s"%("bias_lim","mean_bias","W_score","CCS_SI","CP","NAcc","SNc"))
for _,r in df.iterrows():
    print("%9.3f %10.4f %9.4f %9.4f %6s %6s %6s"%(r.bias_limit,r.mean_bias,r.W_score,r.CCS_SI,
          'YES' if r.has_CP else '-', 'YES' if r.has_NAcc else '-','Y' if r.has_SNc else '-'))
print("\npoints containing CP: %d/%d"%(df.has_CP.sum(),len(df)))
