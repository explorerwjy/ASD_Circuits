import sys,os,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import ScoreCircuit_SI_Joint
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
M="PD_HighConf_DA"; SIZES=list(range(6,71))
obs=pd.read_csv(f"results/STR_ISH/{M}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False).index.values
prof=lambda ss:[ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
o=np.array(prof(obs))
out={'N':SIZES,'CCS_observed':o}
for lbl,pq in [("sibling",f"results/STR_ISH/null_bias/{M}_SibMut_null_bias_sibling.parquet"),
               ("random", f"results/STR_ISH/null_bias/{M}_null_bias_random.parquet")]:
    nb=pd.read_parquet(pq)
    null=np.array(Parallel(n_jobs=12,verbose=0)(
        delayed(prof)(nb[c].sort_values(ascending=False).index.values) for c in nb.columns))
    out[f'p_{lbl}']=[(np.sum(null[:,j]>=o[j])+1)/(null.shape[0]+1) for j in range(len(SIZES))]
    out[f'null_median_{lbl}']=np.median(null,axis=0)
df=pd.DataFrame(out)
df.to_csv('results/PD_HD_validation/bundle/PD_CCS_by_circuit_size.csv',index=False)
sig=df[df.p_sibling<0.05]
print("SIBLING NULL — N where CCS is significant (p<0.05): %d of %d sizes tested (N=6..70)" % (len(sig),len(df)))
print("  significant N:", sig.N.tolist())
print("\n%4s %10s %10s %10s %12s" % ("N","CCS","p_sibling","p_random","null_med_sib"))
for _,r in df.iterrows():
    star="  <== sig (sibling)" if r.p_sibling<0.05 else ""
    if r.N%2==0 or r.p_sibling<0.05:
        print("%4d %10.3f %10.4f %10.4f %12.3f%s" % (r.N,r.CCS_observed,r.p_sibling,r.p_random,r.null_median_sibling,star))
