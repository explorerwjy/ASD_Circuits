import sys, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
from ASD_Circuits import ScoreCircuit_SI_Joint
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
Cont=np.load('dat/allen-mouse-conn/RankScores/RankScore.Ipsi.Cont.npy')
topNs=np.arange(200,5,-1); SIZES=[100,60,46,30,20,10]
idx={N:int(np.where(topNs==N)[0][0]) for N in SIZES}
def prof(sorted_strs): return [ScoreCircuit_SI_Joint(sorted_strs[:N],Info) for N in SIZES]
print("gene set             size  | " + " | ".join("N=%-3d p_matched (p_ASDsib)"%N for N in [46,20]))
print("-"*96)
for s in ["PD_Primary","PD_Sens_Atypical","PD_GWAS_L2G","StriatalDegeneration"]:
    obs=pd.read_csv(f"results/STR_ISH/{s}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False).index.values
    o=prof(obs)
    nb=pd.read_parquet(f"results/STR_ISH/null_bias/{s}_null_bias_random.parquet")
    cols=[str(i) for i in range(nb.shape[1])]
    null=Parallel(n_jobs=10)(delayed(prof)(nb[c].sort_values(ascending=False).index.values) for c in cols)
    null=np.array(null)
    n_genes=len(pd.read_csv(f"dat/Genetics/GeneWeights/{s}.gw",header=None))
    out=[]
    for j,N in enumerate(SIZES):
        if N not in (46,20): continue
        pm=(np.sum(null[:,j]>=o[j])+1)/(null.shape[0]+1)
        pa=(np.sum(Cont[:,idx[N]]>=o[j])+1)/(Cont.shape[0]+1)
        out.append("%.3f  (%.3f)" % (pm,pa))
    print("%-20s %4d  | %s" % (s, n_genes, " | ".join("%-24s"%x for x in out)))
    np.save(f"{sys.argv[1]}/ccsnull_{s}.npy", null)
