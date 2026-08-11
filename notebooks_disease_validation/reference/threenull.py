import sys, os, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from disease_validation import load_ground_truth, recovery_stats, recovery_null_aurocs, empirical_p
GT=load_ground_truth('config/disease_validation_ground_truth.yaml'); gmap=GT['notes']['gene_sets_to_ground_truth']
SETS=["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G","HD_HTT","StriatalDegeneration"]
VAR=[("random  (uniform)","{s}","random"),("expr-matched","{s}_EM","random"),
     ("sibling (uniform)","{s}","sibling"),("sibling (mutability=ASD)","{s}_SibMut","sibling")]
print("%-22s %6s | %-24s %8s %8s | %-24s %8s %8s" % ("gene set","AUROC","null","p_null","q<.10","null","p_null","q<.10"))
for s in SETS:
    core=GT['structures'][gmap[s]]['core']
    base=pd.read_csv(f"results/STR_ISH/{s}_bias_addP_random.csv",index_col=0)
    au=recovery_stats(base,core)['auroc']
    cells=[]
    for lbl,pat,kind in VAR:
        f=f"results/STR_ISH/{pat.format(s=s)}_bias_addP_{kind}.csv"
        nb=f"results/STR_ISH/null_bias/{pat.format(s=s)}_null_bias_{kind}.parquet"
        if not os.path.exists(f): cells.append((lbl,np.nan,np.nan)); continue
        d=pd.read_csv(f,index_col=0); q=int((d['q-value']<0.10).sum())
        try: p=empirical_p(au, recovery_null_aurocs(pd.read_parquet(nb),core))
        except Exception: p=np.nan
        cells.append((lbl,p,q))
    print("%-22s %6.3f | %-24s %8s %8s | %-24s %8s %8s" % (s,au,
          cells[0][0],f"{cells[0][1]:.4f}" if cells[0][1]==cells[0][1] else "nan",cells[0][2],
          cells[3][0],f"{cells[3][1]:.4f}" if cells[3][1]==cells[3][1] else "nan",cells[3][2]))
    print("%-22s %6s | %-24s %8s %8s | %-24s %8s %8s" % ("","",
          cells[1][0],f"{cells[1][1]:.4f}" if cells[1][1]==cells[1][1] else "nan",cells[1][2],
          cells[2][0],f"{cells[2][1]:.4f}" if cells[2][1]==cells[2][1] else "nan",cells[2][2]))
