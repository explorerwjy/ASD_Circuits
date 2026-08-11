import sys, os, re, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from disease_validation import load_ground_truth, recovery_stats, recovery_null_aurocs, empirical_p
GT=load_ground_truth('config/disease_validation_ground_truth.yaml'); gmap=GT['notes']['gene_sets_to_ground_truth']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
sets=["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G","HD_HTT","StriatalDegeneration"]
print("%-24s %6s %5s %8s %10s %9s %8s" % ("gene set","nClust","q<.1","AUROC","p_MannWh","p_geneset","medRank"))
print("-"*82)
for s in sets:
    f=f"results/CT_Z2/{s}_bias_addP_random.csv"
    if not os.path.exists(f): print(f"{s}: MISSING"); continue
    d=pd.read_csv(f,index_col=0)
    tgt=set(GT['cell_type_subclasses'][gmap[s]]['core'])
    cl=[c for c in d.index if sub(c) in tgt]
    st=recovery_stats(d,cl)
    nb=f"results/CT_Z2/null_bias/{s}_null_bias_random.parquet"
    try:
        p=empirical_p(st['auroc'], recovery_null_aurocs(pd.read_parquet(nb),cl))
    except Exception as e: p=float('nan')
    print("%-24s %6d %5d %8.3f %10.2e %9.4f %8.0f" % (s,len(cl),int((d['q-value']<0.10).sum()),
          st['auroc'],st['p_mannwhitney'],p,st['median_rank']))
