import sys, os, re, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from disease_validation import recovery_stats, recovery_null_aurocs, empirical_p
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns], index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
MSN=[c for c in CT.columns if S[c] in {'STR D1 Gaba','STR D2 Gaba','STR D1 Sema5a Gaba','ACB-BST-FS D1 Gaba'}]
print("targets: %d Dopa clusters, %d MSN clusters, of %d\n" % (len(DOPA),len(MSN),CT.shape[1]))
rows=[]
for n in ["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G","StriatalDegeneration",
          "IBD","HDL_C","T2D","hba1c","Alzheimer","ASD_All","DDD_285_ExcludeASD","NT_Dopamine_combined"]:
    f=f"results/CT_Z2/{n}_bias_addP_random.csv"; nb=f"results/CT_Z2/null_bias/{n}_null_bias_random.parquet"
    if not os.path.exists(f): continue
    d=pd.read_csv(f,index_col=0)
    r={"set":n}
    for lbl,tgt in [("Dopa",DOPA),("MSN",MSN)]:
        st=recovery_stats(d,tgt); r[lbl+"_auroc"]=st['auroc']
        if os.path.exists(nb):
            try:
                a=recovery_null_aurocs(pd.read_parquet(nb),tgt)
                r[lbl+"_p"]=empirical_p(st['auroc'],a); r[lbl+"_nullmed"]=float(np.median(a))
            except Exception: r[lbl+"_p"]=np.nan; r[lbl+"_nullmed"]=np.nan
        else: r[lbl+"_p"]=np.nan; r[lbl+"_nullmed"]=np.nan
    rows.append(r)
df=pd.DataFrame(rows)
print("%-24s | %6s %7s %8s | %6s %7s %8s" % ("gene set","Dopa","p_null","nullMed","MSN","p_null","nullMed"))
print("-"*84)
for _,r in df.iterrows():
    print("%-24s | %6.3f %7.4f %8.3f | %6.3f %7.4f %8.3f" % (r['set'],r['Dopa_auroc'],r['Dopa_p'],r['Dopa_nullmed'],
          r['MSN_auroc'],r['MSN_p'],r['MSN_nullmed']))
