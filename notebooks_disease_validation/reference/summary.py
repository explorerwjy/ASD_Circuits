import sys,os,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from disease_validation import load_ground_truth, recovery_stats, recovery_null_aurocs, empirical_p
GT=load_ground_truth('config/disease_validation_ground_truth.yaml')
core=GT['structures']['parkinson']['core']; strc=GT['structures']['striatal']['core']
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
MSN=[c for c in CT.columns if S[c] in {'STR D1 Gaba','STR D2 Gaba','STR D1 Sema5a Gaba','ACB-BST-FS D1 Gaba'}]
rows=[]
for name,gt_s,gt_c,lab in [("PD_HighConf_DA",core,DOPA,"PD"),("PD_HighConf",core,DOPA,"PD"),
                           ("PD_Primary",core,DOPA,"PD"),("PD_GWAS_L2G",core,DOPA,"PD"),
                           ("HD_HTT",strc,MSN,"HD"),("StriatalDegeneration",strc,MSN,"HD")]:
    r={'gene_set':name,'disease':lab}
    f=f"results/STR_ISH/{name}_bias_addP_random.csv"
    if os.path.exists(f):
        d=pd.read_csv(f,index_col=0); r['n_genes']=len(pd.read_csv(f'dat/Genetics/GeneWeights/{name}.gw',header=None))
        r['STR_AUROC']=round(recovery_stats(d,gt_s)['auroc'],3)
        r['STR_q<0.10_random']=int((d['q-value']<0.10).sum())
        fs=f"results/STR_ISH/{name}_SibMut_bias_addP_sibling.csv"
        r['STR_q<0.10_sibling']=int((pd.read_csv(fs,index_col=0)['q-value']<0.10).sum()) if os.path.exists(fs) else None
        for lbl,pat,kind in [('p_random',name,'random'),('p_exprmatched',name+'_EM','random'),('p_siblingMut',name+'_SibMut','sibling')]:
            nb=f"results/STR_ISH/null_bias/{pat}_null_bias_{kind}.parquet"
            try: r[lbl]=round(empirical_p(r['STR_AUROC'],recovery_null_aurocs(pd.read_parquet(nb),gt_s)),4)
            except Exception: r[lbl]=None
    fc=f"results/CT_Z2/{name}_bias_addP_random.csv"
    if os.path.exists(fc):
        d=pd.read_csv(fc,index_col=0); st=recovery_stats(d,gt_c)
        r['CT_AUROC']=round(st['auroc'],3)
        try: r['CT_p_geneset']=round(empirical_p(st['auroc'],recovery_null_aurocs(pd.read_parquet(f"results/CT_Z2/null_bias/{name}_null_bias_random.parquet"),gt_c)),4)
        except Exception: r['CT_p_geneset']=None
    rows.append(r)
t=pd.DataFrame(rows)
t.to_csv('results/tables/PD_HD_validation_summary.csv',index=False)
print(t.to_string(index=False))
