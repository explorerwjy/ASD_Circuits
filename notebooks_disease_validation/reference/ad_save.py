import sys,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseSTR_AvgZ_Weighted, MouseCT_AvgZ_Weighted, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
v2v3=pd.read_csv('dat/ISH_MERFISH_Gene_CorssSTR_Corr.v3.csv',index_col='Genes')['V2_V3_CT_Corr']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
OUT='results/PD_HD_validation/exploratory'
def run(name,ent,wts=None):
    w=dict(zip(ent,wts)) if wts is not None else {g:1.0 for g in ent}
    obs=MouseSTR_AvgZ_Weighted(Z2,w)['EFFECT'].reindex(Z2.columns)
    M=Z2.values; rng=np.random.default_rng(42); N=10000
    nl=np.array([np.nanmean(M[rng.choice(len(Z2.index),size=len(w),replace=False)],axis=0) for _ in range(N)])
    p=((nl>=obs.values).sum(axis=0)+1)/(N+1)
    o=np.argsort(p); q=np.empty_like(p); run_=1.0; m=len(p)
    for k in range(m-1,-1,-1): run_=min(run_,p[o[k]]*m/(k+1)); q[o[k]]=run_
    d=pd.DataFrame({'structure':Z2.columns,'region':[anno.get(c,'?') for c in Z2.columns],
                    'EFFECT':obs.values,'p_value':p,'q_value':q}).sort_values('EFFECT',ascending=False)
    d.insert(0,'rank',np.arange(1,len(d)+1))
    d.to_csv(f"{OUT}/{name}_structure_bias.csv",index=False)
    dn={g:v*(max(v2v3.loc[g],0)**2) for g,v in w.items() if g in v2v3.index and g in CT.index}
    c=MouseCT_AvgZ_Weighted(CT,dn)
    ct=pd.DataFrame({'cluster':c.index,'subclass':[sub(i) for i in c.index],'EFFECT':c['EFFECT'].values})
    ct=ct.sort_values('EFFECT',ascending=False); ct.insert(0,'rank',np.arange(1,len(ct)+1))
    ct.to_csv(f"{OUT}/{name}_celltype_bias.csv",index=False)
    hip=[x for x in ['Field_CA1','Field_CA3','Dentate_gyrus','Entorhinal_area_lateral_part','Subiculum_dorsal_part']]
    rr={s:int(d[d.structure==s]['rank'].iloc[0]) for s in hip if s in d.structure.values}
    print("%-22s n=%3d | q<0.10: %2d | top: %-30s | hippocampal ranks: %s" %
          (name,len(w),(d.q_value<0.10).sum(),d.iloc[0].structure[:30],rr))
t=pd.read_csv('dat/Genetics/OpenTargets/OT-MONDO_0004975-associated-targets-8_14_2026-v26_06.tsv',sep='\t')
t['entrez']=[S2E.get(s) for s in t['symbol']]; t=t.dropna(subset=['entrez'])
t['entrez']=t.entrez.astype(int); t=t[t.entrez.isin(Z2.index)]
for cut in [0.3,0.5,0.6]:
    s=t[t.globalScore>=cut]; run(f"AD_OpenTargets_cut{cut}",list(s.entrez),list(s.globalScore))
c7=[int(S2E[x]) for x in "APP PSEN1 PSEN2 APOE SORL1 ADAM10 CDK5".split() if S2E.get(x) and int(S2E[x]) in Z2.index]
run("AD_causal7",c7)
