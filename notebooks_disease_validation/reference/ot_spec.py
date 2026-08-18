import sys,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseSTR_AvgZ_Weighted, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
F={'PD':'dat/Genetics/OpenTargets/OT-MONDO_0021095-associated-targets-8_12_2026-v26_06.tsv',
   'AD':'dat/Genetics/OpenTargets/OT-MONDO_0004975-associated-targets-8_14_2026-v26_06.tsv'}
prof={}
for lbl,f in F.items():
    t=pd.read_csv(f,sep='\t'); t['entrez']=[S2E.get(s) for s in t['symbol']]
    t=t.dropna(subset=['entrez']); t['entrez']=t['entrez'].astype(int); t=t[t.entrez.isin(Z2.index)]
    for cut in [0.1,0.3,0.5,0.6]:
        sel=t[t.globalScore>=cut]
        prof[(lbl,cut)]=MouseSTR_AvgZ_Weighted(Z2,dict(zip(sel.entrez,sel.globalScore)))['EFFECT']
        prof[(lbl,cut)].name=f"{lbl}@{cut}"
print("SPECIFICITY CHECK — Spearman correlation between the PD and AD structure profiles")
for cut in [0.1,0.3,0.5,0.6]:
    r=prof[('PD',cut)].corr(prof[('AD',cut)],method='spearman')
    ov=len(set(prof[('PD',cut)].nlargest(20).index)&set(prof[('AD',cut)].nlargest(20).index))
    print("   cutoff %.1f :  rho = %+.3f   |  top-20 overlap = %d/20" % (cut,r,ov))
mend=pd.read_csv('results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv',index_col=0)['EFFECT']
print("\nAgainst our 19-gene Mendelian PD profile:")
for lbl in ['PD','AD']:
    for cut in [0.3,0.5,0.6]:
        r=prof[(lbl,cut)].corr(mend,method='spearman')
        ov=len(set(prof[(lbl,cut)].nlargest(20).index)&set(mend.nlargest(20).index))
        print("   OT-%s @%.1f  rho vs Mendelian = %+.3f  | top-20 overlap %d/20" % (lbl,cut,r,ov))
print("\nAD sanity: where do the AD-relevant structures rank in the OT-AD profiles?")
for s in ['Field_CA1','Field_CA3','Dentate_gyrus','Entorhinal_area_lateral_part','Subiculum_dorsal_part']:
    row=" ".join("%s@%.1f r%-4d"%('AD',c,int(prof[('AD',c)].rank(ascending=False)[s])) for c in [0.1,0.3,0.5,0.6])
    print("   %-32s %s" % (s,row))
