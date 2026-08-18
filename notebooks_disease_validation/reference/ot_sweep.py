import sys,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseSTR_AvgZ_Weighted, MouseCT_AvgZ_Weighted, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
v2v3=pd.read_csv('dat/ISH_MERFISH_Gene_CorssSTR_Corr.v3.csv',index_col='Genes')['V2_V3_CT_Corr']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
FILES={'PD':'dat/Genetics/OpenTargets/OT-MONDO_0021095-associated-targets-8_12_2026-v26_06.tsv',
       'AD':'dat/Genetics/OpenTargets/OT-MONDO_0004975-associated-targets-8_14_2026-v26_06.tsv'}
for lbl,f in FILES.items():
    t=pd.read_csv(f,sep='\t')
    t['entrez']=[S2E.get(s) for s in t['symbol']]
    t=t.dropna(subset=['entrez']); t['entrez']=t['entrez'].astype(int)
    t=t[t.entrez.isin(Z2.index)]
    ea=pd.to_numeric(t.get('expressionAtlas'),errors='coerce')
    print("\n########## %s  (%s) ##########" % (lbl,f.split('/')[-1][:22]))
    print("  %d rows -> %d mapped & in Z2 matrix | with expressionAtlas evidence: %d" %
          (len(pd.read_csv(f,sep='\t')),len(t),int(ea.notna().sum())))
    print("  %6s %6s | %-46s | %-40s" % ("cutoff","nGenes","top-5 structures","top subclass (cell type)"))
    for cut in [0.1,0.2,0.3,0.4,0.5,0.6]:
        sel=t[t.globalScore>=cut]
        if len(sel)<3: print("  %6.1f %6d | too few genes" % (cut,len(sel))); continue
        w=dict(zip(sel.entrez,sel.globalScore))
        b=MouseSTR_AvgZ_Weighted(Z2,w).sort_values('EFFECT',ascending=False)
        top5=", ".join(x[:18] for x in b.index[:5])
        dn={g:v*(max(v2v3.loc[g],0)**2) for g,v in w.items() if g in v2v3.index and g in CT.index}
        c=MouseCT_AvgZ_Weighted(CT,dn)
        ms=c['EFFECT'].groupby(S.reindex(c.index)).mean().sort_values(ascending=False)
        dr=list(ms.index).index('SNc-VTA-RAmb Foxa1 Dopa')+1
        print("  %6.1f %6d | %-46s | %-32s (Dopa r%d/340)" % (cut,len(sel),top5[:46],ms.index[0][:32],dr))
