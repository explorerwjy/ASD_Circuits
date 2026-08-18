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
    sel=t[t.globalScore>=0.6].sort_values('globalScore',ascending=False)
    prof[lbl]=(sel,MouseSTR_AvgZ_Weighted(Z2,dict(zip(sel.entrez,sel.globalScore)))['EFFECT'].sort_values(ascending=False))
    print("\n########## OT-%s @ globalScore >= 0.6  (%d genes) ##########" % (lbl,len(sel)))
    print("  top genes:", ", ".join(sel.symbol.head(12)))
    print("  TOP 20 STRUCTURES:")
    for i,(k,v) in enumerate(prof[lbl][1].head(20).items(),1):
        print("   %2d. %6.3f  %-44s %s" % (i,v,k[:44],anno.get(k,'?')))
print("\n=== shared vs unique in the two top-20 lists ===")
a=list(prof['PD'][1].head(20).index); b=list(prof['AD'][1].head(20).index)
print("  shared (%d): %s" % (len(set(a)&set(b)), ", ".join(sorted(set(a)&set(b)))))
print("\n  PD-only : %s" % ", ".join([x for x in a if x not in b]))
print("  AD-only : %s" % ", ".join([x for x in b if x not in a]))
print("\n=== gene overlap between the two sets ===")
pg,ag=set(prof['PD'][0].symbol),set(prof['AD'][0].symbol)
print("  PD %d genes, AD %d genes, shared %d: %s" % (len(pg),len(ag),len(pg&ag),sorted(pg&ag)))
