import sys, re, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseCT_AvgZ_Weighted
from disease_validation import recovery_stats, recovery_null_aurocs, empirical_p
_,_,S2E,E2S=LoadGeneINFO()
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns], index=CT.columns)
htt=CT.loc[int(S2E['HTT'])]
m=htt.groupby(S).agg(['mean','size']); m.columns=['mean','n']; m=m.sort_values('mean',ascending=False)
print("=== 1. HTT across NON-NEURONAL / glial subclasses (of %d subclasses) ===" % len(m))
glia=[s for s in m.index if re.search(r'NN$|Astro|Oligo|OPC|Microglia|Endo|VLMC|Peri|Bergmann|Tanycyte|Ependymal|ABC', s)]
for g in glia:
    print("   %6.3f  n=%-3d rank %3d/%d  %s" % (m.loc[g,'mean'], m.loc[g,'n'], list(m.index).index(g)+1, len(m), g))
print("\n   MSN subclasses for comparison:")
for g in ['STR D1 Gaba','STR D2 Gaba','STR D1 Sema5a Gaba','ACB-BST-FS D1 Gaba']:
    if g in m.index: print("   %6.3f  n=%-3d rank %3d/%d  %s" % (m.loc[g,'mean'],m.loc[g,'n'],list(m.index).index(g)+1,len(m),g))

print("\n=== 2. HD somatic-instability modifier genes (GeM-HD) vs striatal MSNs ===")
MSN=[c for c in CT.columns if S[c] in {'STR D1 Gaba','STR D2 Gaba','STR D1 Sema5a Gaba','ACB-BST-FS D1 Gaba'}]
mods=['FAN1','MSH3','MLH1','MLH3','PMS1','PMS2','LIG1','TCERG1','RRM2B','MSH2','POLD1']
w={}
for g in mods:
    e=S2E.get(g)
    if e is not None and int(e) in CT.index: w[int(e)]=1.0
    else: print("   (missing: %s)" % g)
print("   using %d/%d modifier genes: %s" % (len(w),len(mods),", ".join(sorted(E2S.get(g,str(g)) for g in w))))
b=MouseCT_AvgZ_Weighted(CT,w)
st=recovery_stats(b,MSN)
print("   MODIFIERS vs MSN:  AUROC %.3f   p_MannWhitney %.2e   medRank %.0f/5312" % (st['auroc'],st['p_mannwhitney'],st['median_rank']))
mm=b['EFFECT'].groupby(S.reindex(b.index)).mean().sort_values(ascending=False)
print("   modifier top 8 subclasses:")
for k,v in mm.head(8).items(): print("      %6.3f  %s" % (v,k))
print("   modifier rank of MSN subclasses:")
for g in ['STR D1 Gaba','STR D2 Gaba']:
    print("      %6.3f  %s  (rank %d/%d)" % (mm[g],g,list(mm.index).index(g)+1,len(mm)))
