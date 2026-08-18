import sys,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseSTR_AvgZ_Weighted, MouseCT_AvgZ_Weighted, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
v2v3=pd.read_csv('dat/ISH_MERFISH_Gene_CorssSTR_Corr.v3.csv',index_col='Genes')['V2_V3_CT_Corr']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
SYMS="APP PSEN1 PSEN2 APOE SORL1 ADAM10 CDK5".split()
ent=[int(S2E[s]) for s in SYMS if S2E.get(s) and int(S2E[s]) in Z2.index]
print("AD causal set: %d/%d genes in matrix -> %s" % (len(ent),len(SYMS),
      [E2S.get(g) for g in ent]))
print("missing:",[s for s in SYMS if not(S2E.get(s) and int(S2E[s]) in Z2.index)] or "none")
obs=Z2.loc[ent].mean(axis=0)
# permutation FDR
M=Z2.values; rng=np.random.default_rng(42); N=10000
nl=np.array([np.nanmean(M[rng.choice(len(Z2.index),size=len(ent),replace=False)],axis=0) for _ in range(N)])
p=((nl>=obs.values).sum(axis=0)+1)/(N+1)
o=np.argsort(p); q=np.empty_like(p); run=1.0; m=len(p)
for k in range(m-1,-1,-1): run=min(run,p[o[k]]*m/(k+1)); q[o[k]]=run
d=pd.DataFrame({'structure':Z2.columns,'region':[anno.get(c,'?') for c in Z2.columns],
                'EFFECT':obs.values,'q':q}).sort_values('EFFECT',ascending=False)
d['rank']=np.arange(1,len(d)+1)
print("\nstructures at q<0.10: %d   q<0.05: %d" % ((d.q<0.10).sum(),(d.q<0.05).sum()))
print("\nTOP 20 STRUCTURES:")
for _,r in d.head(20).iterrows():
    f="  <== AD-relevant" if r.region in ('Hippocampus','Cortical_subplate') or 'Entorhinal' in r.structure else ""
    print("  %2d. %6.3f q=%.3f  %-42s %-14s%s" % (r['rank'],r.EFFECT,r.q,r.structure[:42],r.region,f))
print("\nAD hallmark structures:")
for s in ['Field_CA1','Field_CA2','Field_CA3','Dentate_gyrus','Subiculum_dorsal_part',
          'Entorhinal_area_lateral_part','Entorhinal_area_medial_part_dorsal_zone']:
    if s in d.structure.values:
        x=d[d.structure==s].iloc[0]
        print("  rank %3d/213  EFFECT %6.3f  q=%.3f  %s" % (x['rank'],x.EFFECT,x.q,s))
print("\nper-gene: where does CA1 rank in each gene's own profile?")
for g in ent:
    rk=Z2.loc[g].rank(ascending=False)
    print("  %-8s CA1 r%-4d  DG r%-4d  Entorhinal r%-4d  | top: %s" %
          (E2S.get(g),int(rk['Field_CA1']),int(rk['Dentate_gyrus']),
           int(rk['Entorhinal_area_lateral_part']), Z2.loc[g].idxmax()[:26]))
dn={g:1.0*(max(v2v3.loc[g],0)**2) for g in ent if g in v2v3.index and g in CT.index}
c=MouseCT_AvgZ_Weighted(CT,dn)
ms=c['EFFECT'].groupby(S.reindex(c.index)).mean().sort_values(ascending=False)
print("\ncell-type top 8 subclasses:"); print("  "+"\n  ".join("%6.3f  %s"%(v,k) for k,v in ms.head(8).items()))
