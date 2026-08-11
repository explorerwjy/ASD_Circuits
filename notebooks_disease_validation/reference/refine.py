import os,sys,warnings; SP=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import cohen_kappa_score
from ASD_Circuits import STR2Region
anno=STR2Region()
key=pd.read_csv(f"{SP}/blind_key.csv"); key['norm']=key['structure'].str.replace('_',' ')
r1=pd.read_csv(f"{SP}/blind_rater1.csv"); r2=pd.read_csv(f"{SP}/blind_rater2.csv")
for r in (r1,r2): r['norm']=r['structure'].str.replace('_',' ').str.strip()
m=key.merge(r1[['norm','classification']],on='norm').merge(r2[['norm','classification']],on='norm',suffixes=('_1','_2'))
print("INTER-RATER AGREEMENT")
print("  raw agreement: %.2f" % (m.classification_1==m.classification_2).mean())
print("  Cohen kappa  : %.3f" % cohen_kappa_score(m.classification_1,m.classification_2))
hit=lambda s: s.isin(['established','probable'])
m['consensus']=hit(m.classification_1)&hit(m.classification_2)
m['region']=[anno.get(s,'?') for s in m.structure]
BRAINSTEM={'Midbrain','Pons','Medulla','Hypothalamus','Thalamus'}
print("\nCONSENSUS (both raters call established/probable):")
a=int((m.group.eq('TOP20')&m.consensus).sum()); c=int((m.group.eq('DECOY')&m.consensus).sum())
orr,pv=fisher_exact([[a,20-a],[c,20-c]],alternative='greater')
print("  TOP20 %d/20   DECOY %d/20   OR=%.2f  p=%.4g" % (a,c,orr,pv))
print("\nREFINED CRITERION — 'early/selective' = hit AND brainstem/midbrain/diencephalon")
m['early']=m.consensus & m.region.isin(BRAINSTEM)
a2=int((m.group.eq('TOP20')&m.early).sum()); c2=int((m.group.eq('DECOY')&m.early).sum())
orr2,pv2=fisher_exact([[a2,20-a2],[c2,20-c2]],alternative='greater')
print("  TOP20 %d/20   DECOY %d/20   OR=%s  p=%.4g" % (a2,c2,'inf' if c2==0 else '%.2f'%orr2,pv2))
print("\n  consensus hits by anatomy:")
for g in ('TOP20','DECOY'):
    sub=m[(m.group==g)&m.consensus]
    print("   %s: %d hits -> %s" % (g,len(sub),dict(sub.region.value_counts())))
print("\n  TOP20 consensus hits:")
for _,x in m[(m.group=='TOP20')&m.consensus].sort_values('true_rank').iterrows():
    print("     rank %3d  %-44s %s" % (x.true_rank,x.norm[:44],x.region))
print("  DECOY consensus hits:")
for _,x in m[(m.group=='DECOY')&m.consensus].sort_values('true_rank').iterrows():
    print("     rank %3d  %-44s %s" % (x.true_rank,x.norm[:44],x.region))
