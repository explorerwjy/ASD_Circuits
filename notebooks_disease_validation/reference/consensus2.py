import os,warnings; SP=os.path.dirname(os.path.abspath(__file__)); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import cohen_kappa_score
key=pd.read_csv(f"{SP}/blind2_key.csv"); key['norm']=key['structure'].str.replace('_',' ')
r1=pd.read_csv(f"{SP}/blind2_rater1.csv"); r2=pd.read_csv(f"{SP}/blind2_rater2.csv")
for r in (r1,r2): r['norm']=r['structure'].str.replace('_',' ').str.strip()
m=key.merge(r1[['norm','classification']],on='norm').merge(r2[['norm','classification']],on='norm',suffixes=('_1','_2'))
print("\n===== CONSENSUS (anatomy-matched, n=50) =====")
print("  raw agreement %.2f | Cohen kappa %.3f" % ((m.classification_1==m.classification_2).mean(),
      cohen_kappa_score(m.classification_1,m.classification_2)))
hit=lambda s:s.isin(['established','probable'])
for lbl,sel in [("BOTH raters hit (strict)",hit(m.classification_1)&hit(m.classification_2)),
                ("EITHER rater hit (lenient)",hit(m.classification_1)|hit(m.classification_2))]:
    a=int((m.group.eq('TOP20')&sel).sum()); c=int((m.group.eq('DECOY_BS')&sel).sum())
    orr,pv=fisher_exact([[a,20-a],[c,30-c]],alternative='greater')
    u,pr=mannwhitneyu(m.loc[sel,'true_rank'],m.loc[~sel,'true_rank'],alternative='less')
    print("  %-28s TOP20 %2d/20  DECOY %2d/30  OR=%-6s p=%.4g | rank test p=%.4g (med %.0f vs %.0f)"
          % (lbl,a,c,'inf' if c==0 else '%.2f'%orr,pv,pr,m.loc[sel,'true_rank'].median(),m.loc[~sel,'true_rank'].median()))
sel=hit(m.classification_1)&hit(m.classification_2)
print("\n  consensus hits (both raters), by rank:")
for _,x in m[sel].sort_values('true_rank').iterrows():
    print("     rank %3d  %-42s %s" % (x.true_rank,x.norm[:42],x.group))
