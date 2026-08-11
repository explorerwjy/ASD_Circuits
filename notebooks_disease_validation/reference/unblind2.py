import os,sys,warnings; SP=os.path.dirname(os.path.abspath(__file__)); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
key=pd.read_csv(f"{SP}/blind2_key.csv"); key['norm']=key['structure'].str.replace('_',' ')
for f in ["blind2_rater1.csv","blind2_rater2.csv"]:
    p=f"{SP}/{f}"
    if not os.path.exists(p): print(f"\n({f} not ready)"); continue
    r=pd.read_csv(p); r['norm']=r['structure'].str.replace('_',' ').str.strip()
    m=key.merge(r[['norm','classification']],on='norm',how='left')
    print(f"\n===== {f} (matched {m.classification.notna().sum()}/50) =====")
    print(pd.crosstab(m.group,m.classification).to_string())
    for lbl,sel in [("established+probable",m.classification.isin(['established','probable'])),
                    ("established only",m.classification.eq('established'))]:
        a=int((m.group.eq('TOP20')&sel).sum()); na=int(m.group.eq('TOP20').sum())
        c=int((m.group.eq('DECOY_BS')&sel).sum()); nc=int(m.group.eq('DECOY_BS').sum())
        orr,pv=fisher_exact([[a,na-a],[c,nc-c]],alternative='greater')
        print("  %-22s TOP20 %2d/%d  DECOY_BS %2d/%d   OR=%s  p=%.4g"
              % (lbl,a,na,c,nc,'inf' if c==0 else '%.2f'%orr,pv))
    # rank-based: do hits rank higher overall?
    sel=m.classification.isin(['established','probable'])
    if sel.sum()>1 and (~sel).sum()>1:
        u,pv=mannwhitneyu(m.loc[sel,'true_rank'],m.loc[~sel,'true_rank'],alternative='less')
        print("  rank test: PD-affected structures rank higher than unaffected, p=%.4g (median rank %.0f vs %.0f)"
              % (pv,m.loc[sel,'true_rank'].median(),m.loc[~sel,'true_rank'].median()))
    print("  hits by rank:")
    for _,x in m[sel].sort_values('true_rank').iterrows():
        print("     rank %3d  %-42s %-12s %s" % (x.true_rank,x.norm[:42],x.classification,x.group))
