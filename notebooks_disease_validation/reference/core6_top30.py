import sys,warnings,collections; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd
from ASD_Circuits import STR2Region
anno=STR2Region()
STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
c=pd.read_csv('results/PD_HD_validation/bundle/PD_core6_structure_bias.csv').sort_values('rank')
print("CORE-6 SET (LRRK2 SNCA GBA PRKN PINK1 PARK7) — TOP 30 of 213")
print("%4s %8s %9s %9s  %-44s %-16s" % ("rank","EFFECT","q_rand","q_sib","structure","region"))
print("-"*100)
for _,r in c.head(30).iterrows():
    flag="  <== STRIATUM/PALLIDUM" if r.structure in STRIA else ""
    print("%4d %8.3f %9.3f %9.3f  %-44s %-16s%s" % (r['rank'],r.EFFECT,r.q_random,r.q_sibling,r.structure[:44],r.region,flag))
print("-"*100)
print("q<0.10: %d random / %d sibling  (nothing significant)" % ((c.q_random<0.10).sum(),(c.q_sibling<0.10).sum()))
print("\nregion composition of top 30:", dict(collections.Counter(c.head(30).region)))
print("\nstriatal/pallidal structures inside the top 30:",
      [s for s in c.head(30).structure if s in STRIA] or "none")
print("\nfor comparison, 19-gene set top-30 composition:")
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
print(" ",dict(collections.Counter(anno.get(k,'?') for k in d.head(30).index)))
