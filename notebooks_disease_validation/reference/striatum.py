import sys,glob,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
print("Striatum/Pallidum structures in the atlas (%d): %s\n" % (len(STRIA),sorted(STRIA)))
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
ds=pd.read_csv("results/STR_ISH/PD_HighConf_DA_SibMut_bias_addP_sibling.csv",index_col=0)
rk=pd.Series(np.arange(1,len(d)+1),index=d.index)
print("=== 19-gene set: where every striatal/pallidal structure ranks ===")
for s in sorted(STRIA,key=lambda x:rk[x]):
    print("  rank %3d/213  EFFECT %6.3f  q_sib=%.3f  %s" % (rk[s],d.loc[s,'EFFECT'],ds.loc[s,'q-value'],s))
print("\n=== Pareto circuits: any striatal/pallidal member? ===")
for f in sorted(glob.glob("results/CircuitSearch/PD_HighConf_DA/pareto_fronts/*pareto_front.csv")):
    size=int(f.split('size_')[1].split('_')[0])
    p=pd.read_csv(f); base=p[p.circuit_type=='baseline'].iloc[0]
    opt=p[p.circuit_type=='optimized'].copy()
    opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]
    mem=set(sel.structures.split(','))
    hit=sorted(mem&STRIA)
    print("  size %2d : %s" % (size, ("STRIATAL/PALLIDAL: "+", ".join(hit)) if hit else "none"))
print("\n=== 6-gene classic set for comparison ===")
c=pd.read_csv('results/PD_HD_validation/bundle/PD_core6_structure_bias.csv')
for s in sorted(STRIA,key=lambda x:int(c[c.structure==x]['rank'].iloc[0])):
    r=c[c.structure==s].iloc[0]
    print("  rank %3d/213  EFFECT %6.3f  %s" % (r['rank'],r.EFFECT,s))
