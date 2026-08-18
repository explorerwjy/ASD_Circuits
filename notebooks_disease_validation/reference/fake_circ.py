import sys,glob,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
d=pd.read_csv('results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv',index_col=0)
rk=d['Rank']
for f in sorted(glob.glob("results/CircuitSearch/PD_FAKE_cherrypicked/pareto_fronts/*.csv")):
    size=int(f.split('size_')[1].split('_')[0])
    p=pd.read_csv(f); base=p[p.circuit_type=='baseline'].iloc[0]
    opt=p[p.circuit_type=='optimized'].copy()
    opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]
    mem=sel.structures.split(',')
    hits=sorted(set(mem)&STRIA,key=lambda x:rk[x])
    print("=== FAKE size %d @ ASD-matched point (bias %.1f%%, CCS %.3f) ===" % (size,sel.dB,sel.circuit_score))
    print("   SNc present: %s | VTA present: %s" % ('Substantia_nigra_compact_part' in mem,'Ventral_tegmental_area' in mem))
    print("   STRIATAL/PALLIDAL members: %s" % (", ".join("%s(r%d)"%(s,rk[s]) for s in hits) if hits else "NONE"))
    print("   NIGROSTRIATAL PAIR (SNc + CP): %s" % ('Substantia_nigra_compact_part' in mem and 'Caudoputamen' in mem))
    print("   full circuit:", ", ".join(sorted(mem,key=lambda x:rk[x])[:8]),"...")
