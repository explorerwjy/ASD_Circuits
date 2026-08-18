import sys,glob,warnings,collections; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
d=pd.read_csv('results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv',index_col=0); rk=d['Rank']
p_ccs=pd.read_csv('results/PD_HD_validation/exploratory/CCS_profile_real_vs_fake.csv').set_index('N')
for f in sorted(glob.glob("results/CircuitSearch/PD_FAKE_cherrypicked/pareto_fronts/*.csv"),
                key=lambda x:int(x.split('size_')[1].split('_')[0])):
    size=int(f.split('size_')[1].split('_')[0])
    pf=pd.read_csv(f); base=pf[pf.circuit_type=='baseline'].iloc[0]
    opt=pf[pf.circuit_type=='optimized'].copy(); opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]; mem=sel.structures.split(',')
    pv=p_ccs.p_PD_FAKE.get(size,float('nan'))
    print("\n=== FAKE size %2d | CCS(top-N) p=%.4f | Pareto: bias %+.1f%% CCS %.3f ==="%(size,pv,sel.dB,sel.circuit_score))
    st=sorted(set(mem)&STRIA,key=lambda x:rk[x])
    print("   striatal/pallidal: %s" % (", ".join("%s(r%d)"%(s,rk[s]) for s in st) or "NONE"))
    print("   SNc %s | VTA %s | CP %s | NAcc %s" % tuple('Y' if x in mem else 'n' for x in
        ['Substantia_nigra_compact_part','Ventral_tegmental_area','Caudoputamen','Nucleus_accumbens']))
    if size in (6,26):
        print("   members:")
        for s in sorted(mem,key=lambda x:rk[x]): print("      r%-4d %-44s %s"%(rk[s],s[:44],anno.get(s,'?')))
