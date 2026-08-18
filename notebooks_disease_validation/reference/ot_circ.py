import sys,glob,warnings,collections; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
BG={'Substantia_nigra_compact_part','Substantia_nigra_reticular_part','Subthalamic_nucleus','Caudoputamen',
    'Nucleus_accumbens','Globus_pallidus_external_segment','Globus_pallidus_internal_segment',
    'Ventral_medial_nucleus_of_the_thalamus','Primary_motor_area','Secondary_motor_area',
    'Pedunculopontine_nucleus','Fundus_of_striatum','Olfactory_tubercle'}
d=pd.read_csv('results/PD_HD_validation/exploratory/PD_OpenTargets_cut0.6_bias.csv',index_col=0)
rk=d['Rank']
for f in sorted(glob.glob("results/CircuitSearch/PD_OpenTargets_cut06/pareto_fronts/*.csv"),
                key=lambda x:int(x.split('size_')[1].split('_')[0])):
    size=int(f.split('size_')[1].split('_')[0])
    p=pd.read_csv(f); base=p[p.circuit_type=='baseline'].iloc[0]; opt=p[p.circuit_type=='optimized'].copy()
    opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]; mem=sel.structures.split(',')
    print("\n=== OT-PD size %d @ ASD-matched point: bias %+.1f%%  CCS %.3f (baseline %.3f) ==="
          % (size,sel.dB,sel.circuit_score,base.circuit_score))
    print("  striatal/pallidal members: %s" % (", ".join("%s(r%d)"%(s,rk[s]) for s in sorted(set(mem)&STRIA,key=lambda x:rk[x])) or "NONE"))
    print("  basal-ganglia/motor components present:")
    for s in sorted(BG&set(mem),key=lambda x:rk[x]): print("     r%-4d %s" % (rk[s],s))
    print("  absent from BG set:", ", ".join(sorted(BG-set(mem))) or "none")
    print("  regions:",dict(collections.Counter(anno.get(s,'?') for s in mem)))
