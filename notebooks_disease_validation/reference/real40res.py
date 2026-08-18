import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
rk=pd.Series(np.arange(1,len(d)+1),index=d.index)
p=pd.read_csv("results/CircuitSearch/PD_HighConf_DA/pareto_fronts/PD_HighConf_DA_size_40_pareto_front.csv")
base=p[p.circuit_type=='baseline'].iloc[0]; opt=p[p.circuit_type=='optimized'].copy()
opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]; mem=sel.structures.split(',')
print("REAL 19-gene set, size 40 @ ASD-matched point: bias %+.1f%%  CCS %.3f (baseline %.3f)"
      % (sel.dB,sel.circuit_score,base.circuit_score))
BG={'Substantia_nigra_compact_part','Substantia_nigra_reticular_part','Subthalamic_nucleus',
    'Caudoputamen','Nucleus_accumbens','Globus_pallidus_external_segment','Globus_pallidus_internal_segment',
    'Ventral_medial_nucleus_of_the_thalamus','Primary_motor_area','Secondary_motor_area','Pedunculopontine_nucleus'}
print("  basal-ganglia / motor components present:")
for s in sorted(BG&set(mem),key=lambda x:rk[x]): print("     r%-4d %s" % (rk[s],s))
print("  absent:", ", ".join(sorted(BG-set(mem))))
import collections
print("  region composition:",dict(collections.Counter(anno.get(s,'?') for s in mem)))
