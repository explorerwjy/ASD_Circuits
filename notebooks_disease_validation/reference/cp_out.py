import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd
from ASD_Circuits import STR2Region
anno=STR2Region()
W=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv',index_col=0)
print("PF -> CP weight: %.3f   |  CP -> PF: %.3f" % (W.loc['Parafascicular_nucleus','Caudoputamen'],
                                                     W.loc['Caudoputamen','Parafascicular_nucleus']))
print("\nEVERYTHING Caudoputamen projects TO (all %d):" % (W.loc['Caudoputamen']>0).sum())
for k,v in W.loc['Caudoputamen'][W.loc['Caudoputamen']>0].sort_values(ascending=False).items():
    print("   %8.3f  %-46s %s" % (v,k[:46],anno.get(k,'?')))
print("\nCanonical basal-ganglia output pathways — present in the connectome?")
for a,b in [('Caudoputamen','Globus_pallidus_external_segment'),
            ('Caudoputamen','Globus_pallidus_internal_segment'),
            ('Caudoputamen','Substantia_nigra_reticular_part'),
            ('Globus_pallidus_external_segment','Subthalamic_nucleus'),
            ('Subthalamic_nucleus','Substantia_nigra_reticular_part'),
            ('Substantia_nigra_reticular_part','Ventral_medial_nucleus_of_the_thalamus'),
            ('Ventral_medial_nucleus_of_the_thalamus','Primary_motor_area'),
            ('Primary_motor_area','Caudoputamen')]:
    w=W.loc[a,b]; print("   %-42s -> %-42s %8.3f %s" % (a[:42],b[:42],w,"" if w>0 else "  <== ABSENT"))
print("\nfor comparison, outgoing degree of some circuit members:")
for s in ['Substantia_nigra_compact_part','Ventral_tegmental_area','Dorsal_nucleus_raphe','Caudoputamen']:
    print("   %-34s out %3d  in %3d" % (s,(W.loc[s]>0).sum(),(W[s]>0).sum()))
