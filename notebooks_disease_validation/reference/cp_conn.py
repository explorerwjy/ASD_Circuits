import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import STR2Region
anno=STR2Region()
W=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv',index_col=0)
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
print("=== does the connectome contain the thalamostriatal projections? ===")
for a in ['Parafascicular_nucleus','Ventral_medial_nucleus_of_the_thalamus',
          'Ventral_anterior_lateral_complex_of_the_thalamus','Central_lateral_nucleus_of_the_thalamus',
          'Mediodorsal_nucleus_of_thalamus','Subparafascicular_nucleus_parvicellular_part']:
    if a in W.index:
        print("   %-48s -> CP  weight %8.3f | CP -> it  %8.3f" % (a,W.loc[a,'Caudoputamen'],W.loc['Caudoputamen',a]))
print("\n=== CP's overall connectivity degree ===")
out=(W.loc['Caudoputamen']>0).sum(); inn=(W['Caudoputamen']>0).sum()
deg=pd.Series({s:(W.loc[s]>0).sum()+(W[s]>0).sum() for s in W.index}).sort_values(ascending=False)
print("   CP: %d outgoing, %d incoming, total degree %d  -> rank %d/213 by degree"
      % (out,inn,out+inn,list(deg.index).index('Caudoputamen')+1))
print("   median degree across all structures: %d" % deg.median())
print("\n   CP's top incoming partners (who projects TO caudoputamen):")
inc=W['Caudoputamen'][W['Caudoputamen']>0].sort_values(ascending=False)
for k,v in inc.head(12).items(): print("      %8.3f  %-44s %s" % (v,k[:44],anno.get(k,'?')))
print("\n=== the 13-structure circuit: which members are thalamic? ===")
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
c13=list(d.index[:13])
print("   ",[ (s,anno.get(s)) for s in c13 if anno.get(s)=='Thalamus'] or "NONE - no thalamic members at all")
print("\n=== would adding PF + CP together help? ===")
from ASD_Circuits import ScoreCircuit_SI_Joint
b=ScoreCircuit_SI_Joint(c13,Info)
for add in [['Parafascicular_nucleus'],['Caudoputamen'],['Parafascicular_nucleus','Caudoputamen'],
            ['Parafascicular_nucleus','Caudoputamen','Ventral_medial_nucleus_of_the_thalamus','Subthalamic_nucleus']]:
    print("   + %-62s CCS %.4f (%+.4f)" % (", ".join(add)[:62],ScoreCircuit_SI_Joint(c13+add,Info),ScoreCircuit_SI_Joint(c13+add,Info)-b))
