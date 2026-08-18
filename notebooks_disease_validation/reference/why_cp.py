import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import ScoreCircuit_SI_Joint
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
W=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv',index_col=0)
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
core13=list(d.index[:13])
print("TEST 1 — does adding CP to the real top-13 help or hurt CCS?")
b=ScoreCircuit_SI_Joint(core13,Info)
print("   top-13 alone                    CCS %.4f (n=%d)"%(b,len(core13)))
for extra in ['Caudoputamen','Nucleus_accumbens','Substantia_nigra_reticular_part','Subthalamic_nucleus']:
    c=core13+[extra]
    print("   + %-30s CCS %.4f  (%+.4f)"%(extra,ScoreCircuit_SI_Joint(c,Info),ScoreCircuit_SI_Joint(c,Info)-b))
print("\nTEST 2 — CP's actual connectivity to the 13 circuit members")
conn=[(s,W.loc['Caudoputamen',s],W.loc[s,'Caudoputamen'],Info.loc['Caudoputamen',s],Info.loc[s,'Caudoputamen']) for s in core13]
n_out=sum(1 for x in conn if x[1]>0); n_in=sum(1 for x in conn if x[2]>0)
print("   CP -> circuit: %d/13 connections | circuit -> CP: %d/13"%(n_out,n_in))
print("   mean info of CP's pairs with the circuit: %.3f"%np.nanmean([x[3] for x in conn]+[x[4] for x in conn]))
print("   mean info WITHIN the 13-structure circuit: %.3f"%(Info.loc[core13,core13].values[np.nonzero(Info.loc[core13,core13].values)].mean()))
print("   CP's connected partners in the circuit:",[x[0] for x in conn if x[1]>0 or x[2]>0] or "none")
print("\nTEST 3 — is the FAKE set's low CCS caused by CP, or by the whole profile?")
f=pd.read_csv('results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv',index_col=0)
ftop13=list(f.sort_values('EFFECT',ascending=False).index[:13])
print("   fake top-13 (contains CP? %s)  CCS %.4f"%('Caudoputamen' in ftop13,ScoreCircuit_SI_Joint(ftop13,Info)))
print("   real top-13                    CCS %.4f"%b)
print("   fake top-13 minus CP/NAcc      CCS %.4f"%ScoreCircuit_SI_Joint(
      [s for s in ftop13 if s not in ('Caudoputamen','Nucleus_accumbens')],Info))
print("   overlap of the two top-13 sets: %d/13"%len(set(ftop13)&set(core13)))
print("   in fake-13 but not real-13:",sorted(set(ftop13)-set(core13)))
