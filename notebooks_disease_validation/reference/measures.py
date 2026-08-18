import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import ScoreCircuit_SI_Joint, ScoreCircuit_NEdges
W=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv',index_col=0)
I=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
d=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
c13=list(d.index[:13])
def w_mean_nonzero(S,W):
    sub=W.loc[S,S].values; nz=sub[sub>0]; return nz.mean() if len(nz) else 0
def w_mean_all(S,W):
    sub=W.loc[S,S].values; n=len(S)*(len(S)-1); return sub.sum()/n
def w_sum(S,W): return W.loc[S,S].values.sum()
def w_log_mean(S,W):
    sub=W.loc[S,S].values; return np.log1p(sub).sum()/(len(S)*(len(S)-1))
MEAS={'SI (current)':lambda S:ScoreCircuit_SI_Joint(S,I),
      'NEdges (density)':lambda S:ScoreCircuit_NEdges(S,W),
      'mean weight (nonzero pairs)':lambda S:w_mean_nonzero(S,W),
      'mean weight (all pairs)':lambda S:w_mean_all(S,W),
      'sum of weights':lambda S:w_sum(S,W),
      'mean log1p(weight)':lambda S:w_log_mean(S,W)}
print("Does adding a structure to the curated top-13 help under each measure?")
print("%-30s %10s %10s %10s %10s" % ("measure","top13","+CP","+STN","+SNr"))
for name,fn in MEAS.items():
    b=fn(c13)
    vals=[fn(c13+[x]) for x in ['Caudoputamen','Subthalamic_nucleus','Substantia_nigra_reticular_part']]
    arrow=lambda v: ("+" if v>b else "-")
    print("%-30s %10.4f %9.4f%s %9.4f%s %9.4f%s" % (name,b,vals[0],arrow(vals[0]),vals[1],arrow(vals[1]),vals[2],arrow(vals[2])))
print("\nwhy: CP's edges to the 13-member circuit")
sub_w=W.loc[c13+['Caudoputamen'],c13+['Caudoputamen']]
cp_pairs=list(W.loc['Caudoputamen',c13])+list(W.loc[c13,'Caudoputamen'])
print("   CP pairs with circuit: %d of %d nonzero | max weight %.3f | mean over all pairs %.4f"
      % (sum(1 for x in cp_pairs if x>0),len(cp_pairs),max(cp_pairs),np.mean(cp_pairs)))
inner=W.loc[c13,c13].values
print("   circuit-internal:      %d of %d nonzero | mean over all pairs %.4f"
      % ((inner>0).sum(),len(c13)*(len(c13)-1),inner.sum()/(len(c13)*(len(c13)-1))))
