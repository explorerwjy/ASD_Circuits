import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, Fil2Dict, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
base=[int(g) for g in Fil2Dict('dat/Genetics/GeneWeights/PD_HighConf_DA.gw')]
drop=[int(S2E[s]) for s in ['DNAJC6','ATP13A2','GBA','SYNJ1','PLA2G6']]
fake=[g for g in base if g not in drop]+[int(S2E[s]) for s in ['PDE8B','ADCY5','GNAL']]
obs=Z2.loc[fake].mean(axis=0).sort_values(ascending=False)
d=pd.DataFrame({'EFFECT':obs,'Rank':np.arange(1,len(obs)+1),
                'REGION':[anno.get(i,'?') for i in obs.index]})
d.index.name='Structure'
out='results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv'
d.to_csv(out); print("wrote",out,d.shape)
print("CP rank",int(d.loc['Caudoputamen','Rank']),"| SNc rank",int(d.loc['Substantia_nigra_compact_part','Rank']))
