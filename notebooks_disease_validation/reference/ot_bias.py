import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, MouseSTR_AvgZ_Weighted, STR2Region
_,_,S2E,E2S=LoadGeneINFO(); anno=STR2Region()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
t=pd.read_csv('dat/Genetics/OpenTargets/OT-MONDO_0021095-associated-targets-8_12_2026-v26_06.tsv',sep='\t')
t['entrez']=[S2E.get(s) for s in t['symbol']]; t=t.dropna(subset=['entrez'])
t['entrez']=t.entrez.astype(int); t=t[t.entrez.isin(Z2.index)]
sel=t[t.globalScore>=0.6]
w=dict(zip(sel.entrez,sel.globalScore))
b=MouseSTR_AvgZ_Weighted(Z2,w)['EFFECT'].reindex(Z2.columns).sort_values(ascending=False)
d=pd.DataFrame({'EFFECT':b,'Rank':np.arange(1,len(b)+1),'REGION':[anno.get(i,'?') for i in b.index]})
d.index.name='Structure'
d.to_csv('results/PD_HD_validation/exploratory/PD_OpenTargets_cut0.6_bias.csv')
print("OT-PD @0.6: %d genes | floor at rank 50 = %.3f" % (len(w),d['EFFECT'].iloc[49]))
print("\nbasal-ganglia components in the OT-PD profile:")
for s in ['Substantia_nigra_compact_part','Substantia_nigra_reticular_part','Caudoputamen',
          'Nucleus_accumbens','Globus_pallidus_external_segment','Globus_pallidus_internal_segment',
          'Subthalamic_nucleus','Ventral_medial_nucleus_of_the_thalamus','Primary_motor_area',
          'Olfactory_tubercle','Fundus_of_striatum']:
    r=int(d.loc[s,'Rank']); print("   rank %3d  EFFECT %6.3f  %-46s%s" % (r,d.loc[s,'EFFECT'],s,
          "  <= in top-50 candidate pool" if r<=50 else ""))
