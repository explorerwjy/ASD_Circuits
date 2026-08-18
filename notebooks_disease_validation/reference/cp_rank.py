import sys,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, Fil2Dict
_,_,S2E,E2S=LoadGeneINFO()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
w=Fil2Dict('dat/Genetics/GeneWeights/PD_HighConf_DA.gw')
genes=[int(g) for g in w]
DA={'TH','SLC6A3','DDC','GCH1','SPR'}
rows=[]
for g in genes:
    prof=Z2.loc[g]; rk=prof.rank(ascending=False)
    rows.append({'gene':E2S.get(g,str(g)),
                 'Z2_at_CP':prof['Caudoputamen'],
                 'CP_rank_within_gene':int(rk['Caudoputamen']),
                 'Z2_at_SNc':prof['Substantia_nigra_compact_part'],
                 'Z2_at_VTA':prof['Ventral_tegmental_area'],
                 'DA_pathway':E2S.get(g,'') in DA})
d=pd.DataFrame(rows).sort_values('Z2_at_CP',ascending=False)
print("PD_HighConf_DA (19 genes) ranked by bias toward CAUDOPUTAMEN")
print("%-9s %10s %12s | %9s %9s %s" % ("gene","Z2 at CP","CP rank/213","Z2 SNc","Z2 VTA","DA-pathway"))
print("-"*72)
for _,r in d.iterrows():
    print("%-9s %10.3f %12d | %9.3f %9.3f %s" % (r.gene,r.Z2_at_CP,r.CP_rank_within_gene,
          r.Z2_at_SNc,r.Z2_at_VTA,"yes" if r.DA_pathway else ""))
print("-"*72)
print("%-9s %10.3f %12s | %9.3f %9.3f" % ("MEAN",d.Z2_at_CP.mean(),"",d.Z2_at_SNc.mean(),d.Z2_at_VTA.mean()))
print("\npositive at CP: %d/19   |   positive at SNc: %d/19   |   positive at VTA: %d/19"
      % ((d.Z2_at_CP>0).sum(),(d.Z2_at_SNc>0).sum(),(d.Z2_at_VTA>0).sum()))
b=pd.read_csv('results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv',index_col=0)
print("\nSet-level EFFECT at Caudoputamen: %.3f  (rank %d/213)"
      % (b.loc['Caudoputamen','EFFECT'], int(b['EFFECT'].rank(ascending=False)['Caudoputamen'])))
d.to_csv('results/PD_HD_validation/bundle/PD_gene_bias_at_caudoputamen.csv',index=False)
print("wrote bundle/PD_gene_bias_at_caudoputamen.csv")
