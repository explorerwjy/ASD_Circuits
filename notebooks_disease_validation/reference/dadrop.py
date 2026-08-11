import sys, re, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, Fil2Dict, MouseCT_AvgZ_Weighted, MouseSTR_AvgZ_Weighted
from disease_validation import load_ground_truth, recovery_stats, leave_one_out_recovery
_,_,S2E,E2S=LoadGeneINFO()
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
v2v3=pd.read_csv('dat/ISH_MERFISH_Gene_CorssSTR_Corr.v3.csv',index_col='Genes')['V2_V3_CT_Corr']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
core=load_ground_truth('config/disease_validation_ground_truth.yaml')['structures']['parkinson']['core']
g=lambda n:set(int(x) for x in Fil2Dict(f'dat/Genetics/GeneWeights/{n}.gw'))
U29=g('PD_Primary')|g('PD_Sens_DA')|g('PD_Sens_Atypical')
E=lambda s:int(S2E[s])
dn=lambda gs:{x:max(v2v3.loc[x],0)**2 for x in gs if x in CT.index and x in v2v3.index}
def ev(gs,lbl):
    ct=recovery_stats(MouseCT_AvgZ_Weighted(CT,dn(gs)),DOPA)
    st=recovery_stats(MouseSTR_AvgZ_Weighted(Z2,{x:1.0 for x in gs if x in Z2.index}),core)
    print("%-42s n=%2d | CT AUROC %.3f | STR AUROC %.3f" % (lbl,len([x for x in gs if x in CT.index]),ct['auroc'],st['auroc']))
print("Mendelian union (29) and progressive removal of dopamine-pathway genes:")
ev(U29,"all 29 (incl TH,SLC6A3,DDC,GCH1,SPR)")
ev(U29-{E('TH'),E('SLC6A3')},"minus the 2 hardest markers (TH,SLC6A3)")
ev(U29-{E('TH'),E('SLC6A3'),E('DDC')},"minus TH,SLC6A3,DDC")
ev(U29-{E(x) for x in ['TH','SLC6A3','DDC','GCH1','SPR']},"minus all 5 (= PD_Sens_Atypical)")
ev(U29-{E('GCH1'),E('SPR')},"minus only GCH1,SPR (keep TH,SLC6A3,DDC)")
print("\nLeave-one-out at cell-type level on the 29-gene union (most negative = biggest driver):")
loo=leave_one_out_recovery(CT,dn(U29),DOPA,E2S,lambda e,w:MouseCT_AvgZ_Weighted(e,w))
print(loo.head(8)[['dropped_symbol','auroc','delta_auroc']].to_string(index=False))
