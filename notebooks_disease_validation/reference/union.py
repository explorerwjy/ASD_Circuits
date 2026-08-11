import sys, re, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import LoadGeneINFO, Fil2Dict, MouseSTR_AvgZ_Weighted, MouseCT_AvgZ_Weighted
from disease_validation import load_ground_truth, recovery_stats
_,_,S2E,E2S=LoadGeneINFO()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
v2v3=pd.read_csv('dat/ISH_MERFISH_Gene_CorssSTR_Corr.v3.csv',index_col='Genes')['V2_V3_CT_Corr']
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns], index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
GT=load_ground_truth('config/disease_validation_ground_truth.yaml')
core=GT['structures']['parkinson']['core']
g=lambda n: set(int(x) for x in Fil2Dict(f'dat/Genetics/GeneWeights/{n}.gw'))
P,DA,AT,GW = g('PD_Primary'),g('PD_Sens_DA'),g('PD_Sens_Atypical'),g('PD_GWAS_L2G')
DAonly={int(S2E[s]) for s in ['TH','SLC6A3','DDC','GCH1','SPR'] if S2E.get(s)}
sets={
 'PD_Primary (15)':P,
 'PD_Sens_Atypical (24)':AT,
 'PD_GWAS_L2G (40)':GW,
 'UNION all four':P|DA|AT|GW,
 'UNION minus DA markers':(P|DA|AT|GW)-DAonly,
 'UNION Mendelian only (Prim+DA+Atyp)':P|DA|AT,
 'UNION Mendelian noDA (=Atypical)':(P|DA|AT)-DAonly,
}
print("%-38s %5s | %-22s | %-22s" % ("gene set","n","STRUCTURE vs PD core","CELL-TYPE vs Dopa"))
print("%-38s %5s | %8s %12s | %8s %12s" % ("","","AUROC","p_MannWh","AUROC","p_MannWh"))
print("-"*96)
for lbl,gs in sets.items():
    gs_s={x:1.0 for x in gs if x in Z2.index}
    b=MouseSTR_AvgZ_Weighted(Z2,gs_s); s1=recovery_stats(b,core)
    dn={x:1.0*(max(v2v3.loc[x],0)**2) for x in gs if x in CT.index and x in v2v3.index}
    b2=MouseCT_AvgZ_Weighted(CT,dn); s2=recovery_stats(b2,DOPA)
    print("%-38s %5d | %8.3f %12.2e | %8.3f %12.2e" % (lbl,len(gs_s),s1['auroc'],s1['p_mannwhitney'],s2['auroc'],s2['p_mannwhitney']))
u=(P|DA|AT|GW)
print("\nUNION composition: %d unique genes (Primary %d + DA %d + Atyp-extra %d + GWAS %d, overlap %d)"
      % (len(u),len(P),len(DAonly),len(AT-P),len(GW),len(P|DA|AT)&set() if False else len((P|DA|AT)&GW)))
print("genes shared between Mendelian and GWAS tiers: %s" % sorted(E2S.get(x,str(x)) for x in (P|DA|AT)&GW))
