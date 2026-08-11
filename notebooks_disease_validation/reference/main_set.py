import os,sys,re,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import (LoadGeneINFO, Fil2Dict, MouseSTR_AvgZ_Weighted, MouseCT_AvgZ_Weighted,
                          ScoreCircuit_SI_Joint)
from disease_validation import (load_ground_truth, recovery_stats, leave_one_out_recovery,
                                nested_subset_recovery, recovery_null_aurocs, empirical_p)
_,_,S2E,E2S=LoadGeneINFO()
M="PD_HighConf_DA"
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
CT=pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
sub=lambda c: re.sub(r'^\d+\s+','',c).rsplit('_',1)[0]
S=pd.Series([sub(c) for c in CT.columns],index=CT.columns)
DOPA=[c for c in CT.columns if S[c]=='SNc-VTA-RAmb Foxa1 Dopa']
GT=load_ground_truth('config/disease_validation_ground_truth.yaml')
core=GT['structures']['parkinson']['core']
w=Fil2Dict(f'dat/Genetics/GeneWeights/{M}.gw'); wd=Fil2Dict(f'dat/Genetics/GeneWeights_DN/{M}.DN.gw')
print("=== %s leave-one-out at CELL-TYPE level (vs Dopa) ===" % M)
loo=leave_one_out_recovery(CT,wd,DOPA,E2S,lambda e,x:MouseCT_AvgZ_Weighted(e,x))
print(loo.head(6)[['dropped_symbol','auroc','delta_auroc']].to_string(index=False))
print("  full AUROC %.3f | worst-case single-gene drop leaves %.3f" % (
      recovery_stats(MouseCT_AvgZ_Weighted(CT,wd),DOPA)['auroc'], loo['auroc'].min()))
print("\n=== negative-control cross-test: does anything else hit the same targets? ===")
for n in [M,"IBD","HDL_C","T2D","hba1c","ASD_All","DDD_285_ExcludeASD"]:
    f=f"results/CT_Z2/{n}_bias_addP_random.csv"; nb=f"results/CT_Z2/null_bias/{n}_null_bias_random.parquet"
    if not os.path.exists(f): continue
    d=pd.read_csv(f,index_col=0); st=recovery_stats(d,DOPA)
    try: p=empirical_p(st['auroc'],recovery_null_aurocs(pd.read_parquet(nb),DOPA))
    except Exception: p=float('nan')
    print("   %-22s vs Dopa  AUROC %.3f  p_geneset %.4f" % (n,st['auroc'],p))
print("\n=== circuit sizes for SA search ===")
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
for src,lbl in [(f"results/STR_ISH/{M}_bias_addP_random.csv","random"),
                (f"results/STR_ISH/{M}_SibMut_bias_addP_sibling.csv","sibling")]:
    d=pd.read_csv(src,index_col=0)
    n10=int((d['q-value']<0.10).sum()); n05=int((d['q-value']<0.05).sum())
    print("   %s null: q<0.05 -> %d structures, q<0.10 -> %d" % (lbl,n05,n10))
r=pd.read_csv(f"results/STR_ISH/{M}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False).index.values
tn=np.arange(200,5,-1); ccs=np.array([ScoreCircuit_SI_Joint(r[:n],Info) for n in tn])
print("   CCS profile peak at N=%d" % tn[int(np.argmax(ccs))])
