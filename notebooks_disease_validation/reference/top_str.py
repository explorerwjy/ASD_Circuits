import sys, warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd
from ASD_Circuits import STR2Region
from disease_validation import load_ground_truth
anno=STR2Region(); GT=load_ground_truth('config/disease_validation_ground_truth.yaml')
core=set(GT['structures']['parkinson']['core']); braak=set(GT['structures']['parkinson']['braak_early'])
def mark(s): return "  <-- PD CORE" if s in core else ("  <-- Braak-early" if s in braak else "")
for s in ["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G"]:
    d=pd.read_csv(f"results/STR_ISH/{s}_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
    nsig=int((d['q-value']<0.10).sum())
    print(f"\n=== {s}  (top 15 of 213; {nsig} structures at q<0.10) ===")
    for i,(k,r) in enumerate(d.head(15).iterrows(),1):
        print("%2d. %7.3f  q=%.3f  %-52s %-16s%s" % (i, r['EFFECT'], r['q-value'], k[:52], anno.get(k,'?'), mark(k)))
print("\n\n=== Where the 13 pre-registered PD core structures rank in each set ===")
print("%-50s %10s %10s %10s %10s" % ("structure","PD_Primary","Sens_DA","Sens_Atyp","GWAS_L2G"))
ranks={}
for s in ["PD_Primary","PD_Sens_DA","PD_Sens_Atypical","PD_GWAS_L2G"]:
    d=pd.read_csv(f"results/STR_ISH/{s}_bias_addP_random.csv",index_col=0)
    ranks[s]=d['EFFECT'].rank(ascending=False)
for st in GT['structures']['parkinson']['core']:
    print("%-50s %10d %10d %10d %10d" % (st[:50], ranks['PD_Primary'][st], ranks['PD_Sens_DA'][st],
          ranks['PD_Sens_Atypical'][st], ranks['PD_GWAS_L2G'][st]))
