import sys,glob,os,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from ASD_Circuits import ScoreCircuit_SI_Joint, STR2Region
anno=STR2Region()
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
CORE={'Ventral_tegmental_area','Substantia_nigra_compact_part'}
RAPHE=lambda s:'raphe' in s.lower()
rows=[]
for ds,bias in [("PD_HighConf_DA","results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv"),
                ("PD_HighConf_DA_SibMut","results/STR_ISH/PD_HighConf_DA_SibMut_bias_addP_sibling.csv")]:
    d=pd.read_csv(bias,index_col=0).sort_values("EFFECT",ascending=False)
    rank=pd.Series(np.arange(1,len(d)+1),index=d.index)
    for f in sorted(glob.glob(f"results/CircuitSearch/{ds}/pareto_fronts/*pareto_front.csv")):
        size=int(f.split('size_')[1].split('_')[0])
        p=pd.read_csv(f)
        base=p[p.circuit_type=='baseline'].iloc[0]
        opt=p[p.circuit_type=='optimized'].copy()
        opt['dBias']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
        sel=opt.iloc[(opt.dBias-(-20.0)).abs().argmin()]      # ASD-matched -20% bias
        st=sel.structures.split(',')
        rows.append(dict(dataset=ds,size=size,base_CCS=base.circuit_score,base_bias=base.mean_bias,
            sel_CCS=sel.circuit_score,sel_bias=sel.mean_bias,dBias=sel.dBias,
            dCCS=100*(sel.circuit_score-base.circuit_score)/base.circuit_score,
            VTA='Ventral_tegmental_area' in st, SNc='Substantia_nigra_compact_part' in st,
            n_raphe=sum(RAPHE(s) for s in st),
            top5_kept=sum(s in st for s in d.index[:5]),
            structures=st))
r=pd.DataFrame(rows)
print("PARETO SELECTION AT THE ASD-MATCHED OPERATING POINT (bias sacrifice ~20%)")
print("%-22s %4s %8s %8s %7s %7s %5s %5s %6s %8s" % ("dataset","size","baseCCS","selCCS","dBias%","dCCS%","VTA","SNc","raphe","top5kept"))
for _,x in r.iterrows():
    print("%-22s %4d %8.3f %8.3f %7.1f %7.1f %5s %5s %6d %8d/5" % (x.dataset,x['size'],x.base_CCS,x.sel_CCS,
          x.dBias,x.dCCS,"YES" if x.VTA else "no","YES" if x.SNc else "no",x.n_raphe,x.top5_kept))
# core membership stability
print("\nSTABILITY — structures present in ALL 8 selected circuits:")
sets=[set(x) for x in r.structures]
common=set.intersection(*sets)
d0=pd.read_csv("results/STR_ISH/PD_HighConf_DA_bias_addP_random.csv",index_col=0).sort_values("EFFECT",ascending=False)
rk=pd.Series(np.arange(1,len(d0)+1),index=d0.index)
for s in sorted(common,key=lambda x:rk[x]): print("   rank %3d  bias %6.3f  %-42s %s" % (rk[s],d0.loc[s,'EFFECT'],s[:42],anno.get(s,'?')))
print("   (%d structures common to all 8 circuits)" % len(common))
r.drop(columns=['structures']).to_csv('results/tables/PD_circuit_pareto_summary.csv',index=False)
pd.DataFrame({'structure':sorted(common,key=lambda x:rk[x]),'rank':[int(rk[s]) for s in sorted(common,key=lambda x:rk[x])],
              'bias':[d0.loc[s,'EFFECT'] for s in sorted(common,key=lambda x:rk[x])],
              'region':[anno.get(s,'?') for s in sorted(common,key=lambda x:rk[x])]}).to_csv('results/tables/PD_circuit_core_consensus.csv',index=False)
print("\nwrote results/tables/PD_circuit_pareto_summary.csv and PD_circuit_core_consensus.csv")
