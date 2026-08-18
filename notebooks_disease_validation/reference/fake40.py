import sys,glob,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd
from ASD_Circuits import STR2Region
anno=STR2Region(); STRIA={k for k,v in anno.items() if v in ('Striatum','Pallidum')}
d=pd.read_csv('results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv',index_col=0); rk=d['Rank']
for f in sorted(glob.glob("results/CircuitSearch/PD_FAKE_cherrypicked/pareto_fronts/*.csv"),
                key=lambda x:int(x.split('size_')[1].split('_')[0])):
    size=int(f.split('size_')[1].split('_')[0])
    p=pd.read_csv(f); base=p[p.circuit_type=='baseline'].iloc[0]
    opt=p[p.circuit_type=='optimized'].copy(); opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    sel=opt.iloc[(opt.dB-(-20.0)).abs().argmin()]; mem=set(sel.structures.split(','))
    hits=sorted(mem&STRIA,key=lambda x:rk[x])
    print("size %2d | bias %+.1f%% CCS %.3f | SNc %s VTA %s | CP %s NAcc %s | striatal members: %s"
      % (size,sel.dB,sel.circuit_score,
         'Y' if 'Substantia_nigra_compact_part' in mem else 'n',
         'Y' if 'Ventral_tegmental_area' in mem else 'n',
         'YES' if 'Caudoputamen' in mem else 'no',
         'YES' if 'Nucleus_accumbens' in mem else 'no',
         ", ".join("%s(r%d)"%(s,rk[s]) for s in hits) or "NONE"))
    if size==40:
        print("   full size-40 circuit, by bias rank:")
        for s in sorted(mem,key=lambda x:rk[x]):
            print("      r%-4d %-46s %s" % (rk[s],s[:46],anno.get(s,'?')))
