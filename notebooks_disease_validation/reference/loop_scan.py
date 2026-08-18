import sys,glob,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
LOOP={'SNc':'Substantia_nigra_compact_part','SNr':'Substantia_nigra_reticular_part',
      'CP':'Caudoputamen','GPe':'Globus_pallidus_external_segment',
      'GPi':'Globus_pallidus_internal_segment','STN':'Subthalamic_nucleus',
      'VM-thal':'Ventral_medial_nucleus_of_the_thalamus','M1':'Primary_motor_area'}
for tag,pat in [("REAL PD_HighConf_DA","results/CircuitSearch/PD_HighConf_DA/pareto_fronts/*size_40*"),
                ("FAKE cherry-picked","results/CircuitSearch/PD_FAKE_cherrypicked/pareto_fronts/*size_40*")]:
    f=glob.glob(pat)[0]; p=pd.read_csv(f)
    base=p[p.circuit_type=='baseline'].iloc[0]
    opt=p[p.circuit_type=='optimized'].copy().sort_values('mean_bias',ascending=False).reset_index(drop=True)
    opt['dB']=100*(opt.mean_bias-base.mean_bias)/base.mean_bias
    print("\n===== %s, size 40 — loop components across all %d Pareto points =====" % (tag,len(opt)))
    rows=[]
    for i,r in opt.iterrows():
        mem=set(r.structures.split(','))
        present={k:(v in mem) for k,v in LOOP.items()}
        rows.append(dict(idx=i,dB=r.dB,CCS=r.circuit_score,n=sum(present.values()),**present))
    t=pd.DataFrame(rows)
    print("  points containing ALL 8 loop components: %d / %d" % ((t.n==8).sum(),len(t)))
    print("  max components in any single point: %d" % t.n.max())
    print("  per-component presence across the front:")
    for k in LOOP: print("     %-8s %3d/%d points" % (k,t[k].sum(),len(t)))
    best=t.loc[t.n.idxmax()]
    print("  richest point: idx %d (bias %+.1f%%, CCS %.3f) has %d/8 -> %s"
          % (best.idx,best.dB,best.CCS,best.n,[k for k in LOOP if best[k]]))
    print("  missing there: %s" % [k for k in LOOP if not best[k]])
