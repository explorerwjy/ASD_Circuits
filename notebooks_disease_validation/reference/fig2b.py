import sys,os,warnings; sys.path.insert(1,'src'); warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ASD_Circuits import LoadGeneINFO, Fil2Dict, ScoreCircuit_SI_Joint
_,_,S2E,E2S=LoadGeneINFO()
Z2=pd.read_parquet('dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet')
Info=pd.read_csv('dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv',index_col=0)
SIZES=list(range(6,81)); cols=np.array(Z2.columns); M=Z2.values
idx={g:i for i,g in enumerate(Z2.index)}
sw=pd.read_csv('dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv',header=None)
SIB=np.array([idx[g] for g in sw[0].astype(int) if g in idx])
prof=lambda ss:[ScoreCircuit_SI_Joint(ss[:N],Info) for N in SIZES]
def null_band(n_genes,nsim=4000,seed=42):
    rng=np.random.default_rng(seed)
    draws=[rng.choice(SIB,size=n_genes,replace=False) for _ in range(nsim)]
    return np.array(Parallel(n_jobs=12)(delayed(
        lambda dr: prof(cols[np.argsort(-np.nanmean(M[dr],axis=0))]))(d) for d in draws))
sets={}
real=[int(g) for g in Fil2Dict('dat/Genetics/GeneWeights/PD_HighConf_DA.gw')]
sets['PD_HighConf_DA (19, curated)']=(np.array(prof(Z2.loc[real].mean(axis=0).sort_values(ascending=False).index.values)),len(real))
fk=pd.read_csv('results/PD_HD_validation/PD_FAKE_cherrypicked_bias.csv',index_col=0)
nfake=17
sets['PD_FAKE cherry-picked (17)']=(np.array(prof(fk.sort_values('EFFECT',ascending=False).index.values)),nfake)
out={'N':SIZES}
fig,ax=plt.subplots(figsize=(11,6),dpi=150,facecolor='none'); fig.patch.set_alpha(0); ax.patch.set_alpha(0)
colors={'PD_HighConf_DA (19, curated)':'#1f77b4','PD_FAKE cherry-picked (17)':'#d62728'}
for lbl,(o,n) in sets.items():
    nb=null_band(n)
    med=np.median(nb,axis=0); lo=np.percentile(nb,15.9,axis=0); hi=np.percentile(nb,84.1,axis=0)
    p=np.array([(np.sum(nb[:,j]>=o[j])+1)/(nb.shape[0]+1) for j in range(len(SIZES))])
    out[f'CCS_{lbl.split()[0]}']=o; out[f'p_{lbl.split()[0]}']=p
    ax.plot(SIZES,o,color=colors[lbl],lw=2,marker='o',ms=3,label=f"{lbl}")
    ax.fill_between(SIZES,lo,hi,color=colors[lbl],alpha=0.13,lw=0)
    ax.plot(SIZES,med,color=colors[lbl],ls='--',lw=1,alpha=.6)
    sig=np.array(SIZES)[p<0.05]
    ax.plot(sig,o[p<0.05],'o',ms=6,mfc='none',mec=colors[lbl],mew=1.6)
    print("%-32s n=%2d | sig sizes p<0.05: %d/%d | best p=%.4f at N=%d"
          % (lbl,n,(p<0.05).sum(),len(p),p.min(),SIZES[int(p.argmin())]))
ax.set_xlabel("Number of top-ranked structures (circuit size)",fontsize=13)
ax.set_ylabel("Circuit Connectivity Score",fontsize=13)
ax.set_title("CCS vs circuit size, with sibling-derived null bands (shaded 15.9–84.1 pct)",fontsize=12)
ax.grid(alpha=.3,ls='--'); ax.legend(fontsize=10,loc='upper right')
ax.text(.99,.02,"open circles = p < 0.05 vs sibling null",transform=ax.transAxes,ha='right',fontsize=9,style='italic')
plt.tight_layout()
os.makedirs('results/PD_HD_validation/figures',exist_ok=True)
plt.savefig('results/PD_HD_validation/figures/CCS_vs_size_real_vs_fake_siblingnull.png',transparent=True,dpi=300,bbox_inches='tight')
pd.DataFrame(out).to_csv('results/PD_HD_validation/exploratory/CCS_profile_real_vs_fake.csv',index=False)
print("\nwrote figure + CCS_profile_real_vs_fake.csv")
