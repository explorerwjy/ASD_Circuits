#######################################
# Cell type related functions
#######################################

from ASD_Circuits import *
from scipy.stats import fisher_exact
#from scipy.stats import binom_test
from scipy.stats import hypergeom




#################################################
# Z1 Conversion
#################################################
def Z1Conversion(ExpMat, outname=None):
    """Z-score each gene (row) across samples (columns).

    Parameters
    ----------
    ExpMat : pd.DataFrame
        Expression matrix (genes x samples).
    outname : str, optional
        If provided, save the result to this CSV path.

    Returns
    -------
    pd.DataFrame
        Z-scored matrix (same shape as ExpMat).
    """
    z1_data = []
    for idx in ExpMat.index:
        row = ExpMat.loc[idx].values
        z1_data.append(ZscoreConverting(row))
    Z1Mat = pd.DataFrame(data=z1_data, index=ExpMat.index, columns=ExpMat.columns)
    if outname is not None:
        Z1Mat.to_csv(outname)
    return Z1Mat


#################################################
# Allen SC data related Functions
#################################################
ABC_ALL_Class = ['01 IT-ET Glut', '02 NP-CT-L6b Glut', '03 OB-CR Glut',
       '04 DG-IMN Glut', '05 OB-IMN GABA', '06 CTX-CGE GABA',
       '07 CTX-MGE GABA', '08 CNU-MGE GABA', '09 CNU-LGE GABA',
       '11 CNU-HYa GABA', '10 LSX GABA', '12 HY GABA', '13 CNU-HYa Glut',
       '14 HY Glut', '15 HY Gnrh1 Glut', '16 HY MM Glut', '17 MH-LH Glut',
       '18 TH Glut', '19 MB Glut', '20 MB GABA', '21 MB Dopa',
       '22 MB-HB Sero', '23 P Glut', '24 MY Glut', '25 Pineal Glut',
       '26 P GABA', '27 MY GABA', '28 CB GABA', '29 CB Glut',
       '30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular',
       '34 Immune']
ABC_nonNEUR = ['30 Astro-Epen', '31 OPC-Oligo', '32 OEC', '33 Vascular', '34 Immune']
def CompareCT_ABC(Bias1, Bias2, name1="1", name2="2", name0 = "",SuperClusters=ABC_ALL_Class, xmin=0, xmax=0, savefig=""):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["EFFECT_{}".format(name1)] - res["EFFECT_{}".format(name2)]
    #res.to_csv("./test/{}_{}_vs_{}.csv".format(name0, name1, name2))
    res = res[res["class_id_label_{}".format(name1)].isin(SuperClusters)]
    
    columns_to_drop_na = ["EFFECT_{}".format(name1), "EFFECT_{}".format(name2)]
    res = res.dropna(subset=columns_to_drop_na)

    print(pearsonr(res["EFFECT_{}".format(name1)].values, res["EFFECT_{}".format(name2)].values))
    print(spearmanr(res["EFFECT_{}".format(name1)].values, res["EFFECT_{}".format(name2)].values))

    idx = 0
    N_col = 4
    N_rows = math.ceil(len(SuperClusters)/N_col)
    #print(len(SuperClusters))
    if len(SuperClusters) > 22:
        fig = plt.figure(figsize=(12, 20), constrained_layout=False)
    else:
        fig = plt.figure(figsize=(12, 16), constrained_layout=False)
    spec = fig.add_gridspec(ncols=N_col, nrows=N_rows)
    if xmin ==0 and xmax==0:
        xmin = min(min(res["EFFECT_{}".format(name1)].values), min(res["EFFECT_{}".format(name2)].values)) * 1.1
        xmax = max(max(res["EFFECT_{}".format(name1)].values), max(res["EFFECT_{}".format(name2)].values)) * 1.1
    for a in range(N_rows):
        for b in range(N_col):
            ax0 = fig.add_subplot(spec[a, b])
            tmp = res[res["class_id_label_{}".format(name1)]==SuperClusters[idx]]
            ax0.scatter(tmp["EFFECT_{}".format(name1)].values, tmp["EFFECT_{}".format(name2)].values, s=10, )
            ax0.set_title(SuperClusters[idx])
            ax0.plot([xmin,xmax],[xmin,xmax], color="grey", ls="dashed")
            ax0.plot([0,0],[xmin,xmax], color="grey", ls="dotted")
            ax0.plot([xmin,xmax],[0,0], color="grey", ls="dotted")
            ax0.set_xlim((xmin, xmax))
            ax0.set_ylim((xmin, xmax))
            if idx >= len(SuperClusters)-1:
                break
            idx += 1
    plt.tight_layout()
    fig.text(0.5, -0.01, '{} Bias'.format(name1), ha='center', fontsize=20)
    fig.text(-0.02, 0.5, '{} Bias'.format(name2), va='center', rotation='vertical', fontsize=20)
    if savefig != "":
        plt.savefig(savefig, bbox_inches="tight")


#################################################
# Go terms related Functions
#################################################
Go2Uniprot = pk.load(open("/home/jw3514/Work/CellType_Psy/dat3/Goterms/Go2Uniprot.pk", 'rb'))
Uniprot2Entrez = pk.load(open("/home/jw3514/Work/CellType_Psy/dat3/Goterms/Uniprot2Entrez.pk", 'rb'))

def GetALLGo(go, GoID):
    Root = go[GoID]
    all_go = Root.get_all_children()
    all_go.add(GoID)
    return all_go
def GetGeneOfGo2(go, GoID, Go2Uniprot=Go2Uniprot):
    goset = GetALLGo(go, GoID)
    Total_Genes = set([])
    for i, tmpgo in enumerate(goset):
        #print(i, tmpgo)
        if tmpgo in Go2Uniprot:
            geneset = set([Uniprot2Entrez.get(x, 0) for x in Go2Uniprot[tmpgo]])
            Total_Genes = Total_Genes.union(geneset)
    return Total_Genes



# MERFISH Related Functions
def FixSubiculum(DF):
    X = DF.loc["Subiculum_dorsal_part"]
    Y = DF.loc["Subiculum_ventral_part"]
    Z = [(X[0]+Y[0])/2, "Hippocampus", 214]
    DF.loc["Subiculum"] = Z
    DF = DF.drop(["Subiculum_dorsal_part", "Subiculum_ventral_part"])
    return DF


def GetExpQ(pvalues):
    sorted_pvalues = np.sort(pvalues)
    expected = np.linspace(0, 1, len(sorted_pvalues), endpoint=False)[1:]
    expected_quantiles = -np.log10(expected)
    observed_quantiles = -np.log10(sorted_pvalues[1:])
    return expected_quantiles, observed_quantiles


def add_class(BiasDF, ClusterAnn):
    for cluster, row in BiasDF.iterrows():
        BiasDF.loc[cluster, "class_id_label"] = ClusterAnn.loc[cluster, "class_id_label"]
        BiasDF.loc[cluster, "CCF_broad.freq"] = ClusterAnn.loc[cluster, "CCF_broad.freq"]
        BiasDF.loc[cluster, "CCF_acronym.freq"] = ClusterAnn.loc[cluster, "CCF_acronym.freq"]
        BiasDF.loc[cluster, "v3.size"] = ClusterAnn.loc[cluster, "v3.size"]
        BiasDF.loc[cluster, "v2.size"] = ClusterAnn.loc[cluster, "v2.size"]
    return BiasDF


####################################################
# Specificity Score Validation Functions
####################################################
def MergeBiasDF(Bias1, Bias2, name1="1", name2="2"):
    res = Bias1.join(Bias2, how = 'inner', lsuffix="_{}".format(name1), rsuffix="_{}".format(name2))
    res["Diff"] = res["EFFECT_{}".format(name1)] - res["EFFECT_{}".format(name2)]
    return res 
    #res.to_csv("./test/{}_{}_vs_{}.csv".format(name0, name1, name2))

def PlotBiasContrast_v2(MergeDF, name1, name2, dataset="Human", title=""):
    if dataset == "HumanCT":
        NEUR = MergeDF[MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        NonNEUR = MergeDF[~MergeDF["Supercluster_{}".format(name1)].isin(Neurons)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7
        
        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f}\nR_NonNEUR = {r_nonNeur:.2f}",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return
    elif dataset == "MouseCT":
        NEUR = MergeDF[~MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        NonNEUR = MergeDF[MergeDF["class_id_label_{}".format(name1)].isin(ABC_nonNEUR)]
        X_NEUR, Y_NEUR = NEUR["EFFECT_{}".format(name1)], NEUR["EFFECT_{}".format(name2)]
        X_NonNEUR, Y_NonNEUR = NonNEUR["EFFECT_{}".format(name1)], NonNEUR["EFFECT_{}".format(name2)]
        fig, ax = plt.subplots(dpi=300, figsize=(5, 5))

        ax.scatter(X_NEUR, Y_NEUR, s=40, color="blue", edgecolor='black', alpha=0.7)
        ax.scatter(X_NonNEUR, Y_NonNEUR, s=40, color="red", edgecolor='black', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Spearman correlation
        r_Neur, p_Neur = spearmanr(X_NEUR, Y_NEUR)
        r_nonNeur, p_nonNeur = spearmanr(X_NonNEUR, Y_NonNEUR)
        
        xmin = np.min(X_NEUR)
        xmax = np.max(X_NEUR)
        ymax = np.max(Y_NEUR)
        # Adjust text position to avoid overlap
        text_x = xmin * 0.9 if r_Neur < 0 else xmin * 0.7
        text_y = ymax * 0.7
        
        ax.text(text_x, text_y, s=f"R_NEUR = {r_Neur:.2f}\nR_NonNEUR = {r_nonNeur:.2f}",
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        
        ax.set_xlabel(name1, fontsize=12, fontweight='bold')
        ax.set_ylabel(name2, fontsize=12, fontweight='bold')
        
        # Style adjustments
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(fontsize=10, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        return

