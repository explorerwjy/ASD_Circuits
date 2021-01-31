import pandas as pd
import csv 
import re
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os
import sys
import seaborn as sns
import random
import bisect
import collections
import scipy.stats 
import re
import statsmodels.api as sm
import statsmodels.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import binom_test
import itertools
#import mygene
import igraph as ig

plt.rcParams['figure.dpi'] = 80

ConnFil = "/Users/jiayao/Work/ASD_Circuits/dat/allen-mouse-conn/connectome-log.csv"
MajorBrainDivisions = "/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map" 


#####################################################################################
# Preprocess
#####################################################################################
def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : np.sort(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

def ZscoreConverting(values):
    real_values = [x for x in values if x==x]
    mean = np.mean(real_values)
    std = np.std(real_values)
    zscores = []
    for x in values:
        z = (x - mean)/std
        zscores.append(z)
    return np.array(zscores)

def ContGeneSet(prefix, outfil, Ntrail=100):
    res = []
    for i in range(1, Ntrail+1, 1):
        _genes = [int(x.strip()) for x in open(prefix+"/match-{}.txt".format(i), "rt")]
        res.extend(_genes)
    fout = open(outfil, 'wt')
    for gen in res:
        fout.write(str(gen)+"\n")

def loadgenelist(fil):
    return [int(x.strip()) for x in open(fil, "rt")]

def filtergenelist(genelist, allgenelist):
    return [x for x in genelist if x in allgenelist]

def WriteGeneList(genelist, filname):
    fout = open(filname, "wt")
    for gen in genelist:
        fout.write(str(int(gen)) + "\n")

#####################################################################################
#####################################################################################
#### Expression Bias
#####################################################################################
#####################################################################################
def LoadGeneINFO():
    HGNC = pd.read_csv("/Users/jiayao/Work/Resources/protein-coding_gene.txt", delimiter="\t")
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    allen_mouse_genes = loadgenelist("../dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol, allen_mouse_genes

def LoadExpressionMatrices(ExpMat = "../dat/allen-mouse-exp/energy-conn-model.csv", 
                        ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-conn-model.csv",
                        ExpMatNorm = "../dat/allen-mouse-exp/energy-neuronorm.csv",
                        ExpZscoreMatNorm = "../dat/allen-mouse-exp/energy-zscore-neuronorm.csv"):
    ExpMat = pd.read_csv(ExpMat, index_col="ROW")
    ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col="ROW")
    ExpMatNorm = pd.read_csv(ExpMatNorm, index_col="ROW")
    ExpZscoreMatNorm = pd.read_csv(ExpZscoreMatNorm, index_col="ROW")
    return ExpMat, ExpZscoreMat, ExpMatNorm, ExpZscoreMatNorm

def TraceSTR(df, structures, see=None):
    res = {} 
    Coarse = df[df["coarse"]=="X"]["id"].values
    for i, STR in enumerate(structures):
        _df = df[df["KEY"]==STR]
        parent_id = int(_df["parent_structure_id"].values[0])
        parent_df = df[df["id"]==parent_id]
        while 1:
            #_df = df[df["parent_structure_id"]]
            if parent_id == 997: # Root ID = 997
                print("Can't find %s"%STR)
                break
            elif parent_id in Coarse:
                res[STR] = parent_df["KEY"].values[0]
                break
            else:
                parent_id = int(parent_df["parent_structure_id"].values[0])
                parent_df = df[df["id"]==parent_id]
                #if STR == see:
                #    print(parent_df)
    return res

def StructureGeneSetBias(Structure, CaseSubsetDF, ControlSubsetDF, method="mean"):
    Case_Values = CaseSubsetDF[Structure].values
    Case_Values = [x for x in Case_Values if x==x]
    Control_Values = ControlSubsetDF[Structure].values
    Control_Values = [x for x in Control_Values if x==x]
    if method == "mean":
        return np.mean(Case_Values) - np.mean(Control_Values)
    elif method == "median":
        return np.median(Case_Values) - np.median(Control_Values)

def StructureGeneListBias(Structure, ZscoreMat, CaseGeneList, ControlGeneList, method="median"):
    Case_Values = []; Control_Values = [];
    for entrez_id in CaseGeneList:
        try:
            v = ZscoreMat.loc[entrez_id, Structure]
            if v == v:
                Case_Values.append(v)
        except:
            pass
    for entrez_id in ControlGeneList:
        v = ZscoreMat.loc[entrez_id, Structure]
        if v == v:
            Control_Values.append(v)
    t, p = scipy.stats.mannwhitneyu(Case_Values, Control_Values, alternative='greater')
    if method == "mean":
        return np.mean(Case_Values) - np.mean(Control_Values), p
    elif method == "median":
        return np.median(Case_Values) - np.median(Control_Values), p

def ComputeExpressionBias(method, ExpMat, CaseGeneSet, ContGeneSet=None, outfil="region.rank.tsv"):
    assert method in ["zscore", "absolute", "decile"]
    if method == "zscore":
        ExpMat = pd.read_csv(ExpMat, index_col="ROW")
        Structures = list(ExpMat.columns.values)
        Biases_median = {}
        for struc in Structures:
            bias_median, p = StructureGeneListBias(struc, ExpMat, CaseGeneSet, ContGeneSet, method="median")
            Biases_median[struc] = (bias_median, p)
        res = sorted(Biases_median.items(), key=lambda k:k[1], reverse=True)
        fout = open(outfil, 'wt')
        fout.write("STR\tbias\tpvalue\n")
        for k,(bias, p) in res:
            #print(k,v)
            fout.write("%s\t%.3f\t%.3e\n"%(k, bias, p))
    return

def search_decile(gene_str_z, percentile_values):
    for i, v in enumerate(percentile_values):
        if i == len(percentile_values)-1:
            return i
        elif gene_str_z > percentile_values[i] and gene_str_z < percentile_values[i+1]:
            return i

def count_slice(N, values):
    res = []
    for i in range(N):
        res.append(values.count(i))
    return res

#### Decile Correlation Related Functions
def DecileCorrelation(ExpZscoreMat, DZgenes, percentile_slices=np.array(range(0, 100, 10)), pdf_fil="decile.pdf", csv_fil="decile.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    STR_Deciles = {}
    for STR in structures:
        STR_Deciles[STR] = []
        str_zscores = ExpZscoreMat[STR].values
        str_zscores = [x for x in str_zscores if x==x]
        percentile_values = np.percentile(str_zscores, q=percentile_slices)
        for gene in DZgenes:
            try:
                gene_str_z = ExpZscoreMat.loc[gene, STR]
            except:
                continue
            idx = search_decile(gene_str_z, percentile_values)
            STR_Deciles[STR].append(idx)
    if pdf_fil != None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_fil)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    res = []
    for i, STR in enumerate(structures):
        counts = count_slice(len(percentile_values), STR_Deciles[STR])
        r, p = spearmanr(percentile_slices, counts)
        res.append([STR, r, p])
        if pdf_fil != None:
            fig = plt.figure(dpi=120)
            plt.plot(percentile_slices, counts, '--r', marker=".", markersize=15)
            plt.text(0.7*max(percentile_slices), 0.95*max(counts), "spearmanr=%.2f\npvalue=%.3f"%(r, p), fontsize=8,
                verticalalignment='top', bbox=props)
            #plt.title("SCZ genes {} percentiles in {}".format((len(percentile_values)-1), STR))
            plt.title("{}".format(STR))
            plt.xlabel("percentile")
            plt.ylabel("num of genes in each precentile")
            plt.grid(True)
            plt.axhline(y=(len(DZgenes)/(len(percentile_values))), color="black",ls="--")
            pdf.savefig(fig)
        #if i == 1:
        #    break
    if pdf_fil != None:
        pdf.close()
    res = sorted(res, key=lambda x:x[1], reverse=True)
    
    if csv_fil != None:
        writer = csv.writer(open(csv_fil, 'wt'), delimiter="\t")
        writer.writerow(["STR", "Spearmanr", "pvalue.spearman"])
        for STR, r, p in res:
            writer.writerow([STR, r, p])
    top50 = res[:50]
    return top50

def DecileNull(NGenes, Ndecile, Npermute, Weights):
    SET = [x for x in range(Ndecile)]
    res = []
    for i in range(Npermute):
        onerun = np.random.choice(SET, size=NGenes, replace=True)
        onerun = count_slice(Ndecile, list(onerun))
        Score = DecileWeightedScore(onerun, Weights)
        res.append(Score)
    return res

def DecileWeightedScore(DecileValues, Weights):
    assert len(DecileValues) == len(Weights)
    Score = 0
    for V, W in zip(DecileValues, Weights):
        Score += V * W
    return Score/sum(DecileValues)

def DecileOneSTR(STR, ExpZscoreMat, DZGenes, Weights, percentile_slices):
    str_zscores = ExpZscoreMat[STR].values
    str_zscores = [x for x in str_zscores if x==x]
    percentile_values = np.percentile(str_zscores, q=percentile_slices)
    res = []
    for gene in DZGenes:
        try:
            gene_str_z = ExpZscoreMat.loc[gene, STR]
        except:
            continue
        idx = search_decile(gene_str_z, percentile_values)
        res.append(idx)

    counts = count_slice(len(percentile_values), res)
    #print(counts)
    return DecileWeightedScore(counts, Weights)
    
def DecileRankScoring(ExpZscoreMat, DZgenes, Weights, NullDist, percentile_slices=np.array(range(0, 100, 10)), pdf_fil="decile.pdf", csv_fil="decile.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    res = []
    for STR in structures:
        STR_Score = DecileOneSTR(STR, ExpZscoreMat, DZgenes, Weights, percentile_slices)
        P = GetPermutationP(NullDist, STR_Score)
        #STR_Deciles[STR] = (STR_Score, P)
        res.append([STR, STR_Score, P])
    res = sorted(res, key=lambda x:x[1], reverse=True)
    if csv_fil != None:
        writer = csv.writer(open(csv_fil, 'wt'), delimiter="\t")
        writer.writerow(["STR", "Score", "pvalue"])
        for STR, r, p in res:   
            writer.writerow([STR, r, p])

#### Quantile AVG Related Functions
def QuantileNull(NGenes, Npermute, ttest=False):
    slice = 1/18000
    SET = list(range(1, 18000))
    bias, p = [], []
    for i in range(Npermute):
        onerun = np.random.choice(SET, size=NGenes, replace=False)
        onerun = [x/18000 for x in onerun]
        bias.append(np.mean(onerun))
        p.append(ttest_1samp(onerun, 0.5)[1])
    res = pd.DataFrame(data={'Bias':bias, 'P':p})
    res = res.sort_values("Bias", ascending=False)
    return res

def QuantileOneSTR(STR, ZscoreMat, DZgenes):
    zscores = ZscoreMat[STR].values
    zscores = zscores[~np.isnan(zscores)]
    sorted_z = sorted(zscores)
    DZzscore = ZscoreMat[STR].loc[np.array(DZgenes)].values
    res = []
    count = 0
    DZzscore = set(DZzscore)
    TotalLen = len(sorted_z)
    Z = []
    Quantiles = []
    for i,z in enumerate(sorted_z):
        if z==z and z in DZzscore:
            count += 1
            quantile = i/TotalLen
            res.append(quantile)
            Z.append(z)
    return Z, res

def QuantileAVGShowDist(ExpZscoreMat, DZgenes, csv_fil="quantile.rank.tsv", alpha=0.01):
    structures = ExpZscoreMat.columns.values 
    AVGQ, AVGZ, Qs, Zs = [], [], [], []
    for STR in structures:
        DZzscore, Quantiles = QuantileOneSTR(STR, ExpZscoreMat, DZgenes)
        quant_avg = np.mean(Quantiles)
        zscore_avg = np.mean(DZzscore)
        AVGQ.append(quant_avg)
        AVGZ.append(zscore_avg)
        Qs.append(Quantiles)
        Zs.append(DZzscore)
    df = pd.DataFrame(data={"STR":structures, "AVG_Q":AVGQ, "AVG_Z":AVGZ, "Quantiles": Qs, "Zscores":Zs})
    top50_zscores = df.sort_values("AVG_Q", ascending=False).head(50) 
    top50_quantiles = df.sort_values("AVG_Z", ascending=False).head(50)
    return top50_zscores, top50_quantiles

def QuantileAVGScoring(ExpZscoreMat, DZgenes, csv_fil="quantile.rank.tsv", alpha=0.01):
    structures = ExpZscoreMat.columns.values
    res = []
    STRs = []
    Bias = []
    T = []
    P = []
    for STR in structures:
        DZzscore, Quantiles = QuantileOneSTR(STR, ExpZscoreMat, DZgenes) 
        STR_Score = np.mean(Quantiles)
        STRs.append(STR)
        Bias.append(STR_Score)
        #w, p = wilcoxon(DZzscore)
        t, p = ttest_1samp(Quantiles, 0.5)
        T.append(t)
        P.append(p)
    acc, qvalues = stats.multitest.fdrcorrection(P, alpha=alpha)
    df = pd.DataFrame(data={"STR":STRs, "Bias":Bias, "T-stat":T, "P":P, "FDR":qvalues})
    df = df.sort_values("Bias", ascending=False)
    df.index = np.arange(1, len(df) + 1)
    if csv_fil != None:
        df.to_csv(csv_fil, index=False)
    return df

###################################################################################################################
# Expression Specificity Bias Simple
###################################################################################################################
# ExpZscoreMat: z score matrix of expression
# DZgenes: gene set
# RefGenes: reference gene set that sampling from, like brian expressed genes or sibling genes. 
# Nsampling: number of samping from reference gene set. Larger the better estimation but slower
def ZOneSTR(STR, ZscoreMat, DZgenes):
    DZzscore = ZscoreMat[STR].loc[np.array(DZgenes)].values
    DZzscore = [x for x in DZzscore if x==x]
    return np.mean(DZzscore)
def ZscoreAVGWithExpMatch(ExpZscoreMat, DZgenes, Match_DF, csv_fil="specificity.expmatch.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    DZgenes = list(set(DZgenes).intersection(set(ExpZscoreMat.index.values)))
    EFFECTS = []
    EFF_Z = []
    for i, STR in enumerate(structures):
        mean_z = ZOneSTR(STR, ExpZscoreMat, DZgenes)
        Match_mean_Zs = []
        for c in Match_DF.columns:
            match_genes = Match_DF[c].values
            match_z = ZOneSTR(STR, ExpZscoreMat, match_genes)
            Match_mean_Zs.append(match_z)
        est_mean_z = np.mean(Match_mean_Zs)
        effect_z = mean_z - est_mean_z
        EFF_Z.append(effect_z); 
        ZZ = (mean_z - est_mean_z) / np.std(Match_mean_Zs)
        EFFECTS.append(ZZ)
    df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFECTS, "EFF_Z":EFF_Z})
    df = df.sort_values("EFFECT", ascending=False)
    if csv_fil != None:
        df.to_csv(csv_fil, index=False)
    return df
def ZscoreAVGWithExpMatch_SingleGene(ExpZscoreMat, DZgenes, Match_DF, csv_fil="specificity.expmatch.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    STRS, data = [], []
    for i, STR in enumerate(structures):
        EFFECTS = []
        Mean_Z, Match_Z = [],[]
        for j, g in enumerate(DZgenes):
            #print(i, j)
            DZzscore = ExpZscoreMat.loc[g, STR]
            Match_mean_Zs = []
            if not DZzscore == DZzscore:
                continue
            for k, c in enumerate(Match_DF.columns):
                match_gene = Match_DF.loc[g, c]
                match_z = ExpZscoreMat.loc[match_gene, STR]
                if not match_z == match_z:
                        continue
                Match_mean_Zs.append(match_z)
            est_mean_z = np.mean(Match_mean_Zs)
            ZZ = (DZzscore - est_mean_z) / np.std(Match_mean_Zs)
            EFFECTS.append(ZZ)
            if ZZ != ZZ:
                print(STR, g, ZZ)
        data.append(EFFECTS)
        STRS.append(STR)
    #df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFECTS})
    df = pd.DataFrame(data=data, columns=DZgenes)
    df.index=STRS
    return df


###################################################################################################################
# Expression Specificity Bias
###################################################################################################################
# ExpZscoreMat: z score matrix of expression
# DZgenes: gene set
# RefGenes: reference gene set that sampling from, like brian expressed genes or sibling genes. 
# Nsampling: number of samping from reference gene set. Larger the better estimation but slower
def QuantileAVGScoringWithModifiedEffectSize(ExpZscoreMat, DZgenes, RefGenes, Nsamping = 100, csv_fil="quantile.rank.tsv"):
    DF_DZ = QuantileAVGScoring(ExpZscoreMat, DZgenes)
    
    for i in range(Nsampling):
        s
    return df

def ZscoreAVGWithExpMatch2(ExpZscoreMat, DZgenes, Match_DF, csv_fil="specificity.expmatch.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    EFF_Z, EFF_Q = [], []
    Pval_Z, Pval_Q = [], []
    Mean_Z, Match_Z = [], []
    Mean_Q, Match_Q = [], []
    EFFECTS = []
    for i, STR in enumerate(structures):
        DZzscore, Quantiles = QuantileOneSTR(STR, ExpZscoreMat, DZgenes)
        mean_z, mean_q = np.mean(DZzscore), np.mean(Quantiles)
        Match_mean_Zs, Match_mean_Qs = [], []
        for c in Match_DF.columns:
            match_genes = Match_DF[c].values
            match_z, match_q = QuantileOneSTR(STR, ExpZscoreMat, match_genes)
            mean_match_z, mean_match_q = np.mean(match_z), np.mean(match_q)
            Match_mean_Zs.append(mean_match_z)
            Match_mean_Qs.append(mean_match_q)
        est_mean_z = np.mean(Match_mean_Zs)
        est_mean_q = np.mean(Match_mean_Qs)
        effect_z = mean_z - est_mean_z
        effect_q = mean_q - est_mean_q
        EFF_Z.append(effect_z); EFF_Q.append(effect_q)
        Pval_Z.append(ttest_1samp(Match_mean_Zs, mean_z)[1])
        Pval_Q.append(ttest_1samp(Match_mean_Qs, mean_q)[1])
        Mean_Z.append(mean_z); Match_Z.append(est_mean_z); Mean_Q.append(mean_q); Match_Q.append(est_mean_q)
        ZZ = (mean_z - est_mean_z) / np.std(Match_mean_Zs)
        EFFECTS.append(ZZ)
        #print(i,)
    df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFECTS, "EFF_Z":EFF_Z, "Pval_Z":Pval_Z})
    df["Mean_Z"] = Mean_Z
    df["Match_Z"] = Match_Z
    return df

def ZscoreAVGWithExpMatch_SingleGene2(ExpZscoreMat, DZgenes, Match_DF, csv_fil="specificity.expmatch.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    STRS, data = [], []
    for i, STR in enumerate(structures):
        EFFECTS = []
        Mean_Z, Match_Z = [],[]
        for j, g in enumerate(DZgenes):
            #print(i, j)
            DZzscore = ExpZscoreMat.loc[g, STR]
            Match_mean_Zs = []
            if not DZzscore == DZzscore:
                continue
            for k, c in enumerate(Match_DF.columns):
                match_gene = Match_DF.loc[g, c]
                match_z = ExpZscoreMat.loc[match_gene, STR]
                if not match_z == match_z:
                        continue
                Match_mean_Zs.append(match_z)
            est_mean_z = np.mean(Match_mean_Zs)
            ZZ = (DZzscore - est_mean_z) / np.std(Match_mean_Zs)
            EFFECTS.append(ZZ)
            if ZZ != ZZ:
                print(STR, g, ZZ)
        data.append(EFFECTS)
        STRS.append(STR)
    #df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFECTS})
    df = pd.DataFrame(data=data, columns=DZgenes)
    df.index=STRS
    return df

###################################################################################################################

###################################################################################################################
# Expression Level Bias with Constraint
###################################################################################################################
def ZscoreOneSTR_Constraint(STR, ZscoreMat, DZgenes, Gene2LoFZ):
    DZgene_Zs = ZscoreMat[STR].loc[np.array(DZgenes)].values
    DZgene_cons = [Gene2LoFZ.get(g, 1) for g in DZgenes]
    DZgene_Z_cons = [x*y for x,y in zip(DZgene_Zs, DZgene_cons)]
    DZgene_Z_cons = [x for x in DZgene_Z_cons if x==x]
    return np.mean(DZgene_Z_cons)

def ZscoreAVGWithExpMatch_Constraint(ExpZscoreMat, DZgenes, Match_DF, Gene2LoFZ, csv_fil="specificity.expmatch.rank.tsv"):
    structures = ExpZscoreMat.columns.values
    EFFECTS = []; EFF_Z = []
    for i, STR in enumerate(structures):
        mean_z = ZscoreOneSTR_Constraint(STR, ExpZscoreMat, DZgenes, Gene2LoFZ)
        Match_mean_Zs = []; 
        for c in Match_DF.columns:
            match_genes = Match_DF[c].values
            match_z = ZscoreOneSTR_Constraint(STR, ExpZscoreMat, match_genes, Gene2LoFZ)
            Match_mean_Zs.append(match_z)
        est_mean_z = np.mean(Match_mean_Zs)
        effect_z = mean_z - est_mean_z
        EFF_Z.append(effect_z); 
        ZZ = (mean_z - est_mean_z) / np.std(Match_mean_Zs)
        EFFECTS.append(ZZ)
        #print(i,)
    df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFECTS, "EFF_Z":EFF_Z})
    return df

###################################################################################################################
# Expression Level Bias
###################################################################################################################
def ExpLevelOneSTR(STR, ExpMat, DZgenes, Matchgenes):
    #exp_levels = ExpMat[STR].values
    #exp_levels = exp_levels[~np.isnan(exp_levels)]
    g_exp_level_zs = []
    err_gen = 0
    for i, g in enumerate(DZgenes):
        try:
            g_exp_level = ExpMat.loc[g, STR]
            assert g_exp_level == g_exp_level
        except:
            err_gen += 1
            continue
        #print(Matchgenes.head(2))
        g_matches = Matchgenes.loc[g, :].values
        #print(g, g_matches)
        g_matches_exps = ExpMat.loc[g_matches, STR].values
        g_matches_exps = g_matches_exps[~np.isnan(g_matches_exps)]
        g_exp_level_z = (g_exp_level - np.mean(g_matches_exps))/np.std(g_matches_exps)
        g_exp_level_zs.append(g_exp_level_z)
    avg_exp_level_z = np.mean(g_exp_level_zs)
    return avg_exp_level_z
## Match_DF: rows as genes, columns as trails
def ExpAVGWithExpMatch_depricated(ExpMat, DZgenes, Match_DF, csv_fil="explevel.rank.tsv"):
    structures = ExpMat.columns.values
    EFFs = []
    for i, STR in enumerate(structures):
        avg_exp_level_z = ExpLevelOneSTR(STR, ExpMat, DZgenes, Match_DF)
        EFFs.append(avg_exp_level_z)
    df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFs})
    df = df.sort_values("EFFECT", ascending=False)
    df.to_csv(csv_fil, index=False)
    return df

###################################################################################################################
# Expression Level Bias with Constraint
###################################################################################################################
def ExpLevelOneSTR_Constraint(STR, ExpMat, DZgenes, Matchgenes, Gene2LoFZ):
    g_exp_level_zs = []
    err_gen = 0
    for i, g in enumerate(DZgenes):
        try:
            g_exp_level = ExpMat.loc[g, STR]
            assert g_exp_level == g_exp_level
            g_exp_level_cons = g_exp_level * Gene2LoFZ.get(g, 1)
        except:
            err_gen += 1
            continue
        g_matches = Matchgenes.loc[g, :].values
        g_matches_cons = [Gene2LoFZ.get(g, 1) for g in g_matches]
        g_matches_exps = ExpMat.loc[g_matches, STR].values
        g_matches_exps_cons = [x*y for x,y in zip(g_matches_cons, g_matches_exps)]
        #print(g_matches_exps_cons)
        #g_matches_exps_cons = g_matches_exps_cons[~np.isnan(g_matches_exps_cons)]
        g_matches_exps_cons = [x for x in g_matches_exps_cons if x==x]
        g_exp_level_z = (g_exp_level_cons - np.mean(g_matches_exps_cons))/np.std(g_matches_exps_cons)
        g_exp_level_zs.append(g_exp_level_z)
    avg_exp_level_z = np.mean(g_exp_level_zs)
    return avg_exp_level_z
## Match_DF: rows as genes, columns as trails
def ExpAVGWithExpMatch_Constraint(ExpMat, DZgenes, Match_DF, Gene2LoFZ, csv_fil="explevel.rank.tsv"):
    structures = ExpMat.columns.values
    EFFs = []
    for i, STR in enumerate(structures):
        avg_exp_level_z = ExpLevelOneSTR_Constraint(STR, ExpMat, DZgenes, Match_DF, Gene2LoFZ)
        EFFs.append(avg_exp_level_z)
    df = pd.DataFrame(data={"STR":structures, "EFFECT":EFFs})
    df = df.sort_values("EFFECT", ascending=False)
    df.to_csv(csv_fil, index=False)
    return df

def ExpAVGWithExpMatch_SingleGene(ExpMat, DZgenes, Match_DF, csv_fil="explevel.rank.tsv"):
    structures = ExpMat.columns.values
    EFFs = []
    for i, STR in enumerate(structures):
        g_exp_str = []
        for j, g in enumerate(DZgenes):
            g_exp_level = ExpMat.loc[g, STR]
            g_matches = Match_DF.loc[g, :].values
            g_matches_exps = ExpMat.loc[g_matches, STR].values
            g_matches_exps = g_matches_exps[~np.isnan(g_matches_exps)]
            g_exp_level_z = (g_exp_level - np.mean(g_matches_exps))/np.std(g_matches_exps)
            g_exp_str.append(g_exp_level_z)
        #avg_exp_level_z = ExpLevelOneSTR(STR, ExpMat, DZgenes, Match_DF)
        #EFFs.append(avg_exp_level_z)
        EFFs.append(g_exp_str)
    df = pd.DataFrame(data=EFFs, columns=DZgenes)
    df.index = structures
    return df

def QuntileAVGSelectN(BiasDF, name = "STR", plot=False):
    medians = []
    idx = []
    i = 0
    while i<100:
        idx.append(i)
        tmp_df = BiasDF.iloc[i:,:]
        median = np.median(tmp_df["Bias"].values)
        medians.append(median)
        if median <= 0.5:
            N = i-1
            break
        i += 1
    while i<100:
        idx.append(i)
        tmp_df = BiasDF.iloc[i:,:]
        median = np.median(tmp_df["Bias"].values)
        medians.append(median)
        i += 1
    if plot:
        plt.figure(dpi=120)
        plt.hlines(y=0.5, xmin=0, xmax = 100, linestyles="dashed", alpha=0.5, color="grey")
        plt.plot(idx, medians)
        plt.title("%s Median Bias"%name)
        plt.xlabel("Num Removed STRs")
        plt.ylabel("Median Expression Bias")
    return N

def StureturesOverlap(set1, set2, Ntop=50):
    return list(set(set1["STR"].values[:Ntop]).intersection(set2["STR"].values[:Ntop]))

def PlotBiasDistandP(ssc, spark, tada, sib, topN = 50, labels=["ssc", "spark", "tada"]):
    fig, axs = plt.subplots(2, 2, dpi=120)
    dfs = [ssc, spark, tada]
    for idx, ax in enumerate([(0,0), (0,1), (1,0)]):
        i,j = ax
        Case_Values = dfs[idx]["Score"].values[:topN]
        Control_Values = sib["Score"].values[:topN]
        t, p = scipy.stats.mannwhitneyu(Case_Values, Control_Values, alternative='greater')
        u1, u2 = np.mean(Case_Values), np.mean(Control_Values)
        axs[i,j].hist(Case_Values, label=labels[idx], color="red", alpha=0.5, density=1)
        axs[i,j].hist(Control_Values, label=labels[3], color="blue", alpha=0.5, density=1)
        axs[i,j].legend()
        print("%20s\t%.3f\t%.3f\t%.3f\t%.3e"%(labels[idx], u1, u2, (u1-u2), p))
    plt.show()

###################################################################################################################
# Matching genes by overal brain expression level 
###################################################################################################################
def TricubeKernal(u, _min, _mid, _max):
    #_u = np.clip(2*(u-_mid)/(_max - _min), 0, 1)
    _u = 2*(u-_mid)/(_max - _min)
    return 70/81 * (1-abs(_u)**3)**3

def assignProb(xlist):
    #_min, _max = min(xlist), max(xlist)
    _min, _max = min(xlist)-1e-3, max(xlist)+1e-3
    _mid = (_max + _min) / 2
    density = TricubeKernal(xlist, _min, _mid, _max)
    norm = np.sum(density)
    res = density/norm
    res[-1] = 1 - sum(res[:-1])
    return np.array(res)

def ExpressionMatchGeneSet(geneset, match_feature, savefil = None, interval_len = 500, sample_size=1000): 
    genes_to_match = [] # genes
    dat = []
    xx_genes = []
    for i, gene in enumerate(geneset):
        try:
            gene_exp = match_feature.loc[gene, "EXP"]
        except:
            #print("Unmathed:%d"%gene)
            continue
        #print(i, gene)
        if gene in xx_genes:
            continue
        genes_to_match.append(gene)
        gene_rank = match_feature.loc[gene, "Rank"]
        Interval = match_feature[(match_feature["Rank"] >= gene_rank - interval_len) &
                                 (match_feature["Rank"] <= gene_rank + interval_len) & 
                                 (~match_feature.index.isin(geneset))       ]
        Interval_genes = Interval.index.values
        Interval_exps = Interval["EXP"].values
        Interval_genes_probs = assignProb(Interval_exps)
        match_genes = np.random.choice(Interval_genes, size = sample_size, replace=True, p = Interval_genes_probs) # N(sample_size)genes that expression matching with particular gene in gene set
        dat.append(match_genes) 
        xx_genes.append(gene)
    df = pd.DataFrame(data=dat, columns = [x+1 for x in range(sample_size)]) # rows are sampleing of different genes, columns are N sampling
    df.index = genes_to_match
    if savefil != None:
        df.to_csv(savefil)
    return df

def ExpressionMatchGeneSetwithpLI(geneset, match_feature_HIS, match_feature_HS, savefil = None, interval_len = 500, sample_size=1000): 
    genes_to_match = [] # genes, used for giving index of final DF
    dat = [] # To Store match genes
    xx_genes = [] # Remove Duplicated genes in genes to match 
    for i, gene in enumerate(geneset):
        #try:
        #    gene_exp = match_feature.loc[gene, "EXP"]
        #except:
        #    print("Unmathed:%d"%gene)
        #    continue
        #print(i, gene)
        if gene in xx_genes:
            continue
        if gene in match_feature_HIS.index.values:
            match_feature = match_feature_HIS
        elif gene in match_feature_HS.index.values:
            match_feature = match_feature_HS
        else:
            print(gene)
            continue
        genes_to_match.append(gene)
        gene_rank = match_feature.loc[gene, "Rank"]
        Interval = match_feature[(match_feature["Rank"] >= gene_rank - interval_len) &
                                 (match_feature["Rank"] <= gene_rank + interval_len) & 
                                 (~match_feature.index.isin(geneset))       ]
        Interval_genes = Interval.index.values
        Interval_exps = Interval["EXP"].values
        Interval_genes_probs = assignProb(Interval_exps)
        match_genes = np.random.choice(Interval_genes, size = sample_size, replace=True, p = Interval_genes_probs) # N(sample_size)genes that expression matching with particular gene in gene set
        dat.append(match_genes) 
        xx_genes.append(gene)
    df = pd.DataFrame(data=dat, columns = [x+1 for x in range(sample_size)]) # rows are sampleing of different genes, columns are N sampling
    df.index = genes_to_match
    if savefil != None:
        df.to_csv(savefil)
    return df

def write_match(gene, matches, Dir):
    if not os.path.exists(Dir):
        os.makedirs(Dir)
    writer = csv.writer(open("%s/%d.match.csv"%(Dir, gene), 'wt'))
    for match in matches:
        writer.writerow([match["GENE"]])

def STR2Region():
    str2reg_df = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    return str2reg

def RegionDistributions(DF, topN=50, show = False):
    #ax = fig.add_axes([0,0,1,1])
    str2reg_df = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    Regions = list(set(str2reg.values()))
    RegionCount = {}
    for region in Regions:
        RegionCount[region] = []
    for x in DF.head(topN).index:
        region = str2reg[x]
        RegionCount[region].append(x)
    if show:
        for k,v in RegionCount.items():
            if len(v) == 0:
                continue
            print(k, "\t", len(v), "\t", "; ".join(v))
    return RegionCount

def RegionDistributionsList(List, topN=50):
    str2reg_df = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    Regions = list(set(str2reg.values()))
    RegionCount = {}
    for region in Regions:
        RegionCount[region] = []
    for x in List:
        region = str2reg[x]
        RegionCount[region].append(x)
    for k,v in RegionCount.items():
        if len(v) == 0:
            continue
        print(k, "\t", len(v), "\t", "; ".join(v))

###################################################################################################################
# Gene Weights (impact)
###################################################################################################################
def SSC_Gene_Weights(MutFil, gnomad_cons, FDR=0.8):
    #MutFil = pd.read_csv(MutFil)
    if FDR != None:
        MutFil = MutFil[MutFil["FDR"]>FDR]
    gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons, gene2MutNQValue = {}, {}, {}, {}, {}, {}
    for i, row in MutFil.iterrows():
        g = int(row["Entrez"])
        gene2None[g] = 1
        #gene2RR[g] = row["LGD_RR"]*5 + row["misa_RR"]*1 + row["misb_RR"]*2    # Relative Risk is sum over LGD and Missense
        gene2MutN[g] = row["dnLGD"]*0.375 + (row["dnMis"]) * 0.145
        #gene2Cons[g] = gnomad_cons.loc[row["gene"], "lof_z"]*5 + gnomad_cons.loc[row["gene"], "mis_z"]
        #gene2MutNCons[g] = gene2MutN[g] * gene2Cons[g]
        #gene2MutNQValue[g] = gene2MutN[g] * row["qval_dnccPTV"]
    #return gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons, gene2MutNQValue
    return gene2MutN
def ASC_Gene_Weights(MutFil, gnomad_cons=None, FDR=0.1):
    #MutFil = pd.read_csv(MutFil)
    if FDR != None:
        MutFil = MutFil[MutFil["qval_dnccPTV"]<FDR]
    gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons, gene2MutNQValue = {}, {}, {}, {}, {}, {}
    for i, row in MutFil.iterrows():
        g = int(row["entrez_id"])
        gene2None[g] = 1
        gene2RR[g] = row["LGD_RR"]*5 + row["misa_RR"]*1 + row["misb_RR"]*2    # Relative Risk is sum over LGD and Missense
        #gene2MutN[g] = row["dn.ptv"]*0.375 + (row["dn.misa"] + row["dn.misb"]) * 0.145
        gene2MutN[g] = max(row["dn.ptv"]*0.375,  (row["dn.misa"] + row["dn.misb"]) * 0.145)
        #gene2Cons[g] = gnomad_cons.loc[row["gene"], "lof_z"]*5 + gnomad_cons.loc[row["gene"], "mis_z"]
        #gene2MutNCons[g] = gene2MutN[g] * gene2Cons[g]
        gene2MutNQValue[g] = gene2MutN[g] * row["qval_dnccPTV"]
    #return gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons, gene2MutNQValue
    return gene2MutN
def ASC_Gene_Weights_ByIQ(MutFil, FDR=0.2):
    if FDR != None:
        MutFil = MutFil[MutFil["Qvalue"]<FDR]
    gene2MutN = {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["Entrez"])
        except:
            continue
        gene2MutN[g] = row["dnLGD"]*0.375 + (row["dnDmis"]) * 0.145
    return gene2MutN
def ASC_MutCountByLength(MutFil, match_feature, FDR=0.1):
    ASC_NTrio = 6430
    match_feature = pd.read_csv(match_feature, index_col="GENE")
    asc = pd.read_csv(MutFil)
    asc = asc[asc["entrez_id"]!="."]
    asc["entrez_id"] = [int(x) for x in asc["entrez_id"].values]
    if FDR != None:
        MutFil = asc[asc["qval_dnccPTV"]<FDR]
    match_feature = match_feature.sort_values("LENGTH.LOG2")
    Ndiciles = 10
    interval = int(match_feature.shape[0]/Ndiciles)
    Deciles = []
    LengthCut = []
    Gene2Decile = {}
    for i in range(Ndiciles):
        Deciles.append(match_feature.index.values[i * interval:(i+1)*interval])
        LengthCut.append(match_feature.loc[match_feature.index.values[(i+1)*interval-1], "LENGTH.LOG2"])
    #Deciles[-1] = Deciles[-1].append(match_feature.index.values[(i+1)*interval:])
    Deciles[-1] = np.append(Deciles[-1], match_feature.index.values[(i+1)*interval:])
    Percent_True_LGD = []
    Enrichment_LGD = []
    Percent_True_Mis = []
    Enrichment_Mis = []
    for i in range(Ndiciles):
        test_genes = asc[asc["entrez_id"].isin(Deciles[i])]
        for g in test_genes["entrez_id"].values:
            Gene2Decile[g] = i
        N_LGD = sum(test_genes["dn.ptv"])
        N_LGD_exp = sum(test_genes["mut.ptv"]) * 2 * ASC_NTrio
        Percent_True_LGD.append(max(0, (N_LGD - N_LGD_exp)/N_LGD))
        Enrichment_LGD.append(N_LGD/N_LGD_exp)
        N_Mis = sum(test_genes["dn.misa"]) + sum(test_genes["dn.misb"])
        N_Mis_exp = sum(test_genes["mut.misa"]) * 2 * ASC_NTrio + sum(test_genes["mut.misb"]) * 2 * ASC_NTrio
        Percent_True_Mis.append(max(0, (N_Mis - N_Mis_exp)/N_Mis))
        Enrichment_Mis.append(N_Mis/N_Mis_exp)
    gene2MutN_Length = {}
    for i, row in MutFil.iterrows(): # Counting Mutations
        g = int(row["entrez_id"])
        if g in Gene2Decile:
            decile = Gene2Decile[g]
            gene2MutN_Length[g] = row["dn.ptv"] * Percent_True_LGD[decile] + (row["dn.misa"] + row["dn.misb"]) * Percent_True_Mis[decile]
        else:
            print(g)
    return gene2MutN_Length

def SPARK_Gene_Weights(MutFil, gnomad_cons=None, FDR=0.2):
    #MutFil = pd.read_csv(MutFil)
    if FDR != None:
        MutFil = MutFil[MutFil["Qvalue"]<FDR]
    gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons = {}, {}, {}, {}, {} 
    for i, row in MutFil.iterrows():
        try:
            g = int(row["Entrez"])
        except:
            continue
        gene2None[g] = 1
        gene2RR[g] = row["LGD_RR"]*5 + row["Dmis_RR"]*1    # Relative Risk is sum over LGD and Missense
        gene2MutN[g] = row["dnLGD"]*0.347 + row["dnDmis"]*0.194
        #try:
        #    gene2Cons[g] = gnomad_cons.loc[row["HGNC"], "lof_z"]*5 + gnomad_cons.loc[row["HGNC"], "mis_z"]
        #except:
        #    gene2Cons[g] = 1
        #gene2MutNCons[g] = gene2MutN[g] * gene2Cons[g]
    #return gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons
    return gene2MutN

def SPARK_MutCountByLength(MutFil, match_feature, FDR=0.2):
    SPARK_NTrio = 7015
    match_feature = pd.read_csv(match_feature, index_col="GENE")
    spark = pd.read_csv(MutFil)
    spark = spark[spark["Entrez"]!="."]
    spark["Entrez"] = [int(x) for x in spark["Entrez"].values]
    if FDR != None:
        MutFil = spark[spark["Qvalue"]<FDR]
    match_feature = match_feature.sort_values("LENGTH.LOG2")
    Ndiciles = 10
    interval = int(match_feature.shape[0]/Ndiciles)
    Deciles = []
    LengthCut = []
    Gene2Decile = {}
    for i in range(Ndiciles):
        Deciles.append(match_feature.index.values[i * interval:(i+1)*interval])
        LengthCut.append(match_feature.loc[match_feature.index.values[(i+1)*interval-1], "LENGTH.LOG2"])
    Deciles[-1] = np.append(Deciles[-1], match_feature.index.values[(i+1)*interval:])
    Percent_True_LGD = []
    Enrichment_LGD = []
    Percent_True_Mis = []
    Enrichment_Mis = []
    for i in range(Ndiciles):
        test_genes = spark[spark["Entrez"].isin(Deciles[i])]
        for g in test_genes["Entrez"].values:
            Gene2Decile[g] = i
        N_LGD = sum(test_genes["dnLGD"])
        N_LGD_exp = sum(test_genes["mutLGD"]) * 2 * SPARK_NTrio
        Percent_True_LGD.append(max(0, (N_LGD - N_LGD_exp)/N_LGD))
        Enrichment_LGD.append(N_LGD/N_LGD_exp)
        N_Mis = sum(test_genes["dnDmis"])
        N_Mis_exp = sum(test_genes["mutDmis"]) * 2 * SPARK_NTrio 
        Percent_True_Mis.append(max(0, (N_Mis - N_Mis_exp)/N_Mis))
        Enrichment_Mis.append(N_Mis/N_Mis_exp)
    gene2MutN_Length = {}
    for i, row in MutFil.iterrows(): # Counting Mutations
        g = int(row["Entrez"])
        if g in Gene2Decile:
            decile = Gene2Decile[g]
            gene2MutN_Length[g] = row["dnLGD"] * Percent_True_LGD[decile] + row["dnDmis"] * Percent_True_Mis[decile]
        else:
            print(g)
    return gene2MutN_Length

###################################################################################################################
# Weighted Bias Calculattion
###################################################################################################################
# Expression Specificity
def ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights):
    #DZzscore = ZscoreMat[STR].loc[np.array(DZgenes)].values
    #DZzscore = [x for x in DZzscore if x==x]
    res = []
    for gene, weight in Gene2Weights.items():
        if gene not in ExpZscoreMat.index.values:
            continue
        score = weight * ExpZscoreMat.loc[gene, STR]
        if score == score:
            res.append(score)
    return np.mean(res)
def AvgSTRZ_Weighted(ExpZscoreMat, Gene2Weights, Match_DF=0, BS_Weights=False, csv_fil="AvgSTRZ.weighted.csv"):
    STRs = ExpZscoreMat.columns.values
    EFFECTS = []; Normed_EFFECTS = []
    Regions = []
    str2reg_df = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    for i, STR in enumerate(STRs):
        mean_z = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights)
        EFFECTS.append(mean_z)
        #if Match_DF.isinstance(Match_DF, pd.core.frame.DataFrame):
        if type(Match_DF) != int:
            Match_mean_Zs = [];
            for g in Match_DF.columns:
                match_genes = Match_DF[g].values
                if BS_Weights:
                    weights = np.random.choice(list(Gene2Weights.values()), len(match_genes))
                    BS_MG2Weights = dict(zip(match_genes, weights))
                else:
                    BS_MG2Weights = dict(zip(match_genes, [1]*len(match_genes)))
                match_z = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, BS_MG2Weights)
                Match_mean_Zs.append(match_z)
            match_mean_z = np.mean(Match_mean_Zs)
            normed_effect = (mean_z - match_mean_z) / np.std(Match_mean_Zs)
            Normed_EFFECTS.append(normed_effect) 
        Regions.append(str2reg[STR])
    if type(Match_DF) != int:
        df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS, "REGION":Regions, "NormedEFFECT":Normed_EFFECTS})
    else:
        df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS, "REGION":Regions})
    df = df.sort_values("EFFECT", ascending=False)
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    # Trimming Circuit
    g = LoadConnectome2()
    top_structs = df.head(50)["STR"].values
    top_nodes = g.vs.select(label_in=top_structs)
    g2 = g.subgraph(top_nodes)
    graph_size, exp_stats, exp_graphs, rm_vs = CircuitTrimming(g2, g)
    idx = np.argmax(exp_stats[:40])
    print(50-idx)
    CIR_STRS = [v["label"] for v in exp_graphs[idx].vs]
    df['InCircuit'] = 0
    df["TrimRank"] = 1
    df.index = df["STR"].values
    for i, _str in enumerate([x["label"] for x in rm_vs]):
        df.loc[_str, "TrimRank"] = 50-i
        df.loc[_str, "InCircuit"] = 1 if _str in CIR_STRS else 0
    df.to_csv(csv_fil, index=False)
    return df

# Expression Level
def ExpLevelOneSTR_Weighted(STR, ExpMat, Gene2Weights, Match_DF):
    g_exp_level_zs = []
    err_gen = 0
    for g, weights in Gene2Weights.items(): 
        if not g in ExpMat.index.values:
            continue
        if not g in Match_DF.index.values:
            continue
        g_exp_level = ExpMat.loc[g, STR]
        #assert g_exp_level == g_exp_level
        if g_exp_level != g_exp_level:
            continue
        g_matches = Match_DF.loc[g, :].values
        g_matches_exps = ExpMat.loc[g_matches, STR].values
        g_matches_exps = g_matches_exps[~np.isnan(g_matches_exps)]
        g_exp_level_z = (g_exp_level - np.mean(g_matches_exps))/np.std(g_matches_exps)
        g_exp_level_weighted_z = g_exp_level_z * weights 
        if g_exp_level_weighted_z == g_exp_level_weighted_z:
            g_exp_level_zs.append(g_exp_level_weighted_z)
    avg_exp_level_z = np.mean(g_exp_level_zs)
    #print(err_gen)
    return avg_exp_level_z
## Match_DF: rows as genes, columns as trails
def ExpAVGWithExpMatch(ExpMat, Gene2Weights, Match_DF, csv_fil="ExpLevel.weighted.csv"):
    STRs = ExpMat.columns.values
    EFFs = []
    Regions = []
    str2reg_df = pd.read_csv("/Users/jiayao/Work/ASD_Circuits/src/dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    for i, STR in enumerate(STRs):
        avg_exp_level_z = ExpLevelOneSTR_Weighted(STR, ExpMat, Gene2Weights, Match_DF)
        EFFs.append(avg_exp_level_z)
        Regions.append(str2reg[STR])
    df = pd.DataFrame(data={"STR":STRs, "EFFECT":EFFs})
    df = df.sort_values("EFFECT", ascending=False)
    df.reset_index()
    df["Rank"] = df.index + 1
    g = LoadConnectome2()
    top_structs = df.head(50)["STR"].values
    top_nodes = g.vs.select(label_in=top_structs)
    g2 = g.subgraph(top_nodes)
    graph_size, exp_stats, exp_graphs, rm_vs = CircuitTrimming(g2, g)
    idx = np.argmax(exp_stats[:-10])
    print(idx)
    CIR_STRS = [v["label"] for v in exp_graphs[idx].vs]
    df['InCircuit'] = 0
    df["TrimRank"] = 1
    df.index = df["STR"].values
    for i, _str in enumerate([x["label"] for x in rm_vs]):
        df.loc[_str, "TrimRank"] = 50-i
        df.loc[_str, "InCircuit"] = 1 if _str in CIR_STRS else 0
    df.to_csv(csv_fil, index=False)
    return df


#####################################################################################
#####################################################################################
#### Circuits
#####################################################################################
#####################################################################################

def LoadConnectome(RankFil, ConnFil = ConnFil, TopN = 50, Bin = False, columns=["Structure", "Bias"]):
    adj_mat = pd.read_csv(ConnFil)
    node_names = adj_mat["ROW"].values
    adj_mat = adj_mat.drop("ROW", axis=1)
    A = adj_mat.values
    #g = ig.Graph.Adjacency((A > 0).tolist(), mode="ADJ_DIRECTED")
    g = ig.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = node_names  

    #if columns != None:
    #    struct_bias = pd.read_csv(RankFil, delimiter=",", header=None, names = columns)
    #else:
    #    struct_bias = pd.read_csv(RankFil, delimiter=",")
    if isinstance(RankFil, str):
        struct_bias = pd.read_csv(RankFil)
    #elif type(RankFil) == "pandas.core.frame.DataFrame":
    else:
        struct_bias = RankFil
    if "STR" in struct_bias.columns:
        top_structs = struct_bias.head(TopN)["STR"].values
    elif "STRUC" in struct_bias.columns:
        top_structs = struct_bias.head(TopN)["STRUC"].values
    return g, top_structs

def LoadConnectome2(ConnFil = ConnFil,  Bin = False):
    adj_mat = pd.read_csv(ConnFil)
    node_names = adj_mat["ROW"].values
    adj_mat = adj_mat.drop("ROW", axis=1)
    A = adj_mat.values
    g = ig.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = node_names  
    return g 

def EdgePermutation(g, top50nodes, edge_permute_stat, Npermute=1000):
    g2_ecounts = []
    NodeList = np.array(range(0, 213))
    for i in range(Npermute):
        g_ = g.copy()
        for idx_node in NodeList:
            out_edges = g.es.select(_source=idx_node) # select edges source as idx
            out_list = [x.tuple[1] for x in out_edges] # target of the edges
            out_pool = np.delete(NodeList, out_list) # other possible targets
            new_out = np.random.choice(out_pool, len(out_list)) # targets of this permute
            #in_edges = g.es.select(_target=idx_node)
            #in_list = [x.tuple[0] for x in in_edges]
            #in_pool = np.delete(NodeList, in_list)
            #new_in = np.random.choice(in_pool, len(in_list))
            g_.delete_edges(out_edges)
            g_.add_edges([(idx_node, x) for x in new_out])
        g2_ = g_.subgraph(top50nodes)
        g2_ecounts.append(g2_.ecount())
    return g2_ecounts

def LoadSTR2REG():
    df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
    STR2REG = dict(zip(df["STR"].values, df["REG"].values))
    REG2STR = {}
    for k,v in STR2REG.items():
        if v not in REG2STR:
            REG2STR[v] = [k]
        else:
            REG2STR[v].append(k)
    return STR2REG, REG2STR

# Keep N.structures from same region the same
def NodePermutation(g, topNodes, node_permute_stat, Npermute=100, tstat = "conn"):
    STR2REG, REG2STR = LoadSTR2REG()
    REGs = REG2STR.keys()
    REGCounts = dict(zip(REGs, [0]*len(REGs)))
    for node in topNodes:
        reg = STR2REG[node["label"]]
        REGCounts[reg] += 1
    Nulls = []
    for i in range(Npermute):
        _nodes = []
        g_ = g.copy()
        for reg in REGs:
            _strs = np.random.choice(REG2STR[reg], REGCounts[reg])
            _nodes.extend(_strs)
        _nodes = g_.vs.select(label_in=_nodes)
        idx_nodes = [x.index for x in _nodes]
        g2_ = g_.subgraph(idx_nodes)
        if tstat == "cohesive":
            Nulls.append(Cohesiveness(g_, idx_nodes))
        else:
            Nulls.append(g2_.ecount())
    return Nulls

# 
def NodePermutationBinom(g, topNodes, g2, Npermute=100, mode="Region"):
    if mode not in ["Region" ,"Random", "Percent"]:
        print("No such mode, only 'Region', 'random' or 'Percent'.")
        return
    if mode == "Region": # test N_top50_edges/N_top50_possible_edges == N_rand_top_50_edges/N_rand_top50_edges, with control for same number of regions.
        STR2REG, REG2STR = LoadSTR2REG()
        REGs = REG2STR.keys()
        REGCounts = dict(zip(REGs, [0]*len(REGs)))
        for node in g2.vs["label"]:
            reg = STR2REG[node]
            REGCounts[reg] += 1
        Ps = []
        for i in range(Npermute):
            _nodes = []
            g_ = g.copy()
            for reg in REGs:
                _strs = np.random.choice(REG2STR[reg], REGCounts[reg])
                _nodes.extend(_strs)
            _nodes = g_.vs.select(label_in=_nodes)
            idx_nodes = [x.index for x in _nodes]
            g_sub = g_.subgraph(idx_nodes)
            Ps.append(g_sub.ecount() / (topNodes * (topNodes-1)))
        r = np.mean(Ps)
        x = g2.ecount()
        n = topNodes * (topNodes-1)
        p_binom = binom_test(x, n, r)
    elif mode == "Random": # test N_top50_edges/N_top50_possible_edges == N_rand_top_50_edges/N_rand_top50_edges
        Ps = []
        ALL_NODES = g.vs["label"]
        for i in range(Npermute):
            g_ = g.copy()
            _nodes = np.random.choice(ALL_NODES, topNodes)
            _nodes = g_.vs.select(label_in=_nodes)
            idx_nodes = [x.index for x in _nodes]
            g_sub = g_.subgraph(idx_nodes)
            Ps.append(g_sub.ecount() / (topNodes * (topNodes-1)))
        r = np.mean(Ps)
        x = g2.ecount()
        n = topNodes * (topNodes-1)
        p_binom = binom_test(x, n, r)
    elif mode == "Percent": # test N_top50_edges_in/N_top50_total_edges 
        Ps = []
        ALL_NODES = g.vs["label"]
        for i in range(Npermute):
            g_ = g.copy()
            _nodes = np.random.choice(ALL_NODES, topNodes)
            _nodes = g_.vs.select(label_in=_nodes)
            idx_nodes = [x.index for x in _nodes]
            g_sub = g_.subgraph(idx_nodes)
            d = 0
            for v in g_.vs:
                if v["label"] in g_sub.vs["label"]:
                    d += v.degree()
            Ps.append(g_sub.ecount() / d)
        x = g2.ecount()
        n = 0
        for v in g.vs:
            if v["label"] in g2.vs["label"]:
                n += v.degree()
        r = np.mean(Ps)
        p_binom = binom_test(x, n, r)
    return x, n, r, p_binom

# complete random permutation
def NodePermutation2(g, Nnodes = 50, Npermute=1000):
    ALL_NODES = g.vs["label"]
    Nulls = []
    for i in range(Npermute):
        g_ = g.copy()
        _nodes = np.random.choice(ALL_NODES, Nnodes)
        _nodes = g_.vs.select(label_in=_nodes) 
        idx_nodes = [x.index for x in _nodes]
        Nulls.append(Cohesiveness(g_, idx_nodes))
    return Nulls

def GetPermutationP(null, obs):
    count = 0
    for i,v in enumerate(null):
        if obs >= v:
            count += 1
    return 1-float(count)/(len(null)+1)

def PlotPermutationP(Null, Obs):
    Z = (Obs - np.mean(Null)) / np.std(Null)
    n, bins, patches = plt.hist(Null, bins=20, histtype = "barstacked", align = 'mid', facecolor='black', alpha=0.8)
    p = GetPermutationP(Null, Obs)
    plt.vlines(x=Obs, ymin=0, ymax=max(n))
    plt.text(x=Obs, y=max(n), s="p=%.3f, z=%.3f"%(p,Z))
    plt.show()
    return

def splitEdges(g, idx_node, InNodes, OutNodes):
    InEdges, OutEdges = [], [] # edges within circuits or outside the circuits
    src_edges = g.es.select(_source=idx_node) 
    tgt_edges = g.es.select(_target=idx_node) 
    for edge in src_edges:
        _src, _tgt = edge.tuple
        if _tgt in InNodes:
            InEdges.append(edge)
        else:
            OutEdges.append(edge)
    for edge in tgt_edges:
        _src, _tgt = edge.tuple
        if _src in InNodes:
            InEdges.append(edge)
        else:
            OutEdges.append(edge)
    InEdges = sorted(InEdges, key=lambda x:x["weight"], reverse=True)
    OutEdges = sorted(OutEdges, key=lambda x:x["weight"], reverse=True)
    return InEdges, OutEdges

def CohesivenessSingleNode(g, STRList, TopN=1):
    NodeList = np.array(range(0, 213))
    Outside = np.delete(NodeList, STRList)
    skip = 0
    res = []
    for Node in STRList:
        weight_in = 0
        weight_out = 0
        InEdges, OutEdges = splitEdges(g, Node, STRList, Outside)
        weight_in += np.sum([x["weight"] for x in InEdges[:TopN]])
        weight_out += np.sum([x["weight"] for x in OutEdges[:TopN]])
        if weight_out != 0:
            cohe = weight_in / weight_out
            res.append(cohe)
        else:
            skip += 1
    if len(res) > 0:
        for i in range(skip):
            res.append(max(res))
    return res

def Cohesiveness(g, STRList, TopN=1):
    NodeList = np.array(range(0, 213))
    D = len(STRList)
    Outside = np.delete(NodeList, STRList)
    res = 0
    skip = 0
    max_ = 0
    for Node in STRList:
        # TopN connection
        weight_in = 0
        weight_out = 0
        InEdges, OutEdges = splitEdges(g, Node, STRList, Outside)
        weight_in += np.sum([x["weight"] for x in InEdges[:TopN]]) 
        weight_out += np.sum([x["weight"] for x in OutEdges[:TopN]])
        if weight_out != 0:
            res += weight_in / weight_out
            if res > max_:
                max_ = res
        else:
            skip += 1
    for i in range(skip):
        res += max_
    return res/D 

def InOutCohesiveSingleNode(g, g_, STR):
    d,d_ = 0, 0
    for idx, v in enumerate(g.vs):
        if v["label"] == STR:
            d = v.degree()
            break
    for idx_, v_ in enumerate(g_.vs):
        if v_["label"] == STR:
            d_ = v_.degree()
            break
    return d_/d

def InOutCohesiveAVG(g, g_):
    cohesives = []
    for v in g_.vs:
        coh = InOutCohesiveSingleNode(g, g_, v["label"])
        cohesives.append(coh)
    cohesive = np.mean(cohesives)
    return cohesive

def fromSameRegion(str1, str2, STR2REG):
    return STR2REG[str1] == STR2REG[str2] 

def TrimConnMat(ConnMat, STR2REG, distance):
    Structures = ConnMat.columns.values
    if distance == "long":
        for str1 in Structures:
            for str2 in Structures:
                if fromSameRegion(str1, str2, STR2REG):
                    ConnMat.loc[str1, str2] = 0
    elif distance == "long":
        for str1 in Structures:
            for str2 in Structures:
                if not fromSameRegion(str1, str2, STR2REG):
                    ConnMat.loc[str1, str2] = 0
    return ConnMat

def MWU_topN_conns(case_rank, cont_rank, expMatFil, topN=50, mode="connectivity", distance="all"):
    assert mode in ["connectivity", "cohesiveness"]
    case_str_rank = pd.read_csv(case_rank, delimiter="\t")
    case_top50 = case_str_rank.head(topN)["STR"].values
    cont_str_rank = pd.read_csv(cont_rank, delimiter="\t")
    cont_top50 = cont_str_rank.head(topN)["STR"].values
    ConnMat = pd.read_csv(expMatFil, index_col="ROW")
    if distance == "all":
        pass
    else:
        STR2REG, REG2STR = LoadSTR2REG()
        if distance == "long":
            ##
            ConnMat = TrimConnMat(ConnMat, STR2REG, distance="long")
        elif distance == "short":
            ConnMat = TrimConnMat(ConnMat, STR2REG, distance="short")
            ##

    if mode == "connectivity":
        CaseConnMat = ConnMat.filter(items=case_top50, axis=0)
        CaseConnMat = CaseConnMat.filter(items=case_top50, axis=1)
        ContConnMat = ConnMat.filter(items=cont_top50, axis=0)
        ContConnMat = ContConnMat.filter(items=cont_top50, axis=1)

        case_conns = CaseConnMat.values.flatten()
        cont_conns = ContConnMat.values.flatten()
        #case_conns = [min(x, 1) for x in case_conns] 
        #cont_conns = [min(x, 1) for x in cont_conns] 
        case_conns = [x for x in case_conns] 
        cont_conns = [x for x in cont_conns] 
        return case_conns, cont_conns
    if mode == "cohesiveness":
        #g, top_structs = LoadConnectome2("dat/asd.data_vs_match.str.rank.tsv", Bin=True)    
        g = LoadConnectome2(Bin=True)    
        case_top50_idx = [x.index for x in g.vs.select(label_in=case_top50)]
        cont_top50_idx = [x.index for x in g.vs.select(label_in=cont_top50)]
        case_cohe = CohesivenessSingleNode(g, case_top50_idx)
        cont_cohe = CohesivenessSingleNode(g, cont_top50_idx)
        return case_cohe, cont_cohe

def ShowConn(ConnMatFil, ssc_rank, spark_rank, tada_rank, cont_rank, mode="connectivity", labels=["SSC","SPARK","TADA","SIB"]):
    slices = range(10, 255, 5)
    ssc_dat = []
    spark_dat = []
    tada_dat = []
    sib_dat = []
    for i in slices:
        ssc_conns, cont_conns = MWU_topN_conns(ssc_rank, cont_rank, ConnMatFil, topN=i, mode=mode)
        spark_conns, cont_conns = MWU_topN_conns(spark_rank, cont_rank, ConnMatFil, topN=i, mode=mode)
        tada_conns, cont_conns = MWU_topN_conns(tada_rank, cont_rank, ConnMatFil, topN=i, mode=mode)
        t1, p1 = scipy.stats.mannwhitneyu(ssc_conns, cont_conns, alternative='greater')
        t2, p2 = scipy.stats.mannwhitneyu(spark_conns, cont_conns, alternative='greater')
        t3, p3 = scipy.stats.mannwhitneyu(tada_conns, cont_conns, alternative='greater')
        if i == 50:
            print("%s: %.3e\t%s: %.3e\t%s: %.3e" % (labels[0], p1, labels[1], p2, labels[2], p3))
        ssc_dat.append((np.mean(ssc_conns), p1))
        spark_dat.append((np.mean(spark_conns), p2))
        tada_dat.append((np.mean(tada_conns), p3))
        sib_dat.append((np.mean(cont_conns), 0))
    fig, (ax1, ax2) = plt.subplots(2, dpi=120)
    ax1.plot(slices, -np.log10([x[1] for x in ssc_dat]), label=labels[0])
    ax1.plot(slices, -np.log10([x[1] for x in spark_dat]), label=labels[1])
    ax1.plot(slices, -np.log10([x[1] for x in tada_dat]), label=labels[2])
    ax1.hlines(y=-np.log10(0.05), xmin=0, xmax=250)
    ax1.grid()
    ax1.legend(loc="upper right")
    ax1.set_xlabel("N STR")
    ax1.set_ylabel("-log10(P)")
    ax2.plot(slices, ([x[0] for x in ssc_dat]), label=labels[0])
    ax2.plot(slices, ([x[0] for x in spark_dat]), label=labels[1])
    ax2.plot(slices, ([x[0] for x in tada_dat]), label=labels[2])
    ax2.plot(slices, ([x[0] for x in sib_dat]), label=labels[3])
    ax2.grid()
    ax2.legend(loc="upper right")
    ax2.set_xlabel("N STR")
    ax2.set_ylabel("Avg Conn Stregenth")
    plt.show()

def optimize_stat_in_n_out(graph, complate_graph):
    d = 0
    for v in complate_graph.vs:
        if v["label"] in graph.vs["label"]:
            d += v.degree()
    return graph.ecount() / d

def argmax_optimize_stat(graph, complate_graph, optimize_stat="in_n_out"):
    opt_stats, graphs, vs = [], [], []
    for i, v in enumerate(graph.vs):
        tmp_graph = graph.copy()
        tmp_graph.delete_vertices(v)
        #stat = optimize_stat_in_n_out(tmp_graph, complate_graph)
        stat = InOutCohesiveAVG(complate_graph, tmp_graph)
        opt_stats.append(stat)
        graphs.append(tmp_graph)
        vs.append(v)
    idx = np.argmax(opt_stats)
    trimmed_v = vs[idx]
    stat = opt_stats[idx]
    graph = graphs[idx]
    return graph, stat, trimmed_v

def CircuitTrimming(graph, complate_graph, optimize_stat="in_n_out"):
    trim_graph = graph.copy()
    stats, graph_size, graphs, trimmed_Vs = [], [], [], []
    for i in range(len(graph.vs), 1, -1):
        trim_graph, stat, trimmed_v = argmax_optimize_stat(trim_graph, complate_graph, optimize_stat)
        graphs.append(trim_graph.copy())
        stats.append(stat)
        graph_size.append(i)
        trimmed_Vs.append(trimmed_v)
    trimmed_Vs.append(trim_graph.vs[0])
    return graph_size, stats, graphs, trimmed_Vs

def Bias_vs_Cohesiveness(g, df, topN=50, conn="AvgCohesive", title="bias vs cohesiveness"):
    assert conn in ["AvgCohesive", "TotalCohesive", "Top5Edges"]
    topN_STRs = df.head(topN).index.values
    g_ = g.subgraph(g.vs.select(label_in=topN_STRs))
    cohesives = []
    bias = []
    for v in g_.vs:
        if conn == "AvgCohesive":
            coh = InOutCohesiveSingleNode(g, g_, v["label"])
        elif conn == "Top5Edges":
            coh = TopNConn(g, g_)
        cohesives.append(coh)
        bias.append(df.loc[v["label"], "EFFECT"])
    #cohesive = np.mean(cohesives)
    plt.scatter(bias, cohesives)
    r, p = pearsonr(bias, cohesives)
    plt.text(x = min(bias), y = max(cohesives)*0.8, s="r=%.2f, p=%.2e"%(r, p))
    plt.title(title)
    plt.xlabel("bias")
    plt.ylabel("Cohesiveness")
    plt.show()
    return 

def Agg_vs_Indv_gene_Binomial_Test_Cohesiveness(g, agg_df, indv_df, topN=50, conn="TotalCohesive"):
    # Get P_0 from indv res
    if conn == "TotalCohesive":
        #cohesives = []
        Nedge_Inside = 0
        Nedge_Total = 0
        for c in indv_df.columns.values:
            dat = indv_df[c]
            top50 = dat.sort_values(ascending=False).index[:topN]
            g_ = g.subgraph(g.vs.select(label_in=top50))
            d = 0
            for v in g.vs:
                if v["label"] in g_.vs["label"]:
                    d += v.degree()
            Nedge_Inside += g_.ecount()
            Nedge_Total += d
        P_0 = Nedge_Inside / Nedge_Total #np.mean(cohesives)
        top50 = agg_df.head(topN).index.values
        g_agg = g.subgraph(g.vs.select(label_in=top50))
        n = 0
        for v in g.vs:
            if v["label"] in g_agg.vs["label"]:
                n += v.degree()
            x = g_agg.ecount()
        p = binom_test(x, n ,P_0)
        return p, x/n, P_0

def Agg_vs_Indv_gene_PermutationLikeCohesiveness(g, agg_df, indv_df, topN=50):
    cohesives = []
    top50_str = agg_df.head(topN).index.values
    g_ = g.subgraph(g.vs.select(label_in=top50_str))
    for v in g_.vs:
        coh = InOutCohesiveSingleNode(g, g_, v["label"])
        cohesives.append(coh)
    cohesive = np.mean(cohesives)
    cohesives = []
    for c in indv_df.columns.values:
        dat = indv_df[c]
        top50 = dat.sort_values(ascending=False).index[:topN]
        g_ = g.subgraph(g.vs.select(label_in=top50))
        cohs = []
        for v in g_.vs:
            coh = InOutCohesiveSingleNode(g, g_, v["label"])
            cohs.append(coh)
        cohesives.append(np.mean(cohs))
    PlotPermutationP(cohesives, cohesive)

#####################################################################################
# Other Methods
#####################################################################################
def QQplot(pvalues, title="QQ plot"):
    #pvalues.sort(reversed=True)
    pvalues = sorted(list(pvalues), reverse=True)
    Qvalues = []
    for x in pvalues:
        try:
            Qvalues.append(min(10, -math.log(x,10)))
        except:
            print(x)
    top = int(Qvalues[-1]) + 1
    NumTest = len(Qvalues)
    Qvalues = Qvalues
    Qexp = []
    for i in range(len(Qvalues)):
        Qexp.append(float(i+1)/NumTest)
    Qexp.sort(reverse=True)
    Qexp = [-1*math.log(x,10) for x in Qexp]
    plt.figure(dpi=120)
    plt.subplot()
    plt.scatter(Qexp, Qvalues, alpha=0.5, s=40, facecolors='none', edgecolors='b')
    plt.plot([0, top], [0, top], ls="-")
    plt.title(title)
    plt.xlabel('Exp Q')
    plt.ylabel('Obs Q')
    plt.show()


def CompareSTROverlap(DF1, DF2, name1, name2, topN=50):
    RD1 = RegionDistributions(DF1.set_index("STR"))
    RD2 = RegionDistributions(DF2.set_index("STR"))
    Regions = list(set(list(RD1.keys()) + list(RD2.keys())))
    Regions.sort()
    CompareDat = [] # Region, STR_common, STR_only_DF1, STR_only_DF2, N_STR_common, N_STR_only_DF1, N_STR_only_DF2
    for region in Regions:
        STR1s = set(RD1[region])
        STR2s = set(RD2[region])
        if (len(STR1s) + len(STR2s)) == 0:
            continue
        common = list(STR1s.intersection(STR2s))
        only1 = list(STR1s.difference(STR2s))
        only2 = list(STR2s.difference(STR1s))
        CompareDat.append([region, ", ".join(common), ", ".join(only1), ", ".join(only2), len(common), len(only1), len(only2)]) 
    CompareDF = pd.DataFrame(data=CompareDat, columns=["Region", "STR.Common", "STR.only.in.{}".format(name1), "STR.only.in.{}".format(name2), "N.common", "N.only.{}".format(name1), "N.only.{}".format(name2)])
    fig = plt.figure(dpi=200)
    ax = fig.add_axes([0,0,1,1])
    width = 0.3
    commons = CompareDF["N.common"].values
    only1s = CompareDF["N.only.{}".format(name1)].values
    only2s = CompareDF["N.only.{}".format(name2)].values
    #print(commons, only1s, only2s)
    ind = np.arange(len(commons))
    ax.barh(ind+width/2, commons, width, color='b', label="common")
    ax.barh(ind+width/2, only1s, width, left=commons, color='r', label="only.in.{}".format(name1))
    ax.barh(ind-width/2, commons, width, color='b')
    ax.barh(ind-width/2, only2s, width, left=commons, color='yellow', label="only.in.{}".format(name2))
    ax.set_yticks(ind)
    ax.set_yticklabels(CompareDF["Region"].values)
    ax.legend()
    ax.grid(True)
    plt.title("Top{} STRs in {} and {}".format(topN, name1, name2))
    plt.show()
    return CompareDF

def ShowTrimmingProfile(DFs, Names, topN=50):
    g = LoadConnectome2()
    fig, ax = plt.subplots(dpi=120)
    for df, name in zip(DFs, Names):
        g_ = g.copy()
        top_structs = df.head(topN)["STR"]
        top_nodes = g_.vs.select(label_in=top_structs)
        g2 = g_.subgraph(top_nodes)
        graph_size, exp_stats, exp_graphs, trimmed_Vs = CircuitTrimming(g2, g_)
        idx = np.argmax(exp_stats[:topN-10])
        ax.scatter(topN-idx, exp_stats[idx], marker='x', color="black")
        ax.plot(graph_size, exp_stats, marker="o", label=name, alpha=0.7)
    ax.set_xlim(50, 0)
    plt.legend()
    plt.show()

def CompileMethods(DFs, names):
    Circuits = []
    STRs = []
    for DF in DFs:
        STRs.extend(DF.index.values)
        Cirtuit = None
    STRs = list(set(STRs))
    Counts_Regions = {}
    Counts_Circuits = {}
    for STR in STRs:
        for DF_name, DF in zip(DF_names, DFs):
            if STR in DF.index.values:
                Counts[STR] += 1
                DF_name[STR] = 1
            else:
                DF_name[STR] = 0

def CountNDD(DF, Dmis="REVEL", TotalProband=5264):
    #LGD = ['splice_acceptor_variant','splice_donor_variant','stop_lost','frameshift_variant','splice_region_variant','start_lost','stop_gained','protein_altering_variant']
    LGD = ['splice_acceptor_variant','splice_donor_variant','stop_lost','frameshift_variant','start_lost','stop_gained']
    LGD_DF = DF[DF["VEP_functional_class_canonical"].isin(LGD)]
    MIS_DF = DF[DF["VEP_functional_class_canonical"]=="missense_variant"]
    #DMIS = MIS_DF[MIS_DF["MPC"]>0]
    if Dmis == "MPC":
        DMIS = MIS_DF[MIS_DF["MPC"]>1]
    elif Dmis == "REVEL":
        DMIS = MIS_DF[MIS_DF["REVEL"]>0.5]
    GENES = list(set(DF["GENE_NAME"].values))
    RES = {}
    for gene in GENES:
        gene_df_LGD = LGD_DF[LGD_DF["GENE_NAME"]==gene]
        N_LGD = gene_df_LGD.shape[0]
        gene_df_DMIS = DMIS[DMIS["GENE_NAME"]==gene]
        N_DMIS = gene_df_DMIS.shape[0]
        RES[gene] = {}
        RES[gene]["LGD"] = N_LGD
        RES[gene]["Dmis"] = N_DMIS
        RES[gene]["All"] = N_LGD + N_DMIS
    return RES

def myASDNDDClassifier(NASD, NNDD, NASDProband=6430, NNDDProband=5264):
    if NASD/NASDProband > NNDD/NNDDProband:
        return "ASD_p"
    else:
        return "ASD_NDD"

def CompareList(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    print("Common: %d"%len(set1.intersection(set2)), set1.intersection(set2))
    print("Present in 1: %d"%len(set1.difference(set2)), set1.difference(set2))
    print("Present in 2: %d"%len(set2.difference(set1)), set2.difference(set1))

#####################################################################################
# Depricated Methods
#####################################################################################
