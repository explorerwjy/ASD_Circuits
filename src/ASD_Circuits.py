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
from collections import Counter
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
import gzip as gz
from SA import *
import pickle as pk
import copy

plt.rcParams['figure.dpi'] = 80

#ConnFil = "../dat/allen-mouse-conn/connectome-log.csv"
ConnFil = "../dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv"
MajorBrainDivisions = "./dat/structure2region.tsv" 


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

def ZscoreConverting(values, mean = np.nan, std = np.nan):
    real_values = [x for x in values if x==x]
    if mean != mean:
        mean = np.mean(real_values)
    if std != std:
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

def loadgenelist(fil, toint=True):
    if fil.endswith(".gz"):
        if toint:
            return [int(x.strip()) for x in gz.open(fil, "rt")]
        else:
            return [x.strip() for x in gz.open(fil, "rt")]
    else:
        if toint:
            return [int(x.strip()) for x in open(fil, "rt")]
        else:
            return [x.strip() for x in open(fil, "rt")]

def filtergenelist(genelist, allgenelist):
    return [x for x in genelist if x in allgenelist]

def List2Fil(List, filename):
    fout = open(filename, 'wt')
    for x in List:
        fout.write(str(x)+"\n")

def LoadList(filename):
    fin = open(filename, 'rt')
    return [x.strip() for x in fin.readlines()]

def WriteGeneList(genelist, filname):
    fout = open(filname, "wt")
    for gen in genelist:
        try:
            fout.write(str(int(gen)) + "\n")
        except:
            continue

def modify_str(x):
    x = re.sub("[()]", "", x)
    x = re.sub("-", "_", x)
    x = re.sub("reunions", "reuniens", x)
    x = "_".join(x.split(" "))
    return x

def NeuronDensityNorm():
    STRs = [x.strip() for x in open("/Users/jiayao/Work/ASD_Circuits/dat/allen-mouse-exp/Structures.txt", 'rt')]
    STR_cell_comp = cell_comp[cell_comp.index.isin(STRs)]


def clean_name(name):
    name = re.sub('[()]', '', name)
    name = re.sub('[-]', ' ', name)
    name = "_".join(name.split())
    return name

def conver_cartesian_distance(adj_mat_distal):
    Cartesian_distances = pd.read_excel("/Users/jiayao/Work/ASD_Circuits/dat/allen-mouse-conn/Dist_CartesianDistance.xlsx", index_col=0)
    ontology = pd.read_csv("dat/ontology.csv")
    acronym2name = {}
    name2arconym = {}
    for i, row in ontology.iterrows():
        acronym2name[row["acronym"]] = clean_name(row["safe_name"])
        name2arconym[clean_name(row["safe_name"])] = row["acronym"]
    STRs = adj_mat_distal.index.values
    Dat = []
    for STR_i in STRs:
        STR_acr_i = name2arconym[STR_i] + "_ipsi"
        dat_row = []
        for STR_j in STRs:
            STR_acr_j = name2arconym[STR_j] + "_ipsi"
            dist = Cartesian_distances.loc[STR_acr_i, STR_acr_j]
            dat_row.append(dist)
        Dat.append(dat_row)
    Cartesian_dist_DF = pd.DataFrame(data=Dat, index = STRs, columns=STRs)
    Cartesian_dist_DF.to_csv("../dat/allen-mouse-conn/Dist_CartesianDistance.csv")

def Filt_LGD_Mis(DF, Dmis=True):
    dat= []
    for i, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
            dat.append(row.values)
        elif GeneEff == "missense":
            if Dmis:
                if GeneEff == "missense":
                    row["REVEL"] = row["REVEL"].split(";")[0]
                    if row["REVEL"] != ".":
                        if float(row["REVEL"]) > 0.5:
                            dat.append(row.values)
            else:
                if GeneEff == "missense":
                    dat.append(row.values)
    return pd.DataFrame(dat, columns = DF.columns.values)

#####################################################################################
#####################################################################################
#### Expression Bias
#####################################################################################
#####################################################################################
def LoadGeneINFO():
    HGNC = pd.read_csv("../dat/genes/protein-coding_gene.txt", delimiter="\t", low_memory=False)
    ENSID2Entrez = dict(zip(HGNC["ensembl_gene_id"].values, HGNC["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC["symbol"].values, HGNC["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC["entrez_id"].values, HGNC["symbol"].values))
    allen_mouse_genes = loadgenelist("../dat/allen-mouse-exp/allen-mouse-gene_entrez.txt")
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol, allen_mouse_genes

def LoadExpressionMatrices(ExpMat = "../dat/allen-mouse-exp/energy-conn-model.csv", 
                        ExpZscoreMat = "../dat/allen-mouse-exp/energy-zscore-conn-model.0524.csv",
                        ExpMatNorm = "../dat/allen-mouse-exp/energy-neuronorm.csv",
                        ExpZscoreMatNorm = "../dat/allen-mouse-exp/energy-zscore-neuronorm.csv"):
    ExpMat = pd.read_csv(ExpMat, index_col="ROW")
    ExpZscoreMat = pd.read_csv(ExpZscoreMat, index_col="ROW")
    ExpMatNorm = pd.read_csv(ExpMatNorm, index_col="ROW")
    ExpZscoreMatNorm = pd.read_csv(ExpZscoreMatNorm, index_col="ROW")
    return ExpMat, ExpZscoreMat, ExpMatNorm, ExpZscoreMatNorm


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

def ExpressionMatchGeneSet(geneset, match_feature, ExpText="Exp.Volume.Weighted.Mean", savefil = None, interval_len = 500, sample_size=1000): 
    genes_to_match = [] # genes
    dat = []
    xx_genes = []
    for i, gene in enumerate(geneset):
        try:
            gene_exp = match_feature.loc[gene, ExpText]
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
        Interval_exps = Interval[ExpText].values
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
    str2reg_df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    return str2reg

#def ModifySTR2REGTab():
#    str/MajorBrainDivisions


def RegionDistributions(DF, topN=50, show = False, states=np.zeros(5)):
    #ax = fig.add_axes([0,0,1,1])
    str2reg_df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    Regions = list(set(str2reg.values()))
    RegionCount = {}
    for region in Regions:
        RegionCount[region] = []
    if states.sum() == 0:
        for x in DF.head(topN).index:
            region = str2reg[x]
            RegionCount[region].append(x)
        if show:
            for k,v in RegionCount.items():
                if len(v) == 0:
                    continue
                print(k, "\t", len(v), "\t", "; ".join(v))
        return RegionCount
    else:
        CandidateNodes = DF.head(topN).index.values
        Cir_STRs = CandidateNodes[np.where(states==1)[0]]
        for x in DF.head(topN).index:
            if x in Cir_STRs:
                region = str2reg[x]
                RegionCount[region].append(x)
        if show:
            for k,v in RegionCount.items():
                if len(v) == 0:
                    continue
                print(k, "\t", len(v), "\t", "; ".join(v))
        return RegionCount

def RegionDistributionsList(List, topN=50):
    str2reg_df = pd.read_csv(MajorBrainDivisions, delimiter="\t")
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
    #if FDR != None:
    #    MutFil = MutFil[MutFil["Qvalue"]<FDR]
    gene2None, gene2RR, gene2MutN, gene2Cons, gene2MutNCons = {}, {}, {}, {}, {} 
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            continue
        gene2None[g] = 1
        gene2MutN[g] = row["dnLGD"]*0.347 + row["dnDmis"]*0.194
    return gene2None, gene2MutN

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

def Aggregate_top_Genes(MutFil):
    Agg_DWest = MutFil[MutFil["pDenovoWEST"]<1.33e-6]
    Agg_TADA_Rank = MutFil.sort_values("Qvalue").head(100)
    MutFil = pd.concat([Agg_DWest,Agg_TADA_Rank]).drop_duplicates().reset_index(drop=True)
    return MutFil

def Aggregate_Gene_Weights(MutFil, FDR="Candidate", out=None):
    if FDR == "Candidate":
        Agg_DWest = MutFil[MutFil["pDenovoWEST"]<1.33e-6]
        Agg_TADA_Rank = MutFil.sort_values("Qvalue").head(100)
        #print(max(Agg_TADA_Rank["Qvalue"].values))
        MutFil = pd.concat([Agg_DWest,Agg_TADA_Rank]).drop_duplicates().reset_index(drop=True)
    elif FDR == None:
        pass 
    else:
        MutFil = MutFil[MutFil["Qvalue"]<FDR]
        #MutFil = MutFil[MutFil["pDenovoWEST"]<FDR]
    gene2None, gene2MutN = {}, {}
    #print(MutFil.shape)
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            continue
        gene2None[g] = 1
        gene2MutN[g] = row["dnLGD"]*0.357 + row["dnDmis"]*0.231
        #gene2MutN[g] = row["dnLGD"]*0.457 + row["dnDmis"]*0.231
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2None, gene2MutN

def Aggregate_Gene_Weights2(MutFil, allen_mouse_genes, pLI=True, out=None):
    gene2None, gene2MutN = {}, {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
            if g not in allen_mouse_genes:
                continue
        except:
            continue
        gene2None[g] = 1
        #pLI = gnomad_cons.loc[row["HGNC"], "exac_pLI"]
        if pLI:
            try:
                pLI = float(row["ExACpLI"])
            except:
                pLI = 0.0
            if pLI >= 0.5:
                gene2MutN[g] = row["AutismMerged_LoF"]*0.554 + row["AutismMerged_Dmis_REVEL0.5"]*0.333
            else:
                gene2MutN[g] = row["AutismMerged_LoF"]*0.138 + row["AutismMerged_Dmis_REVEL0.5"]*0.130
        else:
            gene2MutN[g] = row["AutismMerged_LoF"]*0.457 + row["AutismMerged_Dmis_REVEL0.5"]*0.231
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2None, gene2MutN

def SCZOwen_Gene_Weights(MutFil, FDR=0.05, out=None):
    MutFil = MutFil[MutFil["Pvalue"]<FDR]
    gene2None, gene2MutN = {}, {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            continue
        gene2None[g] = 1
        gene2MutN[g] = row["N_LGD"]
        #gene2MutN[g] = row["dnLGD"]*0.357 + row["dnDmis"]*0.231
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2None, gene2MutN

def Sibling_Gene_Weights(MutFil, GeneSymbol2Entrez, out=None):
    gene2None, gene2MutN = {}, {}
    for i, row in MutFil.iterrows():
        try:
            #g = int(row["Entrez"])
            g = int(GeneSymbol2Entrez[row["HGNC"]])
        except:
            continue
        gene2None[g] = 1
        #gene2MutN[g] = row["dnv_LGDs_sib"] + row["dnv_missense_sib"] 
        gene2MutN[g] = row["N_LGD"] + row["N_Mis"] 
        #gene2MutN[g] = row["dnLGD"]*0.357 + row["dnDmis"]*0.231
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2None, gene2MutN

def Mut2GeneDF(MutDF, GeneSymbol2Entrez, pLI=True):
    Select_Genes = np.array(list(set(MutDF["HGNC"].values)))
    dat = []
    gene2MutN = {}
    for g in Select_Genes:
        try:
            Entrez = int(GeneSymbol2Entrez[g])
        except:
            Entrez = -1
            continue
        Muts = MutDF[MutDF["HGNC"]==g]
        N_LGD, N_Mis, N_Dmis, N_Syn = CountMut(Muts)
        if pLI:
            try:
                pLI = float(Muts["ExACpLI"].values[0])
            except:
                pLI = 0.0
            if pLI >= 0.5:
                gene2MutN[Entrez] = N_LGD * 0.554 + N_Dmis * 0.333
            else:
                gene2MutN[Entrez] = N_LGD * 0.138 + N_Dmis * 0.130
        #if Dmis:
        #       gene2MutN[Entrez] = N_LGD * 1 + N_Dmis * 1
        else:
            gene2MutN[Entrez] = N_LGD * 0.357 + N_Dmis * 0.231
    return gene2MutN


def CountMut(DF, DmisSTR="REVEL", DmisCut=0.5):
    N_LGD, N_mis, N_Dmis, N_syn = 0,0,0,0
    for i, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor", "start_lost", "stop_gained", "stop_lost"]:
            N_LGD += 1
        elif GeneEff == "missense":
            N_mis += 1
        if row["REVEL"] == ".":
            continue
        elif float(row["REVEL"]) >= 0.5:
            N_Dmis += 1
        elif GeneEff == "synonymous":
            N_syn += 1
    return N_LGD, N_mis, N_Dmis, N_syn

###################################################################################################################
# Weighted Bias Calculattion
###################################################################################################################
# Expression Specificity
def ZscoreOneSTR_Weighted_Indv_Gene(STR, ExpZscoreMat, Gene2Weights, Method = 1, Match_DF = None):
    assert Method in [1, 2]
    res = []
    sum_weight = 0
    if Method == 1:
        for gene, weight in Gene2Weights.items():
            if gene not in ExpZscoreMat.index.values:
                continue
            #score = weight * max(0, ExpZscoreMat.loc[gene, STR])
            score = weight * ExpZscoreMat.loc[gene, STR]
            if score == score:
                res.append(score)
                sum_weight += weight
        return res
    elif Method == 2:
        for gene, weight in Gene2Weights.items():
            if gene not in ExpZscoreMat.index.values:
                continue
            z_gene = ExpZscoreMat.loc[gene, STR]
            g_matches = Match_DF.loc[gene, :].values 
            z_matches = [ExpZscoreMat.loc[g, STR] for g in g_matches]
            z_matches = [x for x in z_matches if x==x]
            b = (z_gene - np.mean(z_matches)) / np.std(z_matches) 
            score = weight * b
            if score == score:
                res.append(score)
                sum_weight += weight
        return res

def Z2_Gene_STR(gene, STR, ExpZscoreMat, Match_DF):
    z_gene = ExpZscoreMat.loc[gene, STR]
    g_matches = Match_DF.loc[gene, :].values
    z_matches = [ExpZscoreMat.loc[g, STR] for g in g_matches]
    z_matches = [x for x in z_matches if x==x]
    b = (z_gene - np.mean(z_matches)) / np.std(z_matches)
    return b

def Z2_GeneSet(ExpMat, Gene2Weights, Match_DF, csv_fil="ExpSpec.csv"):
    STRs = ExpMat.columns.values
    str2reg = STR2Region()
    EFFECTS, Regions = [], []
    for i, STR in enumerate(STRs):
        biases, weights = [], []
        for gene, w in Gene2Weights.items():
            if gene not in ExpMat.index.values:
                continue
            b = Z2_Gene_STR(gene, STR, ExpMat, Match_DF)
            if b==b:
                biases.append(b)
                weights.append(w)
        sorted_bias, sorted_weights = sort_lists([biases, weights])
        trim_bias = sorted_bias[1:-2]
        trim_weights = sorted_weights[1:-2]
        STR_bias = np.average(trim_bias, weights=trim_weights)
        EFFECTS.append(STR_bias)
        Regions.append(str2reg[STR])
    df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS, "REGION":Regions})
    df = df.sort_values("EFFECT", ascending=False)
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    if csv_fil != None:
        df.to_csv(csv_fil, index=False)
    return df



def ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights, Method = 1, Match_DF = None, NonNeg=False):
    assert Method in [1, 2, 3]
    res = []
    sum_weight = 0
    if Method == 1:
        for gene, weight in Gene2Weights.items():
            if gene not in ExpZscoreMat.index.values:
                continue
            if NonNeg:
                score = weight * max(0, ExpZscoreMat.loc[gene, STR])
            else:
                score = weight * ExpZscoreMat.loc[gene, STR]
            if score == score:
                res.append(score)
                sum_weight += weight
        return np.sum(res)/sum_weight
    elif Method == 2:
        for gene, weight in Gene2Weights.items():
            if gene not in ExpZscoreMat.index.values:
                continue
            z_gene = ExpZscoreMat.loc[gene, STR]
            g_matches = Match_DF.loc[gene, :].values 
            g_matches = [x for x in g_matches if x in ExpZscoreMat.index.values]
            z_matches = [ExpZscoreMat.loc[g, STR] for g in g_matches]
            z_matches2 = []
            for xxxx in z_matches:
                try:
                    if xxxx == xxxx:
                        z_matches2.append(xxxx)
                except:
                    continue
            z_matches = z_matches2
            #print(z_matches)
            #print(type(z_matches))
            #print(z_matches[0])
            #z_matches = [x for x in z_matches if x==x]
            b = (z_gene - np.mean(z_matches)) / np.std(z_matches) 
            #b = z_gene - np.mean(z_matches)
            if NonNeg:
                score = weight * max(0, b)
            else:
                score = weight * b
            #print(gene, score, weight)
            if score == score:
                res.append(score)
                sum_weight += weight
        return np.sum(res)/sum_weight
    if Method == 3: ## Quantile Weighted Average
        genes = []
        Values = []
        weights = []
        for gene, weight in Gene2Weights.items():
            if gene in ExpZscoreMat.index.values:
                genes.append(gene)
                Values.append(ExpZscoreMat.loc[gene, STR])
                weights.append(weight)
        minV = min(Values)
        def NanToV(value, V):
            if value != value:
                return V
            else:
                return value
        Values_tosort = [NanToV(x, minV-1) for x in Values]
        Total = len([x for x in Values if x==x])
        idx = np.argsort(Values_tosort) # index to sort Values
        quantiles = []
        for i, (idx, v) in enumerate(zip(idx, Values)):
            if v == v:
                quantile = idx/Total
                quantiles.append(quantile)
            else:
                quantiles.append(0)
        return np.average(quantiles, weights=weights)

def AvgSTRZ_Weighted(ExpZscoreMat, Gene2Weights, Method = 1, Match_DF=0, NonNeg=False, BS_Weights=True, csv_fil=None, ConnFil=ConnFil):
    assert Method in [1,2,3] # 1: AvgZ; 2: gene Z normed by Matched gene Z; 3: gene set avgZ normed by Matched set Z;
    STRs = ExpZscoreMat.columns.values
    EFFECTS = []; Normed_EFFECTS = []
    Regions = []
    str2reg = STR2Region()
    for i, STR in enumerate(STRs):
        if Method == 1:
            mean_z = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights, Method = 1, NonNeg=NonNeg)
            STR_Bias = mean_z
        if Method == 2:
            mean_b = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights, Match_DF=Match_DF, Method = 2, NonNeg=NonNeg)
            STR_Bias = mean_b
        if Method == 3:
            mean_q = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights, Method = 3)
            STR_Bias = mean_q
        if Method == 4: 
            mean_z = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, Gene2Weights, Method = 1, NonNeg=NonNeg)
            Match_mean_Zs = []
            for g in Match_DF.columns:
                match_genes = Match_DF[g].values
                if BS_Weights:
                    weights = np.random.choice(list(Gene2Weights.values()), len(match_genes))
                    BS_MG2Weights = dict(zip(match_genes, weights))
                else:
                    BS_MG2Weights = dict(zip(match_genes, [1]*len(match_genes)))
                match_z = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, BS_MG2Weights)
                Match_mean_Zs.append(match_z)
            normed_effect = (mean_z - np.mean(Match_mean_Zs)) / np.std(Match_mean_Zs)
            STR_Bias = normed_effect
        EFFECTS.append(STR_Bias)
        Regions.append(str2reg[STR])
    df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS, "REGION":Regions})
    #df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS})
    df = df.sort_values("EFFECT", ascending=False)
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    if csv_fil != None:
        df.to_csv(csv_fil, index=False)
    return df

def AvgSTRZ_Weighted_TwoSet(ExpZscoreMat, GeneSet1, GeneSet2, csv_fil=None):
    STRs = ExpZscoreMat.columns.values
    str2reg_df = pd.read_csv("./dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    for i, STR in enumerate(STRs):
        Z1 = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, GeneSet1, Method = 1)
        Z2 = ZscoreOneSTR_Weighted(STR, ExpZscoreMat, GeneSet2, Method = 1)
        Diff = Z1-Z2
        EFFECTS.append(Diff)
        Regions.append(str2reg[STR])
    df = pd.DataFrame(data = {"STR": STRs, "EFFECT":EFFECTS, "REGION":Regions})
    df = df.sort_values("EFFECT", ascending=False)
    df = df.reset_index(drop=True)
    df["Rank"] = df.index + 1
    if csv_fil != None:
        df.to_csv(csv_fil, index=False)
    return df

# Expression Level (Not Used May2021)
def ExpLevelOneSTR_Weighted(STR, ExpMat, Gene2Weights, Match_DF):
    g_exp_level_zs = []
    err_gen = 0
    sum_weight = 0
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
            sum_weight += weights
    avg_exp_level_z = np.sum(g_exp_level_zs)/sum_weight
    #print(err_gen)
    return avg_exp_level_z
## Match_DF: rows as genes, columns as trails
def ExpAVGWithExpMatch(ExpMat, Gene2Weights, Match_DF, csv_fil="ExpLevel.weighted.csv", ConnFil=ConnFil):
    STRs = ExpMat.columns.values
    EFFs = []
    Regions = []
    str2reg_df = pd.read_csv("./dat/structure2region.map", delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    for i, STR in enumerate(STRs):
        avg_exp_level_z = ExpLevelOneSTR_Weighted(STR, ExpMat, Gene2Weights, Match_DF)
        EFFs.append(avg_exp_level_z)
        Regions.append(str2reg[STR])
    df = pd.DataFrame(data={"STR":STRs, "EFFECT":EFFs, "Region":Regions})
    df = df.sort_values("EFFECT", ascending=False)
    df.reset_index()
    #df["Rank"] = df.index + 1
    g = LoadConnectome2(ConnFil=ConnFil)
    top_structs = df.head(50)["STR"].values
    top_nodes = g.vs.select(label_in=top_structs)
    g2 = g.subgraph(top_nodes)
    graph_size, exp_stats, exp_graphs, rm_vs = CircuitTrimming(g2, g)
    idx = np.argmax(exp_stats[:-10])
    #print(idx)
    CIR_STRS = [v["label"] for v in exp_graphs[idx].vs]
    df['InCircuit'] = 0
    df["TrimRank"] = 1
    df.index = df["STR"].values
    for i, _str in enumerate([x["label"] for x in rm_vs]):
        df.loc[_str, "TrimRank"] = 50-i
        df.loc[_str, "InCircuit"] = 1 if _str in CIR_STRS else 0
    df.to_csv(csv_fil, index=False)
    return df

def sibling_gene_weight(df):
    gene2MutN = {}
    for i, row in df.iterrows():
        try:
            g = GeneSymbol2Entrez[row["gene"]]
            gene2MutN[g] = row["dnv_LGDs_sib"]*0.375 + (row["dnv_missense_sib"]) * 0.145
        except:
            continue
    return gene2MutN

def sim_denovo_row2gweight(row, n_gene=101, LGD_Weight = 0.357, DMIS_Weight = 0.231):
    dat = []
    for index, value in row.items():
        N_lgd, N_dmis = map(int, value.split(","))
        #print(index, N_lgd, N_dmis)
        if N_lgd + N_dmis == 0:
            continue
        dat.append([index, N_lgd, N_dmis, N_lgd * LGD_Weight + N_dmis * DMIS_Weight])
    df = pd.DataFrame(data=dat, columns=["Entrez", "NLGD", "NDmis", "Weight"])
    df = df.sort_values("Weight", ascending=False)
    top = df.head(n_gene)
    return dict(zip([int(x) for x in top["Entrez"].values], top["Weight"].values))

def MakeMatchDF(GeneWeightDict, N=1000, DIR = "/Users/jiayao/Work/ASD_Circuits/src/dat/Jon_data/match-exp_avg"):
    Dat = []
    Index = []
    for k, v in list(GeneWeightDict.items()):
        try:
            match_genes = loadgenelist(DIR+"/"+str(k)+".txt")
            Dat.append(match_genes[1:N+1])
            Index.append(k)
        except:
            #print(DIR+"/"+str(k)+".txt")
            #print(k, "Not Found in Dataset")
            GeneWeightDict.pop(k)
    df = pd.DataFrame(data=Dat, index=Index)
    return df

def SeperateGenes(GeneDict):
    df = pd.DataFrame(data=[(k,v) for k,v in GeneDict.items()], index=range(len(GeneDict)))
    #return df
    half = int(df.shape[0]/2)
    print(half)
    df = df.sample(frac=1)
    df = df.reset_index(drop=True)
    df1 = df.loc[0:half-1, :]
    df2 = df.loc[half:, :]
    dict1 = dict(zip(df1[0].values, df1[1].values))
    dict2 = dict(zip(df2[0].values, df2[1].values))
    return dict1, dict2
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
    if type(ConnFil) == str:
        adj_mat = pd.read_csv(ConnFil)
        node_names = adj_mat["ROW"].values
        adj_mat = adj_mat.drop("ROW", axis=1)
    else:
        adj_mat = ConnFil
        node_names = adj_mat.index.values
    A = adj_mat.values
    g = ig.Graph.Adjacency((A > 0).tolist())
    g.es['weight'] = A[A.nonzero()]
    g.vs['label'] = node_names  
    return g 

def subgraph(g_complete, node_names):
    top_nodes = g_complete.vs.select(label_in=node_names)
    g_sub = g_complete.subgraph(top_nodes)
    return g_sub

def GetConnectivity_Edge(g, BiasDF, topN=50):
    g_ = g.copy()
    top_structs = BiasDF.head(topN).index.values
    top_nodes = g_.vs.select(label_in=top_structs)
    g2 = g_.subgraph(top_nodes)
    return g2.ecount(), InOutCohesiveAVG(g_, g2)

# Old, Depricated
def EdgePermutation(g, BiasDF, topN=50, Npermute=1000):
    g2_ecounts = []
    g2_cohe = []
    NodeList = np.array(range(0, 213))
    g_ = g.copy()
    top_structs = BiasDF.head(topN).index.values
    top_nodes = g_.vs.select(label_in=top_structs)
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
        g2_ = g_.subgraph(top_nodes)
        g2_ecounts.append(g2_.ecount())
        g2_cohe.append(InOutCohesiveAVG(g_, g2_))
    return g2_ecounts, g2_cohe


######################################################################################
# Region Preserving graph permutation 
######################################################################################
def CountReg2RegConn(adj_mat, str2reg):
    Reg2Reg = {}
    for STR_i in adj_mat.index.values:
        for STR_j in adj_mat.columns.values:
            conn_w = adj_mat.loc[STR_i, STR_j]
            if conn_w != 0:
                Reg1 = str2reg[STR_i]
                Reg2 = str2reg[STR_j]
                connR = "{}-{}".format(Reg1, Reg2)
                #connS = "{}-{}-{}".format(STR_i, STR_j, conn_w)
                connS = (STR_i, STR_j, conn_w)
                if connR not in Reg2Reg:
                    Reg2Reg[connR] = [connS]
                else:
                    Reg2Reg[connR].append(connS)
    return Reg2Reg

def SortConnByRegion(Reg2Reg):
    WithinRegion = {}
    CrossRegion_SRC, CrossRegion_TGT, CrossRegion_W = [], [], []
    for reg2reg, obj in Reg2Reg.items():
        reg1, reg2 = reg2reg.split("-")
        if reg1 == reg2:
            Sources, Targets, Weights = [], [],[]
            for src, tgt, w in obj:
                Sources.append(src)
                Targets.append(tgt)
                Weights.append(w)
            Sources = [item for items, c in Counter(Sources).most_common()
                                      for item in [items] * c]
            WithinRegion[reg2reg] = (Sources, Targets, Weights)
        else:
            for src, tgt, w in obj:
                CrossRegion_SRC.append(src)
                CrossRegion_TGT.append(tgt)
                CrossRegion_W.append(w)
    CrossRegion_SRC = [item for items, c in Counter(CrossRegion_SRC).most_common()
                                      for item in [items] * c]
    return WithinRegion, CrossRegion_SRC, CrossRegion_TGT, CrossRegion_W

# Keep Node Degree Identity, Keep Same Region Number of Connections
# Allow Cross Region Connection swap
def Region2RegionPert(obj):
    obj_copy = copy.deepcopy(obj)
    Sources, Targets, Weights = obj_copy
    ConnSet = set([])
    for i, w in enumerate(Weights):
        src = Sources[0]
        N_stuck = 0
        while 1:
            tgt = np.random.choice(Targets)
            conn = "{}-{}".format(src, tgt)
            if conn not in ConnSet and src != tgt:
                ConnSet.add(conn)
                N_stuck = 0
                Sources.pop(0)
                Targets.remove(tgt)
                break
            else:
                N_stuck += 1
                if N_stuck >=20:
                    return False, None
    return True, ConnSet

# This version I preserve Same Region Edge Identity but allow swapping of cross region edges to other region
def EdgePermutation_V1(adj_mat, Reg2Reg, CrossRegion_SRC, CrossRegion_TGT, CrossRegion_W, str2reg):
    adj_mat_permut = pd.DataFrame(data=np.zeros((213,213)), index=adj_mat.index.values, columns=adj_mat.columns.values)
    for reg2reg, obj in Reg2Reg.items():
        weights = obj[2]
        random.shuffle(weights)
        cc = 0
        while 1:
            #print(reg2reg, cc)
            flag, ConnSet = Region2RegionPert(obj)
            if flag:
                for i, conn in enumerate(ConnSet):
                    src, tgt = conn.split("-")
                    adj_mat_permut.loc[src, tgt] = weights[i] 
                break
            else:
                cc += 1
    for i, w in enumerate(CrossRegion_W):
        src = CrossRegion_SRC[0]
        N_stuck = 0
        while 1:
            tgt = np.random.choice(CrossRegion_TGT)
            if adj_mat_permut.loc[src, tgt] == 0 and str2reg[src]!=str2reg[tgt]:
                adj_mat_permut.loc[src, tgt] = w
                N_stuck = 0
                CrossRegion_SRC.pop(0)
                CrossRegion_TGT.remove(tgt)
                break
            else:
                N_stuck += 1
                if N_stuck > 10:
                    return False, None
    return True, adj_mat_permut

# This version I swap edge with similar distance.
#def EdgePermutation_V2(adj_mat, ):

######################################################################################
# Distance Preserving graph permutation 
######################################################################################
def SortEdgesToBuckets(adj_mat, adj_mat_dist, DistanceBins):
    RealConnBuckets = []
    PotentialConnBuckets = []
    for i in range(len(DistanceBins)-1):
        RealConnBuckets.append([])
        PotentialConnBuckets.append([])
    for i, str_i in enumerate(adj_mat.index.values):
        for j, str_j in enumerate(adj_mat.columns.values):
            edge_dist = adj_mat_dist.loc[str_i, str_j]
            edge_w = adj_mat.loc[str_i, str_j]
            if edge_w > 0:
                for i in range(len(DistanceBins)-1):
                    if edge_dist >= DistanceBins[i] and edge_dist < DistanceBins[i+1]:
                        RealConnBuckets[i].append((str_i, str_j, edge_w))
            for i in range(len(DistanceBins)-1):
                if edge_dist >= DistanceBins[i] and edge_dist < DistanceBins[i+1]:
                    PotentialConnBuckets[i].append((str_i, str_j))
    return RealConnBuckets, PotentialConnBuckets

def SortConnByDistance(Buckets):
    NewBucket = []
    for Bucket in Buckets:
        Sources, Targets, Weights = [], [], []
        for str_i, str_j, w in Bucket:
            Sources.append(str_i)
            Targets.append(str_j)
            Weights.append(w)
        NewBucket.append((Sources, Targets, Weights))
    return NewBucket
    
def update_weights(Bucket2, OutDCounts, InDCounts):
    res = []
    for xxx in Bucket2:
        str_i, str_j = xxx.split("-")
        res.append(OutDCounts[str_i] + InDCounts[str_j])
    sumw = np.sum(res)
    for i, num in enumerate(res):
        res[i] = num/sumw
    #res[-1] = 1 - np.sum(res[:-1])
    return res

def ShuffleBasedOnDegree(Bucket1, Bucket2):
    Sources, Targets, Weights = copy.deepcopy(Bucket1)
    OutDCounts = {}
    InDCounts = {}
    for pair in Bucket2:
        X,Y = pair
        if X not in OutDCounts:
            OutDCounts[X] = 1 + Sources.count(X)
        if Y not in InDCounts:
            InDCounts[Y] = 1 + Targets.count(Y)
    #print(sorted(OutDCounts.items(), key=lambda x:x[1]))
    res = []
    tmp_bucket2 = ["{}-{}".format(x,y) for x,y in Bucket2]
    weights = update_weights(tmp_bucket2, OutDCounts, InDCounts)
    while len(tmp_bucket2) > 0:
        xxx = np.random.choice(tmp_bucket2, p = weights)
        res.append(xxx)
        tmp_bucket2.remove(xxx)
        weights = update_weights(tmp_bucket2, OutDCounts, InDCounts)
    return res

def PermuteOneBucket(Bucket1, Bucket2):
    Sources, Targets, Weights = copy.deepcopy(Bucket1)
    tmp_bucket2 = copy.deepcopy(Bucket2)
    tmp_weight2 = copy.deepcopy(Weights)
    tmp_bucket2 = ShuffleBasedOnDegree(Bucket1, Bucket2)[::-1]
    #random.shuffle(tmp_bucket2)
    #random.shuffle(tmp_weight2)
    ConnSet = []
    while len(tmp_weight2) > 0:
        if len(tmp_bucket2) == 0:
            return False, ConnSet
        str_i, str_j = tmp_bucket2.pop().split("-")
        if str_i in Sources and str_j in Targets:
            w = tmp_weight2.pop()
            Sources.remove(str_i)
            Targets.remove(str_j)
            ConnSet.append((str_i, str_j, w))
    return True, ConnSet


def EdgePermutation_DistancePreserving(adj_mat, EdgeBuckets, PairBuckets):
    adj_mat_permut = pd.DataFrame(data=np.zeros((213,213)), index=adj_mat.index.values, columns=adj_mat.columns.values)
    for i, (Bucket1, Bucket2) in enumerate(zip(EdgeBuckets, PairBuckets)):
        cc = 0
        while cc < 50:
            Res, ConnSet = PermuteOneBucket(Bucket1, Bucket2)
            if Res:
                for str_i, str_j, w in ConnSet:
                    adj_mat_permut.loc[str_i, str_j] = w
                break
            else:
                cc += 1
        print("Bucket",i,"Failed 50 times")
    return adj_mat_permut

def PermuteOneBucket2(Bucket1, Bucket2):
    SwapSet = copy.deepcopy(Bucket1)
    tmp_bucket2 = ["{}-{}".format(x,y) for x,y in Bucket2]
    for i in range(1000):
        random.shuffle(SwapSet)
        SwapSet_dict = ["{}-{}".format(x,y) for x,y,z in SwapSet]
        str_i, str_j, w = SwapSet[0]
        # Find another pair that: 1. same distance for swapping; 2. Edge to swap not already exist
        Swappable_set = []
        for i in range(1, len(SwapSet)):
            _str_i, _str_j, _w = SwapSet[i]
            Cond1 = "{}-{}".format(_str_i, str_j) in tmp_bucket2
            Cond2 = "{}-{}".format(str_i, _str_j) in tmp_bucket2
            Cond3 = "{}-{}".format(_str_i, str_j) not in SwapSet_dict
            Cond4 = "{}-{}".format(str_i, _str_j) not in SwapSet_dict
            if Cond1 and Cond2 and Cond3 and Cond4:
                Swappable_set.append((i, _str_i, _str_j, _w))
        if len(Swappable_set) > 0:
            random.shuffle(Swappable_set)
            idx, _str_i, _str_j, _w = Swappable_set[0]
            SwapSet[0] = (_str_i, str_j, w)
            SwapSet[idx] = (str_i, _str_j, _w)
            #print(_str_i, _str_j, _w)
            #print(edge_deciles2[0], edge_deciles2[1])
            #print(Cartesian_distancesDF.loc[str_i, str_j])
            #print(Cartesian_distancesDF.loc[_str_i, _str_j])
            #print(Cartesian_distancesDF.loc[str_i, _str_j])
            #print(Cartesian_distancesDF.loc[_str_i, str_j])
    return SwapSet

def EdgePermutation_DistancePreserving_V2(adj_mat, EdgeBuckets, PairBuckets):
    adj_mat_permut = pd.DataFrame(data=np.zeros((213,213)), index=adj_mat.index.values, columns=adj_mat.columns.values)
    for i, (Bucket1, Bucket2) in enumerate(zip(EdgeBuckets, PairBuckets)):
        print("Permutating Bucket",i)
        SwapSet = PermuteOneBucket2(Bucket1, Bucket2)
        for str_i, str_j, w in SwapSet:
            adj_mat_permut.loc[str_i, str_j] = w
    return adj_mat_permut 

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
def NodePermutation(g, BiasDF, topN=50, Npermute=1000, tstat = "conn"):
    g_ = g.copy()
    top_structs = BiasDF.head(topN).index.values
    top_nodes = g_.vs.select(label_in=top_structs)
    STR2REG, REG2STR = LoadSTR2REG()
    REGs = REG2STR.keys()
    REGCounts = dict(zip(REGs, [0]*len(REGs)))
    for node in top_nodes:
        reg = STR2REG[node["label"]]
        REGCounts[reg] += 1
    g2_ecounts = []
    g2_cohe = []
    for i in range(Npermute):
        _nodes = []
        g_ = g.copy()
        for reg in REGs:
            _strs = np.random.choice(REG2STR[reg], REGCounts[reg])
            _nodes.extend(_strs)
        _nodes = g_.vs.select(label_in=_nodes)
        idx_nodes = [x.index for x in _nodes]
        g2_ = g_.subgraph(idx_nodes)
        #if tstat == "cohesive":
        #    Nulls.append(Cohesiveness(g_, idx_nodes))
        #else:
        #    Nulls.append(g2_.ecount())
    #return Nulls
        g2_ecounts.append(g2_.ecount())
        g2_cohe.append(InOutCohesiveAVG(g_, g2_))
    return g2_ecounts, g2_cohe

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

def GetPermutationP(null, obs, gt=True):
    count = 0
    for i,v in enumerate(null):
        if gt:
            if obs >= v:
                count += 1
        else:
            if obs <= v:
                count += 1
    Z = (obs - np.mean(null)) / np.std(null)
    P = 1-float(count)/(len(null)+1)
    return Z, P

def PlotPermutationP(Null, Obs, gt=True, title="", xlabel="", dist_label="", bar_label=""):
    plt.figure(dpi=120)
    n, bins, patches = plt.hist(Null, bins=20, histtype = "barstacked", align = 'mid', facecolor='grey', alpha=0.8, label=dist_label, color="grey", edgecolor="black", linewidth=0.5)
    Z, P = GetPermutationP(Null, Obs, gt=gt)
    plt.vlines(x=Obs, ymin=0, ymax=max(n), label=bar_label, color="black")
    plt.text(x=Obs, y=max(n), s="p=%.3f, z=%.3f"%(P,Z))
    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
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

def CohesivenessSingleNodeMaxInOut2(g, g_, Node, weightDict, weighted=False, Direction=False):
    Whole_EdgeList = []
    Cir_EdgeList = []
    CircuitLables = set(g_.vs["label"])
    Total_Out,Circuit_Out = 0,0
    Total_In, Circuit_In = 0,0
    STR = Node["label"]
    if Direction:
        for _node in Node.successors():
            if weighted:
                weight = weightDict["{}-{}".format(STR, _node["label"])]
            else:
                weight = 1
            Total_Out += weight
            if _node["label"] in CircuitLables:
                Circuit_Out += weight
        if Total_Out == 0:
            Frac_Out = 0
        else:
            Frac_Out = Circuit_Out/Total_Out
        for _node in Node.predecessors():
            if weighted:
                weight = weightDict["{}-{}".format(_node["label"], STR)]
            else:
                weight = 1
            Total_In += weight
            if _node["label"] in CircuitLables:
                Circuit_In += weight
        if Total_In == 0:
            Frac_In = 0
        else:
            Frac_In = Circuit_In/Total_In
        if Frac_Out == 0:
            ITR = np.inf
        else:
            ITR = Frac_In / Frac_Out
        return max(Frac_In, Frac_Out), ITR
    else:
        for _node in Node.successors():
            if weighted:
                weight = weightDict["{}-{}".format(STR, _node["label"])]
            else:
                weight = 1
            Total_Out += weight
            if _node["label"] in CircuitLables:
                Circuit_Out += weight
        for _node in Node.predecessors():
            if weighted:
                weight = weightDict["{}-{}".format(_node["label"], STR)]
            else:
                weight = 1
            Total_In += weight
            if _node["label"] in CircuitLables:
                Circuit_In += weight
        if (Total_In+Total_Out) == 0:
            return 0, None
        else:
            return ((Circuit_In+Circuit_Out)/(Total_In+Total_Out)), None

def CohesivenessSingleNodeMaxInOut(g, g_, STR, weightDict, weighted=False, Direction=False):
    Whole_EdgeList = []
    Cir_EdgeList = []
    Node = g.vs.find(label=STR)
    CircuitLables = set(g_.vs["label"])
    Total_Out,Circuit_Out = 0,0
    Total_In, Circuit_In = 0,0
    if Direction:
        for _node in Node.successors():
            if weighted:
                weight = weightDict["{}-{}".format(STR, _node["label"])]
            else:
                weight = 1
            Total_Out += weight
            if _node["label"] in CircuitLables:
                Circuit_Out += weight
        if Total_Out == 0:
            Frac_Out = 0
        else:
            Frac_Out = Circuit_Out/Total_Out
        for _node in Node.predecessors():
            if weighted:
                weight = weightDict["{}-{}".format(_node["label"], STR)]
            else:
                weight = 1
            Total_In += weight
            if _node["label"] in CircuitLables:
                Circuit_In += weight
        if Total_In == 0:
            Frac_In = 0
        else:
            Frac_In = Circuit_In/Total_In
        if Frac_Out == 0:
            ITR = np.inf
        else:
            ITR = Frac_In / Frac_Out
        return max(Frac_In, Frac_Out), ITR
    else:
        for _node in Node.successors():
            if weighted:
                weight = weightDict["{}-{}".format(STR, _node["label"])]
            else:
                weight = 1
            Total_Out += weight
            if _node["label"] in CircuitLables:
                Circuit_Out += weight
        for _node in Node.predecessors():
            if weighted:
                weight = weightDict["{}-{}".format(_node["label"], STR)]
            else:
                weight = 1
            Total_In += weight
            if _node["label"] in CircuitLables:
                Circuit_In += weight
        if (Total_In+Total_Out) == 0:
            return 0, None
        else:
            return ((Circuit_In+Circuit_Out)/(Total_In+Total_Out)), None

def ScoreSTRSet(Graph, CandidateNodes, WeightDict = {}, Weighted=False, Direction=False): ## Choesiveness
    CandidateNodes = set(CandidateNodes)
    top_nodes = Graph.vs.select(label_in=CandidateNodes)
    g2 = Graph.copy()
    g2 = g2.subgraph(top_nodes)
    cohesives = []
    for v in Graph.vs:
        if v["label"] in CandidateNodes:
            coh, ITR = CohesivenessSingleNodeMaxInOut(Graph, g2, v["label"], WeightDict, Direction=Direction, weighted=Weighted) 
            cohesives.append(coh)
    #return cohesives
    cohesive = np.mean(cohesives)
    return cohesive, len(g2.es)

# Score Connectivity and Cohesiveness between two conponents, 
def ScoreTwoComponents(Graph, STRSet1, STRSet2, WeightDict=None, Weighted=False, Directed=False):
    g = Graph.copy()
    Idx2Label = {}
    for v in g.vs:
        Idx2Label[v.index] = v['label']
    ConnSet1, ConnSet2, Conn_bet = [], [], []
    for e in g.es:
        if Idx2Label[e.source] in STRSet1 and Idx2Label[e.target] in STRSet2:
            Conn_bet.append(e["weight"])
        elif Idx2Label[e.source] in STRSet2 and Idx2Label[e.target] in STRSet1:
            Conn_bet.append(e["weight"])
        if Idx2Label[e.source] in STRSet1 and Idx2Label[e.target] in STRSet1:
            ConnSet1.append(e["weight"])
        elif Idx2Label[e.source] in STRSet2 and Idx2Label[e.target] in STRSet2:
            ConnSet2.append(e["weight"])
    #return ConnSet1, ConnSet2, Conn_bet
    return ConnSet1, ConnSet2, Conn_bet

def InOutCohesiveAVG(g, g_, weightDict):
    cohesives = []
    CandidateNodes = set(g_.vs["label"])
    for v in g.vs:
        if v['label'] in CandidateNodes:
        #coh = InOutCohesiveSingleNode(g, g_, v["label"])
            coh, _ = CohesivenessSingleNodeMaxInOut2(g, g_, v, weightDict)
            cohesives.append(coh)
    #cohesive = np.mean(cohesives) - (len(cohesives)/213) 
    #cohesive = np.mean(cohesives) / (len(cohesives)/213) 
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

def argmax_optimize_stat(graph, complate_graph, WeightDict, optimize_stat="in_n_out"):
    opt_stats, graphs, vs = [], [], []
    for i, v in enumerate(graph.vs):
        tmp_graph = graph.copy()
        tmp_graph.delete_vertices(v)
        #stat = optimize_stat_in_n_out(tmp_graph, complate_graph)
        stat = InOutCohesiveAVG(complate_graph, tmp_graph, WeightDict)
        #stat = sum(tmp_graph.degree())
        #stat = tmp_graph.transitivity_undirected()
        opt_stats.append(stat)
        graphs.append(tmp_graph)
        vs.append(v)
    idx = np.argmax(opt_stats)
    trimmed_v = vs[idx]
    stat = opt_stats[idx]
    graph = graphs[idx]
    return graph, stat, trimmed_v

def CircuitTrimming(graph, complate_graph, WeightDict, optimize_stat="in_n_out"):
    trim_graph = graph.copy()
    stats, graph_size, graphs, trimmed_Vs = [], [], [], []
    for i in range(len(graph.vs), 1, -1):

        trim_graph, stat, trimmed_v = argmax_optimize_stat(trim_graph, complate_graph, WeightDict, optimize_stat)
        graphs.append(trim_graph.copy())
        stats.append(stat)
        graph_size.append(i)
        trimmed_Vs.append(trimmed_v)
    trimmed_Vs.append(trim_graph.vs[0])
    return graph_size, stats, graphs, trimmed_Vs

def GreedyTrim(DF, g, topN, weightDict):
    top_structs = DF.head(topN).index.values
    #top_structs = np.array(DF)
    top_nodes = g.vs.select(label_in=top_structs)
    g2 = g.subgraph(top_nodes)
    graph_size, exp_stats, exp_graphs, rm_vs = CircuitTrimming(g2, g, weightDict)
    idx = np.argmax(exp_stats)
    circuit_STRs = [v["label"] for v in exp_graphs[idx].vs]
    return circuit_STRs, exp_stats[idx], len(exp_graphs[idx].es)

def SiblingTrim(g, topN, InputDir="dat/cont.sib.bias/ASD.sib.Spec.bias.{}.csv"):
    Cohes, Conns = [], []
    for i in range(1000):
        df = pd.read_csv(InputDir.format(i), index_col="STR")
        CirSTRs, cohe, conn = GreedyTrim(df, g, topN)
        Cohes.append(cohe)
        Conns.append(conn)
    return Cohes, Conns


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

class MostCohesiveCirtuis(Annealer):
    def __init__(self, state, Graph, CandidateNodes, WeightDict, Weighted=True, Direction=False):
        self.Graph = Graph
        self.CandidateNodes = CandidateNodes
        self.WeightDict = WeightDict
        self.Weighted = Weighted
        self.Direction = Direction
        super(MostCohesiveCirtuis, self).__init__(state)  # important!
    def move(self):
        # Add node or Remove node
        initial_energy = self.energy()
        for i in range(3):
            idx_change = random.randint(0, len(self.CandidateNodes) - 1)
            self.state[idx_change] = 1 - self.state[idx_change]
        return (self.energy() - initial_energy) 
    def energy2(self):
        return - NeiborBias(self.Graph, self.state)
    def energy(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        top_nodes = self.Graph.vs.select(label_in=InCirtuitNodes)
        g2 = self.Graph.copy()
        g2 = g2.subgraph(top_nodes)
        cohesives = []
        for v in g2.vs:
            coh, ITR = CohesivenessSingleNodeMaxInOut(self.Graph, g2, v["label"], self.WeightDict, Direction=self.Direction, weighted=self.Weighted) 
            cohesives.append(coh)
        cohesive = np.mean(cohesives)
        #cohesive = cohesive / len(top_nodes)
        #cohesive = cohesive - len(top_nodes)/213
        return 1 - cohesive
    def cohesives(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        top_nodes = self.Graph.vs.select(label_in=InCirtuitNodes)
        g2 = self.Graph.copy()
        g2 = g2.subgraph(top_nodes)
        cohesives = []
        for v in g2.vs:
            coh, TIR = CohesivenessSingleNodeMaxInOut(self.Graph, g2, v["label"], self.WeightDict)
            cohesives.append(coh)
        return cohesives

def NeiborBias(g, States, verbose=False):
    NeiborBias = []
    dat = []
    for idx, state in enumerate(States):
        if state == 0:
            continue
        node = g.vs[idx]
        TotalBias, InCircuitBias, OutCircuitBias = [], [], []
        for neighbor in node.neighbors():
            TotalBias.append(neighbor["Bias"])
            if States[neighbor.index] == 1:
                InCircuitBias.append(neighbor["Bias"])
            else:
                OutCircuitBias.append(max(0, neighbor["Bias"]))
        total_bias = np.mean(TotalBias)
        if len(InCircuitBias) == 0:
            incircuit_bias = 0
        else:
            incircuit_bias = np.mean(InCircuitBias)
            #incircuit_bias = np.sum(InCircuitBias)
        if len(OutCircuitBias) == 0:
            outcircuit_bias = 0
        else:
            outcircuit_bias = np.mean(OutCircuitBias)
            #outcircuit_bias = np.sum(OutCircuitBias)
        #NeiborBias.append(incircuit_bias / (incircuit_bias + outcircuit_bias)
        #NeiborBias.append(incircuit_bias)
        #if verbose:
        #    print(node["label"], "%.3f\t%.3f\t%.3f"%(incircuit_bias, outcircuit_bias, incircuit_bias - outcircuit_bias))
        #dat.append([])
    #return np.mean(NeiborBias)  

def run_CircuitOpt_neighborBias(g, EdgeWeightsDict, weighted, directed):
    CandidateNodes = np.array(g.vs["label"]) # all nodes are candidate
    Init_states = np.zeros(len(CandidateNodes))
    for idx in range(len(CandidateNodes)):
        if np.random.rand() > 0.5:
            Init_states[idx] = 1
    #print(CandidateNodes)
    ins = MostCohesiveCirtuis(Init_states, g, CandidateNodes, EdgeWeightsDict,
                              Weighted=weighted, Direction=directed)
    ins.copy_strategy = "deepcopy"
    ins.Tmax = 1
    ins.Tmin = 1e-6
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state

class MostCohesiveCirtuisGivenNCandidates(Annealer):
    def __init__(self, state, Graph, CandidateNodes, WeightDict, Weighted=True, Direction=False):
        self.Graph = Graph
        self.CandidateNodes = CandidateNodes
        self.WeightDict = WeightDict
        self.Weighted = Weighted
        self.Direction = Direction
        super(MostCohesiveCirtuisGivenNCandidates, self).__init__(state)  # important!
    def move(self):
        initial_energy = self.energy()
        # Swipe two nodes
        idx_in = np.where(self.state==1)[0]
        idx_out = np.where(self.state==0)[0]
        idx_change_i = np.random.choice(idx_in, 1)
        idx_change_j = np.random.choice(idx_out, 1)
        self.state[idx_change_i] = 1 - self.state[idx_change_i]
        self.state[idx_change_j] = 1 - self.state[idx_change_j]
        return (self.energy() - initial_energy) 
    def energy(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        top_nodes = self.Graph.vs.select(label_in=InCirtuitNodes)
        g2 = self.Graph.copy()
        g2 = g2.subgraph(top_nodes)
        cohesives = []
        for v in g2.vs:
            coh, ITR = CohesivenessSingleNodeMaxInOut(self.Graph, g2, v["label"], self.WeightDict, Direction=self.Direction, weighted=self.Weighted) 
            cohesives.append(coh)
        cohesive = np.mean(cohesives)
        return 1 - cohesive
    def cohesives(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        top_nodes = self.Graph.vs.select(label_in=InCirtuitNodes)
        g2 = self.Graph.copy()
        g2 = g2.subgraph(top_nodes)
        cohesives = []
        for v in g2.vs:
            coh, TIR = CohesivenessSingleNodeMaxInOut(self.Graph, g2, v["label"], self.WeightDict)
            cohesives.append(coh)
        return cohesives

def EdgeDict(graph, keyon='label'):
    EdgeDict = {}
    for idx in range(len(graph.vs)):
        for edge in graph.es.select(_source=idx):
            if keyon == 'index':
                key = "{}-{}".format(edge.source, edge.target)
            elif keyon == 'label':
                key = "{}-{}".format(graph.vs[edge.source]["label"], graph.vs[edge.target]["label"]) 
            EdgeDict[key] = edge["weight"]
    return EdgeDict

def CircuitOptimize(topN=100, Weighted = False, Direction = False, Weight_Dict = None):
    CandidateNodes = np.random.choice(meta_jon.head(213).index.values, 100, replace=False)
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if np.random.rand() > 0.5:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes, None)
    ins.copy_strategy = "deepcopy"
    ins.Tmax=1
    ins.Tmin=0.00001
    ins.Steps=topN * 500
    Tmps, Energys, state, e = ins.anneal()
    return ins

def run_CircuitOpt(g, EdgeWeightsDict, BiasDF, weighted, directed, topN=50):
    CandidateNodes = BiasDF.head(topN).index.values
    Init_States = np.ones(len(CandidateNodes))
    for idx in range(len(Init_States)):
        if np.random.rand() > 0.5:
            Init_States[idx] = 0
    ins = MostCohesiveCirtuis(Init_States, g, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed)
    ins.copy_strategy = "deepcopy"
    ins.Tmax=1
    ins.Tmin=0.00001
    Tmps, Energys, state, e = ins.anneal()
    print("Done")
    return state


def LocalDistal_Region():
    adj_mat = pd.read_csv("../dat/allen-mouse-conn/norm_density-max_ipsi_contra-pval_0.05-deg_min_1-by_weight_pvalue.csv", index_col="ROW")
    str2reg = STR2Region()
    ALL_STRs = adj_mat.index.values
    adj_mat_local = []
    adj_mat_distal = []
    for str_i in ALL_STRs:
        tmp_local = []
        tmp_distal = []
        for str_j in ALL_STRs:
            weight = adj_mat.loc[str_i, str_j]
            if weight == 0:
                tmp_local.append(0)
                tmp_distal.append(0)
            else:
                rg_i = str2reg[str_i]
                rg_j = str2reg[str_j]
                if rg_i == rg_j:
                    tmp_local.append(weight)
                    tmp_distal.append(0)
                else:
                    tmp_local.append(0)
                    tmp_distal.append(weight)
        adj_mat_local.append(tmp_local)
        adj_mat_distal.append(tmp_distal)
    adj_mat_local = pd.DataFrame(data=adj_mat_local, index=ALL_STRs, columns=ALL_STRs)
    adj_mat_distal = pd.DataFrame(data=adj_mat_distal, index=ALL_STRs, columns=ALL_STRs)
    return adj_mat_local, adj_mat_distal

def GetBestCoheSAFil(SAFil, Weighted, Direction, g, EdgeWeightsDict):
    EdgeWeightsDict = EdgeDict(g, keyon="label")
    sim = pd.read_csv(SAFil)
    candidates = sim.columns.values
    Cohes = []
    for i, row in sim.iterrows():
        state = row.values
        InCirtuitNodes = candidates[np.where(state==1)[0]]
        InCirtuitNodes_str = ",".join(InCirtuitNodes)
        score,x = ScoreSTRSet(g, InCirtuitNodes, EdgeWeightsDict, Weighted=Weighted, Direction=Direction)
        Cohes.append(np.mean(score))
    best_id = np.argmax(Cohes)
    state = sim.loc[1, :].values
    InCirtuitNodes = candidates[np.where(state==1)[0]]
    best_cohe = max(Cohes)
    exp = len(candidates)/213
    print("Best Cohe %.3f\tExp %.3f\tRate %.3f" %(best_cohe, exp, best_cohe/exp))
    return InCirtuitNodes

def Complete_Local_Distal_Cohesivesness_TopN_Case(BiasDF, g, g_local_region, g_distal_region, EdgeWeightsDict, Weighted, Directed, topN=50):
    CandidateSTRs = BiasDF.index.values[:topN]
    complete_cohes = ScoreSTRSet(g, CandidateSTRs, EdgeWeightsDict, Weighted=Weighted, Direction=Directed)
    local_cohes = ScoreSTRSet(g_local_region, CandidateSTRs, EdgeWeightsDict, Weighted=Weighted, Direction=Directed)
    dist_cohes = ScoreSTRSet(g_distal_region, CandidateSTRs, EdgeWeightsDict, Weighted=Weighted, Direction=Directed)
    return complete_cohes, local_cohes, dist_cohes

def Complete_Local_Distal_Cohesiveness_TopN_Cont(InputDir, g, g_local_region, g_distal_region, EdgeWeightsDict, Weighted, Directed):
    Complete, Local, Distal = [], [], []
    for i in range(1000):
        df = pd.read_csv(InputDir.format(i), index_col="STR")
        InCirtuitNodes = df.index.values[:50]
        complete = ScoreSTRSet(g, InCirtuitNodes, EdgeWeightsDict, Weighted, Directed)
        local = ScoreSTRSet(g_local_region, InCirtuitNodes, EdgeWeightsDict, Weighted, Directed)
        distal = ScoreSTRSet(g_distal_region, InCirtuitNodes, EdgeWeightsDict, Weighted, Directed)
        #Complete.append(np.mean(complete))
        #Local.append(np.mean(local))
        #Distal.append(np.mean(distal))
        Complete.append(complete)
        Local.append(local)
        Distal.append(distal)
    Complete = np.array(Complete)
    local = np.array(local)
    distal = np.array(distal)
    return Complete, Local, Distal

class DiffusionGraph():
    def __init__(self, biasDF, ConnMat):
        self.biasDF = biasDF
        self.Matrix = ConnMat
        self.NodeSeq = self.Matrix.index.values
        self.InitBias = np.array([self.biasDF.loc[X, "EFFECT"] for X in self.NodeSeq])
    def Diffuse(self, N=100):
        States = [self.InitBias]
        last_state = States[0]
        self.Beta = 0.9 * np.ones(213)
        for n in range(N):
            state = np.zeros(self.Matrix.values.shape[0])
            for i, tgt_node in enumerate(self.NodeSeq):
                retained_heat = self.Beta[i] * last_state[i]
                received_heat = 0
                for j, src_node in enumerate(self.Matrix.index.values):
                    received_heat += self.Matrix.loc[src_node, tgt_node] * (1-self.Beta[j]) * last_state[j] 
                state[i] = retained_heat + received_heat
            last_state = state
            States.append(state)
        return np.array(States)
    def Diffuse_vec(self, N=100):
        States = [self.InitBias]
        last_state = States[0]
        self.Beta = 0.9 * np.ones(213)
        for n in range(N):
            state = last_state * self.Beta + (np.eye(213) @ ((1-self.Beta) * last_state) @ self.Matrix.values).sum(axis=1)
            States.append(state)
            last_state = state
        return np.array(States)
    def Diffuse2(self, N=100): # Source & Sink 
        self.InitBias = self.InitBias - np.mean(self.InitBias)# Center Bias at 0 
        States = [self.InitBias]
        last_state = States[0]
        for n in range(N):
            state = np.zeros(self.Matrix.values.shape[0])
            for i, tgt_node in enumerate(self.NodeSeq):
                heat = 0.95 * last_state[i] + 0.05 * self.InitBias[i] # Each Step Node will Gian heat equal to their bias
                for j, src_node in enumerate(self.Matrix.index.values):
                    heat += 0.05 * self.Matrix.loc[src_node, tgt_node] * last_state[j]
                state[i] = heat
            last_state = state
            States.append(state)
        return np.array(States)
    def Diffuse2_vec(self, N=100, alpha=1e-3, beta = 1e-3):
        InitBias = self.InitBias - np.mean(self.InitBias) # Center Bias at 0
        States = [InitBias]
        sumheat = []
        T = self.Matrix.values
        for n in range(N):
            A = (1-alpha) * States[-1] + beta * InitBias
            B = alpha * T.transpose() @ States[-1].transpose()
            state = A + B
            States.append(state)
            sumheat.append(sum(state))
        self.States = np.array(States)
        return self.States

#####################################################################################
# Other Methods
#####################################################################################
def sort_lists(list_of_lists):
    return list(map(list, list(zip(*sorted(zip(*list_of_lists), key=lambda sublist_to_sort_by: sublist_to_sort_by[0])))))


def queryConnections(STR, ConnMat, BiasDF, STR2REG):
    #print("Project to")
    XX = ConnMat.loc[STR, :].sort_values(ascending=False)
    res = pd.DataFrame(data={'Region': [STR2REG[x] for x in XX.index.values], 
                             'STR':XX.index.values, 'Weight':XX.values,
                            "Bias":[BiasDF.loc[x, "EFFECT"] for x in XX.index.values]})
    res = res[res["Weight"] > 0]
    Project_To = res.shape[0]
    NeighborBias = res["Bias"].sum()
    #print("Receive From")
    XX = ConnMat.loc[:, STR].sort_values(ascending=False)
    res = pd.DataFrame(data={'Region': [STR2REG[x] for x in XX.index.values], 
                             'STR':XX.index.values, 'Weight':XX.values, 
                             "Bias":[BiasDF.loc[x, "EFFECT"] for x in XX.index.values]})
    res = res[res["Weight"] > 0]
    Receive_From = res.shape[0]
    NeighborBias += res["Bias"].sum()
    return Project_To, Receive_From, NeighborBias

def queryDist(adj_mat, Cartesian_distances_w_edge, dist_min, dist_max, directed=False):
    str2reg = STR2Region()
    REG2REG = []
    STR2STR = []
    for STR_i in adj_mat.index.values:
        for STR_j in adj_mat.index.values:
            if adj_mat.loc[STR_i, STR_j] == 0:
                continue
            dist = Cartesian_distances_w_edge.loc[STR_i, STR_j]
            if dist > dist_min and dist < dist_max:
                str2str = " - ".join([STR_i, STR_j])
                region_i = str2reg[STR_i]
                region_j = str2reg[STR_j]
                STR2STR.append(str2str)
                if directed:
                    REG2REG.append(" - ".join([region_i, region_j]))
                else:
                    REG2REG.append(" - ".join(sorted([region_i, region_j])))
    return STR2STR, REG2REG


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
    #RD1 = RegionDistributions(DF1.set_index("STR"))
    #RD2 = RegionDistributions(DF2.set_index("STR"))
    RD1 = RegionDistributions(DF1)
    RD2 = RegionDistributions(DF2)
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

def ShowTrimmingProfile(DFs, Names, topN=50, ConnFil="../dat/allen-mouse-conn/jw-conn-al1.csv"):
    g = LoadConnectome2(ConnFil=ConnFil)
    fig, ax = plt.subplots(dpi=120)
    for df, name in zip(DFs, Names):
        g_ = g.copy()
        top_structs = df.head(topN).index
        top_nodes = g_.vs.select(label_in=top_structs)
        g2 = g_.subgraph(top_nodes)
        graph_size, exp_stats, exp_graphs, trimmed_Vs = CircuitTrimming(g2, g_)
        idx = np.argmax(exp_stats[:topN-10])
        print(idx, exp_stats[idx])
        ax.scatter(topN-idx, exp_stats[idx], marker='x', color="black")
        ax.plot(graph_size, exp_stats, marker="o", label=name, alpha=0.7)
    ax.set_xlim(topN, 0)
    plt.legend()
    plt.show()
    return graph_size, exp_stats, exp_graphs, trimmed_Vs

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

def PlotEffectDist(BiasDF, title="Effect size Distribution"):
    Min, Max = min(BiasDF["EFFECT"].values), max(BiasDF["EFFECT"].values)
    bins = np.arange(Min, Max,  (Max-Min)/20)

    plt.hist(BiasDF["EFFECT"].values[:50], color="red", label="Top50", bins=bins, rwidth=0.7)
    plt.hist(BiasDF["EFFECT"].values[50:], color="blue", label="Rest", bins=bins, rwidth=0.7)
    plt.xlabel("EFFECT")
    plt.ylabel("Freq")
    plt.legend()
    plt.title(title)
    plt.show()

def sim_denovo_row2gweight(row, n_gene=101, LGD_Weight = 0.357, DMIS_Weight = 0.231):
    dat = []
    for index, value in row.items():
        N_lgd, N_dmis = map(int, value.split(","))
        #print(index, N_lgd, N_dmis)
        if N_lgd + N_dmis == 0:
            continue
        dat.append([index, N_lgd, N_dmis, N_lgd * LGD_Weight + N_dmis * DMIS_Weight])
    df = pd.DataFrame(data=dat, columns=["Entrez", "NLGD", "NDmis", "Weight"])
    df = df.sort_values("Weight", ascending=False)
    top = df.head(n_gene)
    return dict(zip([int(x) for x in top["Entrez"].values], top["Weight"].values))

def Loading_sim_denovo_dat(DIRi, N_sim=1000):
    EFFECT = {}
    for i in range(N_sim):
        tmp_df = pd.read_csv("{}/dnv_sm.{}.csv".format(DIR, i), index_col="STR")
        for STR, row in tmp_df.iterrows():
            #print(STR, EFFECT)
            if STR in EXP_LEVEL_EFFECT:
                EFFECT[STR].append(row["EFFECT"])
            else:
                EFFECT[STR] = [row["EFFECT"]]

def movingAVG(Input, smoothLen=3):
    res = []
    for i in range(len(Input)):
        if i - smoothLen < 0:
            new = np.mean(Input[0:i+smoothLen])
        elif i+ smoothLen > len(Input):
            new = np.mean(Input[i-smoothLen:])
        else:
            new = np.mean(Input[i-smoothLen:i+smoothLen])
        res.append(new)
    return res

def CI(simulations, p):
    simulations = sorted(simulations, reverse=False)
    n = len(simulations)
    u_pval = (1+p)/2.
    l_pval = (1-u_pval)
    print(l_pval, u_pval)
    l_indx = int(np.floor(n*l_pval))
    u_indx = int(np.floor(n*u_pval))
    print(l_indx, u_indx)
    return(simulations[l_indx],simulations[u_indx])

#####################################################################################
# Processes
#####################################################################################

# Show how cohesiveness change with TopN includes as candidates. 
# Could be complete Graph, local or distal connections
# Control could be 1. randomly selected STRs 2. subsampled siblings 3. Expression Matched genes
def ContTopNvsCohe(biasDF, topN, weighted, directed, g, EdgeWeightsDict, cont="sib", N_sim=100):
    YYY = []
    for i in range(N_sim):
        if cont == "rand":
            CandidateNodes = np.random.choice(biasDF.index.values, topN, replace=False)
        elif cont == "sib":
            BiasDF_cont = pd.read_csv("dat/cont.sib.bias/ASD.sib.Spec.bias.{}.csv".format(i), index_col="STR")
            CandidateNodes = BiasDF_cont.head(topN).index.values
        elif cont == "match":
            BiasDF_cont = pd.read_csv("dat/cont.bias/ASD.MetaMatch.Spec.bias.{}.csv".format(i), index_col="STR")
            CandidateNodes = BiasDF_cont.head(topN).index.values
        yyy = ScoreSTRSet(g, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed)
        #YYY.append(np.mean(yyy))
        YYY.append(yyy)
    return np.array(YYY)

def TopNAndCohesiveness(BiasDF, graph, weighted, directed, EdgeWeightsDict, cont, N_sim):
    ASD_Cohe, Cont_Cohe = [], []
    for topN in range(213):
        CandidateNodes = BiasDF.head(topN).index.values
        asd = ScoreSTRSet(graph, CandidateNodes, EdgeWeightsDict, Weighted=weighted, Direction=directed)
        asd = np.mean(asd)
        cont_cohe = ContTopNvsCohe(BiasDF, topN, weighted, directed, graph, EdgeWeightsDict, cont, N_sim)
        cont_cohe = np.mean(cont_cohe)
        ASD_Cohe.append(asd)
        Cont_Cohe.append(cont_cohe)
    unweight = movingAVG(np.array(ASD_Cohe[1:213])/np.array(Cont_Cohe[1:213]))
    return unweight

def MaskCortex2Cortex(adj_mat, Cortex_STRs):
    adj_mat_c2c = adj_mat.copy(deep=True)
    adj_mat_other = adj_mat.copy(deep=True)
    for STR_i in adj_mat.index.values:
        for STR_j in adj_mat.columns.values:
            if STR_i in Cortex_STRs and STR_j in Cortex_STRs:
                #adj_mat_c2c.loc[STR_i, STR_j] = adj_mat.loc[]
                adj_mat_other.loc[STR_i, STR_j] = 0
            else:
                adj_mat_c2c.loc[STR_i, STR_j] = 0
    return adj_mat_c2c, adj_mat_other


def MaskCortex2Cortex(adj_mat, Cortex_STRs):
    adj_mat_c2c = adj_mat.copy(deep=True)
    adj_mat_other = adj_mat.copy(deep=True)
    for STR_i in adj_mat.index.values:
        for STR_j in adj_mat.columns.values:
            if STR_i in Cortex_STRs and STR_j in Cortex_STRs:
                adj_mat_other.loc[STR_i, STR_j] = 0
            else:
                adj_mat_c2c.loc[STR_i, STR_j] = 0
    return adj_mat_c2c, adj_mat_other

def MaskLocalRegionalConnection(adj_mat):
    str2reg = STR2Region()
    ALL_STRs = adj_mat.index.values
    adj_mat_local = []
    adj_mat_distal = []
    for str_i in ALL_STRs:
        tmp_local = []
        tmp_distal = []
        for str_j in ALL_STRs:
            weight = adj_mat.loc[str_i, str_j]
            if weight == 0:
                tmp_local.append(0)
                tmp_distal.append(0)
            else:
                rg_i = str2reg[str_i]
                rg_j = str2reg[str_j]
                if rg_i == rg_j:
                    tmp_local.append(weight)
                    tmp_distal.append(0)
                else:
                    tmp_local.append(0)
                    tmp_distal.append(weight)
        adj_mat_local.append(tmp_local)
        adj_mat_distal.append(tmp_distal)
    adj_mat_local = pd.DataFrame(data=adj_mat_local, index=ALL_STRs, columns=ALL_STRs)
    adj_mat_distal = pd.DataFrame(data=adj_mat_distal, index=ALL_STRs, columns=ALL_STRs)
    return adj_mat_local, adj_mat_distal

def CohesivenessProfile(BiasDF, g, topNs, perm_conn_dir = "dat/permut_connectome_oct07", EdgeWeightsDict={}):
    ASD_Conn_Z, ASD_Cohe_Z, ASD_Conn_P, ASD_Cohe_P, ASD_Conn_E, ASD_Cohe_E = [],[],[],[],[],[]
    for topN in topNs:
        Permuted_ASD_cohe = []
        Permuted_ASD_conn = []
        for i in range(1, 1001, 1):
            adj_mat_perm = pd.read_csv("{}/{}.csv".format(perm_conn_dir, i), index_col=0)
            g_perm = LoadConnectome2(adj_mat_perm)
            asd_cohe, asd_conn = ScoreSTRSet(g_perm, BiasDF.head(topN).index.values, EdgeWeightsDict)
            Permuted_ASD_cohe.append(asd_cohe)
            Permuted_ASD_conn.append(asd_conn)
        asd_cohe, asd_conn = ScoreSTRSet(g, BiasDF.head(topN).index.values, EdgeWeightsDict)

        asd_z_conn, asd_p_conn = GetPermutationP(Permuted_ASD_conn, asd_conn)
        asd_z_cohe, asd_p_cohe = GetPermutationP(Permuted_ASD_cohe, asd_cohe)
        effect_conn = asd_conn/np.mean(Permuted_ASD_conn)
        effect_cohe = asd_cohe/np.mean(Permuted_ASD_cohe)
        ASD_Conn_Z.append(asd_z_conn); ASD_Cohe_Z.append(asd_z_cohe)
        ASD_Conn_P.append(asd_p_conn); ASD_Cohe_P.append(asd_p_cohe)
        ASD_Conn_E.append(effect_conn); ASD_Cohe_E.append(effect_cohe)
    return ASD_Cohe_E, ASD_Cohe_P

def addline(TopNs, Eff, ax, label = "", color="grey", ls="dashed"):
    ax.plot(TopNs, Eff, label=label, color=color, ls="solid", lw=3, )

def PlotCohesivenessProfile(topNs, EWE, EWE2):
    fig, ax = plt.subplots(dpi=720, figsize=(16,8))
    plt.style.use('seaborn-talk')
    matplotlib.rcParams.update({'font.size': 32})

    addline(topNs, EWE, ax, "ASD", "blue")
    addline(topNs, EWE2, ax, "Sib", "green")

    ax.hlines(xmin=min(topNs), xmax=max(topNs), y=1, ls="--", color="grey")
    ax.grid(True)

    ax.set_ylabel("Cohesiveness Effect Size",fontsize=28)
    ax.set_xlabel("Number of Most Biased Structures",fontsize=28)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

def BiasCorrelation(DF1, DF2):
    Bias1, Bias2 = [], []
    Rank1, Rank2 = [], []
    for STR in DF1.index.values:
        bias1 = DF1.loc[STR, "EFFECT"]
        bias2 = DF2.loc[STR, "EFFECT"]
        rank1 = DF1.loc[STR, "Rank"]
        rank2 = DF2.loc[STR, "Rank"]
        Bias1.append(bias1); Bias2.append(bias2)
        Rank1.append(rank1); Rank2.append(rank2)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6), dpi=120)
    ax1.scatter(Bias1, Bias2, s=3)
    ax1.set_xlabel("Bias1")
    ax1.set_ylabel("Bias2")
    print(pearsonr(Bias1, Bias2))
    ax2.scatter(Rank1, Rank2, s=3)
    ax2.set_xlabel("Rank1")
    ax2.set_ylabel("Rank2")
    ax2.hlines(xmin=0, xmax=213, y=50, ls="--", color="grey")
    ax2.vlines(ymin=0, ymax=213, x=50, ls="--", color="grey")
    print(pearsonr(Rank1, Rank2))
    plt.tight_layout()
    plt.show()


#####################################################################################
#####################################################################################
#####################################################################################
### Depricated Methods
### Depricated Methods
### Depricated Methods
#####################################################################################
#####################################################################################
#####################################################################################
"""
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

"""
