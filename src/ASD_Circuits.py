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
#import statsmodels.api as sm
import statsmodels.stats as stats
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
#from scipy.stats import binom_test
import itertools
#import mygene
import igraph as ig
import gzip as gz
import pickle as pk
import copy
import zipfile
from tabulate import tabulate
import logging
from typing import Dict, Any
import gc
import yaml

from SA import Annealer

# Load config to get project directory and data paths
def load_config(config_path="../config/config.yaml"):
    """Load configuration file with paths"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # If config not found in current directory, try parent directory
        parent_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/config.yaml")
        with open(parent_config, 'r') as f:
            return yaml.safe_load(f)

# Load config once at module import
_config = load_config()
_ProjDIR = _config["ProjDIR"]


# properly handle NA values
def quantileNormalize_withNA(expression_matrix):
    """
    Perform quantile normalization on gene expression matrix while ignoring NA values.

    Reimplements R's preprocessCore::normalize.quantiles (qnorm_c_handleNA).
    Key behavior for columns with NAs: the column's sorted non-NA values are
    linearly interpolated to fill all row positions before averaging, so every
    column contributes equally at every position regardless of NA count.

    Parameters:
    -----------
    expression_matrix : pandas.DataFrame or numpy.ndarray
        Expression matrix with genes as rows and structures as columns.
        May contain NA/NaN values.

    Returns:
    --------
    pandas.DataFrame
        Quantile normalized expression matrix with the same shape as input.
        NA values remain NA in the output.
    """
    if isinstance(expression_matrix, np.ndarray):
        df = pd.DataFrame(expression_matrix)
    else:
        df = expression_matrix.copy()

    original_columns = df.columns if hasattr(df, 'columns') else None
    original_index = df.index if hasattr(df, 'index') else None

    n_rows = len(df)
    n_cols = len(df.columns)

    # --- Step 1: Compute target distribution (matches R's determine_target) ---
    # For each column, sort non-NA values and interpolate to n_rows positions.
    # Then average across columns (each contributing 1/n_cols).
    target = np.zeros(n_rows)

    sorted_values = {}
    for col in df.columns:
        valid = df[col].dropna().values.copy()
        valid.sort()
        sorted_values[col] = valid
        n_valid = len(valid)

        if n_valid == n_rows:
            # No NAs: direct contribution
            target += valid / n_cols
        elif n_valid > 1:
            # Has NAs: interpolate sorted values to fill all n_rows positions.
            # R maps target position i to sorted index:
            #   idx = 1 + (n_valid - 1) * i / (n_rows - 1)  (1-based)
            # which is linear interpolation from sorted[0] to sorted[n_valid-1].
            source_positions = np.linspace(0, n_valid - 1, n_rows)
            interpolated = np.interp(source_positions,
                                     np.arange(n_valid, dtype=float), valid)
            target += interpolated / n_cols
        elif n_valid == 1:
            target += valid[0] / n_cols

    # --- Step 2: Assign target values back (matches R's using_target) ---
    # Vectorized: use np.interp for interpolation, handle ties via averaging.
    normalized_df = pd.DataFrame(np.full(df.shape, np.nan),
                                 index=df.index, columns=df.columns)
    target_xp = np.arange(n_rows, dtype=float)  # 0-based positions in target

    for col in df.columns:
        col_data = df[col].dropna()
        n_valid = len(col_data)
        if n_valid == 0:
            continue

        ranks = col_data.rank(method='average').values  # 1-based

        if n_valid == n_rows:
            # No NAs: map rank directly to target index (0-based)
            target_idx = ranks - 1.0  # continuous 0-based indices
            # Integer ranks: direct lookup; fractional ranks (ties): interp
            result = np.interp(target_idx, target_xp, target)
        elif n_valid > 1:
            # Has NAs: map rank percentile to target position via interpolation
            pctiles = (ranks - 1.0) / (n_valid - 1.0)
            target_idx = pctiles * (n_rows - 1.0)
            result = np.interp(target_idx, target_xp, target)
        else:
            # Single value: maps to target[0]
            result = np.array([target[0]])

        normalized_df.loc[col_data.index, col] = result

    if original_columns is not None:
        normalized_df.columns = original_columns
    if original_index is not None:
        normalized_df.index = original_index

    return normalized_df

def ZscoreConverting(values, mean = np.nan, std = np.nan):
    real_values = [x for x in values if x==x]
    #real_values = values 
    if mean != mean:
        mean = np.mean(real_values)
    if std != std:
        std = np.std(real_values)
    zscores = []
    #print(mean, std)
    #print(mean, std)
    for x in values:
        try:
            z = (x - mean)/std
            zscores.append(z)
        except:
            zscores.append(0)
    return np.array(zscores)
    

def LoadGeneINFO():
    protein_coding_genes_path = os.path.join(_ProjDIR, _config["data_files"]["protein_coding_genes"])
    HGNC = pd.read_csv(protein_coding_genes_path, delimiter="\t", low_memory=False)
    HGNC["entrez_id"] = pd.to_numeric(HGNC["entrez_id"], errors='coerce').astype('Int64')
    HGNC_valid = HGNC.dropna(subset=['entrez_id'])
    ENSID2Entrez = dict(zip(HGNC_valid["ensembl_gene_id"].values, HGNC_valid["entrez_id"].values))
    GeneSymbol2Entrez = dict(zip(HGNC_valid["symbol"].values, HGNC_valid["entrez_id"].values))
    Entrez2Symbol = dict(zip(HGNC_valid["entrez_id"].values, HGNC_valid["symbol"].values))
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol

def STR2Region():
    major_brain_divisions_path = os.path.join(_ProjDIR, _config["data_files"]["major_brain_divisions"])
    str2reg_df = pd.read_csv(major_brain_divisions_path, delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    return str2reg

str2reg = STR2Region()

def GeneList2GW(GeneListFil):
    with open(GeneListFil, 'r') as f:
        GeneList = []
        for g in f.readlines():
            gene_symbol = g.strip()
            entrez_id = GeneSymbol2Entrez.get(gene_symbol)
            if entrez_id is not None:
                GeneList.append(int(entrez_id))
            else:
                GeneList.append(0)
    GW = dict(zip(GeneList, np.ones(len(GeneList))))
    return GW

def LoadList(filename):
    """Read a text file and return a list of stripped lines."""
    with open(filename, 'rt') as f:
        return [x.strip() for x in f.readlines()]


def modify_str(x):
    """Clean up Allen Brain Atlas structure names for use as identifiers."""
    x = re.sub("[()]", "", x)
    x = re.sub("-", "_", x)
    x = re.sub("reunions", "reuniens", x)
    x = "_".join(x.split(" "))
    return x


def Dict2Fil(dict_, fil_):
    with open(fil_, 'wt') as f:
        writer = csv.writer(f)
        for k,v in dict_.items():
            writer.writerow([k, v])

def Fil2Dict(fil_):
    df = pd.read_csv(fil_, header=None)
    return dict(zip(df[0].values, df[1].values))


#####################################################################################
#####################################################################################
#### Annotation Functions
#####################################################################################
#####################################################################################
def add_class(BiasDF, ClusterAnn):
    for cluster, row in BiasDF.iterrows():
        if cluster in ClusterAnn.index:
            BiasDF.loc[cluster, "class_id_label"] = ClusterAnn.loc[cluster, "class_id_label"]
            BiasDF.loc[cluster, "subclass_id_label"] = ClusterAnn.loc[cluster, "subclass_id_label"]
            BiasDF.loc[cluster, "supertype_id_label"] = ClusterAnn.loc[cluster, "supertype_id_label"]
            if "CCF_broad.freq" in ClusterAnn.columns:
                BiasDF.loc[cluster, "CCF_broad.freq"] = ClusterAnn.loc[cluster, "CCF_broad.freq"]
            if "CCF_acronym.freq" in ClusterAnn.columns:
                BiasDF.loc[cluster, "CCF_acronym.freq"] = ClusterAnn.loc[cluster, "CCF_acronym.freq"]
            if "v3.size" in ClusterAnn.columns:
                BiasDF.loc[cluster, "v3.size"] = ClusterAnn.loc[cluster, "v3.size"]
            if "v2.size" in ClusterAnn.columns:
                BiasDF.loc[cluster, "v2.size"] = ClusterAnn.loc[cluster, "v2.size"]
        else:
            # Set default values if cluster not found in annotation
            BiasDF.loc[cluster, "class_id_label"] = "Unknown"
    return BiasDF

def AnnotateCTDat(df, Anno):
    for i, row in df.iterrows():
        df.loc[i, "Class"] = Anno.loc[int(i), "Class auto-annotation"]
        df.loc[i, "Supercluster"] = Anno.loc[int(i), "Supercluster"]
        df.loc[i, "Subtype"] = Anno.loc[int(i), "Subtype auto-annotation"]
        df.loc[i, "Neurotransmitter"] = Anno.loc[int(i), "Neurotransmitter auto-annotation"]
        df.loc[i, "Top three regions"] = Anno.loc[int(i), "Top three regions"]
        df.loc[i, "Top three dissections"] = Anno.loc[int(i), "Top three dissections"]
        df.loc[i, "Number of cells"] = Anno.loc[int(i), "Number of cells"] #Top three dissections
        df.loc[i, "Neuropeptide auto-annotation"] = Anno.loc[int(i), "Neuropeptide auto-annotation"] #Top three dissections
    df.index = [int(i) for i in df.index.values]
    return df

#####################################################################################
#####################################################################################
#### Gene Weights Functions
#####################################################################################
#####################################################################################

def SPARK_Gene_Weights(
    MutFil, BGMR=None, Nproband=42607, UsepLI=True, Bmis=False, out=None
):
    """
    Calculate mutation-derived gene weights for the SPARK dataset.

    Parameters:
    - MutFil: pandas DataFrame of mutations (must contain columns: "EntrezID", "ExACpLI", "AutismMerged_LoF",
              "AutismMerged_Dmis_REVEL0.5", "AutismMerged_Bmis_REVEL0.5")
    - BGMR: optional background mutation rate DataFrame (indexed by Entrez ID, must contain: "p_LGD", "prevel_0.5", "p_misense")
    - Nproband: Number of probands in cohort
    - UsepLI: if True, apply pLI-based gene weighting
    - Bmis: if True, include Bmis category, otherwise exclude
    - out: optional output CSV file for writing sorted gene weights

    Returns:
    - gene2None: dict mapping gene EntrezID to 1
    - gene2MutN: dict mapping gene EntrezID to computed weight
    """
    gene2None = {}
    gene2MutN = {}

    # Add requested printout for BGMR usage
    # if BGMR is not None:
    #     print("SPARK_Gene_Weights: BGMR correction applied.")
    # else:
    #     print("SPARK_Gene_Weights: BGMR correction NOT applied.")

    for _, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except Exception:
            continue
        gene2None[g] = 1

        # Initialize expected mutation counts
        LGD_exp = Dmis_exp = Bmis_exp = 0
        if BGMR is not None:
            try:
                LGD_MR = BGMR.loc[g, "p_LGD"]
                Dmis_MR = BGMR.loc[g, "prevel_0.5"]
                Bmis_MR = Dmis_MR - BGMR.loc[g, "p_misense"]
                LGD_exp = LGD_MR * 0.138 * Nproband * 2
                Dmis_exp = Dmis_MR * 0.130 * Nproband * 2
                Bmis_exp = Bmis_MR * 0.022 * Nproband * 2
            except Exception:
                LGD_exp = Dmis_exp = Bmis_exp = 0

        pLI = 0.0
        if UsepLI:
            try:
                pLI = float(row["ExACpLI"])
            except Exception:
                pLI = 0.0

        # Compute gene weights according to options and thresholds
        if UsepLI:
            if pLI >= 0.5:
                if Bmis:
                    score = (
                        (row["AutismMerged_LoF"] - LGD_exp) * 0.554 +
                        (row["AutismMerged_Dmis_REVEL0.5"] - Dmis_exp) * 0.333 +
                        (row["AutismMerged_Bmis_REVEL0.5"] - Bmis_exp) * 0.099
                    )
                else:
                    score = (
                        (row["AutismMerged_LoF"] - LGD_exp) * 0.554 +
                        (row["AutismMerged_Dmis_REVEL0.5"] - Dmis_exp) * 0.333
                    )
            else:
                if Bmis:
                    score = (
                        (row["AutismMerged_LoF"] - LGD_exp) * 0.138 +
                        (row["AutismMerged_Dmis_REVEL0.5"] - Dmis_exp) * 0.130 +
                        (row["AutismMerged_Bmis_REVEL0.5"] - Bmis_exp) * 0.022
                    )
                else:
                    score = (
                        (row["AutismMerged_LoF"] - LGD_exp) * 0.138 +
                        (row["AutismMerged_Dmis_REVEL0.5"] - Dmis_exp) * 0.130
                    )
        else:
            score = (
                (row["AutismMerged_LoF"] - LGD_exp) * 0.457 +
                (row["AutismMerged_Dmis_REVEL0.5"] - Dmis_exp) * 0.231
            )
        gene2MutN[g] = score

    if out is not None:
        with open(out, 'wt', newline='') as f:
            writer = csv.writer(f)
            for k, v in sorted(gene2MutN.items(), key=lambda x: x[1], reverse=True):
                writer.writerow([k, v])

    return gene2None, gene2MutN


def CountMut(DF, DmisSTR="REVEL", DmisCut=0.5):
    """Count LGD, missense, damaging missense, and synonymous mutations.

    Parameters
    ----------
    DF : DataFrame
        Mutation-level DataFrame with columns 'GeneEff' and 'REVEL'.
    DmisSTR : str
        Column name for damaging missense score (default: 'REVEL').
    DmisCut : float
        Threshold for damaging missense (default: 0.5).

    Returns
    -------
    tuple : (N_LGD, N_mis, N_Dmis, N_syn)
    """
    N_LGD, N_mis, N_Dmis, N_syn = 0, 0, 0, 0
    for _, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor",
                        "start_lost", "stop_gained", "stop_lost"]:
            N_LGD += 1
        elif GeneEff == "missense":
            N_mis += 1
            revel = str(row[DmisSTR]).split(";")[0]
            if revel != ".":
                try:
                    if float(revel) > DmisCut:
                        N_Dmis += 1
                except (ValueError, TypeError):
                    pass
        elif GeneEff == "synonymous":
            N_syn += 1
    return N_LGD, N_mis, N_Dmis, N_syn


def Mut2GeneDF(MutDF, PPVs=(0.554, 0.333, 0.138, 0.130), LGD=True, Dmis=True,
               gene_col="HGNC", gene_symbol_to_entrez=None):
    """Convert mutation-level DataFrame to gene-level weights using PPV + pLI.

    Parameters
    ----------
    MutDF : DataFrame
        Mutation-level DataFrame with columns for gene symbol, GeneEff, REVEL, ExACpLI.
    PPVs : tuple
        (PPV_LGD_highpLI, PPV_Dmis_highpLI, PPV_LGD_lowpLI, PPV_Dmis_lowpLI).
    LGD : bool
        Include LGD mutations in weights.
    Dmis : bool
        Include damaging missense mutations in weights.
    gene_col : str
        Column containing gene identifiers. If 'HGNC', uses gene_symbol_to_entrez
        to map to Entrez IDs. If 'Entrez', uses values directly.
    gene_symbol_to_entrez : dict or None
        Mapping from gene symbol to Entrez ID. Required when gene_col='HGNC'.
        If None and gene_col='HGNC', loads via LoadGeneINFO().

    Returns
    -------
    dict : {Entrez_ID: weight}
    """
    if gene_col == "HGNC" and gene_symbol_to_entrez is None:
        _, _, gene_symbol_to_entrez, _ = LoadGeneINFO()

    genes = list(set(MutDF[gene_col].values))
    gene2MutN = {}
    for g in genes:
        if gene_col == "HGNC":
            try:
                entrez = int(gene_symbol_to_entrez[g])
            except (KeyError, ValueError, TypeError):
                continue
        else:
            try:
                entrez = int(g)
            except (ValueError, TypeError):
                continue

        Muts = MutDF[MutDF[gene_col] == g]
        try:
            pLI = float(Muts["ExACpLI"].values[0])
        except (ValueError, TypeError, IndexError):
            pLI = 0.0

        N_LGD, _, N_Dmis, _ = CountMut(Muts)
        if not LGD:
            N_LGD = 0
        if not Dmis:
            N_Dmis = 0
        if pLI >= 0.5:
            gene2MutN[entrez] = N_LGD * PPVs[0] + N_Dmis * PPVs[1]
        else:
            gene2MutN[entrez] = N_LGD * PPVs[2] + N_Dmis * PPVs[3]
    return gene2MutN


def Filt_LGD_Mis(DF, Dmis=True):
    """Filter mutation DataFrame to keep only LGD and (optionally) damaging missense.

    Parameters
    ----------
    DF : DataFrame
        Mutation-level DataFrame with columns 'GeneEff' and 'REVEL'.
    Dmis : bool
        If True, keep only damaging missense (REVEL > 0.5).
        If False, keep all missense.

    Returns
    -------
    DataFrame : filtered mutations
    """
    dat = []
    for _, row in DF.iterrows():
        GeneEff = row["GeneEff"].split(";")[0]
        if GeneEff in ["frameshift", "splice_acceptor", "splice_donor",
                        "start_lost", "stop_gained", "stop_lost"]:
            dat.append(row.values)
        elif GeneEff == "missense":
            if Dmis:
                revel = str(row["REVEL"]).split(";")[0]
                if revel != ".":
                    try:
                        if float(revel) > 0.5:
                            dat.append(row.values)
                    except (ValueError, TypeError):
                        pass
            else:
                dat.append(row.values)
    return pd.DataFrame(dat, columns=DF.columns.values)


def RegionDistributionsList(List, topN=50):
    str2reg_df = pd.read_csv(os.path.join(_ProjDIR, _config["data_files"]["major_brain_divisions"]), delimiter="\t")
    str2reg_df = str2reg_df.sort_values("REG")
    #str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    str2reg = dict(zip(str2reg_df["STR"].values, str2reg_df["REG"].values))
    Regions = list(set(str2reg.values()))
    RegionCount = {}
    for region in Regions:
        RegionCount[region] = []
    for x in List:
        region = str2reg[x]
        RegionCount[region].append(x)
    outstr = ""
    for k,v in RegionCount.items():
        if len(v) == 0:
            continue
        outstr = outstr + k + "\t" +  str(len(v)) + "\t" + "; ".join(v) + "\n"
    return outstr

def BiasCorrelation(DF1, DF2, name1="1", name2="2", dpi=80):
    Bias1, Bias2 = [], []
    Rank1, Rank2 = [], []
    for STR in DF1.index.values:
        bias1 = DF1.loc[STR, "EFFECT"]
        bias2 = DF2.loc[STR, "EFFECT"]
        rank1 = DF1.loc[STR, "Rank"]
        rank2 = DF2.loc[STR, "Rank"]
        if bias1 != bias1 or bias2 != bias2:
            continue
        Bias1.append(bias1); Bias2.append(bias2)
        Rank1.append(rank1); Rank2.append(rank2)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,4), dpi=dpi)
    ax1.scatter(Bias1, Bias2, edgecolor='k', alpha=0.7, s=20)
    ax1.set_xlabel("Bias {}".format(name1), fontsize=14, weight='bold')
    ax1.set_ylabel("Bias {}".format(name2), fontsize=14, weight='bold')
    print(pearsonr(Bias1, Bias2))
    #print(np.corrcoef(Bias1, Bias2))
    ax2.scatter(Rank1, Rank2, s=3)
    ax2.set_xlabel("Rank {}".format(name1))
    ax2.set_ylabel("Rank {}".format(name2))
    ax2.hlines(xmin=0, xmax=213, y=50, ls="--", color="grey")
    ax2.vlines(ymin=0, ymax=213, x=50, ls="--", color="grey")

    STR_Com = len(set(DF1.head(50).index.values).intersection(set(DF2.head(50).index.values)))
    ax2.text(x=80, y=20, s="{} STR in Common Among top 50".format(STR_Com))
    #print(pearsonr(Rank1, Rank2))
    plt.tight_layout()
    plt.show()

def Aggregate_Gene_Weights_NDD(MutFil, usepLI=False, Bmis=False, out=None):
    gene2MutN = {}
    for i, row in MutFil.iterrows():
        try:
            g = int(row["EntrezID"])
        except:
            print(g, "Error converting Entrez ID")

        nLGD = row["frameshift_variant"] + row["splice_acceptor_variant"] + row["splice_donor_variant"] + row["stop_gained"] + row["stop_lost"] 
        nMis = row["missense_variant"] 

        gene2MutN[g] = nLGD * 0.347 + nMis * 0.194
    if out != None:
        writer = csv.writer(open(out, 'wt'))
        for k,v in sorted(gene2MutN.items(), key=lambda x:x[1], reverse=True):
           writer.writerow([k,v]) 
    return gene2MutN
    
# This function is used to bootstrap the gene mutations 
def bootstrap_gene_mutations(
    df,
    n_boot=10,
    weighted=True,
    lof_col="AutismMerged_LoF",
    dmis_col="AutismMerged_Dmis_REVEL0.5",
    rng=None,
):
    """
    Bootstrap mutation counts at the mutation level, preserving gene identity
    and total mutation load. Supports weighted (mutation-rate) or uniform resampling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with gene-level LOF and Dmis mutation counts.
    n_boot : int, optional
        Number of bootstrap replicates (default = 10).
    weighted : bool, optional
        If True, mutations are resampled with probability proportional to observed counts.
        If False, each gene has equal probability of receiving any mutation.
    lof_col, dmis_col : str
        Column names for LOF and Dmis counts.
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.

    Returns
    -------
    boot_DFs : list of pd.DataFrame
        List of bootstrapped dataframes with resampled mutation counts and 'bootstrap_iter' column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(df)
    boot_DFs = []

    # Ensure integer non-negative mutation counts
    lof = df[lof_col].astype(int).clip(lower=0)
    dmis = df[dmis_col].astype(int).clip(lower=0)

    total_lof = lof.sum()
    total_dmis = dmis.sum()

    # Probability vectors for mutation assignment
    if weighted:
        # Weighted by observed mutation burden per gene
        p_lof = lof / total_lof if total_lof > 0 else np.ones(n) / n
        p_dmis = dmis / total_dmis if total_dmis > 0 else np.ones(n) / n
    else:
        # Uniform: every gene equally likely
        p_lof = np.ones(n) / n
        p_dmis = np.ones(n) / n

    for i in range(1, n_boot + 1):
        # Draw total_lof mutation events, assign to genes
        new_lof_counts = np.bincount(
            rng.choice(n, size=total_lof, replace=True, p=p_lof),
            minlength=n
        )
        new_dmis_counts = np.bincount(
            rng.choice(n, size=total_dmis, replace=True, p=p_dmis),
            minlength=n
        )

        # Create bootstrap replicate
        df_boot = df.copy().reset_index(drop=True)
        df_boot[lof_col] = new_lof_counts
        df_boot[dmis_col] = new_dmis_counts
        df_boot["bootstrap_iter"] = i
        df_boot["bootstrap_type"] = "weighted" if weighted else "uniform"
        boot_DFs.append(df_boot)

    return boot_DFs

#####################################################################################
#####################################################################################
#### IQ function
#####################################################################################
#####################################################################################
# Global cache for expression matrices
_matrix_cache: Dict[str, pd.DataFrame] = {}

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by converting to appropriate dtypes.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Memory-optimized DataFrame
    """
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        # Check if column has only integers
        if df[col].dtype in ['int64', 'int32']:
            # Try to downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype in ['float64']:
            # Convert float64 to float32 if precision allows
            max_val = df[col].abs().max()
            if pd.notna(max_val) and max_val < np.finfo(np.float32).max:
                # Check if conversion preserves precision reasonably
                temp_col = df[col].astype(np.float32)
                if np.allclose(df[col].dropna(), temp_col.dropna(), rtol=1e-6):
                    df[col] = temp_col
    
    # Optimize categorical columns with low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
    
    new_memory = df.memory_usage(deep=True).sum()
    reduction = (original_memory - new_memory) / original_memory * 100
    
    logging.info(f"Memory optimization: {reduction:.1f}% reduction ({original_memory/1e6:.1f}MB → {new_memory/1e6:.1f}MB)")
    
    return df

def load_expression_matrix_cached(file_path: str, force_reload: bool = False) -> pd.DataFrame:
    """
    Load expression matrix with caching to avoid redundant I/O operations.
    
    Args:
        file_path: Path to the expression matrix file
        force_reload: Force reload even if cached
    
    Returns:
        Expression matrix DataFrame
    """
    cache_key = os.path.abspath(file_path)
    
    # Return cached version if available
    if cache_key in _matrix_cache and not force_reload:
        logging.info(f"Using cached matrix: {file_path}")
        return _matrix_cache[cache_key]
    
    # Load matrix
    logging.info(f"Loading expression matrix: {file_path}")
    if '.parquet' in file_path:
        matrix = pd.read_parquet(file_path)
    elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
        matrix = pd.read_csv(file_path, index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Optimize data types
    matrix = optimize_dataframe_dtypes(matrix)
    
    # Preserve column names as strings (cell type names)
    if matrix.columns.dtype == 'object':
        matrix.columns = matrix.columns.astype(str)
        logging.info(f"Column names preserved as strings: {len(matrix.columns)} cell types")
    
    # Cache the optimized matrix
    _matrix_cache[cache_key] = matrix
    logging.info(f"Cached matrix with shape {matrix.shape}")
    
    return matrix

def clear_matrix_cache():
    """Clear the expression matrix cache and free memory."""
    global _matrix_cache
    cache_size = len(_matrix_cache)
    _matrix_cache.clear()
    gc.collect()
    logging.info(f"Cleared matrix cache ({cache_size} matrices)")

def cleanup_memory():
    """Explicit memory cleanup."""
    gc.collect()
    logging.info("Memory cleanup completed")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current cache state."""
    cache_info = {}
    total_memory = 0
    
    for path, matrix in _matrix_cache.items():
        memory_mb = matrix.memory_usage(deep=True).sum() / 1e6
        cache_info[path] = {
            'shape': matrix.shape,
            'memory_mb': memory_mb,
            'dtypes': matrix.dtypes.value_counts().to_dict()
        }
        total_memory += memory_mb
    
    cache_info['total_memory_mb'] = total_memory
    cache_info['num_matrices'] = len(_matrix_cache)
    
    return cache_info
#####################################################################################
#####################################################################################
#### Bias Calculation
#####################################################################################
#####################################################################################
def MouseSTR_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None, results_dir="./results"):
    """
    Calculate weighted average bias scores for brain structures

    Parameters:
    -----------
    ExpZscoreMat : pd.DataFrame
        Expression z-score matrix (genes x structures)
    Gene2Weights : dict
        Gene weights dictionary (entrez_id -> weight)
    csv_fil : str, optional
        Filename or full path to save results
    results_dir : str, default "./results"
        Directory to save results if csv_fil is provided as a filename only

    Returns:
    --------
    pd.DataFrame : Structure bias results with EFFECT and Rank columns
    """
    # Convert Gene2Weights to pandas Series for faster lookup
    weights_series = pd.Series(Gene2Weights)

    # Get intersection of genes using index operations
    valid_genes = ExpZscoreMat.index.intersection(weights_series.index)

    # Get weights and expression values using vectorized operations
    weights = weights_series[valid_genes].values
    expr_mat = ExpZscoreMat.loc[valid_genes]

    # Compute weighted averages for all cell types at once using matrix operations
    mask = ~np.isnan(expr_mat)
    # Broadcasting weights to match expr_mat shape
    weights_broadcast = weights[:, np.newaxis]
    # Multiply weights with expression values, handling nans
    weighted_vals = expr_mat.values * weights_broadcast * mask
    # Sum of weights where values are not nan
    weight_sums = weights_broadcast * mask
    # Compute weighted average across genes for each cell type
    EFFECTS = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)

    # Create results dataframe efficiently
    df = pd.DataFrame({
        'Structure': ExpZscoreMat.columns,
        'EFFECT': EFFECTS
    }).set_index('Structure')

    # Sort and add rank
    df = df.sort_values('EFFECT', ascending=False)
    df['Rank'] = np.arange(1, len(df) + 1)
    df["REGION"] = df.index.map(str2reg)

    if csv_fil is not None:
        # If csv_fil has a directory component, use it as a full path (do not prepend results_dir)
        if os.path.dirname(csv_fil):
            output_path = csv_fil
            os.makedirs(os.path.dirname(csv_fil), exist_ok=True)
        else:
            # Otherwise, use results_dir as base directory
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, csv_fil)
        df.to_csv(output_path)
        print(f"Results saved to: {output_path}")

    return df

def HumanCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None):
    # Filter out genes not in expression matrix
    valid_genes = [g for g in Gene2Weights.keys() if g in ExpZscoreMat.index.values]
    
    # Get weights and expression values for valid genes
    weights = np.array([Gene2Weights[g] for g in valid_genes])
    expr_mat = ExpZscoreMat.loc[valid_genes]
    
    # Compute weighted average for each cell type
    CellTypes = ExpZscoreMat.columns.values
    EFFECTS = []
    for CT in CellTypes:
        expr_values = expr_mat[CT].values
        # Remove nan values and corresponding weights
        mask = ~np.isnan(expr_values)
        filtered_expr = expr_values[mask]
        filtered_weights = weights[mask]
        # Weighted average for this cell type, ignoring nans
        weighted_avg = np.average(filtered_expr, weights=filtered_weights)
        EFFECTS.append(weighted_avg)
        
    # Create results dataframe
    df = pd.DataFrame(data={"ct_idx": CellTypes, "EFFECT": EFFECTS}).set_index('ct_idx')
    df = df.sort_values("EFFECT", ascending=False)
    #df = df.reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    #df = df.set_index("ct_idx")
    
    if csv_fil is not None:
        df.to_csv(csv_fil)
        
    return df

def MouseCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights, csv_fil=None):
    # Convert Gene2Weights to pandas Series for faster lookup
    weights_series = pd.Series(Gene2Weights)
    
    # Get intersection of genes using index operations
    valid_genes = ExpZscoreMat.index.intersection(weights_series.index)
    
    # Get weights and expression values using vectorized operations
    weights = weights_series[valid_genes].values
    expr_mat = ExpZscoreMat.loc[valid_genes]
    
    # Compute weighted averages for all cell types at once using matrix operations
    mask = ~np.isnan(expr_mat)
    # Broadcasting weights to match expr_mat shape
    weights_broadcast = weights[:, np.newaxis]
    # Multiply weights with expression values, handling nans
    weighted_vals = expr_mat.values * weights_broadcast * mask
    # Sum of weights where values are not nan
    weight_sums = weights_broadcast * mask
    # Compute weighted average across genes for each cell type
    EFFECTS = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)
    
    # Create results dataframe efficiently
    df = pd.DataFrame({
        'ct_idx': ExpZscoreMat.columns,
        'EFFECT': EFFECTS
    }).set_index('ct_idx')
    
    # Sort and add rank
    df = df.sort_values('EFFECT', ascending=False)
    df['Rank'] = np.arange(1, len(df) + 1)
    
    if csv_fil is not None:
        df.to_csv(csv_fil)
        
    return df


def GetPermutationP_vectorized(null_matrix, observed_vals, greater_than=True):
    """Vectorized permutation p-value calculation for multiple tests."""
    # null_matrix shape: (n_permutations, n_cell_types)
    # observed_vals shape: (n_cell_types,)
    
    # Remove NaN values per cell type
    mask = ~np.isnan(null_matrix)
    n_valid = np.sum(mask, axis=0)
    
    # Calculate means and stds, handling NaNs
    means = np.nanmean(null_matrix, axis=0)
    stds = np.nanstd(null_matrix, axis=0)
    
    # Calculate z-scores
    z_scores = (observed_vals - means) / stds
    
    # Calculate p-values vectorized
    if greater_than:
        counts = np.sum(observed_vals[None, :] <= null_matrix, axis=0)
    else:
        counts = np.sum(observed_vals[None, :] >= null_matrix, axis=0)
    
    p_values = (counts + 1) / (n_valid + 1)
    obs_adjs = observed_vals - means
    
    return z_scores, p_values, obs_adjs

#####################################################################################
#####################################################################################
####  Circuit Related Functions
#####################################################################################
#####################################################################################

#####################################################################################
# Normalized Cohesiveness
#####################################################################################
def BiasLim(BiasDF, size):
    max_mean_bias = BiasDF.head(size)["EFFECT"].mean()
    bias_constranit_p1 = np.arange(0, 0.2, 0.05)
    bias_constranit_p2 = np.arange(0.2, 0.3, 0.01)
    bias_constranit_p3 = np.arange(0.3, max_mean_bias, 0.005)
    bias_constranit = np.concatenate((bias_constranit_p1, bias_constranit_p2, bias_constranit_p3), axis=0)
    res = []
    for j, b in enumerate(bias_constranit):
        res.append((size, round(b, 3)))
    return res


def calculate_circuit_scores(pc_scores_df, IpsiInfoMat, sort_by="PC1"):
    STR_Ranks = pc_scores_df.sort_values(sort_by, ascending=False).index.values
    topNs = list(range(200, 5, -1))
    SC_Agg_topN_score = []
    
    for topN in topNs:
        top_strs = STR_Ranks[:topN]
        score = ScoreCircuit_SI_Joint(top_strs, IpsiInfoMat)
        SC_Agg_topN_score.append(score)
        
    return np.array(SC_Agg_topN_score)
    
def ScoreCircuit_SI_Joint(STRs, InfoMat):
    CirInfo = InfoMat.loc[STRs, STRs]
    #N_events = len(STRs) * (len(STRs) - 1)
    N_events = np.count_nonzero(CirInfo)
    CirInfo = np.nan_to_num(CirInfo, nan=0)
    score = np.sum(CirInfo)
    return score/N_events

def ScoreCircuit_NEdges(STRs, AdjMat):
    CirMat = AdjMat.loc[STRs, STRs]
    N_edges = np.count_nonzero(CirMat)
    N_edges_Possible = len(STRs) * len(STRs)
    return N_edges/N_edges_Possible
    
def ScoreCircuit_NEdges_ALL(STRs, AdjMat):
    CirMat = AdjMat.loc[STRs, STRs]
    CirScource = AdjMat.loc[STRs, :]
    CirTarget  = AdjMat.loc[:, STRs]
    N_edges = np.count_nonzero(CirMat)
    N_edges_S = np.count_nonzero(CirScource)
    N_edges_T = np.count_nonzero(CirTarget)
    return (N_edges_S + N_edges_T - N_edges) / (2 * len(STRs) * 213 - len(STRs)*len(STRs))


class CircuitSearch_SA_InfoContent(Annealer):
    def __init__(self, BiasDF, state, adjMat, InfoMat, CandidateNodes, minbias):
        self.BiasDF = BiasDF
        self.adjMat = adjMat
        self.InfoMat = InfoMat
        self.CandidateNodes = CandidateNodes
        self.minbias = minbias
        super(CircuitSearch_SA_InfoContent, self).__init__(state)  # important!
    def move(self):
        initial_energy = self.energy()
        # Swipe two nodes
        idx_in = np.where(self.state==1)[0]
        idx_out = np.where(self.state==0)[0]
        idx_change_i = np.random.choice(idx_in, 1)
        idx_change_j = np.random.choice(idx_out, 1)
        self.state[idx_change_i] = 1 - self.state[idx_change_i]
        self.state[idx_change_j] = 1 - self.state[idx_change_j]
        strs = self.CandidateNodes[np.where(self.state==1)]
        if self.BiasDF.loc[strs, "EFFECT"].mean() < self.minbias:
            self.state[idx_change_i] = 1 - self.state[idx_change_i]
            self.state[idx_change_j] = 1 - self.state[idx_change_j]
        return (self.energy() - initial_energy) 
    def energy(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        #score = ScoreCircuit_v7(InCirtuitNodes, self.adjMat, self.ProbMat1, self.ProbMat2)
        score = ScoreCircuit_SI_Joint(InCirtuitNodes, self.InfoMat)
        return -score

class CircuitSearch_SA_Connectivity(Annealer):
    def __init__(self, BiasDF, state, adjMat, InfoMat, CandidateNodes, minbias):
        self.BiasDF = BiasDF
        self.adjMat = adjMat
        self.InfoMat = InfoMat
        self.CandidateNodes = CandidateNodes
        self.minbias = minbias
        super(CircuitSearch_SA_Connectivity, self).__init__(state)  # important!
    def move(self):
        initial_energy = self.energy()
        # Swipe two nodes
        idx_in = np.where(self.state==1)[0]
        idx_out = np.where(self.state==0)[0]
        idx_change_i = np.random.choice(idx_in, 1)
        idx_change_j = np.random.choice(idx_out, 1)
        self.state[idx_change_i] = 1 - self.state[idx_change_i]
        self.state[idx_change_j] = 1 - self.state[idx_change_j]
        strs = self.CandidateNodes[np.where(self.state==1)]
        if self.BiasDF.loc[strs, "EFFECT"].mean() < self.minbias:
            self.state[idx_change_i] = 1 - self.state[idx_change_i]
            self.state[idx_change_j] = 1 - self.state[idx_change_j]
        return (self.energy() - initial_energy) 
    def energy(self):
        InCirtuitNodes = self.CandidateNodes[np.where(self.state==1)[0]]
        #score = ScoreCircuit_v7(InCirtuitNodes, self.adjMat, self.ProbMat1, self.ProbMat2)
        #score = ScoreCircuit_SI_Joint(InCirtuitNodes, self.InfoMat)
        score = ScoreCircuit_NEdges(InCirtuitNodes, self.adjMat)
        return -score


#####################################################################################
#####################################################################################
#### Analyze Neurotransmitter Systems
#####################################################################################
#####################################################################################
def create_gene_weights_by_system_category(neural_system_df, system=None, category=None, group=None, SOURCE_CATEGORIES=None, TARGET_CATEGORIES=None, save_dir="dat/Genetics/GeneWeights/", weights = "EXP"):
    """
    Create gene weights dictionary for specific neurotransmitter system and/or category
    
    Parameters:
    -----------
    neural_system_df : pd.DataFrame
        DataFrame with neurotransmitter system data
    system : str, optional
        Specific neurotransmitter system ('dopamine', 'serotonin', 'oxytocin', 'acetylcholine')
    category : str, optional  
        Specific category ('synthesis', 'receptor', etc.)
    group : str, optional
        Either 'source' or 'target' to group categories
    SOURCE_CATEGORIES : list, optional
        Categories for source genes (default: ['synthesis', 'transporter', 'storage_release'])
    TARGET_CATEGORIES : list, optional  
        Categories for target genes (default: ['receptor', 'degradation'])
    
    Returns:
    --------
    dict : Gene weights dictionary (entrez_id -> 1.0)
    """
    # Set default categories if not provided
    if SOURCE_CATEGORIES is None:
        SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release']
    if TARGET_CATEGORIES is None:
        TARGET_CATEGORIES = ['receptor', 'degradation']
        
    df = neural_system_df.copy()
    
    if system is not None:
        df = df[df['neurotransmitter_system'] == system]
    
    if category is not None:
        df = df[df['category'] == category]
    
    if group is not None:
        if group == 'source':
            df = df[df['category'].isin(SOURCE_CATEGORIES)]
        elif group == 'target':
            df = df[df['category'].isin(TARGET_CATEGORIES)]
    
    # Create gene weights dictionary
    gene_weights = {}
    for _, row in df.iterrows():
        entrez_id = row['entrez_id']
        if pd.notna(entrez_id) and entrez_id != 0:
            if weights == "EXP":
                gene_weights[int(entrez_id)] = row['EXP']
            else:
                gene_weights[int(entrez_id)] = 1.0
    if save_dir is not None:
        full_save_dir = os.path.join(_ProjDIR, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        fil_ = os.path.join(full_save_dir, f"NT_{system}_{group}.gw")
        Dict2Fil(gene_weights, fil_)
    return gene_weights


def analyze_neurotransmitter_systems_source_target(
    neural_system_df,
    bias_mat=None,
    SOURCE_CATEGORIES=None,
    TARGET_CATEGORIES=None,
    save_dir="dat/Genetics/GeneWeights/",
    weights = "EXP",
    generate_bias=False
):
    """
    Analyze all neurotransmitter systems with source, target, and combined biases,
    or just generate .gw files for each (default).
    Only uses genes from SOURCE_CATEGORIES and TARGET_CATEGORIES

    Parameters:
    -----------
    neural_system_df : pd.DataFrame
        DataFrame with neurotransmitter system data
    bias_mat : pd.DataFrame, optional
        Structure bias matrix (required if generate_bias=True)
    SOURCE_CATEGORIES : list, optional
        Categories for source genes (default: ['synthesis', 'transporter', 'storage_release'])
    TARGET_CATEGORIES : list, optional  
        Categories for target genes (default: ['receptor', 'degradation'])
    save_dir : str
        Directory to save .gw files
    generate_bias : bool, default False
        If True, calculate bias (requires bias_mat); if False, only generate .gw files

    Returns:
    --------
    dict : Results for each system with 'source', 'target', 'combined' DataFrames (empty if generate_bias is False)
    """

    # Set default categories if not provided
    if SOURCE_CATEGORIES is None:
        SOURCE_CATEGORIES = ['synthesis', 'transporter', 'storage_release']
    if TARGET_CATEGORIES is None:
        TARGET_CATEGORIES = ['receptor', 'degradation']

    results = {}

    if generate_bias:
        if bias_mat is None:
            raise ValueError("bias_mat must be provided when generate_bias=True")
        Anno = STR2Region()

    systems = neural_system_df['neurotransmitter_system'].unique()

    print("Analyzing neurotransmitter systems with SOURCE-TARGET grouping:")
    print(f"SOURCE categories: {SOURCE_CATEGORIES}")
    print(f"TARGET categories: {TARGET_CATEGORIES}")
    print("Note: Other categories (metabolism, processing) will be excluded\n")

    for system in systems:
        print(f"Analyzing {system.upper()} system...")
        results[system] = {}

        # Source genes analysis
        source_weights = create_gene_weights_by_system_category(
            neural_system_df, 
            system=system, 
            group='source', 
            SOURCE_CATEGORIES=SOURCE_CATEGORIES, 
            TARGET_CATEGORIES=TARGET_CATEGORIES, 
            save_dir=save_dir,
            weights=weights
        )
        if generate_bias and source_weights:
            source_bias = MouseSTR_AvgZ_Weighted(bias_mat, source_weights)
            source_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in source_bias.index.values]
            results[system]['source'] = source_bias
            print(f"  Source genes: {len(source_weights)}")
        else:
            if not source_weights:
                print(f"  Source genes: 0 (no genes found)")
            else:
                print(f"  Source genes: {len(source_weights)} (gw file generated)")

        # Target genes analysis
        target_weights = create_gene_weights_by_system_category(
            neural_system_df,
            system=system,
            group='target',
            SOURCE_CATEGORIES=SOURCE_CATEGORIES,
            TARGET_CATEGORIES=TARGET_CATEGORIES,
            save_dir=save_dir,
            weights=weights
        )
        if generate_bias and target_weights:
            target_bias = MouseSTR_AvgZ_Weighted(bias_mat, target_weights)
            target_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in target_bias.index.values]
            results[system]['target'] = target_bias
            print(f"  Target genes: {len(target_weights)}")
        else:
            if not target_weights:
                print(f"  Target genes: 0 (no genes found)")
            else:
                print(f"  Target genes: {len(target_weights)} (gw file generated)")

        # Combined analysis (source + target genes only)
        combined_weights = {}
        combined_weights.update(source_weights if source_weights else {})
        combined_weights.update(target_weights if target_weights else {})
        full_save_dir = os.path.join(_ProjDIR, save_dir)
        os.makedirs(full_save_dir, exist_ok=True)
        fil_ = os.path.join(full_save_dir, f"NT_{system}_combined.gw")
        Dict2Fil(combined_weights, fil_)

        if generate_bias and combined_weights:
            combined_bias = MouseSTR_AvgZ_Weighted(bias_mat, combined_weights)
            combined_bias["Region"] = [Anno.get(ct_idx, "Unknown") for ct_idx in combined_bias.index.values]
            results[system]['combined'] = combined_bias
            print(f"  Combined genes: {len(combined_weights)} (source + target only)")
        else:
            if not combined_weights:
                print(f"  Combined genes: 0 (no genes found)")
            else:
                print(f"  Combined genes: {len(combined_weights)} (gw file generated)")

        # Show which categories are present for this system
        system_df = neural_system_df[neural_system_df['neurotransmitter_system'] == system]
        available_categories = system_df['category'].unique()
        used_categories = [cat for cat in available_categories if cat in SOURCE_CATEGORIES + TARGET_CATEGORIES]
        excluded_categories = [cat for cat in available_categories if cat not in SOURCE_CATEGORIES + TARGET_CATEGORIES]

        print(f"  Used categories: {used_categories}")
        if excluded_categories:
            print(f"  Excluded categories: {excluded_categories}")
        print()

    if generate_bias:
        return results
    else:
        return {}



def plot_source_target_heatmap(results, top_n=15, save_plot=False, results_dir="./results"):
    """
    Create heatmap comparing source vs target across all systems
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    """
    comparison_data = {}
    all_structures = set()
    
    for system, system_data in results.items():
        for group_type in ['source', 'target', 'combined']:
            if group_type in system_data:
                key = f"{system}_{group_type}"
                comparison_data[key] = system_data[group_type]['EFFECT']
                all_structures.update(system_data[group_type].head(top_n).index)
    
    # Create comparison matrix
    comparison_df = pd.DataFrame(index=list(all_structures), columns=list(comparison_data.keys()))
    
    for key, effects in comparison_data.items():
        comparison_df[key] = effects
    
    # Fill NaN with 0 and sort by mean effect
    comparison_df = comparison_df.fillna(0)
    comparison_df = comparison_df.loc[comparison_df.mean(axis=1).sort_values(ascending=False).index]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, max(8, len(comparison_df) * 0.3)))
    sns.heatmap(comparison_df.head(top_n), 
                annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
                yticklabels=[s.replace('_', ' ') for s in comparison_df.head(top_n).index],
                ax=ax)
    ax.set_title('Neurotransmitter Source vs Target Comparison - Top Structures')
    ax.set_xlabel('System_Group')
    ax.set_ylabel('Brain Structure')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_heatmap_top{top_n}.svg')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {plot_path}")
    
    plt.show()
    return fig

def plot_source_target_only(
    results, 
    top_n=15, 
    save_plot=False, 
    results_dir="./results",
    pvalue_label="q-value",
    dpi=300
):
    """
    Visualize source vs target neurotransmitter system bias results, 
    with significance stars (* for q < 0.05, ** for q < 0.01, *** for q < 0.001)
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    top_n : int, default 15
        Number of top structures to show
    save_plot : bool, default False
        Whether to save the plot to results directory
    results_dir : str, default "./results"
        Directory to save plots
    pvalue_label : str, default "q-value"
        Column/field name used to extract p-value for significance stars (e.g., "q-value" or "p-value")
    """
    
    def get_significance_star(q):
        if q < 0.001:
            return '***'
        elif q < 0.01:
            return '**'
        elif q < 0.05:
            return '*'
        return ''
    
    systems = list(results.keys())
    n_systems = len(systems)
    
    # Create subplots
    fig, axes = plt.subplots(n_systems, 3, figsize=(25, 4*n_systems), dpi=480)
    if n_systems == 1:
        axes = axes.reshape(1, -1)
    
    for i, system in enumerate(systems):
        system_data = results[system]
        
        # Plot source
        if 'source' in system_data:
            ax = axes[i, 0] if n_systems > 1 else axes[0]
            top_source = system_data['source'].head(top_n)
            bars = ax.barh(range(len(top_source)), top_source['EFFECT'])
            ax.set_yticks(range(len(top_source)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_source.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Source\n(synthesis, transporter, storage_release)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('orange')
            # Add significance stars
            if pvalue_label in top_source.columns:
                top_source_pval = top_source[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'source'.")
            for j, (eff_val, p_val) in enumerate(zip(top_source['EFFECT'], top_source_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset to right/left of bar depending on sign
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
        
        # Plot target
        if 'target' in system_data:
            ax = axes[i, 1] if n_systems > 1 else axes[1]
            top_target = system_data['target'].head(top_n)
            bars = ax.barh(range(len(top_target)), top_target['EFFECT'])
            ax.set_yticks(range(len(top_target)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_target.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Target\n(receptor, degradation)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('blue')
            # Add significance stars
            if pvalue_label in top_target.columns:
                top_target_pval = top_target[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'target'.")
            for j, (eff_val, p_val) in enumerate(zip(top_target['EFFECT'], top_target_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
        
        # Plot combined
        if 'combined' in system_data:
            ax = axes[i, 2] if n_systems > 1 else axes[2]
            top_combined = system_data['combined'].head(top_n)
            bars = ax.barh(range(len(top_combined)), top_combined['EFFECT'])
            ax.set_yticks(range(len(top_combined)))
            ax.set_yticklabels([s.replace('_', ' ') for s in top_combined.index], fontsize=16)
            ax.set_xlabel('Bias Effect', fontsize=16)
            ax.set_title(f'{system.capitalize()} - Combined\n(source + target)', fontsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.invert_yaxis()
            for bar in bars:
                bar.set_color('gray')
            # Add significance stars
            if pvalue_label in top_combined.columns:
                top_combined_pval = top_combined[pvalue_label]
            else:
                raise ValueError(f"P-value column '{pvalue_label}' not found in the dataframe for system '{system}', 'combined'.")
            for j, (eff_val, p_val) in enumerate(zip(top_combined['EFFECT'], top_combined_pval)):
                star = get_significance_star(p_val)
                if star:
                    ax.text(
                        eff_val + 0.03 * (1 if eff_val >= 0 else -1),  # Offset
                        j,
                        star,
                        va='center',
                        ha='left' if eff_val >= 0 else 'right',
                        color='black',
                        fontsize=18,
                        fontweight='bold'
                    )
    
    plt.tight_layout()
    
    if save_plot:
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        plot_path = os.path.join(results_dir, f'neurotransmitter_barplots_top{top_n}.svg')
        fig.savefig(plot_path, dpi=480, bbox_inches='tight')
        print(f"Bar plots saved to: {plot_path}")
    
    plt.show()
    
    return fig

def save_neurotransmitter_results(results, results_dir="./results"):
    """
    Save neurotransmitter analysis results to CSV files in results directory
    
    Parameters:
    -----------
    results : dict
        Results from analyze_neurotransmitter_systems_source_target
    results_dir : str, default "./results"
        Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Saving neurotransmitter results to {results_dir}/")
    saved_files = []
    
    for system, system_data in results.items():
        for analysis_type, bias_df in system_data.items():
            filename = f"{system}_{analysis_type}_bias.csv"
            filepath = os.path.join(results_dir, filename)
            bias_df.to_csv(filepath)
            saved_files.append(filepath)
            print(f"  Saved: {filepath}")
    
    return saved_files


def GetPermutationP(null_dist, observed_val, greater_than=True):
    """Calculate permutation-based p-value and z-score.
    
    Args:
        null_dist: List/array of null distribution values
        observed_val: Observed value to test
        greater_than: If True, calculates P(null >= observed), else P(null <= observed)
    
    Returns:
        z_score: Z-score of observed value relative to null distribution
        p_value: Permutation-based p-value
        obs_adj: Mean-adjusted observed value
    """
    # Remove NaN values
    null_dist = np.array([x for x in null_dist if not np.isnan(x)])
    
    # Calculate statistics
    z_score = (observed_val - np.mean(null_dist)) / np.std(null_dist)
    count = sum(observed_val <= x for x in null_dist) if greater_than else sum(observed_val >= x for x in null_dist)
    p_value = (count + 1) / (len(null_dist) + 1)
    obs_adj = observed_val - np.mean(null_dist)
    
    return z_score, p_value, obs_adj


#####################################################################################
#####################################################################################
#### Bias Dataset Comparison Utilities
#####################################################################################
#####################################################################################

def fit_structure_bias_linear_model(merged_data, metric='EFFECT', suffixes=('_1', '_2')):
    """Fit a linear regression between two bias datasets and compute residuals.

    Parameters
    ----------
    merged_data : DataFrame
        Merged bias data with columns ``{metric}{suffix}`` for each suffix.
    metric : str
        Column base name to use (default ``'EFFECT'``).
    suffixes : tuple of str
        Column suffixes identifying the two datasets.

    Returns
    -------
    DataFrame
        Copy of *merged_data* with added ``predicted`` and ``residual`` columns.
    """
    from sklearn.linear_model import LinearRegression

    col_x = f'{metric}{suffixes[1]}'
    col_y = f'{metric}{suffixes[0]}'

    # Drop rows with NaN in either column before fitting
    valid_mask = merged_data[col_x].notna() & merged_data[col_y].notna()
    X_all = merged_data[col_x].values.reshape(-1, 1)
    y_all = merged_data[col_y].values

    model = LinearRegression()
    model.fit(X_all[valid_mask], y_all[valid_mask])

    results_df = merged_data.copy()
    results_df['predicted'] = np.nan
    results_df.loc[valid_mask, 'predicted'] = model.predict(X_all[valid_mask])
    results_df['residual'] = results_df[col_y] - results_df['predicted']
    return results_df


def merge_bias_datasets(dataset1, dataset2, suffixes=('_1', '_2'),
                        cols1=None, cols2=None):
    """Merge two bias datasets and compute difference / residual metrics.

    Works for both structure-level and cell-type-level data — just pass the
    appropriate column lists.

    Parameters
    ----------
    dataset1, dataset2 : DataFrame
        Bias DataFrames indexed by structure or cell-type name.
    suffixes : tuple of str
        Suffixes appended to overlapping column names after merge.
    cols1 : list of str or None
        Columns to keep from *dataset1*.  If ``None``, defaults to
        ``['Rank', 'EFFECT', 'Region']`` (structure level).
    cols2 : list of str or None
        Columns to keep from *dataset2*.  If ``None``, defaults to
        ``['Rank', 'EFFECT']``.

    Returns
    -------
    DataFrame
        Merged data sorted by ``ABS_DIFF_EFFECT``, with ``predicted`` and
        ``residual`` columns from a linear model fit.
    """
    if cols1 is None:
        cols1 = ['Rank', 'EFFECT', 'Region']
    if cols2 is None:
        cols2 = ['Rank', 'EFFECT']

    # Only keep columns that actually exist in each dataset
    cols1 = [c for c in cols1 if c in dataset1.columns]
    cols2 = [c for c in cols2 if c in dataset2.columns]

    merged_data = pd.merge(dataset1[cols1], dataset2[cols2],
                           left_index=True, right_index=True,
                           suffixes=suffixes)

    merged_data['DIFF_Rank'] = merged_data[f'Rank{suffixes[0]}'] - merged_data[f'Rank{suffixes[1]}']
    merged_data['ABS_DIFF_Rank'] = np.abs(merged_data['DIFF_Rank'])
    merged_data['DIFF_EFFECT'] = merged_data[f'EFFECT{suffixes[0]}'] - merged_data[f'EFFECT{suffixes[1]}']
    merged_data['ABS_DIFF_EFFECT'] = np.abs(merged_data['DIFF_EFFECT'])

    merged_data = merged_data.sort_values('ABS_DIFF_EFFECT', ascending=False)
    merged_data = fit_structure_bias_linear_model(merged_data, metric='EFFECT',
                                                  suffixes=suffixes)
    return merged_data


def batch_permutation_bias(STR_BiasMat, gene_pool, weights, n_perm=10000,
                           seed=42):
    """Run batched permutation bias for structure-level analysis.

    Instead of calling ``MouseSTR_AvgZ_Weighted`` *n_perm* times, this
    pre-computes the valid gene intersection once and performs a single
    batched matrix multiply.

    Parameters
    ----------
    STR_BiasMat : DataFrame
        Expression z-score matrix (genes × structures).
    gene_pool : list
        Pool of Entrez gene IDs to sample from.
    weights : list
        Weights to assign to each sampled gene (length determines sample
        size per permutation).
    n_perm : int
        Number of permutations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of DataFrame
        One bias DataFrame per permutation, each with ``EFFECT``, ``Rank``,
        and ``Region`` columns, matching the format of
        ``MouseSTR_AvgZ_Weighted``.
    """
    rng = np.random.default_rng(seed)
    k = len(weights)
    weights_arr = np.array(weights, dtype=float)

    # Valid genes = intersection of gene_pool with expression matrix index
    gene_pool_set = set(gene_pool)
    valid_pool = [g for g in STR_BiasMat.index if g in gene_pool_set]
    valid_pool_idx = {g: i for i, g in enumerate(valid_pool)}
    expr_sub = STR_BiasMat.loc[valid_pool].values  # (N_valid_genes, N_structures)

    str2reg = STR2Region()
    structures = STR_BiasMat.columns

    results = []
    for _ in range(n_perm):
        sampled = rng.choice(len(valid_pool), size=k, replace=False)
        expr_sample = expr_sub[sampled]  # (k, N_structures)
        mask = ~np.isnan(expr_sample)
        w_broadcast = weights_arr[:, np.newaxis]
        weighted_vals = expr_sample * w_broadcast * mask
        weight_sums = w_broadcast * mask
        effects = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)

        df = pd.DataFrame({'Structure': structures, 'EFFECT': effects}).set_index('Structure')
        df = df.sort_values('EFFECT', ascending=False)
        df['Rank'] = np.arange(1, len(df) + 1)
        df['Region'] = [str2reg.get(s, 'Unknown') for s in df.index]
        results.append(df)

    return results