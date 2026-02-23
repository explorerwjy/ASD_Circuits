# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: gencic
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import (
    LoadGeneINFO, GetPermutationP, ScoreCircuit_SI_Joint,
)

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")

# %% [markdown]
# # 07. Stratified Distance Analysis
#
# Compare the connectivity of ASD circuit structures vs sibling-derived control
# structures, stratified by pairwise Cartesian distance.
#
# **Sections**:
# 1. Load data (connectivity, distance, circuits)
# 2. Distance-stratified connectivity — top-bias sibling structures as control
# 3. Distance-stratified connectivity — SA-derived sibling circuits as control
# 4. Summary plots and significance tests

# %% [markdown]
# ## 1. Load Data

# %%
# ASD circuit structures
ASD_CircuitsSet = pd.read_csv(
    "../results/STR_ISH/ASD.SA.Circuits.Size46.csv", index_col="idx"
)
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")

# Sibling top-bias structures (top 46 of 61-gene sibling bias)
RealSibSTRs = pd.read_csv(
    "../dat/Unionize_bias/sib.top61.Z2.csv", index_col="STR"
).head(46).index.values

print(f"ASD circuit: {len(ASD_Circuits)} structures")
print(f"Sibling top-bias: {len(RealSibSTRs)} structures")

# %%
# Connectivity and distance matrices
CONN_DIR = "../dat/allen-mouse-conn/ConnectomeScoringMat"
adj_mat = pd.read_csv(f"{CONN_DIR}/WeightMat.Ipsi.csv", index_col=0)
dist_mat = pd.read_csv("../dat/allen-mouse-conn/Dist_CartesianDistance.csv", index_col=0)
dist_mat.columns = dist_mat.index.values

InfoMat = pd.read_csv(f"{CONN_DIR}/InfoMat.Ipsi.csv", index_col=0)

print(f"Adjacency matrix: {adj_mat.shape}")
print(f"Distance matrix: {dist_mat.shape}")

# %% [markdown]
# ## 2. Distance-Stratified Connectivity: Top-Bias Structures

# %%
# Utility: mask connectivity by distance range
def MaskDistMat_xy(distance_mat, Conn_mat, cutoff, cutoff2, keep='bw'):
    """Zero out entries outside a distance range. Excludes self-connections."""
    Conn_new = Conn_mat.copy(deep=True)
    dist_new = distance_mat.copy(deep=True)
    for STR_i in distance_mat.index.values:
        for STR_j in distance_mat.columns.values:
            if STR_i == STR_j:
                Conn_new.loc[STR_i, STR_j] = 0
                dist_new.loc[STR_i, STR_j] = 0
                continue
            d = distance_mat.loc[STR_i, STR_j]
            if keep == 'bw':
                inside = (d >= cutoff) and (d <= cutoff2)
            elif keep == 'gt':
                inside = d >= cutoff
            elif keep == 'lt':
                inside = d <= cutoff
            else:
                inside = False
            if not inside:
                Conn_new.loc[STR_i, STR_j] = 0
                dist_new.loc[STR_i, STR_j] = 0
    return Conn_new, dist_new


# %%
# Bin all structure pairs by distance range
Distance_Cuts = [0, 1000, 2000, 3000, 4000, 100000]
N_bins = len(Distance_Cuts) - 1

Cutted_AdjMat = {}
Cutted_DistMat = {}
N_Connections_total = np.zeros(N_bins, dtype=int)
N_Pairs_total = np.zeros(N_bins, dtype=int)

for i in range(N_bins):
    Conn_new, dist_new = MaskDistMat_xy(
        dist_mat, adj_mat, Distance_Cuts[i], Distance_Cuts[i + 1], keep="bw"
    )
    Cutted_AdjMat[i] = Conn_new
    Cutted_DistMat[i] = dist_new
    N_Connections_total[i] = np.count_nonzero(Conn_new)
    N_Pairs_total[i] = np.count_nonzero(dist_new)

print("Distance bins (um):", list(zip(Distance_Cuts[:-1], Distance_Cuts[1:])))
print("Total connections per bin:", N_Connections_total)
print("Total pairs per bin:", N_Pairs_total)


# %%
# Count connections/pairs for a set of structures across distance bins
def count_connections_by_distance(structures, Cutted_AdjMat, Cutted_DistMat, N_bins):
    """Count nonzero connections and pairs per distance bin for given structures."""
    connections = np.zeros(N_bins, dtype=int)
    pairs = np.zeros(N_bins, dtype=int)
    for i in range(N_bins):
        adj_sub = Cutted_AdjMat[i].loc[structures, structures]
        dist_sub = Cutted_DistMat[i].loc[structures, structures]
        connections[i] = np.count_nonzero(adj_sub)
        pairs[i] = np.count_nonzero(dist_sub)
    return connections, pairs


# %%
# ASD circuit connectivity by distance
ASD_Connections, ASD_Pairs = count_connections_by_distance(
    ASD_Circuits, Cutted_AdjMat, Cutted_DistMat, N_bins
)

# Sibling top-bias connectivity
SIB_Connections, SIB_Pairs = count_connections_by_distance(
    RealSibSTRs, Cutted_AdjMat, Cutted_DistMat, N_bins
)

# Subsampled sibling controls (10K random structure sets)
CACHE_DIR = "../results/Distance_analysis"
os.makedirs(CACHE_DIR, exist_ok=True)
subsib_cache = os.path.join(CACHE_DIR, "subsib_distance_counts.npz")

if os.path.exists(subsib_cache):
    print(f"Loading cached SubSampleSib counts from {subsib_cache}...")
    data = np.load(subsib_cache)
    subsib_connections = data["connections"]
    subsib_pairs = data["pairs"]
else:
    print("Computing SubSampleSib distance counts (10K files, caching for next time)...")
    Sim_dir = "../dat/Unionize_bias/SubSampleSib/"
    subsib_connections = []
    subsib_pairs = []
    for file in sorted(os.listdir(Sim_dir)):
        if file.startswith("cont.genes"):
            continue
        df = pd.read_csv(os.path.join(Sim_dir, file), index_col="STR")
        strs = df.head(46).index.values
        conn, pair = count_connections_by_distance(
            strs, Cutted_AdjMat, Cutted_DistMat, N_bins
        )
        subsib_connections.append(conn)
        subsib_pairs.append(pair)
    subsib_connections = np.array(subsib_connections)
    subsib_pairs = np.array(subsib_pairs)
    np.savez(subsib_cache, connections=subsib_connections, pairs=subsib_pairs)
    print(f"  Cached to {subsib_cache}")

subsib_conns_mean = subsib_connections.mean(axis=0)
subsib_pairs_mean = subsib_pairs.mean(axis=0)
print(f"Loaded {len(subsib_connections)} subsampled controls")

# %%
# Significance tests: connection enrichment per distance bin
print("Connection enrichment per distance bin (ASD vs subsampled siblings):")
print(f"{'Bin':>10s} {'ASD':>8s} {'Ctrl mean':>10s} {'Ratio':>8s} {'p-value':>10s}")
print("-" * 50)
for i, (asd, sibs) in enumerate(zip(ASD_Connections, subsib_connections.T)):
    z, p, _ = GetPermutationP(sibs, asd)
    label = f"{Distance_Cuts[i]}-{Distance_Cuts[i+1]}"
    print(f"{label:>10s} {asd:>8d} {np.mean(sibs):>10.1f} {asd/np.mean(sibs):>8.2f} {p:>10.4f}")

# %%
# Plot: Connection Density by distance
matplotlib.rcParams.update({'font.size': 20})

fig, axes = plt.subplots(1, 3, dpi=240, figsize=(18, 6))
X = np.arange(N_bins)
xlabels = ["0-1", "1-2", "2-3", "3-4", r">4"]

# Panel 1: Connection Density
ax = axes[0]
ax.plot(X, ASD_Connections / N_Connections_total, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, subsib_conns_mean / N_Connections_total,
            yerr=np.nanstd(subsib_connections / N_Connections_total, axis=0),
            marker="o", color="grey", label="Sibling Structures")
ax.set_ylabel("Connection Density", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.5)

# Panel 2: Connection Likelihood Ratio
ax = axes[1]
norm = N_Connections_total / N_Pairs_total
ax.plot(X, ASD_Connections / ASD_Pairs / norm, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, subsib_conns_mean / subsib_pairs_mean / norm,
            yerr=np.nanstd(subsib_connections / subsib_pairs / norm, axis=0),
            marker="o", color="grey", label="Sibling Structures")
ax.set_ylabel("Connection Likelihood Ratio", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.5)

# Panel 3: Structure Pairs Density
ax = axes[2]
ax.plot(X, ASD_Pairs / N_Pairs_total, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, subsib_pairs_mean / N_Pairs_total,
            yerr=np.nanstd(subsib_pairs / N_Pairs_total, axis=0),
            marker="o", color="grey", label="Sibling Structures")
ax.set_ylabel("Structure Pairs Density", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Distance-Stratified Connectivity: SA-Derived Sibling Circuits
#
# Instead of using top-bias sibling structures, use circuits found by
# simulated annealing on subsampled sibling bias profiles.

# %%
# Helpers for loading SA results
def searchFil(text, DIR):
    return [f for f in os.listdir(DIR) if text in f]


def LoadSA3(fname, DIR, InfoMat, topL=100):
    """Load SA result file and find the highest-scoring circuit."""
    max_score, max_bias, max_STRs = 0, 0, []
    with open(os.path.join(DIR, fname), 'rt') as fin:
        for i, l in enumerate(fin):
            if i > topL:
                break
            parts = l.strip().split()
            bias = float(parts[1])
            STRs = parts[2].split(",")
            score = ScoreCircuit_SI_Joint(STRs, InfoMat)
            if score > max_score:
                max_score = score
                max_bias = bias
                max_STRs = STRs
    return max_score, max_bias, max_STRs


def ExtractSibSTR(bias_cut, DIR, InfoMat):
    """Extract sibling circuit structures at a given bias cutoff."""
    fil = searchFil(f"topN_121-keepN_46-minbias_{bias_cut}.txt", DIR)[0]
    _, _, STRs = LoadSA3(fil, DIR, InfoMat)
    return STRs


# %%
# Load SA-found sibling circuits (with caching)
SIB_SA_DIR = "../dat/Circuits/SA/SubSib_ScoreInfo_Dec21/"
bias_cut = 0.37
sa_sib_cache = os.path.join(CACHE_DIR, f"sa_sib_distance_counts_bias{bias_cut}.npz")

if os.path.exists(sa_sib_cache):
    print(f"Loading cached SA sibling circuit counts from {sa_sib_cache}...")
    data = np.load(sa_sib_cache)
    sa_sib_connections = data["connections"]
    sa_sib_pairs = data["pairs"]
else:
    print(f"Computing SA sibling circuits (6K dirs, bias_cut={bias_cut}, caching for next time)...")
    Sib_Cir_STRs = []
    n_failed = 0
    for file in sorted(os.listdir(SIB_SA_DIR)):
        d = os.path.join(SIB_SA_DIR, file)
        if os.path.isdir(d):
            try:
                STRs = ExtractSibSTR(bias_cut, d + "/", InfoMat)
                Sib_Cir_STRs.append(STRs)
            except Exception:
                n_failed += 1
                continue
    print(f"  Extracted {len(Sib_Cir_STRs)} circuits ({n_failed} failed)")

    sa_sib_connections = []
    sa_sib_pairs = []
    for strs in Sib_Cir_STRs:
        conn, pair = count_connections_by_distance(
            strs, Cutted_AdjMat, Cutted_DistMat, N_bins
        )
        sa_sib_connections.append(conn)
        sa_sib_pairs.append(pair)
    sa_sib_connections = np.array(sa_sib_connections)
    sa_sib_pairs = np.array(sa_sib_pairs)
    np.savez(sa_sib_cache, connections=sa_sib_connections, pairs=sa_sib_pairs)
    print(f"  Cached to {sa_sib_cache}")

sa_sib_conns_mean = sa_sib_connections.mean(axis=0)
sa_sib_pairs_mean = sa_sib_pairs.mean(axis=0)
print(f"SA sibling circuits: {sa_sib_connections.shape[0]} circuits × {N_bins} distance bins")

# %%
# Significance tests: ASD vs SA sibling circuits
print("Connection enrichment per distance bin (ASD vs SA sibling circuits):")
print(f"{'Bin':>10s} {'ASD':>8s} {'Ctrl mean':>10s} {'Ratio':>8s} {'p-value':>10s}")
print("-" * 50)
for i, (asd, sibs) in enumerate(zip(ASD_Connections, sa_sib_connections.T)):
    z, p, _ = GetPermutationP(sibs, asd)
    label = f"{Distance_Cuts[i]}-{Distance_Cuts[i+1]}"
    print(f"{label:>10s} {asd:>8d} {np.mean(sibs):>10.1f} {asd/np.mean(sibs):>8.2f} {p:>10.4f}")

# Grouped significance (short-range vs long-range)
split_idx = 3
print(f"\nGrouped test (short-range: 0-{Distance_Cuts[split_idx]}um, "
      f"long-range: {Distance_Cuts[split_idx]}um+):")

CaseShort = np.sum(ASD_Connections[:split_idx]) / np.sum(N_Connections_total[:split_idx])
CaseLong = np.sum(ASD_Connections[split_idx:]) / np.sum(N_Connections_total[split_idx:])
CtrlShort = np.sum(sa_sib_connections[:, :split_idx], axis=1) / np.sum(N_Connections_total[:split_idx])
CtrlLong = np.sum(sa_sib_connections[:, split_idx:], axis=1) / np.sum(N_Connections_total[split_idx:])

z_s, p_s, _ = GetPermutationP(CtrlShort, CaseShort)
z_l, p_l, _ = GetPermutationP(CtrlLong, CaseLong)
print(f"  Short-range: ASD={CaseShort:.4f}, ctrl={CtrlShort.mean():.4f}, p={p_s:.4f}")
print(f"  Long-range:  ASD={CaseLong:.4f}, ctrl={CtrlLong.mean():.4f}, p={p_l:.4f}")

# %% [markdown]
# ## 4. Summary Plots (SA Sibling Circuits)

# %%
matplotlib.rcParams.update({'font.size': 20})

fig, axes = plt.subplots(1, 3, dpi=240, figsize=(18, 6))
X = np.arange(N_bins)
xlabels = ["0-1", "1-2", "2-3", "3-4", r">4"]

# Panel 1: Connection Density
ax = axes[0]
ax.plot(X, ASD_Connections / N_Connections_total, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, sa_sib_conns_mean / N_Connections_total,
            yerr=np.nanstd(sa_sib_connections / N_Connections_total, axis=0),
            marker="o", color="grey", label="Sibling Circuits")
ax.set_ylabel("Connection Density", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(loc="lower left", fontsize=12)
ax.grid(True, alpha=0.5)

# Panel 2: Connection Likelihood Ratio
ax = axes[1]
norm = N_Connections_total / N_Pairs_total
sa_clr_all = sa_sib_connections / sa_sib_pairs / norm  # may have NaN from 0 pairs
ax.plot(X, ASD_Connections / ASD_Pairs / norm, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, np.nanmean(sa_clr_all, axis=0),
            yerr=np.nanstd(sa_clr_all, axis=0),
            marker="o", color="grey", label="Sibling Circuits")
ax.set_ylabel("Connection Likelihood Ratio", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(loc="upper left", fontsize=12)
ax.grid(True, alpha=0.5)

# Panel 3: Structure Pairs Density
ax = axes[2]
ax.plot(X, ASD_Pairs / N_Pairs_total, marker="o", color="blue",
        label="ASD Circuits")
ax.errorbar(X + 0.05, sa_sib_pairs_mean / N_Pairs_total,
            yerr=np.nanstd(sa_sib_pairs / N_Pairs_total, axis=0),
            marker="o", color="grey", label="Sibling Circuits")
ax.set_ylabel("Structure Pairs Density", fontsize=18)
ax.set_xlabel(r"Distance range (mm)", fontsize=16)
ax.set_xticks(X)
ax.set_xticklabels(xlabels, fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()

# Likelihood ratio significance (filter NaN from circuits with 0 pairs)
print("\nLikelihood ratio significance per distance bin (ASD vs SA sibling circuits):")
print(f"{'Bin':>10s} {'ASD CLR':>10s} {'Ctrl CLR':>10s} {'p-value':>10s}")
print("-" * 45)
sa_clr = sa_sib_connections / sa_sib_pairs  # may have NaN/inf from 0 pairs
for i in range(N_bins):
    asd_clr = ASD_Connections[i] / ASD_Pairs[i] / norm[i]
    sibs_clr = sa_clr[:, i] / norm[i]
    valid = np.isfinite(sibs_clr)
    z, p, _ = GetPermutationP(sibs_clr[valid], asd_clr)
    label = f"{Distance_Cuts[i]}-{Distance_Cuts[i+1]}"
    print(f"{label:>10s} {asd_clr:>10.4f} {np.nanmean(sibs_clr):>10.4f} {p:>10.4f}")

# %% [markdown]
# ## Summary
#
# | Metric | Description |
# |--------|-------------|
# | Connection Density | Fraction of all connections in a distance bin that involve circuit structures |
# | Connection Likelihood Ratio | Connection density normalized by pair density |
# | Structure Pairs Density | Fraction of all structure pairs in a distance bin from circuit |
