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
ProjDIR = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.insert(1, os.path.join(ProjDIR, 'src'))
from ASD_Circuits import *
from plot import *

try:
    os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
    print(f"Current working directory: {os.getcwd()}")
except FileNotFoundError as e:
    print(f"Error: Could not change directory - {e}")
except Exception as e:
    print(f"Unexpected error: {e}")


HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()

# %%
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

expr_matrix_path = config["analysis_types"]["STR_ISH"]["expr_matrix"]
STR_BiasMat = pd.read_parquet(f"../{expr_matrix_path}")
Anno = STR2Region()

# Dat used for generating figures
Cartesian_distancesDF = pd.read_csv("../dat/allen-mouse-conn/Dist_CartesianDistance.csv",
                                   index_col=0)
STR_ExpL = pd.read_csv("../dat/allen-mouse-exp/ExpLevel.csv.gz", index_col=0)

# Connectivity adjacency matrix and igraph graph
adj_mat = pd.read_csv("../dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv", index_col=0)
import igraph as ig
graph = ig.Graph.Adjacency((adj_mat.values > 0).tolist())
graph.es['weight'] = adj_mat.values[adj_mat.values.nonzero()]
graph.vs['label'] = adj_mat.index.values

# Aliases used by supplementary figure cells
ExpZ2 = STR_BiasMat
ExpL = STR_ExpL


# %% [markdown] heading_collapsed=true
# # Main Figures

# %% [markdown] hidden=true
# #### Figure 2

# %% code_folding=[29] hidden=true
def LoadBiasData2(Real_Bias, Sim_dir):
    #ASD_Bias = pd.read_csv(ASD_Bias, index_col="STR")
    ASD_STR_Biases = {}
    for STR, row in Real_Bias.iterrows():
        ASD_STR_Biases[STR] = STRBias(STR, row["EFFECT"], row["Rank"])     
            
    biases_match_rank = {}
    biases_match_STR = {}
    for file in os.listdir(Sim_dir):
        if file.startswith("cont.genes"):
            continue
        df = pd.read_csv(Sim_dir+file, index_col="STR")
        for STR, row in df.iterrows():
            if STR not in biases_match_STR:
                biases_match_STR[STR] = []
            biases_match_STR[STR].append(row["EFFECT"])
            if row["Rank"] not in biases_match_rank:
                biases_match_rank[row["Rank"]] = []
            biases_match_rank[row["Rank"]].append(row["EFFECT"])
    return ASD_STR_Biases, biases_match_rank, biases_match_STR
def CI(simulations, p):
    simulations = sorted(simulations, reverse=False)
    n = len(simulations)
    u_pval = (1+p)/2.
    l_pval = (1-u_pval)
    l_indx = int(np.floor(n*l_pval))
    u_indx = int(np.floor(n*u_pval))
    return(simulations[l_indx],simulations[u_indx])
class STRBias:
    def __init__(self, STR, Bias, Rank):
        self.STR = STR
        self.Bias = Bias
        self.Rank = Rank
        self.Boots = []
    def GetCI(self, p):
        return CI(self.Boots, p)


# %% hidden=true
# ASD Z2 and 61 Rand Genes
ASD_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")
ASD_Sim_dir = "../dat/Unionize_bias/SubSampleSib/"
ASD_Z2_Bias, ASD_Z2_Match_Bias_Rank, ASD_Z2_Sim_Bias_STR = LoadBiasData2(
    ASD_Bias, ASD_Sim_dir)

# %% hidden=true
mpl.style.use('default')
fig, ax = plt.subplots(figsize=(6, 8), dpi=480)

# Define region color mapping
REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                'Hippocampus','Amygdala','Striatum', 
                "Thalamus", "Hypothalamus", "Midbrain", 
                "Medulla", "Pallidum", "Pons", 
                "Cerebellum"]
REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                    "#2c9d39", "#742eb5", "#ed8921", 
                                    "#e82315", "#E6B0AA", "#f6b26b",  
                                    "#20124d", "#2ECC71", "#D2B4DE", 
                                    "#ffd966", ]))

plt.style.use('seaborn-v0_8-whitegrid')

# Try to find region annotation from the dataframe
region_column_candidates = ['REGION', 'Region', 'region', 'Category', 'Structure']
bias_region_dic = None
if hasattr(ASD_Bias, 'columns'):
    available_cols = set(ASD_Bias.columns)
    for col in region_column_candidates:
        if col in available_cols:
            bias_region_dic = ASD_Bias[col].to_dict()
            break

# Plotting with color region annotation
proband_scatters = []
sim_scatters = []
regions_seen = {}  # To avoid redundant labels in legend

for i, (STR, STR_bias) in enumerate(sorted(ASD_Z2_Bias.items(), key=lambda x: x[1].Rank)):
    # Determine region/color for STR
    region = None
    if bias_region_dic and STR in bias_region_dic:
        region = bias_region_dic[STR]
        color = REG_COR_Dic.get(region, "#555555")
    else:
        color = "#555555"
    # Only assign label for legend to first seen region
    region_label = region if region and region not in regions_seen else None
    if region_label:
        regions_seen[region] = True

    # Proband bias (SWAPPED: now marker "v" for ASD)
    sc_prob = ax.scatter(STR_bias.Bias, i + 1, marker="v", s=15, color=color, label=region_label, edgecolor="k", alpha=0.85)
    proband_scatters.append(sc_prob)
    # Sibling simulation mean and CI (SWAPPED: now marker "o" for siblings)
    data = ASD_Z2_Sim_Bias_STR[STR]
    ci_low, ci_high = CI(data, 0.68)
    ax.hlines(i + 1, ci_low, ci_high, color=color, lw=1.3, alpha=0.7, linestyle="dashed")
    sc_sim = ax.scatter(np.mean(data), i + 1, marker="o", s=10, color=color)
    sim_scatters.append(sc_sim)

ax.invert_yaxis()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

# Region color legend
from matplotlib.lines import Line2D
region_legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
    for region, color in REG_COR_Dic.items()
]
region_legend = ax.legend(region_legend_handles, REG_COR_Dic.keys(),
                          title="Region", loc="center left", bbox_to_anchor=(1.03, 0.5),
                          fontsize=10, title_fontsize=12, frameon=True)
ax.add_artist(region_legend)

# Proband/Sibling marker type legend (SWAPPED marker shapes for ASD/Siblings)
marker_handles = [
    Line2D([0], [0], marker='v', color='k', linestyle="None", markersize=10, label='ASD Probands'),
    Line2D([0], [0], marker='o', color='k', linestyle="None", markerfacecolor='grey', markersize=8, label='Siblings')
]
main_legend = ax.legend(marker_handles, ['ASD Probands', 'Siblings'],
            prop={'size': 11}, loc="lower right", bbox_to_anchor=(1.02, 0.01), frameon=True)
main_legend.get_frame().set_edgecolor('grey')
main_legend.get_frame().set_linewidth(0.5)

REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                'Hippocampus','Amygdala','Striatum', 
                "Thalamus", "Hypothalamus", "Midbrain", 
                "Medulla", "Pallidum", "Pons", 
                "Cerebellum"]
REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                    "#2c9d39", "#742eb5", "#ed8921", 
                                    "#e82315", "#E6B0AA", "#f6b26b",  
                                    "#20124d", "#2ECC71", "#D2B4DE", 
                                    "#ffd966", ]))

# Add a legend to highlight region color mapping
from matplotlib.lines import Line2D
region_legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
    for region, color in REG_COR_Dic.items()
]

region_legend = ax.legend(region_legend_handles, REG_COR_Dic.keys(),
                          title="Region", loc="center left", bbox_to_anchor=(1.03, 0.5),
                          fontsize=10, title_fontsize=12, frameon=True)
ax.add_artist(region_legend)

ax.set_ylim(121, 0)
ax.set_xlim(-0.5, 0.75)
ax.set_xlabel("Mutation Bias", fontsize=20, labelpad=10)
ax.set_ylabel("Structure Rank", fontsize=20)
ax.grid(True)

# %%
mpl.style.use('default')
fig, ax = plt.subplots(figsize=(6, 8), dpi=480)

# Define region color mapping
REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                'Hippocampus','Amygdala','Striatum', 
                "Thalamus", "Hypothalamus", "Midbrain", 
                "Medulla", "Pallidum", "Pons", 
                "Cerebellum"]
REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                    "#2c9d39", "#742eb5", "#ed8921", 
                                    "#e82315", "#E6B0AA", "#f6b26b",  
                                    "#20124d", "#2ECC71", "#D2B4DE", 
                                    "#ffd966", ]))

plt.style.use('seaborn-v0_8-whitegrid')

# Try to find region annotation from the dataframe
region_column_candidates = ['REGION', 'Region', 'region', 'Category', 'Structure']
bias_region_dic = None
if hasattr(ASD_Bias, 'columns'):
    available_cols = set(ASD_Bias.columns)
    for col in region_column_candidates:
        if col in available_cols:
            bias_region_dic = ASD_Bias[col].to_dict()
            break

# Plotting with color region annotation
proband_scatters = []
sim_scatters = []
regions_seen = {}  # To avoid redundant labels in legend

for i, (STR, STR_bias) in enumerate(sorted(ASD_Z2_Bias.items(), key=lambda x: x[1].Rank)):
    # Determine region/color for STR
    region = None
    if bias_region_dic and STR in bias_region_dic:
        region = bias_region_dic[STR]
        color = REG_COR_Dic.get(region, "#555555")
    else:
        color = "#555555"
    # Only assign label for legend to first seen region
    region_label = region if region and region not in regions_seen else None
    if region_label:
        regions_seen[region] = True

    # Proband bias (SWAPPED: now marker "v" for ASD)
    sc_prob = ax.scatter(STR_bias.Bias, i + 1, marker="v", s=15, color=color, label=region_label, edgecolor="k", alpha=0.85)
    proband_scatters.append(sc_prob)
    # Sibling simulation mean and CI (SWAPPED: now marker "o" for siblings)
    data = ASD_Z2_Sim_Bias_STR[STR]
    ci_low, ci_high = CI(data, 0.68)
    ax.hlines(i + 1, ci_low, ci_high, color=color, lw=1.3, alpha=0.7, linestyle="dashed")
    sc_sim = ax.scatter(np.mean(data), i + 1, marker="o", s=10, color=color)
    sim_scatters.append(sc_sim)

ax.invert_yaxis()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.78, box.height])

# Region color legend
from matplotlib.lines import Line2D
region_legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
    for region, color in REG_COR_Dic.items()
]
region_legend = ax.legend(region_legend_handles, REG_COR_Dic.keys(),
                          title="Region", loc="center left", bbox_to_anchor=(1.03, 0.5),
                          fontsize=10, title_fontsize=12, frameon=True)
ax.add_artist(region_legend)

# Proband/Sibling marker type legend (SWAPPED marker shapes for ASD/Siblings)
marker_handles = [
    Line2D([0], [0], marker='v', color='k', linestyle="None", markersize=10, label='ASD Probands'),
    Line2D([0], [0], marker='o', color='k', linestyle="None", markerfacecolor='grey', markersize=8, label='Siblings')
]
main_legend = ax.legend(marker_handles, ['ASD Probands', 'Siblings'],
            prop={'size': 11}, loc="lower right", bbox_to_anchor=(1.02, 0.01), frameon=True)
main_legend.get_frame().set_edgecolor('grey')
main_legend.get_frame().set_linewidth(0.5)

REGIONS_seq = ['Isocortex','Olfactory_areas', 'Cortical_subplate', 
                'Hippocampus','Amygdala','Striatum', 
                "Thalamus", "Hypothalamus", "Midbrain", 
                "Medulla", "Pallidum", "Pons", 
                "Cerebellum"]
REG_COR_Dic = dict(zip(REGIONS_seq, ["#268ad5", "#D5DBDB", "#7ac3fa", 
                                    "#2c9d39", "#742eb5", "#ed8921", 
                                    "#e82315", "#E6B0AA", "#f6b26b",  
                                    "#20124d", "#2ECC71", "#D2B4DE", 
                                    "#ffd966", ]))

# Add a legend to highlight region color mapping
from matplotlib.lines import Line2D
region_legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
    for region, color in REG_COR_Dic.items()
]

region_legend = ax.legend(region_legend_handles, REG_COR_Dic.keys(),
                          title="Region", loc="center left", bbox_to_anchor=(1.03, 0.5),
                          fontsize=10, title_fontsize=12, frameon=True)
ax.add_artist(region_legend)

ax.set_ylim(121, 0)
ax.set_xlim(-0.5, 0.75)
ax.set_xlabel("Mutation Bias", fontsize=20, labelpad=10)
ax.set_ylabel("Structure Rank", fontsize=20)
ax.grid(True)

# %%
ASD_Bias['REGION'] = ASD_Bias['REGION'].replace('Amygdalar', 'Amygdala')
ASD_Bias.head(50)

# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Reset style and create figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(6, 8), dpi=480)

# ----------------------------------------------------
# Define region color mapping
# ----------------------------------------------------
REGIONS_SEQ = [
    'Isocortex', 'Olfactory_areas', 'Cortical_subplate',
    'Hippocampus', 'Amygdala', 'Striatum',
    'Thalamus', 'Hypothalamus', 'Midbrain',
    'Medulla', 'Pallidum', 'Pons', 'Cerebellum'
]

REG_COLORS = [
    "#268ad5", "#D5DBDB", "#7ac3fa",
    "#2c9d39", "#742eb5", "#ed8921",
    "#e82315", "#E6B0AA", "#f6b26b",
    "#20124d", "#2ECC71", "#D2B4DE",
    "#ffd966"
]

REG_COLOR_DICT = dict(zip(REGIONS_SEQ, REG_COLORS))

# ----------------------------------------------------
# Determine region annotations from dataframe
# ----------------------------------------------------
region_column_candidates = ['REGION', 'Region', 'region', 'Category', 'Structure']
bias_region_dic = None

if hasattr(ASD_Bias, 'columns'):
    available_cols = set(ASD_Bias.columns)
    for col in region_column_candidates:
        if col in available_cols:
            bias_region_dic = ASD_Bias[col].to_dict()
            break

# ----------------------------------------------------
# Plot ASD vs. sibling mutation biases per structure
# ----------------------------------------------------
regions_seen = {}
for i, (STR, STR_bias) in enumerate(
    sorted(ASD_Z2_Bias.items(), key=lambda x: x[1].Rank)
):
    # Determine region color
    region = bias_region_dic.get(STR) if bias_region_dic else None
    color = REG_COLOR_DICT.get(region, "#555555")

    # Add region label to legend only once
    region_label = region if region and region not in regions_seen else None
    if region_label:
        regions_seen[region] = True

    # ASD probands (v markers)
    ax.scatter(
        STR_bias.Bias, i + 1, marker="v", s=18,
        color=color, label=region_label,
        edgecolor="k", alpha=0.85, linewidth=0.3
    )

    # Sibling simulations (mean ± CI)
    data = ASD_Z2_Sim_Bias_STR[STR]
    ci_low, ci_high = CI(data, 0.68)
    ax.hlines(i + 1, ci_low, ci_high, color=color, lw=1.3, alpha=0.7, linestyle="dashed")
    ax.scatter(np.mean(data), i + 1, marker="o", s=14, color=color, edgecolor="none", alpha=0.8)

# ----------------------------------------------------
# Axis settings
# ----------------------------------------------------
ax.invert_yaxis()
ax.set_ylim(121, 0)
ax.set_xlim(-0.5, 0.75)
ax.set_xlabel("Mutation Bias", fontsize=18, labelpad=8)
ax.set_ylabel("Structure Rank", fontsize=18)
ax.grid(True, linestyle="--", alpha=0.5)

# Adjust position to make space for legends
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.76, box.height])

# ----------------------------------------------------
# Legends
# ----------------------------------------------------

# (1) Region color legend -- FIX: Use a dedicated legend artist and set bbox_to_anchor
region_legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
           markersize=9, label=region)
    for region, color in REG_COLOR_DICT.items()
]
region_legend = fig.legend(
    handles=region_legend_handles,
    title="Region",
    loc="center left",
    bbox_to_anchor=(0.70, 0.40),
    fontsize=9,
    title_fontsize=11,
    frameon=True,
    borderaxespad=0.0
)
region_legend.get_frame().set_edgecolor('grey')
region_legend.get_frame().set_linewidth(0.5)
region_legend.get_frame().set_alpha(0.0)  # Make frame transparent

# (2) ASD vs. Sibling marker legend
marker_handles = [
    Line2D([0], [0], marker='v', color='k', linestyle="None", markersize=9,
           label='ASD Probands', markerfacecolor='k'),
    Line2D([0], [0], marker='o', color='k', linestyle="None", markersize=8,
           label='Siblings', markerfacecolor='grey')
]
main_legend = ax.legend(
    handles=marker_handles,
    loc="lower right",
    bbox_to_anchor=(0.95, 0.04),
    fontsize=9,
    frameon=True
)
main_legend.get_frame().set_edgecolor('grey')
main_legend.get_frame().set_linewidth(0.5)
main_legend.get_frame().set_alpha(0.0)  # Make frame transparent

# Make sure the region legend is on top
region_legend.set_zorder(10)

# ----------------------------------------------------
# Save and show
# ----------------------------------------------------
plt.tight_layout()
plt.show()
fig.savefig(
    "../results/ASD_Gene_Clustering_CircuitStructures_flipped.pdf",
    dpi=300, bbox_inches="tight"
)

# %% [markdown] hidden=true
# ###### Test Average abs Z

# %% hidden=true
#ASD
asd_mean_abs_bias = np.mean([abs(x) for x in ASD_Bias["EFFECT"].values])
#sib_mean_abs_bias = np.mean([abs(x) for x in SIB_Bias["EFFECT"].values])

subsib_mean_abs_biases = []
for file in os.listdir(ASD_Sim_dir):
    if file.startswith("cont.genes"):
        continue
    df = pd.read_csv(ASD_Sim_dir+file, index_col="STR")
    mean_abs_bias = np.mean([abs(x) for x in df["EFFECT"].values])
    subsib_mean_abs_biases.append(mean_abs_bias)

# %% hidden=true
print(GetPermutationP(subsib_mean_abs_biases, asd_mean_abs_bias))
#print(GetPermutationP(subsib_mean_abs_biases, sib_mean_abs_bias))

# %% hidden=true
#np.mean(subsib_mean_abs_biases)

# %% hidden=true
import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')
fig, ax = plt.subplots(dpi=120)
n, bins, patches = ax.hist(subsib_mean_abs_biases, bins=20, histtype = "barstacked", align="mid", 
                          facecolor="grey", alpha=0.8, label="subsampled sib", edgecolor="black", 
                          linewidth=0.5)
Z1, P1, _ = GetPermutationP(subsib_mean_abs_biases, asd_mean_abs_bias)
#Z2, P2, _ = GetPermutationP(subsib_mean_abs_biases, sib_mean_abs_bias)
ax.vlines(asd_mean_abs_bias, ymin=0, ymax = max(n), linewidth = 5, color="blue", label="ASD")
#ax.vlines(sib_mean_abs_bias, ymin=0, ymax = max(n), linewidth = 5, color="orange", label="Sib")
ax.text(x=asd_mean_abs_bias*1.01, y=max(n)*0.7, s = "p=%.2e"%P1)
#ax.text(x=sib_mean_abs_bias*1.01, y=max(n)*0.7, s = "p=%.2e"%P2)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
ax.set_xlabel("Mean Absolute Bias")
plt.show()


# %% hidden=true
def cut_posotive(BiasDF):
    last_eff = 1
    for str_, row in BiasDF.iterrows():
        eff = row["EFFECT"]
        if eff <=0 and last_eff > 0:
            return row["Rank"]
        last_eff = eff
def GetMeanLogP_Positive(BiasDF):
    cut_size = cut_posotive(BiasDF)
    return np.mean(BiasDF[BiasDF["Rank"]<=cut_size]["EFFECT"].values)


# %% hidden=true
asd_mean_abs_bias = GetMeanLogP_Positive(ASD_Bias)
#sib_mean_abs_bias = GetMeanLogP_Positive(SIB_Bias)

subsib_mean_abs_biases = []
for file in os.listdir(ASD_Sim_dir):
    if file.startswith("cont.genes"):
        continue
    df = pd.read_csv(ASD_Sim_dir+file, index_col="STR")
    mean_abs_bias = GetMeanLogP_Positive(df)
    subsib_mean_abs_biases.append(mean_abs_bias)

# %%
DIR = "../results/STR_ISH/"
control_files = {
    "T2D": "T2D_bias_addP_sibling.csv",
    #"Parkinson": "Parkinson_bias_addP_sibling.csv",
    #"HBALC": "hba1c_bias_addP_sibling.csv",
    "IBD": "IBD_bias_addP_sibling.csv",
    "HDL_C": "HDL_C_bias_addP_sibling.csv",
    #"Alzheimer": "Alzheimer_bias_addP_sibling.csv"
}

control_dfs = {}
control_mean_abs_bias = {}

for name, fname in control_files.items():
    df = pd.read_csv(DIR + fname, index_col=0)
    control_dfs[name] = df
    control_mean_abs_bias[name] = GetMeanLogP_Positive(df)

asd_mean_abs_bias = GetMeanLogP_Positive(ASD_Bias)

# %% hidden=true
# plot_avg_positive_bias_histogram: moved to src/plot.py
fig, ax = plot_avg_positive_bias_histogram(
    subsib_mean_abs_biases,
    asd_mean_abs_bias,
    control_biases=control_mean_abs_bias,
)
plt.show()

# %% [markdown] hidden=true
# #### Figure 3

# %% hidden=true
topNs = np.arange(200, 5, -1)
RANKSCORE_DIR = "../dat/allen-mouse-conn/RankScores"
ASD_Distance = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.ASD.npy")
Cont_Distance = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.Cont.npy")

ASD_DistanceShort = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.Short.3900.ASD.npy")
Cont_DistanceShort = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.Short.3900.Cont.npy")

ASD_DistanceLong = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.Long.3900.ASD.npy")
Cont_DistanceLong = np.load(f"{RANKSCORE_DIR}/RankScore.Ipsi.Long.3900.Cont.npy")

# %% hidden=true
import matplotlib.backends.backend_pdf

fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=480, figsize=(7,11))

BarLen = 34.1
#BarLen = 47.5

cont = np.median(Cont_Distance, axis=0)
ax1.plot(topNs, ASD_Distance, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax1.legend(fontsize=13)

cont = np.nanmean(Cont_DistanceLong, axis=0)
ax2.plot(topNs, ASD_DistanceLong, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.nanpercentile(Cont_DistanceLong, 50-BarLen, axis=0)
upper = np.nanpercentile(Cont_DistanceLong, 50+BarLen, axis=0)
ax2.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, abs(upper - cont) ), ls="dashed", label="Siblings")
ax2.set_xlabel("Structure Rank\n", fontsize=17)
ax2.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax2.legend(fontsize=13)

cont = np.median(Cont_DistanceShort, axis=0)
ax3.plot(topNs, ASD_DistanceShort, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_DistanceShort, 50-BarLen, axis=0)
upper = np.percentile(Cont_DistanceShort, 50+BarLen, axis=0)
ax3.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax3.set_xlabel("Structure Rank\n", fontsize=17)
ax3.set_ylabel("Circuit Connectivity Score", fontsize=15)
ax3.legend(fontsize=13)

#fig.text(0.5, -0.03, 'Top Number of Structuress', ha='center')
#fig.text(-0.03, 0.5, 'SI Score', va='center', rotation='vertical')
ax1.set_xlim(0, 121)
ax2.set_xlim(0, 121)
ax3.set_xlim(0, 121)
plt.tight_layout()
#plt.legend(fontsize=15)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
#plt.savefig("../figs/main/Fig2.BCD.pdf")
#plt.savefig("../figs/main/Fig2.BCD.png")

# %% hidden=true
import matplotlib.backends.backend_pdf

fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=480, figsize=(6,9))

BarLen = 34.1
#BarLen = 47.5

# Define region color mapping
REGIONS_SEQ = [
    'Isocortex', 'Olfactory_areas', 'Cortical_subplate',
    'Hippocampus', 'Amygdala', 'Striatum', 
    'Thalamus', 'Hypothalamus', 'Midbrain',
    'Medulla', 'Pallidum', 'Pons', 'Cerebellum'
]

REG_COLORS = [
    "#268ad5", "#D5DBDB", "#7ac3fa",
    "#2c9d39", "#742eb5", "#ed8921",
    "#e82315", "#E6B0AA", "#f6b26b", 
    "#20124d", "#2ECC71", "#D2B4DE",
    "#ffd966"
]

REG_COLOR_DICT = dict(zip(REGIONS_SEQ, REG_COLORS))

cont = np.median(Cont_Distance, axis=0)
for i, (score, region) in enumerate(zip(ASD_Distance, ASD_Bias['REGION'])):
    color = REG_COLOR_DICT.get(region, "#555555")
    ax1.plot(topNs[i], score, color=color, marker="o", markersize=5)

ax1.plot([], [], color="blue", marker="o", markersize=5, lw=1,
         ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax1.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax1.set_xticklabels(labels=[])
ax1.legend(fontsize=13)

cont = np.nanmean(Cont_DistanceLong, axis=0)
for i, (score, region) in enumerate(zip(ASD_DistanceLong, ASD_Bias['REGION'])):
    color = REG_COLOR_DICT.get(region, "#555555")
    ax2.plot(topNs[i], score, color=color, marker="o", markersize=5)

ax2.plot([], [], color="blue", marker="o", markersize=5, lw=1,
         ls="dashed", label="ASD\nProbands")

lower = np.nanpercentile(Cont_DistanceLong, 50-BarLen, axis=0)
upper = np.nanpercentile(Cont_DistanceLong, 50+BarLen, axis=0)
ax2.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, abs(upper - cont) ), ls="dashed", label="Siblings")
ax2.set_xticklabels(labels=[])
ax2.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax2.legend(fontsize=13)

cont = np.median(Cont_DistanceShort, axis=0)
for i, (score, region) in enumerate(zip(ASD_DistanceShort, ASD_Bias['REGION'])):
    color = REG_COLOR_DICT.get(region, "#555555")
    ax3.plot(topNs[i], score, color=color, marker="o", markersize=5)

ax3.plot([], [], color="blue", marker="o", markersize=5, lw=1,
         ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_DistanceShort, 50-BarLen, axis=0)
upper = np.percentile(Cont_DistanceShort, 50+BarLen, axis=0)
ax3.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax3.set_xlabel("\nStructure Rank\n", fontsize=18)
ax3.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax3.tick_params(axis='both', which='minor', labelsize=12)
ax3.legend(fontsize=13)

# Add region color legend
region_legend_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
           markersize=9, label=region)
    for region, color in REG_COLOR_DICT.items()
]
region_legend = fig.legend(
    handles=region_legend_handles,
    title="Region",
    loc="center right",
    bbox_to_anchor=(1.15, 0.5),
    fontsize=9,
    title_fontsize=11,
    frameon=True
)

ax1.set_xlim(0, 121)
ax2.set_xlim(0, 121)
ax3.set_xlim(0, 121)
plt.tight_layout()
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

# %%
import matplotlib.backends.backend_pdf

fig, (ax1, ax2, ax3) = plt.subplots(3,1, dpi=480, figsize=(6,9))

BarLen = 34.1
#BarLen = 47.5

cont = np.median(Cont_Distance, axis=0)
ax1.plot(topNs, ASD_Distance, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_Distance, 50-BarLen, axis=0)
upper = np.percentile(Cont_Distance, 50+BarLen, axis=0)
ax1.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
#ax1.set_xlabel("Structure Rank\n", fontsize=17)
ax1.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax1.set_xticklabels(labels=[])
ax1.legend(fontsize=13)
#plt.yticks(fontsize=15, rotation=0)
cont = np.nanmean(Cont_DistanceLong, axis=0)
ax2.plot(topNs, ASD_DistanceLong, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.nanpercentile(Cont_DistanceLong, 50-BarLen, axis=0)
upper = np.nanpercentile(Cont_DistanceLong, 50+BarLen, axis=0)
ax2.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, abs(upper - cont) ), ls="dashed", label="Siblings")
#ax2.set_xlabel("Structure Rank\n", fontsize=17)
ax2.set_xticklabels(labels=[])
ax2.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax2.legend(fontsize=13)
#plt.yticks(fontsize=15, rotation=0)
cont = np.median(Cont_DistanceShort, axis=0)
ax3.plot(topNs, ASD_DistanceShort, color="blue", marker="o", markersize=5, lw=1,
                     ls="dashed", label="ASD\nProbands")

lower = np.percentile(Cont_DistanceShort, 50-BarLen, axis=0)
upper = np.percentile(Cont_DistanceShort, 50+BarLen, axis=0)
ax3.errorbar(topNs, cont, color="grey", marker="o", markersize=1.5, lw=1,
            yerr=(cont - lower, upper - cont ), ls="dashed", label="Siblings")
ax3.set_xlabel("\nStructure Rank\n", fontsize=18)
ax3.set_ylabel("Circuit Connectivity Score", fontsize=12)
ax3.tick_params(axis='both', which='minor', labelsize=12)
ax3.legend(fontsize=13)

#fig.text(0.5, -0.03, 'Top Number of Structuress', ha='center')
#fig.text(-0.03, 0.5, 'SI Score', va='center', rotation='vertical')
ax1.set_xlim(0, 121)
ax2.set_xlim(0, 121)
ax3.set_xlim(0, 121)
plt.tight_layout()
#plt.legend(fontsize=15)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
#plt.xticks(fontsize=15, rotation=0)
#plt.yticks(fontsize=15, rotation=0)
#plt.savefig("../figs/main/Fig2.BCD.pdf")
#plt.savefig("../figs/main/Fig2.BCD.png")

# %% [markdown]
# # Supplementary Figures

# %% [markdown] heading_collapsed=true
# #### Gene top5 expression bias versus expression level. 

# %% code_folding=[0, 16] hidden=true
def topNSpect(Mat, GeneSet, topN=5):
    topSpecs = []
    topSelectSpec = []
    idx_all = []
    idx_select = []
    for i, g in enumerate(Mat.index.values):
        exps = [x for x in Mat.loc[g, :].values if x == x]
        exps.sort()
        top5_spec = np.mean(exps[-topN:])
        topSpecs.append(top5_spec)
        idx_all.append(i)
        if g in GeneSet:
            topSelectSpec.append(top5_spec)
            idx_select.append(i)
    return idx_all, topSpecs, idx_select, topSelectSpec

def topNSpecVsExpLevel(MatSpec, MatLevel, GeneSet, topN=5):
    expLevel = []
    topSpecs = []
    topSelectSpec = []
    expLevelSelect = []
    idx_all = []
    idx_select = []
    common_genes = MatSpec.index.intersection(MatLevel.index)
    for i, g in enumerate(common_genes):
        spec = [x for x in MatSpec.loc[g, :].values if x == x]
        spec.sort()
        top5_spec = np.mean(spec[-topN:])
        topSpecs.append(top5_spec)
        idx_all.append(i)
        exp = MatLevel.loc[g, :].mean()
        expLevel.append(exp)
        if g in GeneSet:
            topSelectSpec.append(top5_spec)
            idx_select.append(i)
            expLevelSelect.append(exp)
    return idx_all, topSpecs, expLevel, idx_select, topSelectSpec, expLevelSelect


# %% hidden=true
Spark_Meta_2stage = pd.read_excel("../dat/Genetics/TabS_DenovoWEST_Stage1+2.xlsx",
                           skiprows=2, sheet_name="TopDnEnrich")
Spark_Meta_ExomeWide = Spark_Meta_2stage[Spark_Meta_2stage["pDenovoWEST_Meta"]<=1.3e-6]
GeneSet = set(Spark_Meta_ExomeWide["EntrezID"].values)
idx_all, topSpecs, expLevel, idx_select, topSelectSpec, expLevelSelect = topNSpecVsExpLevel(
    ExpZ2, ExpL, GeneSet, topN=5)
sibling_genes = pd.read_csv("../dat/Genetics/GeneWeights/sibling_weights_LGD_Dmis.csv", index_col=0).index.values
idx_all, topSpecs, expLevel, idx_select2, topSelectSpec2, expLevelSelect2 = topNSpecVsExpLevel(
    ExpZ2, ExpL, sibling_genes, topN=5)


# %% code_folding=[] hidden=true
def eeee(expLevelSelect, topSelectSpec, expLevelSelect2, topSelectSpec2, cutoff=1):
    xx1, xx2 = [], []
    yy1, yy2 = [], []
    for level, bias in zip(expLevelSelect, topSelectSpec):
        if level > cutoff:
            xx2.append(bias)
        else:
            xx1.append(bias)
    for level, bias in zip(expLevelSelect2, topSelectSpec2):
        if level > cutoff:
            yy2.append(bias)
        else:
            yy1.append(bias)
    print(np.mean(topSelectSpec), np.mean(xx1), np.mean(xx2))
    print(np.mean(topSelectSpec2), np.mean(yy1), np.mean(yy2))
eeee(expLevelSelect, topSelectSpec, expLevelSelect2, topSelectSpec2, cutoff=1)

# %% hidden=true
plt.style.use('classic')
ax.set_facecolor('white')
fig, ax = plt.subplots(dpi=120)
fig.patch.set_facecolor('white')
ax.scatter(expLevel, topSpecs, s=10, color="grey", marker="o", facecolors='none', alpha=0.3,
          label="Other genes")
ax.scatter(expLevelSelect2, topSelectSpec2, s=15, color="darkorange", marker="o", alpha=1, 
            facecolors='none', label="Sibling Genes")
ax.scatter(expLevelSelect, topSelectSpec, s=20, color="blue", marker="o", 
            facecolors='none', label="Proband Genes")
ax.annotate(xy=(0.05,7), text="ASD mean=2.80\nSibling mean=3.48")
ax.annotate(xy=(1.5,7), text="ASD mean=2.25\nSibling mean=2.91")
ax.vlines(ymin=0, ymax=10, x=1, color="grey", ls="dashed")
print(spearmanr(expLevelSelect, topSelectSpec))
ax.set_xlabel("Expression Level")
ax.set_ylim(0, 9)
ax.set_xscale('log')
plt.legend(loc="lower left")
plt.grid(False)
ax.set_ylabel("Mean Top 5 Expression Bias")

# %% [markdown] heading_collapsed=true
# #### Structure overlap among different ASD genesets. 

# %% [markdown] heading_collapsed=true
# #### Normalization for neuronal density and neuron-to-glia ratio.

# %% hidden=true
ASD_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")
ASD_Neuron_den_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuron.density.norm.bias.csv", 
                                      index_col="STR")
ASD_Glia_norm_bias = pd.read_csv("../dat/Unionize_bias/ASD.neuro2glia.norm.bias.csv", 
                                      index_col="STR")

# %% hidden=true
dat1, dat2 = [], []
for STR in ASD_Bias.index.values:
    bias1 = ASD_Bias.loc[STR, "EFFECT"]
    bias2 = ASD_Neuron_den_norm_bias.loc[STR, "EFFECT"]
    dat1.append(bias1)
    dat2.append(bias2)
dat11, dat22 = [], []
for STR in ASD_Bias.index.values:
    bias1 = ASD_Bias.loc[STR, "EFFECT"]
    bias2 = ASD_Glia_norm_bias.loc[STR, "EFFECT"]
    dat11.append(bias1)
    dat22.append(bias2)

# %% hidden=true
#print(pearsonr(dat1,dat2))
#print()
r1, p1 = pearsonr(dat1,dat2)
r2, p2 = pearsonr(dat11,dat22)


# %% hidden=true
def addxeqy(ax):
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)


# %% hidden=true
plt.style.use('classic')
fig, (ax1, ax2) = plt.subplots(1,2,dpi=480, figsize=(8,4))
fig.set_facecolor('white')
ax1.scatter(dat2, dat1, s=5, color="black")
ax2.scatter(dat22, dat11, s=5, color="black")
addxeqy(ax1)
addxeqy(ax2)
ax1.set_xlabel("Neuronal Density Normalized Bias\nZhou et al. 61 ASD genes")
ax2.set_xlabel("Neuro-to-Glia Ratio Normalized Bias\nZhou et al. 61 ASD genes")
ax1.set_ylabel("Mutation Bias \nZhou et al. 61 ASD genes")
#ax1.text(x=-0.6, y=0.5, s="r=%.2f, p=%.1e"%(r1, p1))
#ax2.text(x=-0.6, y=0.5, s="r=%.2f, p=%.1e"%(r2, p2))
ax1.text(x=-0.6, y=0.5, s="r=%.2f, p<%.0e"%(r1, 1e-10))
ax2.text(x=-0.6, y=0.5, s="r=%.2f, p<%.0e"%(r2, 1e-10))
import string
ax1.text(-0.1, 1.1, string.ascii_lowercase[2], transform=ax1.transAxes, 
            size=15, weight='bold')
ax2.text(-0.1, 1.1, string.ascii_lowercase[3], transform=ax2.transAxes, 
            size=15, weight='bold')
plt.tight_layout()
ax1.locator_params(nbins=6)
ax2.locator_params(nbins=6)
ax1.set_ylim(-0.7, 0.7)
ax1.set_xlim(-0.7, 0.7)
ax2.set_ylim(-0.7, 0.7)
ax2.set_xlim(-0.7, 0.7)
plt.show()

# %% [markdown] heading_collapsed=true
# #### ASC 102, Spark 159, Spark 61

# %% hidden=true
ASD_Bias = pd.read_csv("../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv", index_col="STR")
ASD_ASC = pd.read_csv("../dat/Unionize_bias/ASD.ASC102.Z2.bias.csv", 
                                      index_col="STR")
ASD_159 = pd.read_csv("../dat/Unionize_bias/ASD.159.pLI.z2.csv", 
                                      index_col="STR")

# %% hidden=true
dat1, dat2 = [], []
for STR in ASD_Bias.index.values:
    bias1 = ASD_Bias.loc[STR, "EFFECT"]
    bias2 = ASD_ASC.loc[STR, "EFFECT"]
    dat1.append(bias1)
    dat2.append(bias2)
dat11, dat22 = [], []
for STR in ASD_Bias.index.values:
    bias1 = ASD_Bias.loc[STR, "EFFECT"]
    bias2 = ASD_159.loc[STR, "EFFECT"]
    dat11.append(bias1)
    dat22.append(bias2)

# %% hidden=true
r1, p1 = pearsonr(dat1,dat2)
r2, p2 = pearsonr(dat11,dat22)

plt.style.use('classic')
fig, (ax1, ax2) = plt.subplots(1,2,dpi=480, figsize=(8,4))
fig.set_facecolor('white')
ax1.scatter(dat2, dat1, s=5, color="black")
ax2.scatter(dat22, dat11, s=5, color="black")

ax1.set_xlabel("Mutation Bias \nSatterstorm et al. 102 ASD genes")
ax2.set_xlabel("Mutation Bias \nZhou et al. 159 ASD genes")
ax1.set_ylabel("Mutation Bias \nZhou et al. 61 ASD genes")
ax1.text(x=-0.6, y=0.5, s="r=%.2f, p<%.0e"%(r1, 1e-10))
ax2.text(x=-0.6, y=0.5, s="r=%.2f, p<%.0e"%(r2, 1e-10))
import string
ax1.text(-0.1, 1.1, string.ascii_lowercase[0], transform=ax1.transAxes, 
            size=15, weight='bold')
ax2.text(-0.1, 1.1, string.ascii_lowercase[1], transform=ax2.transAxes, 
            size=15, weight='bold')
plt.tight_layout()
ax1.locator_params(nbins=6)
ax2.locator_params(nbins=6)
ax1.set_ylim(-0.7, 0.7)
ax1.set_xlim(-0.7, 0.7)
ax2.set_ylim(-0.7, 0.7)
ax2.set_xlim(-0.7, 0.7)
addxeqy(ax1)
addxeqy(ax2)
plt.show()

# %% [markdown] heading_collapsed=true
# #### Male vs Female

# %% hidden=true
ASD_Male = pd.read_csv("../dat/Unionize_bias/ASD.Male.ALL.bias.csv",
                                      index_col="Structure")
ASD_Female = pd.read_csv("../dat/Unionize_bias/ASD.Female.ALL.bias.csv",
                                      index_col="Structure")

# %% hidden=true
dat1, dat2 = [], []
for STR in ASD_Bias.index.values:
    bias1 = ASD_Male.loc[STR, "EFFECT"]
    bias2 = ASD_Female.loc[STR, "EFFECT"]
    dat1.append(bias1)
    dat2.append(bias2)
r1, p1 = pearsonr(dat1,dat2)

plt.style.use('classic')
fig, ax1 = plt.subplots(dpi=480, figsize=(4,4))
fig.set_facecolor('white')
ax1.scatter(dat1, dat2, s=5, color="black")

ax1.set_xlabel("Male Mutation Bias\nZhou et al. 61 ASD genes")
ax1.set_ylabel("Female Mutation Bias\nZhou et al. 61 ASD genes")
ax1.text(x=-0.6, y=0.5, s="r=%.2f, p<%.0e"%(r1, 1e-10))
import string
ax1.text(-0.1, 1.1, string.ascii_lowercase[4], transform=ax1.transAxes, 
            size=15, weight='bold')

plt.tight_layout()
ax1.locator_params(nbins=6)

ax1.set_ylim(-0.7, 0.7)
ax1.set_xlim(-0.7, 0.7)

addxeqy(ax1)

plt.show()

# %% [markdown] heading_collapsed=true
# #### Disttance/Weights distribution between ASD and Sib

# %% hidden=true
W_Ipsi = pd.read_csv(
    "../dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv", index_col=0)
D_ipsi = pd.read_csv(
    "../dat/allen-mouse-conn/Dist_CartesianDistance.csv", index_col=0)
D_ipsi.columns = D_ipsi.index.values
ASD_CircuitsSet = pd.read_csv(
    "../dat/Circuits/ASD.SA.Circuits.Size46.csv",
    index_col="idx")
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")
SIB_Bias = pd.read_csv("../dat/Unionize_bias/sib.top61.Z2.csv", index_col="STR")
SIB_Circuits = SIB_Bias.index.values[:46]

# %% hidden=true
ASD_Ws = W_Ipsi.loc[ASD_Circuits, ASD_Circuits].values
SIB_Ws = W_Ipsi.loc[SIB_Circuits, SIB_Circuits].values
ASD_Dists = D_ipsi.loc[ASD_Circuits, ASD_Circuits].values
SIB_Dists = D_ipsi.loc[SIB_Circuits, SIB_Circuits].values
ASD_Dists = ASD_Dists[ASD_Ws>0].flatten()
SIB_Dists = SIB_Dists[SIB_Ws>0].flatten()

# %% hidden=true
ASD_Ws = np.array([x for x in ASD_Ws.flatten() if x>0])
SIB_Ws = np.array([x for x in SIB_Ws.flatten() if x>0])

# %% hidden=true
fig, (ax1, ax2) = plt.subplots(1,2, dpi=480, figsize=(8,3))
ax1.hist(ASD_Dists/1000, color="blue", alpha=0.5, density=1, bins=20, edgecolor="black", linewidth=0.5,
        label="ASD Circuits")
ax1.hist(SIB_Dists/1000, color="orange", alpha=0.5, density=1, bins=20, edgecolor="black", linewidth=0.5, 
        label="Sibling Circuits")
ax1.set_xlabel(r"Distance range ($\times 10^3$ $\mu$m)")
ax1.text(5, 0.3, s="p_ks = 9e-09", fontsize=10)
ax1.set_ylabel("Density")
ax1.legend()
data1 = np.log2(1+ASD_Ws)
data2 = np.log2(1+SIB_Ws)
ax2.hist(data1, color="blue", alpha=0.5, density=1, bins=20, edgecolor="black", linewidth=0.5,
        label="ASD Circuits")
ax2.hist(data2, color="orange", alpha=0.5, density=1, bins=20, edgecolor="black", linewidth=0.5, 
        label="Sibling Circuits")
ax2.set_xlabel("Log Connection Strength")
ax2.text(2, 1.5, s="p_ks = 0.19", fontsize=10)
ax2.legend()
plt.show()


# %% hidden=true
scipy.stats.ks_2samp(ASD_Ws, SIB_Ws)

# %% hidden=true
scipy.stats.ks_2samp(ASD_Dists, SIB_Dists)

# %% [markdown] heading_collapsed=true
# #### Phenotype

# %% [markdown]
# ## Figure S6: CCS Significance
#
# Panels a–d: Histograms of sibling null CCS vs ASD observed at circuit size 46.
# Panel e: -log10(P) profile across structure ranks for full/short/long connectome.

# %%
# Load connection count distributions (not loaded earlier)
ASD_Conn = np.load(f"{RANKSCORE_DIR}/RankConn.C.Ipsi.ASD.npy")
Cont_Conn = np.load(f"{RANKSCORE_DIR}/RankConn.C.Ipsi.Cont.npy")


def PlotPermutationP(Null, Obs, ax, alt="auto", title="", xlabel="", dist_label="", bar_label=""):
    """Plot histogram of null distribution with observed value line."""
    n, bins, patches = ax.hist(Null, bins=20, histtype="barstacked", align='mid',
                                facecolor='#a1d99b', alpha=0.8, label=dist_label,
                                edgecolor="black", linewidth=0.5)
    if alt == "auto":
        gt = Obs >= np.mean(Null)
    else:
        gt = (alt == "greater")
    Z, P, _ = GetPermutationP(Null, Obs, greater_than=gt)
    ax.vlines(x=Obs, ymin=0, ymax=max(n), label=bar_label, color="#8c510a", linewidth=3)
    ax.text(x=Obs, y=max(n) * 0.7, s=" p=%.0e" % P, fontsize=20)
    return ax


size = 46
idx = np.where(topNs == size)[0][0]

fig = plt.figure(figsize=(16, 18), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.3)

# Panel a: Full Connectome CCS
ax_a = fig.add_subplot(gs[0, 0])
PlotPermutationP(Cont_Distance[:, idx], ASD_Distance[idx], ax_a,
                 title="Full Connectome", xlabel="CCS",
                 dist_label="Siblings", bar_label="ASD")
ax_a.text(-0.12, 1.08, 'a', transform=ax_a.transAxes, fontsize=32, fontweight='bold')
ax_a.set_title("Full Connectome", fontsize=24)
ax_a.set_xlabel("CCS", fontsize=22)
ax_a.tick_params(labelsize=18)
ax_a.legend(fontsize=18)

# Panel b: Full Connectome Number of Connections
ax_b = fig.add_subplot(gs[0, 1])
PlotPermutationP(Cont_Conn[:, idx], ASD_Conn[idx], ax_b,
                 title="Full Connectome", xlabel="Number of Connections",
                 dist_label="Siblings", bar_label="ASD")
ax_b.text(-0.12, 1.08, 'b', transform=ax_b.transAxes, fontsize=32, fontweight='bold')
ax_b.set_title("Full Connectome", fontsize=24)
ax_b.set_xlabel("Number of Connections", fontsize=22)
ax_b.tick_params(labelsize=18)
ax_b.xaxis.set_major_locator(plt.MaxNLocator(5))
ax_b.legend(fontsize=18)

# Panel c: Long Distance Connectome CCS
ax_c = fig.add_subplot(gs[1, 0])
PlotPermutationP(Cont_DistanceLong[:, idx], ASD_DistanceLong[idx], ax_c,
                 title="Long Distance Connectome", xlabel="CCS",
                 dist_label="Siblings", bar_label="ASD")
ax_c.text(-0.12, 1.08, 'c', transform=ax_c.transAxes, fontsize=32, fontweight='bold')
ax_c.set_title("Long Distance Connectome", fontsize=24)
ax_c.set_xlabel("CCS", fontsize=22)
ax_c.tick_params(labelsize=18)
ax_c.legend(fontsize=18)

# Panel d: Short Distance Connectome CCS
ax_d = fig.add_subplot(gs[1, 1])
PlotPermutationP(Cont_DistanceShort[:, idx], ASD_DistanceShort[idx], ax_d,
                 title="Short Distance Connectome", xlabel="CCS",
                 dist_label="Siblings", bar_label="ASD")
ax_d.text(-0.12, 1.08, 'd', transform=ax_d.transAxes, fontsize=32, fontweight='bold')
ax_d.set_title("Short Distance Connectome", fontsize=24)
ax_d.set_xlabel("CCS", fontsize=22)
ax_d.tick_params(labelsize=18)
ax_d.legend(fontsize=18)

# Panel e: -log10(P) profile across structure ranks
ax_e = fig.add_subplot(gs[2, :])


def _compute_pvalues(asd_scores, cont_scores):
    """Compute permutation p-values across structure ranks."""
    pvals = []
    for case, cont in zip(asd_scores, cont_scores.T):
        _, P, _ = GetPermutationP(cont, case, greater_than=True)
        pvals.append(P)
    return np.array(pvals)


pvals_full = _compute_pvalues(ASD_Distance, Cont_Distance)
pvals_short = _compute_pvalues(ASD_DistanceShort, Cont_DistanceShort)
pvals_long = _compute_pvalues(ASD_DistanceLong, Cont_DistanceLong)

ax_e.plot(topNs, -np.log10(pvals_full), label="Full Connectome",
          marker="o", markersize=4, lw=1, ls="dashed", color='#1f77b4')
ax_e.plot(topNs, -np.log10(pvals_short), label="Short Distance Connectome",
          marker="o", markersize=4, lw=1, ls="dashed", color='#ff7f0e')
ax_e.plot(topNs, -np.log10(pvals_long), label="Long Distance Connectome",
          marker="o", markersize=4, lw=1, ls="dashed", color='#2ca02c')
ax_e.axhline(y=-np.log10(0.05), color='grey', linestyle='--', lw=1.5)
ax_e.set_xlabel("Structure Rank", fontsize=24)
ax_e.set_ylabel("-log10(P)", fontsize=24)
ax_e.set_xlim(5, 121)
ax_e.tick_params(labelsize=18)
ax_e.legend(loc="upper left", fontsize=20)
ax_e.text(-0.06, 1.08, 'e', transform=ax_e.transAxes, fontsize=32, fontweight='bold')

fig.patch.set_alpha(0)
for ax in [ax_a, ax_b, ax_c, ax_d, ax_e]:
    ax.patch.set_alpha(0)

fig.savefig("../results/figs/FigureS6_CCS_Significance.pdf",
            transparent=True, dpi=300, bbox_inches='tight')
plt.show()
print("Saved: results/figs/FigureS6_CCS_Significance.pdf")

