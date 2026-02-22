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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07. Figures — Cell-Type Analysis (Figure 5)
#
# Generate Figure 5 panels for the manuscript:
# - **5A**: ISH vs MERFISH structure-level bias correlation
# - **5B**: QQ plot and boxplot of cell-type bias by class
# - **5F**: ASD circuit structure × cell-type composition heatmap
# - **Connectivity**: MERFISH-derived structure bias connectivity scoring

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)

# %% [markdown]
# ## 1. Load Data

# %%
# ISH structure bias
ASD_STR_Bias = pd.read_csv(f"../{config['data_files']['str_bias_raw']}", index_col=0)

# Merge Subiculum dorsal/ventral
if "Subiculum_dorsal_part" in ASD_STR_Bias.index:
    sub_d = ASD_STR_Bias.loc["Subiculum_dorsal_part", "EFFECT"]
    sub_v = ASD_STR_Bias.loc["Subiculum_ventral_part", "EFFECT"]
    ASD_STR_Bias.loc["Subiculum"] = [(sub_d + sub_v) / 2, "Hippocampus", 214]
    ASD_STR_Bias = ASD_STR_Bias.drop(["Subiculum_dorsal_part", "Subiculum_ventral_part"])
ASD_STR_Bias["REGION"] = ASD_STR_Bias["REGION"].replace("Amygdalar", "Amygdala")

# MERFISH structure bias (Neuron Mean)
SC_Agg_Bias = pd.read_csv(f"../{config['data_files']['merfish_z2_neur_mean']}", index_col=0)
SC_Agg_Bias.columns = [c.replace(" ", "_") for c in SC_Agg_Bias.columns]
SC_Agg_Bias = MouseSTR_AvgZ_Weighted(
    SC_Agg_Bias,
    Fil2Dict(f"../{config['data_files']['asd_gene_weights_v2']}"),
)

# ASD circuit structures (Size46 Pareto front, index 3)
ASD_CircuitsSet = pd.read_csv(f"../{config['data_files']['asd_circuit_size46']}", index_col="idx")
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")
ASD_Circuits.append("Subiculum")

# Cell-type bias with p-values
CT_Bias = pd.read_csv(f"../{config['data_files']['ct_bias_addp']}", index_col=0)

print(f"ISH structures: {len(ASD_STR_Bias)}")
print(f"MERFISH structures: {len(SC_Agg_Bias)}")
print(f"ASD circuit: {len(ASD_Circuits)} structures")
print(f"Cell-type clusters: {len(CT_Bias)}")

# %% [markdown]
# ## 2. Figure 5A — ISH vs MERFISH Structure Bias Correlation

# %%
REGIONS_seq = [
    "Isocortex", "Olfactory_areas", "Cortical_subplate",
    "Hippocampus", "Amygdala", "Striatum",
    "Thalamus", "Hypothalamus", "Midbrain",
    "Medulla", "Pallidum", "Pons", "Cerebellum",
]
REG_COLORS = dict(zip(REGIONS_seq, [
    "#268ad5", "#D5DBDB", "#7ac3fa",
    "#2c9d39", "#742eb5", "#ed8921",
    "#e82315", "#E6B0AA", "#f6b26b",
    "#20124d", "#2ECC71", "#D2B4DE", "#ffd966",
]))

# %%
# Region-level correlation
ish_reg = []
mf_reg = []
for reg in REGIONS_seq:
    ish_vals = ASD_STR_Bias[ASD_STR_Bias["REGION"] == reg]["EFFECT"]
    mf_vals = SC_Agg_Bias[SC_Agg_Bias["REGION"] == reg]["EFFECT"]
    ish_reg.append(np.nanmean(ish_vals))
    mf_reg.append(np.nanmean(mf_vals))

r_p, _ = pearsonr(ish_reg, mf_reg)
r_s, _ = spearmanr(ish_reg, mf_reg)

fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

for i, reg in enumerate(REGIONS_seq):
    ax.scatter(ish_reg[i], mf_reg[i], color=REG_COLORS[reg], s=60, zorder=5)
    ax.annotate(reg.replace("_", " "), (ish_reg[i], mf_reg[i]),
                fontsize=7, ha="left", va="bottom")

lims = [-0.6, 0.5]
ax.plot(lims, lims, color="grey", alpha=0.3, ls="--")
ax.text(-0.4, 0.3, f"r = {r_p:.2f}\nrho = {r_s:.2f}", fontsize=12)
ax.set_xlabel("ISH Z2 Bias", fontsize=12)
ax.set_ylabel("MERFISH NM Z2 Bias", fontsize=12)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title("Fig 5A: ISH vs MERFISH Structure Bias")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Figure 5B — QQ Plot of Cell-Type Bias P-values

# %%
def qq_expected(pvalues):
    """Compute expected vs observed -log10 quantiles for QQ plot."""
    sorted_p = np.sort(pvalues)
    expected = np.linspace(0, 1, len(sorted_p), endpoint=False)[1:]
    return -np.log10(expected), -np.log10(sorted_p[1:])


DISPLAY_CLASSES = [
    "01 IT-ET Glut", "02 NP-CT-L6b Glut",
    "09 CNU-LGE GABA", "18 TH Glut",
    "11 CNU-HYa GABA", "13 CNU-HYa Glut",
]

# FDR thresholds
p_q005 = CT_Bias[CT_Bias["qvalues"] < 0.05]["Pvalue"].iloc[-1] if (CT_Bias["qvalues"] < 0.05).any() else -1
p_q010 = CT_Bias[CT_Bias["qvalues"] < 0.10]["Pvalue"].iloc[-1] if (CT_Bias["qvalues"] < 0.10).any() else -1

fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

for cls in DISPLAY_CLASSES:
    pvals = CT_Bias[CT_Bias["class_id_label"] == cls]["Pvalue"].values
    exp_q, obs_q = qq_expected(pvals)
    ax.scatter(exp_q, obs_q, alpha=0.7, s=50, label=cls)

# Other classes
other_pvals = CT_Bias[~CT_Bias["class_id_label"].isin(DISPLAY_CLASSES)]["Pvalue"].values
exp_q, obs_q = qq_expected(other_pvals)
ax.scatter(exp_q, obs_q, alpha=0.3, s=20, color="grey", label="Other")

max_val = 4.1
ax.plot([0, max_val], [0, max_val], "k--", lw=1, alpha=0.5)
if p_q005 > 0:
    ax.axhline(-np.log10(p_q005), color="red", ls=":", lw=1, label="FDR < 0.05")
if p_q010 > 0:
    ax.axhline(-np.log10(p_q010), color="orange", ls=":", lw=1, label="FDR < 0.10")
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel("Expected -log10(p)", fontsize=14)
ax.set_ylabel("Observed -log10(p)", fontsize=14)
ax.set_title("Fig 5B: Cell-Type Bias QQ Plot")
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Figure 5B — Class-Level Bias Boxplot

# %%
# Build class hierarchy
CellTypesDF = pd.read_csv(f"../{config['data_files']['cell_type_hierarchy']}")
Class2Cluster = {}
for _, row in CellTypesDF.iterrows():
    cls = row.iloc[1]  # class
    if cls not in Class2Cluster:
        Class2Cluster[cls] = []
    Class2Cluster[cls].append(row.iloc[0])  # cluster

# Boxplot sorted by median
classes_sorted = sorted(Class2Cluster.keys())
class_data = []
class_medians = []
for cls in classes_sorted:
    vals = CT_Bias[CT_Bias["class_id_label"] == cls]["EFFECT"].dropna().values
    class_data.append(vals)
    class_medians.append(np.median(vals) if len(vals) > 0 else 0)

sort_idx = np.argsort(class_medians)
class_data = [class_data[i] for i in sort_idx]
classes_sorted = [classes_sorted[i] for i in sort_idx]

fig, ax = plt.subplots(dpi=300, figsize=(8, 8))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

bp = ax.boxplot(class_data, labels=classes_sorted, vert=False, patch_artist=True,
                flierprops=dict(marker=".", markersize=2, alpha=0.3))
colors = sns.color_palette("muted", len(classes_sorted))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axvline(0, color="grey", ls="--", lw=0.5)
ax.set_xlabel("EFFECT (weighted avg Z2)", fontsize=12)
ax.set_title("Cell-Type Bias by Class")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Figure 5F — Circuit Structure × Cell-Type Heatmap
#
# For the 46 ASD circuit structures, compute per-structure cell-type composition
# and mean bias using MERFISH cell-level data.

# %%
# Load MERFISH cell data (with ASD bias annotation from 06)
MERFISH = pd.read_parquet(f"../{config['data_files']['merfish_cells_bias_annotated']}")

# Convert circuit names to space-separated (matching MERFISH ISH_STR column)
ASD_Circuits_clean = [" ".join(x.split("_")) for x in ASD_Circuits]
# Remove subiculum parts if present (merged to Subiculum)
for name in ["Subiculum ventral part", "Subiculum dorsal part"]:
    if name in ASD_Circuits_clean:
        ASD_Circuits_clean.remove(name)

# Filter MERFISH to circuit structures
MERFISH_Circuit = MERFISH[MERFISH["ISH_STR"].isin(ASD_Circuits_clean)]
print(f"MERFISH cells in circuit: {MERFISH_Circuit.shape[0]:,}")

# %%
# Subclass counts (filter to subclasses with >500 cells in circuit)
subclass_counts = MERFISH_Circuit["subclass"].value_counts()
subclass_counts = subclass_counts[subclass_counts > 500]
print(f"Subclasses with >500 cells: {len(subclass_counts)}")

# Compute: subclass bias per structure + cell composition
bias_data = []
comp_data = []
for str_name in ASD_Circuits_clean:
    str_cells = MERFISH_Circuit[MERFISH_Circuit["ISH_STR"] == str_name]
    str_bias = []
    str_comp = []
    for subclass in subclass_counts.index:
        sub_cells = str_cells[str_cells["subclass"] == subclass]
        str_bias.append(np.nanmean(sub_cells["ASD.Bias"]) if len(sub_cells) > 0 else 0)
        str_comp.append(len(sub_cells) / len(str_cells) if len(str_cells) > 0 else 0)
    bias_data.append(str_bias)
    comp_data.append(str_comp)

Subclass_STR_Bias = pd.DataFrame(bias_data, index=ASD_Circuits_clean,
                                  columns=subclass_counts.index).T
SubclassCellComp = pd.DataFrame(comp_data, index=ASD_Circuits_clean,
                                 columns=subclass_counts.index).T.sort_index()

# %%
# Sort structures by brain region
Cir_ISH_Bias = ASD_STR_Bias[ASD_STR_Bias.index.isin(ASD_Circuits)]
region_order = ["Isocortex", "Olfactory_areas", "Cortical_subplate", "Hippocampus",
                "Striatum", "Amygdala", "Pallidum", "Thalamus", "Midbrain"]
region_colors = dict(zip(region_order,
    ["#0098D4", "#783F04", "#9FC5E8", "#6AA84F", "#E69138",
     "#674EA7", "#674EA7", "#F44336", "#783F04"]))

STR_Sort_Names = []
STR_Reg = {}
for reg in region_order:
    tmp = Cir_ISH_Bias[Cir_ISH_Bias["REGION"] == reg]
    clean_names = [" ".join(x.split("_")) for x in tmp.index.values]
    for name in clean_names:
        STR_Reg[name] = reg
    STR_Sort_Names.extend(clean_names)

# %%
def size_transform(comp):
    return 0 if comp == 0 else 100 + comp * 2500

# Build scatter data
x_labels = SubclassCellComp.index
y_labels = STR_Sort_Names[::-1]

x_pos, y_pos, sizes, colors = [], [], [], []
norm = plt.Normalize(-0.6, 0.6)
cmap = plt.cm.coolwarm

for i, subclass in enumerate(x_labels):
    for j, structure in enumerate(y_labels):
        x_pos.append(i)
        y_pos.append(j)
        comp = SubclassCellComp.loc[subclass, structure] if structure in SubclassCellComp.columns else 0
        bias = Subclass_STR_Bias.loc[subclass, structure] if structure in Subclass_STR_Bias.columns else 0
        sizes.append(size_transform(comp))
        colors.append(cmap(norm(bias)))

fig, ax = plt.subplots(dpi=300, figsize=(35, 20))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

scatter = ax.scatter(x_pos, y_pos, s=sizes, alpha=0.8, c=colors)

ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=90, fontsize=10)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels, fontsize=12)

# Color y-labels by brain region
for tick_label in ax.get_yticklabels():
    name = tick_label.get_text()
    reg = STR_Reg.get(name, None)
    if reg:
        tick_label.set_color(region_colors.get(reg, "black"))

ax.grid(True, ls="--", alpha=0.1)
ax.set_xlim(-0.5, len(x_labels))
ax.set_ylim(-0.5, len(y_labels))

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.3, pad=0.02)
cbar.set_label("ASD Bias", fontsize=12)

ax.set_title("Fig 5F: Circuit Structure × Cell-Type Composition & Bias", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. MERFISH Structure Bias — Connectivity Scoring
#
# Score MERFISH-derived structure bias against ISH null connectivity
# distributions (RankScore permutations). Three distance variants:
# full, short (<3900 µm), long (>3900 µm).

# %%
# Load connectivity matrices
IpsiInfoMat = pd.read_csv(f"../{config['data_files']['infomat_ipsi']}", index_col=0)
IpsiInfoMatShort = pd.read_csv(f"../{config['data_files']['infomat_ipsi_short']}", index_col=0)
IpsiInfoMatLong = pd.read_csv(f"../{config['data_files']['infomat_ipsi_long']}", index_col=0)

# Load MERFISH bias variants (need Subiculum split for InfoMat compatibility)
def split_subiculum(df):
    """Split merged Subiculum back into dorsal/ventral for InfoMat compatibility."""
    if "Subiculum" in df.index:
        z = df.loc["Subiculum"]
        df.loc["Subiculum_dorsal_part"] = z
        df.loc["Subiculum_ventral_part"] = z
        df = df.drop("Subiculum")
        df = df.sort_values("EFFECT", ascending=False)
    return df

# Load pre-computed MERFISH CM structure bias (from 04.MERFISH_Structure_Bias)
MERFISH_CM = pd.read_csv("dat/Bias/STR/ASD.MERFISH_Allen.CM.ISHMatch.Z2.csv", index_col=0)
MERFISH_CM = split_subiculum(MERFISH_CM)

# Load null distributions
Cont_Full = np.load(f"../{config['data_files']['rankscore_ipsi']}")
Cont_Short = np.load(f"../{config['data_files']['rankscore_ipsi_short']}")
Cont_Long = np.load(f"../{config['data_files']['rankscore_ipsi_long']}")

topNs = np.arange(200, 5, -1)
print(f"Null shape: {Cont_Full.shape}, topN range: [{topNs[-1]}, {topNs[0]}]")

# %%
# Compute connectivity scores for MERFISH CM
STR_Ranks = MERFISH_CM.index.values
scores_full, scores_short, scores_long = [], [], []
for topN in topNs:
    top_strs = STR_Ranks[:topN]
    scores_full.append(ScoreCircuit_SI_Joint(top_strs, IpsiInfoMat))
    scores_short.append(ScoreCircuit_SI_Joint(top_strs, IpsiInfoMatShort))
    scores_long.append(ScoreCircuit_SI_Joint(top_strs, IpsiInfoMatLong))

# %%
# 3-panel connectivity plot
BarLen = 34.1  # percentile CI half-width

fig, axes = plt.subplots(3, 1, dpi=300, figsize=(7, 11))
fig.patch.set_alpha(0)

labels_dist = ["Full Distance", "Short (<3900 µm)", "Long (>3900 µm)"]
score_sets = [scores_full, scores_short, scores_long]
null_sets = [Cont_Full, Cont_Short, Cont_Long]

for ax, label, scores, null in zip(axes, labels_dist, score_sets, null_sets):
    ax.patch.set_alpha(0)
    median_null = np.median(null, axis=0)
    lower = np.percentile(null, 50 - BarLen, axis=0)
    upper = np.percentile(null, 50 + BarLen, axis=0)

    ax.plot(topNs, scores, color="blue", marker="o", markersize=3, lw=1,
            ls="dashed", label="MERFISH CM")
    ax.errorbar(topNs, median_null, color="grey", marker="o", markersize=1, lw=1,
                yerr=[median_null - lower, upper - median_null], alpha=0.5, label="ISH Null")

    ax.set_xlabel("Top-N Structures")
    ax.set_ylabel("Connectivity Score (SI)")
    ax.set_title(label)
    ax.legend(fontsize=8)

plt.suptitle("MERFISH Structure Bias — Connectivity Scoring", y=1.01)
plt.tight_layout()
plt.show()
