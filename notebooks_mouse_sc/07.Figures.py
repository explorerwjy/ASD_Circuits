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

# %% [markdown]
# # 07. Figures — Cell-Type Analysis (Figure 5)
#
# Generate Figure 5 panels for the manuscript:
# - **5A**: ISH vs MERFISH structure-level bias correlation (per-structure)
# - **5B**: QQ plot and boxplot of cell-type bias by class
# - **5C**: ASD circuit structure × cell-type composition & bias dotplot
# - **Subclass**: Subclass-level bias boxplots for selected cell classes
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
from matplotlib.patches import Patch

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

# MERFISH CM structure bias (pre-computed by 04.MERFISH_Structure_Bias)
MERFISH_CM_Bias = pd.read_csv("dat/Bias/STR/ASD.MERFISH_Allen.CM.ISHMatch.Z2.csv", index_col=0)
# Merge subiculum if split
if "Subiculum_dorsal_part" in MERFISH_CM_Bias.index:
    sub_d = MERFISH_CM_Bias.loc["Subiculum_dorsal_part", "EFFECT"]
    sub_v = MERFISH_CM_Bias.loc["Subiculum_ventral_part", "EFFECT"]
    reg = MERFISH_CM_Bias.loc["Subiculum_dorsal_part", "REGION"]
    MERFISH_CM_Bias.loc["Subiculum"] = [(sub_d + sub_v) / 2, reg, len(MERFISH_CM_Bias) + 1]
    MERFISH_CM_Bias = MERFISH_CM_Bias.drop(["Subiculum_dorsal_part", "Subiculum_ventral_part"])

# ASD circuit structures (Size46 Pareto front, index 3)
ASD_CircuitsSet = pd.read_csv(f"../{config['data_files']['asd_circuit_size46']}", index_col="idx")
ASD_Circuits = ASD_CircuitsSet.loc[3, "STRs"].split(";")
ASD_Circuits.append("Subiculum")

# Cell-type bias with p-values
CT_Bias = pd.read_csv(f"../{config['data_files']['ct_bias_addp']}", index_col=0)

# Cell type hierarchy
CellTypesDF = pd.read_csv(f"../{config['data_files']['cell_type_hierarchy']}")
Class2Cluster = {}
Subclass2Cluster = {}
for _, row in CellTypesDF.iterrows():
    cluster, cls, sub = row.iloc[0], row.iloc[1], row.iloc[2]
    Class2Cluster.setdefault(cls, []).append(cluster)
    Subclass2Cluster.setdefault(sub, []).append(cluster)

# Brain region definitions
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

print(f"ISH structures: {len(ASD_STR_Bias)}")
print(f"MERFISH CM structures: {len(MERFISH_CM_Bias)}")
print(f"ASD circuit: {len(ASD_Circuits)} structures")
print(f"Cell-type clusters: {len(CT_Bias)}")

# %% [markdown]
# ## 2. Figure 5A — ISH vs MERFISH Structure Bias Correlation

# %%
# Per-structure scatter plot colored by brain region
shared = ASD_STR_Bias.index.intersection(MERFISH_CM_Bias.index)
ish_eff = ASD_STR_Bias.loc[shared, "EFFECT"].values
mf_eff = MERFISH_CM_Bias.loc[shared, "EFFECT"].values
reg_arr = ASD_STR_Bias.loc[shared, "REGION"].values

r_val, p_val = pearsonr(ish_eff, mf_eff)

fig, ax = plt.subplots(dpi=300, figsize=(6, 4))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

for reg in REGIONS_seq:
    mask = reg_arr == reg
    if not mask.any():
        continue
    ax.scatter(ish_eff[mask], mf_eff[mask], color=REG_COLORS[reg],
               s=20, alpha=0.7, label=reg.replace("_", " "))

lims = [-1.1, 0.8]
ax.plot(lims, lims, color="grey", alpha=0.3)
ax.text(-0.95, 0.5, f"r={r_val:.2f}\nP<{max(p_val, 1e-10):.1e}",
        fontsize=12, weight="bold")
ax.set_xlabel("ISH Bias", fontsize=14, weight="bold")
ax.set_ylabel("MERFISH Bias", fontsize=14, weight="bold")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.grid(True, ls="--", alpha=0.5)
ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
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
# Draw FDR lines and label above each
if p_q005 > 0:
    y_005 = -np.log10(p_q005)
    ax.axhline(y_005, color="red", ls=":", lw=1)
    ax.text(
        max_val*0.82, y_005 + 0.06, 'FDR < 0.05', 
        color='red', fontsize=10, ha='left', va='bottom', weight='bold', 
        backgroundcolor='white'
    )
if p_q010 > 0:
    y_010 = -np.log10(p_q010)
    ax.axhline(y_010, color="orange", ls=":", lw=1)
    ax.text(
        max_val*0.82, y_010 + 0.06, 'FDR < 0.10', 
        color='orange', fontsize=10, ha='left', va='bottom', weight='bold', 
        backgroundcolor='white'
    )
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.set_xlabel("Expected -log10(p)", fontsize=14)
ax.set_ylabel("Observed -log10(p)", fontsize=14)
#ax.set_title("Fig 5B: Cell-Type Bias QQ Plot")
# FDR labels are above lines, remove from legend
handles, labels = ax.get_legend_handles_labels()
# Remove possible FDR legends if present
legend_labels = [l for l in labels if not l.startswith("FDR ")]
legend_handles = [h for (h, l) in zip(handles, labels) if not l.startswith("FDR ")]
ax.legend(legend_handles, legend_labels, fontsize=8, loc="lower right")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Figure 5B — Class-Level Bias Boxplot

# %%
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
ax.set_xlabel("ASD Mutation Bias", fontsize=12)
#ax.set_title("Cell-Type Bias by Class")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Subclass-Level Bias Boxplots
#
# ASD bias distributions at the subclass level within selected cell classes.

# %%
def subclass_bias_boxplot(target_classes, CT_Bias, CellTypesDF, figsize=(8, 8)):
    """Horizontal boxplot of subclass-level bias for given cell classes."""
    sub_df = CellTypesDF[CellTypesDF["class"].isin(target_classes)]
    subclasses = sorted(sub_df["subclass"].unique())

    dat, labels = [], []
    for sub in subclasses:
        clusters = sub_df[sub_df["subclass"] == sub]["cluster"].values
        valid = [c for c in clusters if c in CT_Bias.index]
        vals = CT_Bias.loc[valid, "EFFECT"].dropna().values
        if len(vals) > 0:
            dat.append(vals)
            labels.append(sub)

    medians = [np.median(d) for d in dat]
    idx = np.argsort(medians)[::-1]
    dat = [dat[i] for i in idx]
    labels = [labels[i] for i in idx]

    fig, ax = plt.subplots(dpi=300, figsize=figsize)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    sns.boxplot(data=dat, orient="h", ax=ax, palette="deep")
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("ASD Bias", fontsize=14)
    ax.axvline(0, color="grey", ls="--", lw=0.5)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%
# Cortical Glutamatergic (IT-ET + NP-CT-L6b)
subclass_bias_boxplot(["01 IT-ET Glut", "02 NP-CT-L6b Glut"],
                      CT_Bias, CellTypesDF, figsize=(8, 10))

# %%
# Forebrain GABAergic (CTX-CGE + CNU-LGE)
subclass_bias_boxplot(["06 CTX-CGE GABA", "09 CNU-LGE GABA"],
                      CT_Bias, CellTypesDF, figsize=(8, 6))

# %%
# Subcortical (CNU-HYa GABA + CNU-HYa Glut + TH Glut)
subclass_bias_boxplot(["11 CNU-HYa GABA", "13 CNU-HYa Glut", "18 TH Glut"],
                      CT_Bias, CellTypesDF, figsize=(8, 10))

# %% [markdown]
# ## 6. Figure 5C — Circuit Structure × Cell-Type Dotplot
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
region_colors_dotplot = {
    "Isocortex": "#268ad5", "Olfactory_areas": "#783F04",
    "Cortical_subplate": "#9FC5E8", "Hippocampus": "#6AA84F",
    "Striatum": "#E69138", "Amygdala": "#674EA7",
    "Pallidum": "#2ECC71", "Thalamus": "#F44336", "Midbrain": "#783F04",
}
region_display = {
    "Isocortex": "Isocortex", "Olfactory_areas": "Olfactory areas",
    "Cortical_subplate": "Cortical subplate", "Hippocampus": "Hippocampus",
    "Striatum": "Striatum", "Amygdala": "Amygdala",
    "Pallidum": "Pallidum", "Thalamus": "Thalamus", "Midbrain": "Midbrain",
}

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


def shorten_name(x):
    if "Lateral septal nucleus rostral" in x:
        return "Lateral septal nucleus"
    return x


# Build scatter data
x_labels = SubclassCellComp.index
y_labels = STR_Sort_Names[::-1]

x_pos, y_pos, sizes, colors_arr = [], [], [], []
norm = plt.Normalize(-0.6, 0.6)
cmap = plt.cm.coolwarm

for i, subclass in enumerate(x_labels):
    for j, structure in enumerate(y_labels):
        x_pos.append(i)
        y_pos.append(j)
        comp = SubclassCellComp.loc[subclass, structure] if structure in SubclassCellComp.columns else 0
        bias = Subclass_STR_Bias.loc[subclass, structure] if structure in Subclass_STR_Bias.columns else 0
        sizes.append(size_transform(comp))
        colors_arr.append(cmap(norm(bias)))

# Region sequence for y-axis color bar
y_region_seq = []
for name in y_labels:
    reg = STR_Reg.get(name, None)
    if reg is None:
        underscore_name = "_".join(name.split())
        reg = ASD_STR_Bias.loc[underscore_name, "REGION"] if underscore_name in ASD_STR_Bias.index else "Unknown"
    y_region_seq.append(reg)

# --- Create figure with region color bar ---
fig, ax = plt.subplots(dpi=480, figsize=(45, 25))
fig.patch.set_alpha(0)
ax.patch.set_alpha(0)
plt.subplots_adjust(left=0.15, right=0.88, top=0.95, bottom=0.15)

# Main scatter
ax.scatter(x_pos, y_pos, s=sizes, alpha=0.8, c=colors_arr)

# Axis labels
y_labels_display = [f"{shorten_name(x)}     " for x in y_labels]
ax.set_xticks(np.arange(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=90, fontsize=28)
ax.set_yticks(np.arange(len(y_labels)))
ax.set_yticklabels(y_labels_display, fontsize=30)

ax.grid(True, ls="--", alpha=0.1)
ax.set_xlim(-0.5, len(x_labels))
ax.set_ylim(-0.5, len(y_labels))
for spine in ax.spines.values():
    spine.set_visible(False)

# --- Region color bar (thin strip between y-labels and dots) ---
ax_region = fig.add_axes([0.135, 0.149, 0.008, 0.80])
for i, reg in enumerate(y_region_seq):
    color = region_colors_dotplot.get(reg, "grey")
    ax_region.add_patch(plt.Rectangle((0, i - 0.5), 1, 1, color=color, lw=0))
ax_region.set_xlim(0, 1)
ax_region.set_ylim(-0.5, len(y_labels) - 0.5)
ax_region.set_xticks([])
ax_region.set_yticks([])
for spine in ax_region.spines.values():
    spine.set_visible(False)

# --- Colorbar for Bias (right side, stretched) ---
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cax = fig.add_axes([0.90, 0.35, 0.012, 0.55])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label("Bias", fontsize=35, fontweight="bold")
cbar.ax.tick_params(labelsize=30)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

# --- Cell composition size legend (right side, lower) ---
comp_vals = [0.001, 0.01, 0.1, 0.5]
size_of_comps = [size_transform(c) for c in comp_vals]
size_handles = [plt.scatter([], [], s=sz, color="grey", edgecolors="none",
                            label=f" : {c}") for c, sz in zip(comp_vals, size_of_comps)]
size_leg = fig.legend(
    handles=size_handles,
    title="Cell composition",
    loc="center right",
    bbox_to_anchor=(1.01, 0.18),
    fontsize=30, 
    title_fontsize=35,
    frameon=False,
    labelspacing=1.8,
    handletextpad=0.5,
)
# Make legend font bold
for text in size_leg.get_texts():
    text.set_fontweight("bold")
if size_leg.get_title() is not None:
    size_leg.get_title().set_fontweight("bold")

# --- Region legend (bottom left, 2 columns) ---
region_patches = [Patch(facecolor=region_colors_dotplot[r], label=region_display[r])
                  for r in region_order if r in set(y_region_seq)]
region_leg = fig.legend(
    handles=region_patches, 
    loc="lower left", 
    bbox_to_anchor=(-0.08, -0.02),
    fontsize=32, ncol=2, frameon=False, 
    handlelength=1.5, handletextpad=0.5,
)
# Make region legend font bold
for text in region_leg.get_texts():
    text.set_fontweight("bold")
if region_leg.get_title() is not None:
    region_leg.get_title().set_fontweight("bold")

plt.savefig("../results/figs/ASD_top60_SubClass_CellCompBias_with_regions.pdf",
            bbox_inches="tight", transparent=True)
plt.show()

# %% [markdown]
# ## 7. MERFISH Structure Bias — Connectivity Scoring
#
# Score MERFISH-derived structure bias against ISH null connectivity
# distributions (RankScore permutations). Three distance variants:
# full, short (<3900 µm), long (>3900 µm).

# %%
# Load connectivity matrices
IpsiInfoMat = pd.read_csv(f"../{config['data_files']['infomat_ipsi']}", index_col=0)
IpsiInfoMatShort = pd.read_csv(f"../{config['data_files']['infomat_ipsi_short']}", index_col=0)
IpsiInfoMatLong = pd.read_csv(f"../{config['data_files']['infomat_ipsi_long']}", index_col=0)

# Load MERFISH CM bias for connectivity (needs Subiculum split)
def split_subiculum(df):
    """Split merged Subiculum back into dorsal/ventral for InfoMat compatibility."""
    if "Subiculum" in df.index:
        z = df.loc["Subiculum"]
        df.loc["Subiculum_dorsal_part"] = z
        df.loc["Subiculum_ventral_part"] = z
        df = df.drop("Subiculum")
        df = df.sort_values("EFFECT", ascending=False)
    return df


MERFISH_CM_conn = pd.read_csv("dat/Bias/STR/ASD.MERFISH_Allen.CM.ISHMatch.Z2.csv", index_col=0)
MERFISH_CM_conn = split_subiculum(MERFISH_CM_conn)

# Load null distributions
Cont_Full = np.load(f"../{config['data_files']['rankscore_ipsi']}")
Cont_Short = np.load(f"../{config['data_files']['rankscore_ipsi_short']}")
Cont_Long = np.load(f"../{config['data_files']['rankscore_ipsi_long']}")

topNs = np.arange(200, 5, -1)
print(f"Null shape: {Cont_Full.shape}, topN range: [{topNs[-1]}, {topNs[0]}]")

# %%
# Compute connectivity scores for MERFISH CM
STR_Ranks = MERFISH_CM_conn.index.values
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

labels_dist = ["Full Distance", "Short (<3900 \u00b5m)", "Long (>3900 \u00b5m)"]
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

plt.suptitle("MERFISH Structure Bias \u2014 Connectivity Scoring", y=1.01)
plt.tight_layout()
plt.show()
