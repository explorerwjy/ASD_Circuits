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
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f'{ProjDIR}/src/')
from ASD_Circuits import *

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# # Circuit Size Robustness Analysis
#
# The CCS (Circuit Connectivity Score) peaks at size 46, but this peak is not
# perfectly sharp. To demonstrate that the identified circuit is not an artifact
# of the specific size choice, we run the full Pareto-front SA search at sizes
# 32, 40, 45, 46, and 50 and compare the resulting circuits.
#
# Size 32 corresponds to the number of structures with FDR < 0.10 under the
# SubSampleSib null model — a statistically motivated circuit size.

# %% [markdown]
# ## 1. Load Pareto Fronts

# %%
SIZES = [32, 40, 45, 46, 50]  # 32 = FDR<0.10 (SubSampleSib null)
RESULT_DIR = "../results/CircuitSearch/ASD_SPARK_61/pareto_fronts"

pareto = {}
for s in SIZES:
    path = os.path.join(RESULT_DIR, f"ASD_SPARK_61_size_{s}_pareto_front.csv")
    pareto[s] = pd.read_csv(path)
    print(f"Size {s}: {len(pareto[s])} Pareto points")

# %% [markdown]
# ## 2. Extract Selected Circuits
#
# The main analysis (notebook 05) uses Pareto index 3 (`bias_limit=0.37`)
# as the selected circuit. We compare this same index across sizes.

# %%
SELECTED_IDX = 3  # Same as notebook 05 (index 0=baseline, 3=3rd optimized point, CCS~0.7)

selected = {}
baselines = {}

for s in SIZES:
    df = pareto[s].sort_values("mean_bias", ascending=False).reset_index(drop=True)
    pareto[s] = df  # store sorted version
    bl = df[df["circuit_type"] == "baseline"]
    baselines[s] = set(bl["structures"].values[0].split(","))

    row = df.iloc[SELECTED_IDX]
    selected[s] = set(row["structures"].split(","))

    print(f"Size {s} (index {SELECTED_IDX}):")
    print(f"  bias_limit={row['bias_limit']}, mean_bias={row['mean_bias']:.4f}, SI={row['circuit_score']:.4f}")
    print(f"  {len(selected[s])} structures")

# %% [markdown]
# ## 3. Jaccard Similarity (Selected Circuits)

# %%
def jaccard_matrix(circuit_dict, sizes):
    """Compute pairwise Jaccard similarity matrix."""
    n = len(sizes)
    J = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(circuit_dict[sizes[i]] & circuit_dict[sizes[j]])
            union = len(circuit_dict[sizes[i]] | circuit_dict[sizes[j]])
            J[i, j] = J[j, i] = inter / union
    return pd.DataFrame(J, index=sizes, columns=sizes)

J_selected = jaccard_matrix(selected, SIZES)

print(f"Selected circuit (Pareto index {SELECTED_IDX}) Jaccard similarity:")
print(J_selected.to_string(float_format="{:.3f}".format))
print()

# Also print shared counts
for i, s1 in enumerate(SIZES):
    for s2 in SIZES[i + 1:]:
        inter = len(selected[s1] & selected[s2])
        union = len(selected[s1] | selected[s2])
        print(f"  {s1} vs {s2}: {inter} shared / {union} union = {inter/union:.3f}")

# %% [markdown]
# ## 4. Structure Membership Heatmap

# %%
all_sel_strs = sorted(set().union(*selected.values()))

def membership_matrix(circuit_dict, all_strs, sizes):
    """Binary matrix: rows=structures, cols=sizes."""
    mat = pd.DataFrame(0, index=all_strs, columns=sizes)
    for s in sizes:
        for st in circuit_dict[s]:
            mat.loc[st, s] = 1
    return mat

mem_sel = membership_matrix(selected, all_sel_strs, SIZES)
mem_sel["n_sizes"] = mem_sel.sum(axis=1)
mem_sel = mem_sel.sort_values("n_sizes", ascending=False)

# %%
Anno = STR2Region()

REGION_COLORS_MAP = {
    'Isocortex': '#268ad5', 'Olfactory_areas': '#5ab4ac',
    'Cortical_subplate': '#7ac3fa', 'Hippocampus': '#2c9d39',
    'Amygdala': '#742eb5', 'Striatum': '#ed8921',
    'Thalamus': '#e82315', 'Hypothalamus': '#c27ba0',
    'Midbrain': '#f6b26b', 'Pallidum': '#2ECC71',
    'Cerebellum': '#8B4513', 'Medulla': '#708090',
    'Pons': '#A0522D',
}

def _prep_heatmap_data(mem_df, anno):
    """Shared data prep for both heatmap styles."""
    sizes = [c for c in mem_df.columns if c != "n_sizes"]
    mat = mem_df[sizes].values
    strs = mem_df.index.tolist()
    n_sizes_col = mem_df["n_sizes"].values
    n_str, n_sz = mat.shape
    region_for_str = [anno.get(s, "Other") for s in strs]
    row_colors = [REGION_COLORS_MAP.get(r, "#cccccc") for r in region_for_str]
    str_labels = [s.replace("_", " ") for s in strs]
    seen = {}
    for r, c in zip(region_for_str, row_colors):
        if r not in seen:
            seen[r] = c
    leg_handles = [Patch(facecolor=c, edgecolor="k", label=r.replace("_", " "),
                         linewidth=0.5) for r, c in seen.items()]
    return dict(sizes=sizes, mat=mat, strs=strs, n_sizes_col=n_sizes_col,
                n_str=n_str, n_sz=n_sz, region_for_str=region_for_str,
                row_colors=row_colors, str_labels=str_labels, leg_handles=leg_handles)

def _make_region_bar(ax, row_colors, str_labels, n_str):
    for i, c in enumerate(row_colors):
        ax.barh(i, 1, color=c, edgecolor="none")
    ax.set_ylim(-0.5, n_str - 0.5); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_yticks(range(n_str))
    ax.set_yticklabels(str_labels, fontsize=18)
    ax.set_xticks([])
    ax.set_xlabel("Region", fontsize=20, fontweight='bold')
    for sp in ['top', 'right', 'bottom']:
        ax.spines[sp].set_visible(False)

def _make_count_col(ax, n_sizes_col, n_str, n_sz, stripe_color='#f0f0f0'):
    for i in range(n_str):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color=stripe_color, zorder=0)
        ct = int(n_sizes_col[i])
        ax.text(0.5, i, f"{ct}/{n_sz}", ha='center', va='center', fontsize=18,
                fontweight='bold' if ct == n_sz else 'normal',
                color='#1a5276' if ct == n_sz else '#555555')
    ax.set_ylim(-0.5, n_str - 0.5); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Count", fontsize=20, fontweight='bold')
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_membership_heatmap(mem_df, title, anno):
    """Dot plot membership heatmap with count column."""
    from matplotlib.colors import Normalize
    d = _prep_heatmap_data(mem_df, anno)
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0, vmax=d['n_sz'])
    fig_h = max(16, d['n_str'] * 0.42)
    fig = plt.figure(figsize=(18, fig_h))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.3, 4, 0.5], wspace=0.02)

    _make_region_bar(fig.add_subplot(gs[0]), d['row_colors'], d['str_labels'], d['n_str'])

    ax = fig.add_subplot(gs[1])
    for i in range(d['n_str']):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)
    for i in range(d['n_str'] + 1):
        ax.axhline(i - 0.5, color='#e0e0e0', linewidth=0.5, zorder=1)
    for j in range(d['n_sz'] + 1):
        ax.axvline(j - 0.5, color='#e0e0e0', linewidth=0.5, zorder=1)

    xs_p, ys_p, cs_p = [], [], []
    xs_a, ys_a = [], []
    for i in range(d['n_str']):
        for j in range(d['n_sz']):
            if d['mat'][i, j] == 1:
                xs_p.append(j); ys_p.append(i)
                cs_p.append(cmap(norm(d['n_sizes_col'][i])))
            else:
                xs_a.append(j); ys_a.append(i)
    if xs_a:
        ax.scatter(xs_a, ys_a, s=100, facecolors='none', edgecolors='#d0d0d0',
                   linewidths=0.8, zorder=2)
    if xs_p:
        ax.scatter(xs_p, ys_p, s=350, c=cs_p, edgecolors='white',
                   linewidths=1.2, zorder=3)

    ax.set_xlim(-0.5, d['n_sz'] - 0.5)
    ax.set_ylim(-0.5, d['n_str'] - 0.5); ax.invert_yaxis()
    ax.set_xticks(range(d['n_sz']))
    ax.set_xticklabels([str(s) for s in d['sizes']], fontsize=28, fontweight='bold')
    ax.set_xlabel("Circuit Size", fontsize=24, fontweight='bold'); ax.set_yticks([])
    ax.set_title(title, fontsize=22, fontweight='bold', pad=12)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(handles=d['leg_handles'], loc="lower left", fontsize=14, ncol=2, framealpha=0.8)

    _make_count_col(fig.add_subplot(gs[2]), d['n_sizes_col'], d['n_str'], d['n_sz'], '#f5f5f5')
    fig.patch.set_alpha(0)
    for a in fig.axes:
        a.patch.set_alpha(0)
    return fig


# %%
fig_sel = plot_membership_heatmap(mem_sel, f"Selected Circuit Membership (Pareto index {SELECTED_IDX})", Anno)
plt.show()

# %% [markdown]
# ## 5. Core vs Size-Specific Structures

# %%
core = set.intersection(*selected.values())
print(f"Core structures (in all {len(SIZES)} sizes): {len(core)} / {len(all_sel_strs)} total")
print()

# Structures unique to each size
for s in SIZES:
    others = [selected[s2] for s2 in SIZES if s2 != s]
    unique = selected[s] - set.union(*others)
    if unique:
        print(f"Size {s} only ({len(unique)}): {sorted(unique)}")

# Structures in size 46 but not in other sizes
ref = selected[46]
for s in SIZES:
    if s == 46:
        continue
    in_46_not_s = ref - selected[s]
    in_s_not_46 = selected[s] - ref
    if in_46_not_s or in_s_not_46:
        print(f"\n46 vs {s}:")
        if in_46_not_s:
            print(f"  In 46 but not {s}: {sorted(in_46_not_s)}")
        if in_s_not_46:
            print(f"  In {s} but not 46: {sorted(in_s_not_46)}")

# %% [markdown]
# ## 6. Bias Profile of Selected Circuits

# %%
import yaml
with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)
STR_BiasMat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")

ASD_GW = Fil2Dict(ProjDIR + "dat/Genetics/GeneWeights/Spark_Meta_EWS.GeneWeight.csv")
ASD_bias = MouseSTR_AvgZ_Weighted(STR_BiasMat, ASD_GW)

print("Per-structure bias in selected circuits:")
for s in SIZES:
    strs = sorted(selected[s])
    biases = ASD_bias.loc[strs, "EFFECT"]
    print(f"  Size {s}: mean={biases.mean():.4f}, min={biases.min():.4f}, max={biases.max():.4f}")

# %% [markdown]
# ## 7. Pareto Front Comparison
#
# Overlay Pareto fronts from all sizes. Black X = selected circuit (index 3).

# %%
fig, ax = plt.subplots(dpi=120, figsize=(7, 5))

colors = {32: "#a65628", 40: "#e41a1c", 45: "#377eb8", 46: "#542788", 50: "#984ea3"}
size_labels = {
    32: "Size 32 (FDR<0.10)",
    40: "Size 40",
    45: "Size 45",
    46: "Size 46 (CCS peak)",
    50: "Size 50",
}

for s in SIZES:
    df = pareto[s]
    front = df.sort_values("circuit_score")
    sel_row = df.iloc[SELECTED_IDX]

    # Pareto front line + dots
    ax.plot(front["circuit_score"], front["mean_bias"],
            marker='.', color=colors[s], lw=2, markersize=8, ls='-',
            label=size_labels[s], alpha=0.85)
    # Selected circuit — black X
    ax.scatter(sel_row["circuit_score"], sel_row["mean_bias"],
               marker='x', s=80, color='k', lw=2, zorder=10)

# Add "Selected" annotation with arrow pointing to size 46 X marker
sel46 = pareto[46].iloc[SELECTED_IDX]
#ax.annotate("Selected\ncircuits", (sel46["circuit_score"], sel46["mean_bias"]),
#            textcoords="offset points", xytext=(40, 20), fontsize=13,
#            fontweight='bold', color='k',
#            arrowprops=dict(arrowstyle='->', color='k', lw=1.2))

ax.set_xlabel("Circuit Connectivity Score", fontsize=18)
ax.set_ylabel("Average Mutation Bias", fontsize=18)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14, frameon=False, loc='lower left')
ax.grid(True, alpha=0.2)
ax.patch.set_alpha(0)
ax.set_ylim(0.25, 0.42)
fig.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Summary
#
# Circuit robustness across sizes 32, 40, 45, 46, 50 using Pareto index 3
# (high-bias knee of each Pareto front), the same operating point as the
# main analysis.
#
# Size 32 = number of structures at FDR < 0.10 (SubSampleSib null).
# Core structures present at all sizes: see Section 5.

# %% [markdown]
# ## Supplementary Figure 10 Caption
#
# **Supplementary Figure 10: Robustness of ASD circuit identification across circuit sizes.**
# **a)** Structure membership heatmap showing overlap of selected circuits across five circuit
# sizes (32, 40, 45, 46, and 50 structures). Rows represent brain structures, colored by brain
# region; columns represent circuit sizes. Filled dots indicate membership, and the count column
# shows the number of sizes each structure appears in. Of 50 total structures across all sizes,
# 32 are shared by all five circuits, demonstrating a stable core.
# **b)** Pareto fronts from GENCIC searches across circuit sizes. The X-axis shows circuit
# connectivity scores (CCS) and the Y-axis shows average ASD mutation biases. Each colored line
# represents the Pareto front for a different circuit size: 32 (brown; corresponding to the
# number of structures with ASD mutation bias FDR < 0.1), 40 (red), 45 (blue), 46 (purple;
# CCS peak), and 50 (magenta). Black X markers indicate the selected operating point (Pareto
# index 3) for each size.
# **c)** The selected ASD circuit including 32 structures (nodes in the network) from the
# isocortex (dark blue), striatum (orange), thalamus (red), cortical subplate (light blue),
# hippocampus (green), amygdala (purple), and other brain regions (brown). Node sizes are
# proportional to ASD mutation biases of the corresponding brain structures and edges indicate
# the directions of anatomical connectome projections between the circuit structures.
