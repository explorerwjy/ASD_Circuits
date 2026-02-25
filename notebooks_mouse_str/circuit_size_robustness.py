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
# 40, 45, 46, and 50 and compare the resulting circuits.

# %% [markdown]
# ## 1. Load Pareto Fronts

# %%
SIZES = [40, 45, 46, 50]
RESULT_DIR = "../results/CircuitSearch/ASD_SPARK_61/pareto_fronts"

pareto = {}
for s in SIZES:
    path = os.path.join(RESULT_DIR, f"ASD_SPARK_61_size_{s}_pareto_front.csv")
    pareto[s] = pd.read_csv(path)
    print(f"Size {s}: {len(pareto[s])} Pareto points")

# %% [markdown]
# ## 2. Extract Selected Circuits
#
# The main analysis (notebook 05) uses Pareto index 2 (`bias_limit=0.31`)
# as the selected circuit. We compare this same index across sizes.

# %%
SELECTED_IDX = 2  # Same as notebook 05

selected = {}
baselines = {}

for s in SIZES:
    df = pareto[s]
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

def plot_membership_heatmap(mem_df, title, anno):
    sizes = [c for c in mem_df.columns if c != "n_sizes"]
    mat = mem_df[sizes].values
    strs = mem_df.index.tolist()
    n_sizes_col = mem_df["n_sizes"].values

    region_colors_map = {
        'Isocortex': '#268ad5', 'Olfactory_areas': '#5ab4ac',
        'Cortical_subplate': '#7ac3fa', 'Hippocampus': '#2c9d39',
        'Amygdala': '#742eb5', 'Striatum': '#ed8921',
        'Thalamus': '#e82315', 'Hypothalamus': '#c27ba0',
        'Midbrain': '#f6b26b', 'Pallidum': '#2ECC71',
        'Cerebellum': '#8B4513', 'Medulla': '#708090',
        'Pons': '#A0522D',
    }
    region_for_str = [anno.get(s, "Other") for s in strs]
    row_colors = [region_colors_map.get(r, "#cccccc") for r in region_for_str]

    fig, axes = plt.subplots(1, 2, figsize=(10, max(8, len(strs) * 0.22)),
                             gridspec_kw={"width_ratios": [0.3, 4]})

    # Region color bar
    ax_reg = axes[0]
    for i, c in enumerate(row_colors):
        ax_reg.barh(i, 1, color=c, edgecolor="none")
    ax_reg.set_ylim(-0.5, len(strs) - 0.5)
    ax_reg.invert_yaxis()
    ax_reg.set_xlim(0, 1)
    ax_reg.set_yticks(range(len(strs)))
    ax_reg.set_yticklabels(strs, fontsize=7)
    ax_reg.set_xticks([])
    ax_reg.set_xlabel("Region", fontsize=8)

    # Membership heatmap
    ax = axes[1]
    cmap_data = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == 1:
                cmap_data[i, j] = n_sizes_col[i] / len(sizes)

    im = ax.imshow(cmap_data, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                   interpolation="nearest")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] == 1:
                ax.plot(j, i, "s", color=row_colors[i], markersize=6,
                        markeredgecolor="k", markeredgewidth=0.3)

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(s) for s in sizes], fontsize=10)
    ax.set_xlabel("Circuit Size")
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)

    seen = {}
    for r, c in zip(region_for_str, row_colors):
        if r not in seen:
            seen[r] = c
    legend_handles = [Patch(facecolor=c, edgecolor="k", label=r, linewidth=0.5)
                      for r, c in seen.items()]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              ncol=2, framealpha=0.8)

    fig.patch.set_alpha(0)
    for a in axes:
        a.patch.set_alpha(0)
    plt.tight_layout()
    return fig


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
# Overlay Pareto fronts from all sizes. Stars = selected circuit (index 2).

# %%
fig, ax = plt.subplots(figsize=(8, 6))

colors = {40: "#e41a1c", 45: "#377eb8", 46: "#4daf4a", 50: "#984ea3"}
markers = {40: "o", 45: "s", 46: "D", 50: "^"}

for s in SIZES:
    df = pareto[s]
    optimized = df[df["circuit_type"] == "optimized"]
    bl = df[df["circuit_type"] == "baseline"]
    sel_row = df.iloc[SELECTED_IDX]

    ax.scatter(optimized["circuit_score"], optimized["mean_bias"],
               c=colors[s], marker=markers[s], s=50, alpha=0.8,
               label=f"Size {s}", edgecolors="k", linewidths=0.3)
    ax.scatter(bl["circuit_score"], bl["mean_bias"],
               c=colors[s], marker=markers[s], s=50, alpha=0.3,
               edgecolors="k", linewidths=0.3)
    # Selected circuit — large star
    ax.scatter(sel_row["circuit_score"], sel_row["mean_bias"],
               c=colors[s], marker="*", s=300, edgecolors="k", linewidths=0.8,
               zorder=10)
    # Connect Pareto front
    front = df.sort_values("circuit_score")
    ax.plot(front["circuit_score"], front["mean_bias"],
            color=colors[s], alpha=0.4, linewidth=1)

ax.set_xlabel("Circuit Connectivity Score (SI)", fontsize=12)
ax.set_ylabel("Average Mutation Bias", fontsize=12)
ax.set_title("Pareto Fronts Across Circuit Sizes")
ax.legend(fontsize=10)
ax.patch.set_alpha(0)
fig.patch.set_alpha(0)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Summary
#
# Circuit robustness across sizes 40, 45, 46, 50 using Pareto index 2
# (`bias_limit=0.31`), the same operating point as the main analysis.
#
# Core structures present at all sizes: see Section 5.
