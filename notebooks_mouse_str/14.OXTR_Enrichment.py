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

# %% [markdown]
# # OXTR Expression Enrichment in ASD Circuit Structures
#
# Tests whether the 46 ASD circuit structures have higher oxytocin receptor
# (OXTR) gene expression than expected by chance.
#
# **Method:** Compute mean OXTR Z2 expression across the 46 ASD circuit
# structures. Compare against a null distribution where, for each of 10,000
# sibling gene sets, the top 46 structures (by mutation bias) are selected
# and their mean OXTR Z2 expression is computed.
#
# **Manuscript claim:** P = 1.6×10⁻² (Supplementary Figure 18)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, "../src")
from ASD_Circuits import LoadGeneINFO

# %% [markdown]
# ## 1. Load Data

# %%
# Z2 expression matrix (genes × 213 structures)
Z2 = pd.read_parquet("../dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet")

# OXTR Entrez ID
_, _, GeneSymbol2Entrez, Entrez2Symbol = LoadGeneINFO()
OXTR_ENTREZ = GeneSymbol2Entrez["OXTR"]  # 5021
oxtr_z2 = Z2.loc[OXTR_ENTREZ]
print(f"OXTR (Entrez {OXTR_ENTREZ}): {len(oxtr_z2)} structures")
print(f"  mean={oxtr_z2.mean():.4f}, min={oxtr_z2.min():.4f}, max={oxtr_z2.max():.4f}")

# %%
# ASD bias — top 46 structures define the ASD circuit
asd_bias = pd.read_csv(
    "../dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.SubSampleSib.csv", index_col=0
)
asd_top46 = asd_bias.nlargest(46, "EFFECT").index.tolist()
print(f"ASD circuit: {len(asd_top46)} structures")

# Observed: mean OXTR Z2 across ASD circuit structures
asd_oxtr_mean = oxtr_z2[asd_top46].mean()
print(f"Observed mean OXTR Z2 in ASD circuit: {asd_oxtr_mean:.4f}")

# %% [markdown]
# ## 2. Null Distribution — Mutability Model
#
# For each of 10,000 sibling gene sets (mutability-matched), find the top 46
# structures by mutation bias and compute their mean OXTR Z2 expression.

# %%
sib_mut = pd.read_parquet(
    "../results/Sibling_bias/Mutability_61gene/sibling_mutability_bias.parquet"
)
print(f"Mutability null: {sib_mut.shape[1]} simulations × {sib_mut.shape[0]} structures")

n_sims = sib_mut.shape[1]
null_mut = np.zeros(n_sims)
for i in range(n_sims):
    top46 = sib_mut.iloc[:, i].nlargest(46).index.tolist()
    null_mut[i] = oxtr_z2[top46].mean()

p_mut = np.mean(null_mut >= asd_oxtr_mean)
print(f"Mutability null: mean={null_mut.mean():.4f}, std={null_mut.std():.4f}")
print(f"P-value (mutability): {p_mut:.4f}")

# %% [markdown]
# ## 3. Null Distribution — Random Model

# %%
sib_rand = pd.read_parquet(
    "../results/Sibling_bias/Random_61gene/sibling_random_bias.parquet"
)
print(f"Random null: {sib_rand.shape[1]} simulations × {sib_rand.shape[0]} structures")

null_rand = np.zeros(sib_rand.shape[1])
for i in range(sib_rand.shape[1]):
    top46 = sib_rand.iloc[:, i].nlargest(46).index.tolist()
    null_rand[i] = oxtr_z2[top46].mean()

p_rand = np.mean(null_rand >= asd_oxtr_mean)
print(f"Random null: mean={null_rand.mean():.4f}, std={null_rand.std():.4f}")
print(f"P-value (random): {p_rand:.4f}")

# %% [markdown]
# ## 4. Supplementary Figure 18 — OXTR Enrichment Plot

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=300)
fig.patch.set_alpha(0)

for ax, null_vals, p_val, label in zip(
    axes,
    [null_mut, null_rand],
    [p_mut, p_rand],
    ["Mutability null", "Random null"],
):
    ax.patch.set_alpha(0)
    ax.hist(null_vals, bins=50, color="cyan", alpha=0.7, edgecolor="white", label="Siblings")
    ax.axvline(asd_oxtr_mean, color="brown", lw=2, label="ASD circuit")
    ax.set_xlabel("Mean OXTR Z2 expression", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{label}\nP = {p_val:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("../results/figs/SuppFig18_OXTR_enrichment.png",
            transparent=True, dpi=300, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Summary
#
# | Null model | Null mean | Observed | P-value |
# |---|---|---|---|
# | Mutability | see above | 0.587 | ~0.005 |
# | Random | see above | 0.587 | ~0.016 |
#
# The manuscript reports P = 1.6×10⁻². This matches the **random** null model,
# which is consistent with the old codebase (`Manuscript Analysis.ipynb`) using
# `SubSampleSib` (random gene subsampling) as the null.
#
# **Recommendation:** Use the mutability null (P ≈ 0.005) for consistency with
# the rest of the paper, or keep the random null (P ≈ 0.016) and note which
# model is used.
