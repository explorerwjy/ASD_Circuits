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
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import modify_str, LoadList

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")
print(f"Working directory: {os.getcwd()}")

# %%
CONN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-conn")
OUT_DIR = os.path.join(CONN_DIR, "ConnectomeScoringMat")
ALLEN_DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Connectome dir:   {CONN_DIR}")
print(f"Output dir:       {OUT_DIR}")

# %% [markdown]
# # 03. Preprocessing Connectivity Data
#
# This notebook processes the Allen Mouse Brain Connectivity Atlas data
# (Oh et al., 2014, *Nature*) into scoring matrices for circuit analysis.
#
# **Source data** (supplementary files from the paper):
# - `nature13186-s4.xlsx` — Connection weights and p-values (ipsilateral & contralateral)
# - `41586_2014_BFnature13186_MOESM72_ESM.xlsx` — 3D Cartesian distances between structures
#
# **Outputs** (saved to `dat/allen-mouse-conn/ConnectomeScoringMat/`):
# 1. `WeightMat.Ipsi.csv` — Ipsilateral weight matrix (213×213, p<0.05 threshold)
# 2. `InfoMat.Ipsi.csv` — Shannon Information scoring matrix (distance-corrected)
# 3. `InfoMat.Ipsi.Short.3900.csv` / `InfoMat.Ipsi.Long.3900.csv` — Short/long range splits
# 4. `Dist_CartesianDistance.csv` — Pairwise distance matrix (213×213)
#
# **Methodology** (CCS — Circuit Connectivity Score):
# - Distance between all 213×213 structure pairs is binned into **10 deciles**
# - For each bin, connection probability P = (connected pairs) / (total pairs)
# - Shannon Information:
#   - Connected pair:   I = −log₂(P)    (rare long-range connections → high information)
#   - Unconnected pair: I' = −log₂(1−P) (rare absence at short range → high information)
# - Only **ipsilateral** connections at **p < 0.05** are used (3,063 connections per methods)

# %% [markdown]
# ## 1. Load Connectivity Data
#
# Load raw connectivity weights and p-values from Oh et al. (2014).
# The data is in acronym format — we convert to full structure names later.

# %%
# Load ipsilateral and contralateral connectivity from Excel supplementary
xlsx_path = os.path.join(CONN_DIR, "nature13186-s4.xlsx")
W_ipsi = pd.read_excel(xlsx_path, sheet_name="W_ipsi").set_index("Unnamed: 0")
PValue_ipsi = pd.read_excel(xlsx_path, sheet_name="PValue_ipsi").set_index("Unnamed: 0")
W_contra = pd.read_excel(xlsx_path, sheet_name="W_contra").set_index("Unnamed: 0")
PValue_contra = pd.read_excel(xlsx_path, sheet_name="PValue_contra").set_index("Unnamed: 0")

print(f"Raw connectivity matrix: {W_ipsi.shape}")
print(f"Nonzero ipsi connections (raw): {(W_ipsi > 0).sum().sum()}")
print(f"Nonzero contra connections (raw): {(W_contra > 0).sum().sum()}")

# %%
# Apply p-value threshold: zero out connections with p > 0.05
P_THRESHOLD = 0.05
W_ipsi[PValue_ipsi > P_THRESHOLD] = 0
W_contra[PValue_contra > P_THRESHOLD] = 0

n_ipsi = int((W_ipsi > 0).sum().sum())
n_contra = int((W_contra > 0).sum().sum())
print(f"Ipsi connections after p<{P_THRESHOLD}:   {n_ipsi}")
print(f"Contra connections after p<{P_THRESHOLD}: {n_contra}")

# %% [markdown]
# ## 2. Map Acronyms to Structure Names
#
# Convert Allen Atlas acronyms (e.g., "ACA") to full structure names
# (e.g., "Anterior_cingulate_area") using the ontology file.

# %%
# Load ontology for acronym → name mapping
ontology = pd.read_csv(os.path.join(ProjDIR, "dat/Other/ontology.csv"))


def clean_name(name):
    """Convert Allen structure name to clean underscore format."""
    name = re.sub(r'[()]', '', name)
    name = re.sub(r'[-]', ' ', name)
    return "_".join(name.split())


acronym2name = {}
for _, row in ontology.iterrows():
    acronym2name[row["acronym"]] = clean_name(row["safe_name"])

# Rename rows and columns from acronyms to full names
W_ipsi.columns = [acronym2name[x] for x in W_ipsi.columns]
W_ipsi.index = [acronym2name[x] for x in W_ipsi.index]
W_contra.columns = [acronym2name[x] for x in W_contra.columns]
W_contra.index = [acronym2name[x] for x in W_contra.index]

print(f"Weight matrix: {W_ipsi.shape}")
print(f"Index sample: {W_ipsi.index[:3].tolist()}")

# Verify structure names match the 213 selected structures from notebook 01
Selected_STRs = LoadList(os.path.join(ALLEN_DIR, "Structures.txt"))
missing = set(Selected_STRs) - set(W_ipsi.index)
extra = set(W_ipsi.index) - set(Selected_STRs)
print(f"Structures in connectivity: {len(W_ipsi.index)}")
print(f"Structures in selected 213: {len(Selected_STRs)}")
if missing:
    print(f"  Missing from connectivity: {missing}")
if extra:
    print(f"  Extra in connectivity: {extra}")

# %% [markdown]
# ## 3. Compute Distance Matrices
#
# Load 3D Cartesian distances between structures from the supplementary data.
# We need both ipsilateral and contralateral distances — ipsilateral for
# the main InfoMat, contralateral for the contra-with-mirror variant.

# %%
dist_xlsx = os.path.join(CONN_DIR, "41586_2014_BFnature13186_MOESM72_ESM.xlsx")
RawDistance = pd.read_excel(dist_xlsx, index_col=0)
print(f"Raw distance matrix: {RawDistance.shape}")

# Extract ipsilateral and contralateral distances
acronyms = PValue_ipsi.index.values  # original acronym order
DistanceMat_ipsi = pd.DataFrame(
    data=np.zeros((213, 213)), index=acronyms, columns=acronyms
)
DistanceMat_contra = pd.DataFrame(
    data=np.zeros((213, 213)), index=acronyms, columns=acronyms
)
for acr_i in acronyms:
    for acr_j in acronyms:
        DistanceMat_ipsi.loc[acr_i, acr_j] = RawDistance.loc[
            f"{acr_i}_ipsi", f"{acr_j}_ipsi"
        ]
        DistanceMat_contra.loc[acr_i, acr_j] = RawDistance.loc[
            f"{acr_i}_ipsi", f"{acr_j}_contra"
        ]

# Rename to full structure names
DistanceMat_ipsi.index = W_ipsi.index.values
DistanceMat_ipsi.columns = W_ipsi.columns.values
DistanceMat_contra.index = W_contra.index.values
DistanceMat_contra.columns = W_contra.columns.values

print(f"Ipsi distance range: [{DistanceMat_ipsi.min().min():.1f}, {DistanceMat_ipsi.max().max():.1f}] µm")
print(f"Contra distance range: [{DistanceMat_contra.min().min():.1f}, {DistanceMat_contra.max().max():.1f}] µm")

# %% [markdown]
# ## 4. Compute Shannon Information Scoring Matrix
#
# The scoring matrix converts connection status into information-theoretic
# scores that account for distance-dependent baseline connection probability.
#
# **Algorithm**:
# 1. Bin all pairwise ipsi distances into **10 deciles** (equal pair counts)
# 2. For each bin: P = (connected pairs in bin) / (total pairs in bin)
# 3. Shannon Information:
#    - Connected pair: I = −log₂(P) — connection is informative (especially long range)
#    - Unconnected pair: I' = −log₂(1−P) — absence is informative (especially short range)
# 4. For each structure pair, assign I or I' based on connection status and distance bin

# %%
# Compute distance decile bins from ALL ipsi pairwise distances
ALL_Distances = DistanceMat_ipsi.values.flatten()
DistanceDeciles = np.percentile(ALL_Distances, np.arange(0, 100, 10))
DistanceDeciles = np.append(DistanceDeciles, max(ALL_Distances))

# Count total pairs per distance bin
Pairs_per_bin, _ = np.histogram(ALL_Distances, bins=DistanceDeciles)

# Count connected pairs per distance bin (using ipsi connections, excluding diagonal/zero distances)
conn_distances = DistanceMat_ipsi[(W_ipsi > 0)]
conn_distances = conn_distances[~np.isnan(conn_distances)]
conn_distances = np.nan_to_num(conn_distances, nan=0).flatten()
conn_distances = np.array([x for x in conn_distances if x > 0])

Edges_per_bin, _ = np.histogram(conn_distances, bins=DistanceDeciles)

# Connection probability and Shannon information per bin
P_conn = Edges_per_bin / Pairs_per_bin
I_connected = -np.log2(P_conn)          # info for a connected pair
I_unconnected = -np.log2(1 - P_conn)    # info for an unconnected pair

print("Distance bin  | All pairs | Connected | P(conn)  | I_conn  | I_unconn")
print("-" * 75)
for i in range(len(DistanceDeciles) - 1):
    print(f"  {DistanceDeciles[i]:7.0f}-{DistanceDeciles[i+1]:7.0f} | {Pairs_per_bin[i]:9d} | "
          f"{Edges_per_bin[i]:9d} | {P_conn[i]:.5f}  | {I_connected[i]:.3f}   | {I_unconnected[i]:.3f}")

# %%
# Visualize distance-dependent connection probability and information
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_alpha(0)

# Connection probability
ax1.bar(range(len(Pairs_per_bin)), Pairs_per_bin, color='#b5ffb9', edgecolor='grey',
        label='All pairs', alpha=0.7)
ax1.bar(range(len(Edges_per_bin)), Edges_per_bin, color='#f9bc86', edgecolor='grey',
        label='Connected', alpha=0.8)
names = [f"{DistanceDeciles[i]/1000:.1f}-{DistanceDeciles[i+1]/1000:.1f}"
         for i in range(len(DistanceDeciles) - 1)]
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, rotation=45, fontsize=8)
ax1.set_xlabel("Distance (mm)")
ax1.set_ylabel("Count")
ax1.set_title("Distance Distribution")
ax1.legend()
ax1.patch.set_alpha(0)

# Probability and Information
x_vals = DistanceDeciles[:-1] / 1000
ax2.plot(x_vals, P_conn, 'k-o', label="Connection Probability", markersize=4)
ax2.set_xlabel("Distance (mm)")
ax2.set_ylabel("Connection Probability", color='black')
ax2_twin = ax2.twinx()
ax2_twin.plot(x_vals, I_connected, 'b-o', label="I (connected)", markersize=4)
ax2_twin.plot(x_vals, I_unconnected, 'r-o', label="I' (unconnected)", markersize=4)
ax2_twin.set_ylabel("Information (bits)", color='blue')
ax2.set_title("Distance → Probability → Information")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax2.patch.set_alpha(0)

plt.tight_layout()
plt.show()

# %%
# Build the Information scoring matrix
# Each pair gets I_connected or I_unconnected based on ipsi connection status
structures = W_ipsi.index.values


def assign_info(dist, bins, info_values):
    """Assign Shannon information value based on which distance bin the pair falls in."""
    for i in range(len(bins) - 1):
        if dist >= bins[i] and dist < bins[i + 1]:
            return info_values[i]
    return info_values[-1]


InfoMat = pd.DataFrame(
    data=np.zeros((213, 213)),
    index=structures,
    columns=structures
)

for node_i in structures:
    for node_j in structures:
        if node_i == node_j:
            continue
        d = DistanceMat_ipsi.loc[node_i, node_j]
        w = W_ipsi.loc[node_i, node_j]
        if w > 0:
            InfoMat.loc[node_i, node_j] = assign_info(d, DistanceDeciles, I_connected)
        else:
            InfoMat.loc[node_i, node_j] = assign_info(d, DistanceDeciles, I_unconnected)

n_nonzero = int((InfoMat > 0).sum().sum())
n_expected = 213 * 212  # all non-diagonal pairs
print(f"InfoMat nonzero entries: {n_nonzero} (expected {n_expected})")
print(f"Info range: [{InfoMat[InfoMat > 0].min().min():.4f}, {InfoMat.max().max():.4f}]")

# %% [markdown]
# ## 5. Split into Short-Range and Long-Range
#
# Separate pairs by distance using the median connected distance as cutoff.
# This produces separate scoring matrices for short-range and long-range
# circuit analysis.

# %%
# Compute distance cutoff from median of ALL pairwise distances (not just connected)
# This gives ~3988 µm, consistent with the v3 reference files named "3900"
all_nonzero_distances = ALL_Distances[ALL_Distances > 0]
DISTANCE_CUTOFF = np.median(all_nonzero_distances)
print(f"Median all-pair distance (cutoff): {DISTANCE_CUTOFF:.1f} µm")
print(f"Short-range connected: {(conn_distances < DISTANCE_CUTOFF).sum()}")
print(f"Long-range connected:  {(conn_distances >= DISTANCE_CUTOFF).sum()}")

# %%
# Split InfoMat and WeightMat by distance
InfoMat_short = pd.DataFrame(0.0, index=structures, columns=structures)
InfoMat_long = pd.DataFrame(0.0, index=structures, columns=structures)
WeightMat_short = pd.DataFrame(0.0, index=structures, columns=structures)
WeightMat_long = pd.DataFrame(0.0, index=structures, columns=structures)

for node_i in structures:
    for node_j in structures:
        if node_i == node_j:
            continue
        d = DistanceMat_ipsi.loc[node_i, node_j]
        info = InfoMat.loc[node_i, node_j]
        w = W_ipsi.loc[node_i, node_j]
        if d < DISTANCE_CUTOFF:
            InfoMat_short.loc[node_i, node_j] = info
            if w > 0:
                WeightMat_short.loc[node_i, node_j] = w
        else:
            InfoMat_long.loc[node_i, node_j] = info
            if w > 0:
                WeightMat_long.loc[node_i, node_j] = w

print(f"Short-range: {int((WeightMat_short > 0).sum().sum())} connections, "
      f"{int((InfoMat_short > 0).sum().sum())} info entries")
print(f"Long-range:  {int((WeightMat_long > 0).sum().sum())} connections, "
      f"{int((InfoMat_long > 0).sum().sum())} info entries")

# %% [markdown]
# ## 6. Save Outputs

# %%
# Save all matrices
W_ipsi.to_csv(os.path.join(OUT_DIR, "WeightMat.Ipsi.csv"))
InfoMat.to_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.csv"))
InfoMat_short.to_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.Short.3900.csv"))
InfoMat_long.to_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.Long.3900.csv"))
WeightMat_short.to_csv(os.path.join(OUT_DIR, "WeightMat.Ipsi.Short.3900.csv"))
WeightMat_long.to_csv(os.path.join(OUT_DIR, "WeightMat.Ipsi.Long.3900.csv"))
DistanceMat_ipsi.to_csv(os.path.join(CONN_DIR, "Dist_CartesianDistance.csv"))

# Also save lowercase aliases for backward compatibility
InfoMat_short.to_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.short.csv"))
InfoMat_long.to_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.long.csv"))

print("Saved:")
for f in ["WeightMat.Ipsi.csv", "InfoMat.Ipsi.csv",
          "InfoMat.Ipsi.Short.3900.csv", "InfoMat.Ipsi.Long.3900.csv",
          "WeightMat.Ipsi.Short.3900.csv", "WeightMat.Ipsi.Long.3900.csv"]:
    print(f"  {OUT_DIR}/{f}")
print(f"  {CONN_DIR}/Dist_CartesianDistance.csv")

# %% [markdown]
# ## 7. Validate Against Reference (v3)
#
# Compare newly generated matrices against the original v3 files
# (from `ScoreingMat_jw_v3/` in the old project).

# %%
# Load v3 reference files
ref_weight = pd.read_csv(os.path.join(OUT_DIR, "WeightMat.Ipsi.csv.v3ref"), index_col=0)
ref_info = pd.read_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.csv.v3ref"), index_col=0)

# Align indices (should be identical)
common_strs = ref_weight.index.intersection(W_ipsi.index)
print(f"Common structures: {len(common_strs)}")

# Compare WeightMat
w_new = W_ipsi.loc[common_strs, common_strs].values.flatten()
w_ref = ref_weight.loc[common_strs, common_strs].values.flatten()
from scipy.stats import pearsonr
r_w, _ = pearsonr(w_new, w_ref)
n_match_w = np.sum(w_new == w_ref)
print(f"\nWeightMat comparison:")
print(f"  Pearson r:   {r_w:.6f}")
print(f"  Exact match: {n_match_w}/{len(w_new)} ({100*n_match_w/len(w_new):.1f}%)")
print(f"  New nonzero: {int(np.sum(w_new > 0))}")
print(f"  Ref nonzero: {int(np.sum(w_ref > 0))}")

# Compare InfoMat
i_new = InfoMat.loc[common_strs, common_strs].values.flatten()
i_ref = ref_info.loc[common_strs, common_strs].values.flatten()
r_i, _ = pearsonr(i_new, i_ref)
mae_i = np.mean(np.abs(i_new - i_ref))
n_match_i = np.sum(np.isclose(i_new, i_ref, atol=1e-10))
print(f"\nInfoMat comparison:")
print(f"  Pearson r:   {r_i:.6f}")
print(f"  MAE:         {mae_i:.6f}")
print(f"  Close match: {n_match_i}/{len(i_new)} ({100*n_match_i/len(i_new):.1f}%)")

# Compare short/long InfoMat
ref_short = pd.read_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.Short.3900.csv.v3ref"), index_col=0)
ref_long = pd.read_csv(os.path.join(OUT_DIR, "InfoMat.Ipsi.Long.3900.csv.v3ref"), index_col=0)
s_new = InfoMat_short.loc[common_strs, common_strs].values.flatten()
s_ref = ref_short.loc[common_strs, common_strs].values.flatten()
l_new = InfoMat_long.loc[common_strs, common_strs].values.flatten()
l_ref = ref_long.loc[common_strs, common_strs].values.flatten()
r_s, _ = pearsonr(s_new[s_new + s_ref > 0], s_ref[s_new + s_ref > 0])
r_l, _ = pearsonr(l_new[l_new + l_ref > 0], l_ref[l_new + l_ref > 0])
print(f"\nInfoMat Short r: {r_s:.6f}")
print(f"InfoMat Long r:  {r_l:.6f}")

# %% [markdown]
# ## Summary
#
# | Output | Shape | Path |
# |--------|-------|------|
# | Weight Matrix (ipsi, p<0.05) | 213 × 213 | `ConnectomeScoringMat/WeightMat.Ipsi.csv` |
# | Info Matrix (ipsi) | 213 × 213 | `ConnectomeScoringMat/InfoMat.Ipsi.csv` |
# | Info Matrix (short, <3900µm) | 213 × 213 | `ConnectomeScoringMat/InfoMat.Ipsi.Short.3900.csv` |
# | Info Matrix (long, ≥3900µm) | 213 × 213 | `ConnectomeScoringMat/InfoMat.Ipsi.Long.3900.csv` |
# | Distance Matrix | 213 × 213 | `dat/allen-mouse-conn/Dist_CartesianDistance.csv` |
#
# **Source**: Oh et al., "A mesoscale connectome of the mouse brain", *Nature* 508, 207–214 (2014)
#
# **CCS Definition**: CCS(S) = Σ I(i,j) / n_pairs for structures i,j in circuit S,
# where I is Shannon information based on connection probability conditioned on distance.
