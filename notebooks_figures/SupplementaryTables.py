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
import yaml
import numpy as np
import pandas as pd

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from ASD_Circuits import STR2Region

os.chdir(f"{ProjDIR}/notebooks_figures/")
print(f"Working directory: {os.getcwd()}")

# %%
# Load config
with open("../config/supplementary_tables.yaml", "r") as f:
    config = yaml.safe_load(f)

output_excel = os.path.join(ProjDIR, config["output_excel"])
tables_cfg = config["tables"]
print(f"Output: {output_excel}")
print(f"Tables to compile: {len(tables_cfg)}")

# %% [markdown]
# # Helper functions

# %%
def resolve_path(path):
    """Resolve a path relative to project root, or absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(ProjDIR, path)


def read_bias_csv(path, index_col="STR"):
    """Read a bias CSV and return DataFrame."""
    return pd.read_csv(resolve_path(path), index_col=index_col)


def read_bias_csv_no_index(path):
    """Read a bias CSV without setting index."""
    return pd.read_csv(resolve_path(path))

# %% [markdown]
# # Compile all tables

# %% [markdown]
# ## Table S1: ASD mutation biases across 213 mouse brain structures

# %%
cfg = tables_cfg["S1"]
src = cfg["sources"]

# Start from the FDR bias file (has EFFECT, Rank, REGION, Pvalue columns)
bias_fdr = read_bias_csv(src["bias_fdr"])

# Build Table S1
S1 = pd.DataFrame(index=bias_fdr.index)
S1.index.name = "STR"
S1["Bias"] = bias_fdr["EFFECT"]
S1["REGION"] = bias_fdr["REGION"] if "REGION" in bias_fdr.columns else STR2Region().reindex(bias_fdr.index)["Region"]
S1["Rank"] = bias_fdr["Rank"]

# P-values (sibling null)
if "Pvalue" in bias_fdr.columns:
    S1["Pvalue"] = bias_fdr["Pvalue"]
if "qvalues" in bias_fdr.columns:
    S1["Pvalue_adj"] = bias_fdr["qvalues"]
elif "Pvalue_adj" in bias_fdr.columns:
    S1["Pvalue_adj"] = bias_fdr["Pvalue_adj"]

# Additional bias columns
bias_male = read_bias_csv(src["bias_male"])
bias_female = read_bias_csv(src["bias_female"])
bias_fu72 = read_bias_csv_no_index(src["bias_fu72"])
bias_fu185 = read_bias_csv_no_index(src["bias_fu185"])
bias_spark159 = read_bias_csv(src["bias_spark159"])
bias_neuron_den = read_bias_csv(src["bias_neuron_density"])
bias_glia = read_bias_csv(src["bias_neuro_glia"])

S1["Bias.Male"] = bias_male.reindex(S1.index)["EFFECT"]
S1["Bias.Female"] = bias_female.reindex(S1.index)["EFFECT"]
S1["Bias.Fu_72"] = S1.index.map(dict(zip(bias_fu72["Structure"], bias_fu72["EFFECT"])))
S1["Bias.Fu_185"] = S1.index.map(dict(zip(bias_fu185["Structure"], bias_fu185["EFFECT"])))
S1["Bias.Spark159"] = bias_spark159.reindex(S1.index)["EFFECT"]
S1["Bias.Neuron_density_normalized"] = bias_neuron_den.reindex(S1.index)["EFFECT"]
S1["Bias.Neuron_glia_ratio_normalized"] = bias_glia.reindex(S1.index)["EFFECT"]

# Circuit membership
circuit_46_file = resolve_path(cfg["circuit_46_file"])
circuit_46_row = cfg["circuit_46_row"]
circuit_46_df = pd.read_csv(circuit_46_file, index_col="idx")
circuit_46 = circuit_46_df.loc[circuit_46_row, "STRs"].split(";")
circuit_32 = cfg["circuit_32"]

S1["Circuits.46"] = S1.index.isin(circuit_46).astype(int)
S1["Circuits.32"] = S1.index.isin(circuit_32).astype(int)

S1 = S1.reset_index()
print(f"Table S1: {S1.shape}")
S1.head(3)

# %% [markdown]
# ## Table S2: Connectome information score matrix

# %%
cfg = tables_cfg["S2"]
S2 = pd.read_csv(resolve_path(cfg["sources"]["info_mat"]), index_col=0)
S2 = S2.reset_index()
S2.rename(columns={"index": "Structure"}, inplace=True)
print(f"Table S2: {S2.shape}")
S2.iloc[:3, :5]

# %% [markdown]
# ## Table S3: ASD cell type bias (ABC atlas)

# %%
cfg = tables_cfg["S3"]
bias = read_bias_csv_no_index(cfg["sources"]["bias"])
anno = pd.read_csv(resolve_path(cfg["sources"]["annotation"]), index_col="cluster_id_label")

# Merge annotation columns
S3 = bias.copy()
if "Structure" in S3.columns:
    S3.rename(columns={"Structure": "Cell Type (Cluster)"}, inplace=True)

# Add annotation columns if not already present
merge_col = "Cell Type (Cluster)" if "Cell Type (Cluster)" in S3.columns else "Structure"
for col in ["class_id_label", "subclass_id_label", "CCF_broad.freq", "CCF_acronym.freq"]:
    if col not in S3.columns and col in anno.columns:
        mapping = anno[col].to_dict()
        S3[col] = S3[merge_col].map(mapping) if merge_col in S3.columns else np.nan

print(f"Table S3: {S3.shape}")
S3.head(3)

# %% [markdown]
# ## Table S4: Cell composition (MERFISH)

# %%
cfg = tables_cfg["S4"]
S4 = pd.read_csv(resolve_path(cfg["sources"]["composition"]), index_col=0)
S4 = S4.reset_index()
print(f"Table S4: {S4.shape}")
S4.iloc[:3, :5]

# %% [markdown]
# ## Table S5: Structure bias stratified by IQ

# %%
cfg = tables_cfg["S5"]
src = cfg["sources"]

# Load IQ pvalues (213 structures) and subset to 46-node circuit
S5_all = pd.read_csv(resolve_path(src["iq_pvalues"]))
if cfg.get("filter_circuit_46", False):
    circuit_strs = set(circuit_46)  # from S1 cell above
    S5 = S5_all[S5_all["STR"].isin(circuit_strs)].copy()
else:
    S5 = S5_all.copy()
print(f"Table S5: {S5.shape} (filtered from {len(S5_all)} to {len(S5)} structures)")
S5.head(3)

# %% [markdown]
# ## Table S6: Neurotransmitter genes

# %%
cfg = tables_cfg["S6"]
nt_genes = pd.read_csv(resolve_path(cfg["sources"]["genes"]), index_col=0)

# Filter to specified systems
systems = cfg.get("systems", None)
if systems:
    S6 = nt_genes[nt_genes["neurotransmitter_system"].str.lower().isin([s.lower() for s in systems])].copy()
else:
    S6 = nt_genes.copy()

S6 = S6.reset_index()
print(f"Table S6: {S6.shape}")
S6.head(3)

# %% [markdown]
# ## Tables S7-S9: Neurotransmitter bias (Dopamine, Serotonin, Oxytocin)

# %%
S7 = read_bias_csv_no_index(tables_cfg["S7"]["sources"]["bias"])
S8 = read_bias_csv_no_index(tables_cfg["S8"]["sources"]["bias"])
S9 = read_bias_csv_no_index(tables_cfg["S9"]["sources"]["bias"])
print(f"Table S7 (Dopamine):  {S7.shape}")
print(f"Table S8 (Serotonin): {S8.shape}")
print(f"Table S9 (Oxytocin):  {S9.shape}")

# %% [markdown]
# ## Tables S10-S11: DD bias (structure and cell type)

# %%
S10 = read_bias_csv_no_index(tables_cfg["S10"]["sources"]["bias"])
S11 = read_bias_csv_no_index(tables_cfg["S11"]["sources"]["bias"])
print(f"Table S10 (DD structure): {S10.shape}")
print(f"Table S11 (DD cell type): {S11.shape}")

# %% [markdown]
# ## Tables S12-S13: Constrained gene bias (structure and cell type)

# %%
S12 = read_bias_csv_no_index(tables_cfg["S12"]["sources"]["bias"])
S13 = read_bias_csv_no_index(tables_cfg["S13"]["sources"]["bias"])
print(f"Table S12 (Constrained structure): {S12.shape}")
print(f"Table S13 (Constrained cell type): {S13.shape}")

# %% [markdown]
# ## Table S14: Human-mouse region mapping

# %%
cfg = tables_cfg["S14"]
S14 = pd.read_excel(resolve_path(cfg["sources"]["mapping"]))
print(f"Table S14: {S14.shape}")
S14.head(3)

# %% [markdown]
# # Build description sheet and write Excel

# %%
# Description sheet
desc_rows = []
for key in sorted(tables_cfg.keys(), key=lambda k: int(k[1:])):
    cfg = tables_cfg[key]
    desc_rows.append({
        "Table": cfg["sheet_name"],
        "Title": cfg["title"],
        "Description": cfg["description"].strip()
    })
desc_df = pd.DataFrame(desc_rows)
print("Description sheet:")
desc_df

# %%
# Collect all tables in order
all_tables = {
    "Table Descriptions": desc_df,
    tables_cfg["S1"]["sheet_name"]: S1,
    tables_cfg["S2"]["sheet_name"]: S2,
    tables_cfg["S3"]["sheet_name"]: S3,
    tables_cfg["S4"]["sheet_name"]: S4,
    tables_cfg["S5"]["sheet_name"]: S5,
    tables_cfg["S6"]["sheet_name"]: S6,
    tables_cfg["S7"]["sheet_name"]: S7,
    tables_cfg["S8"]["sheet_name"]: S8,
    tables_cfg["S9"]["sheet_name"]: S9,
    tables_cfg["S10"]["sheet_name"]: S10,
    tables_cfg["S11"]["sheet_name"]: S11,
    tables_cfg["S12"]["sheet_name"]: S12,
    tables_cfg["S13"]["sheet_name"]: S13,
    tables_cfg["S14"]["sheet_name"]: S14,
}

# Ensure output directory exists
os.makedirs(os.path.dirname(output_excel), exist_ok=True)

with pd.ExcelWriter(output_excel, engine="openpyxl", mode="w") as writer:
    for sheet_name, df in all_tables.items():
        # Excel sheet names max 31 chars
        name = sheet_name[:31]
        df.to_excel(writer, sheet_name=name, index=False)

print(f"\nWrote {len(all_tables)} sheets to {output_excel}")
for name, df in all_tables.items():
    print(f"  {name[:31]:35s}  {df.shape}")
