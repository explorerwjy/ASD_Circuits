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
import requests
import json
import pickle as pk
import pandas as pd
import numpy as np

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import modify_str, LoadList

os.chdir(os.path.join(ProjDIR, "notebooks_mouse_str"))
print(f"Project root: {ProjDIR}")
print(f"Working directory: {os.getcwd()}")

# %%
# Load config for data paths
with open(os.path.join(ProjDIR, "config/config.yaml"), "r") as f:
    config = yaml.safe_load(f)

ISH_DIR = config["data_files"]["ish_expression_dir"]
print(f"ISH expression directory: {ISH_DIR}")

# %% [markdown]
# # 01. Download ISH Data and Build Expression Matrix
#
# This notebook downloads input data from the Allen Brain Atlas and builds the
# gene expression matrix used for all structure-level analyses.
#
# 1. **Allen ISH experiment metadata** — list of all mouse brain ISH experiments with gene info
# 2. **Allen gene catalog** — all genes with Entrez IDs in the Allen Mouse Brain Atlas
# 3. **ISH expression energy** — per-structure unionized expression for each experiment (bulk download)
# 4. **Human-Mouse gene homology** — ortholog mapping from MGI/JAX
# 5. **HGNC ortholog mappings** — human↔mouse gene mappings via homology groups
# 6. **Discontinued mouse Entrez IDs** — update Allen data with current Entrez IDs
# 7. **Human gene → Allen section IDs** — union of Entrez-based and symbol-based routes (Jon's strategy)
# 8. **Structure metadata** — 213 selected brain structures
# 9. **Expression matrix** — arithmetic mean of raw expression energy across ISH sections per gene
#
# All outputs are saved to `dat/allen-mouse-exp/`.
# Data downloads are guarded by `os.path.exists()` checks to avoid re-downloading.
#
# ### Gene mapping strategy
#
# Following Jon's R pipeline (`human_gene.R`), each human gene is mapped to
# Allen ISH section IDs via **two independent routes**, then the results are
# **unioned**:
#
# - **Entrez route**: human Entrez → mouse Entrez (HGNC) → Allen section IDs
# - **Symbol route**: human Entrez → human symbol → mouse symbol (HGNC) → Allen section IDs
#
# Before mapping, discontinued mouse Entrez IDs in the Allen data are updated
# to their current equivalents (414 sections affected, 253 mouse genes recovered).
# See `docs/Jon_Exp_Mat.md` for full documentation.

# %% [markdown]
# ## 1. Download Allen ISH Experiment Metadata

# %%
# Download list of all ISH experiments with gene and plane-of-section information
out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv")
if not os.path.exists(out_path):
    url = (
        "http://api.brain-map.org/api/v2/data/query.csv?criteria=model::SectionDataSet,"
        "rma::criteria,[failed$eqfalse],products[abbreviation$eq'Mouse'],"
        "treatments[name$eq'ISH'],genes,plane_of_section,"
        "rma::options,[tabular$eq'plane_of_sections.name+as+plane',"
        "'genes.acronym+as+gene_acronym',"
        "'genes.entrez_id+as+genes_entrez_id',"
        "'genes.ensembl_id+as+gene_ensembl_id',"
        "'genes.alias_tags+as+gene_alias_tags',"
        "'data_sets.id+as+section_data_set_id'],"
        "[order$eq'plane_of_sections.name,genes.acronym,data_sets.id']&"
        "start_row=0&num_rows=all"
    )
    print(f"Downloading ISH experiments from:\n{url}")
    r = requests.get(url, allow_redirects=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {out_path}")
else:
    print(f"Already exists: {out_path}")

# %%
# Download complete gene catalog (gene symbols, aliases, Entrez IDs)
out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_genes.csv")
if not os.path.exists(out_path):
    url = (
        "http://api.brain-map.org/api/v2/data/query.csv?criteria=model::Gene,"
        "rma::criteria,[failed$eqfalse],products[abbreviation$eq'Mouse'],"
        "[tabular$eq'acronym+as+gene_acronym',"
        "'gene_aliases.name+as+gene_aliases',"
        "'entrez_id+as+genes_entrez_id',"
        "[order$eq'acronym']&"
        "start_row=0&num_rows=all"
    )
    print(f"Downloading gene catalog from:\n{url}")
    r = requests.get(url, allow_redirects=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {out_path}")
else:
    print(f"Already exists: {out_path}")

# %%
# Download gene metadata with homologene IDs
out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_Genes.csv")
if not os.path.exists(out_path):
    url = (
        "http://api.brain-map.org/api/v2/data/query.csv?criteria="
        "model::Gene,"
        "rma::criteria,products[abbreviation$eq'Mouse'],"
        "rma::options,[tabular$eq'genes.id','genes.acronym+as+gene_symbol',"
        "'genes.name+as+gene_name','genes.entrez_id+as+entrez_gene_id',"
        "'genes.homologene_id+as+homologene_group_id',"
        "'genes.alias_tags+as+gene_alias_tags'],"
        "[order$eq'genes.acronym']&num_rows=all&start_row=0"
    )
    print(f"Downloading gene metadata from:\n{url}")
    r = requests.get(url, allow_redirects=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {out_path}")
else:
    print(f"Already exists: {out_path}")

# %% [markdown]
# ## 2. Download ISH Expression Energy (Bulk)
#
# Each ISH experiment has a `section_data_set_id`. For each, we download the
# structure-unionized expression energy from the Allen API. This produces one
# CSV per experiment in the ISH directory configured in `config/config.yaml`.
#
# **This is a large download (~26,000 files). Skip if already present.**

# %%
API_PATH = "http://api.brain-map.org/api/v2/data"
GRAPH_ID = 1

experiments_csv = os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv")

# Check how many are already downloaded
existing_files = set(os.listdir(ISH_DIR)) if os.path.exists(ISH_DIR) else set()
print(f"ISH directory: {ISH_DIR}")
print(f"Existing expression files: {len(existing_files)}")

if len(existing_files) < 25000:
    os.makedirs(ISH_DIR, exist_ok=True)
    experiments = pd.read_csv(experiments_csv)
    section_ids = experiments["section_data_set_id"].unique()
    print(f"Total unique section IDs: {len(section_ids)}")

    url_base = (
        f"{API_PATH}/StructureUnionize/query.csv"
        f"?criteria=[section_data_set_id$eq%d]"
        f",structure[graph_id$eq{GRAPH_ID}]"
        f"&numRows=all"
    )
    n_downloaded = 0
    for i, section_id in enumerate(section_ids):
        fname = f"{section_id}.csv"
        if fname in existing_files:
            continue
        url = url_base % section_id
        r = requests.get(url, allow_redirects=True)
        with open(os.path.join(ISH_DIR, fname), "wb") as f:
            f.write(r.content)
        n_downloaded += 1
        if n_downloaded % 1000 == 0:
            print(f"  Downloaded {n_downloaded} / {len(section_ids) - len(existing_files)} remaining...")
    print(f"Downloaded {n_downloaded} new files. Total: {len(existing_files) + n_downloaded}")
else:
    print(f"ISH expression data already downloaded ({len(existing_files)} files)")

# %% [markdown]
# ## 3. Download Human-Mouse Gene Homology

# %%
# Download MGI Human-Mouse homology report
hom_path = os.path.join(ProjDIR, "dat/HOM_MouseHumanSequence.rpt")
if not os.path.exists(hom_path):
    url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"
    print(f"Downloading homology data from:\n{url}")
    r = requests.get(url, allow_redirects=True)
    with open(hom_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {hom_path}")
else:
    print(f"Already exists: {hom_path}")

# %% [markdown]
# ## 4. Load Data for Gene Mapping

# %%
AllenMouseGenes = pd.read_csv(
    os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv")
)
Human2MouseHom = pd.read_csv(hom_path, delimiter="\t")
print(f"Allen ISH experiments: {len(AllenMouseGenes)}")
print(f"Homology entries: {len(Human2MouseHom)}")

# %% [markdown]
# ## 5. Build HGNC Ortholog Mappings
#
# Three mappings needed for Jon's union strategy:
# - human Entrez → mouse Entrez (via DB Class Key homology groups)
# - human symbol → mouse symbol (via DB Class Key homology groups)
# - human Entrez → human symbol (for the symbol route)

# %%
hum_dat = Human2MouseHom[Human2MouseHom["NCBI Taxon ID"] == 9606]
mus_dat = Human2MouseHom[Human2MouseHom["NCBI Taxon ID"] == 10090]

# human Entrez → mouse Entrez
# For each human gene, find its homology groups, expand to related human genes,
# then find all mouse genes in those groups (Jon: mapl$human.entrez.to.mouse.entrez)
human_entrez_to_mouse_entrez = {}
for hentrez in hum_dat["EntrezGene ID"].unique():
    db_keys = set(hum_dat[hum_dat["EntrezGene ID"] == hentrez]["DB Class Key"])
    related_human = set(hum_dat[hum_dat["DB Class Key"].isin(db_keys)]["EntrezGene ID"])
    all_db_keys = set(hum_dat[hum_dat["EntrezGene ID"].isin(related_human)]["DB Class Key"])
    mouse_genes = set(mus_dat[mus_dat["DB Class Key"].isin(all_db_keys)]["EntrezGene ID"])
    if mouse_genes:
        human_entrez_to_mouse_entrez[int(hentrez)] = [int(m) for m in mouse_genes]

# human symbol → mouse symbol (Jon: mapl$human.symbol.to.mouse.symbol)
human_symbol_to_mouse_symbol = {}
for hsym in hum_dat["Symbol"].unique():
    db_keys = set(hum_dat[hum_dat["Symbol"] == hsym]["DB Class Key"])
    related_human = set(hum_dat[hum_dat["DB Class Key"].isin(db_keys)]["Symbol"])
    all_db_keys = set(hum_dat[hum_dat["Symbol"].isin(related_human)]["DB Class Key"])
    mouse_syms = set(mus_dat[mus_dat["DB Class Key"].isin(all_db_keys)]["Symbol"])
    if mouse_syms:
        human_symbol_to_mouse_symbol[hsym] = list(mouse_syms)

# human Entrez → human symbol (Jon uses org.Hs.eg.db; we use HGNC directly)
human_entrez_to_symbol = {}
for _, row in hum_dat[["EntrezGene ID", "Symbol"]].drop_duplicates().iterrows():
    eid = int(row["EntrezGene ID"])
    human_entrez_to_symbol.setdefault(eid, []).append(row["Symbol"])

print(f"human Entrez → mouse Entrez: {len(human_entrez_to_mouse_entrez)} genes")
print(f"human symbol → mouse symbol: {len(human_symbol_to_mouse_symbol)} genes")
print(f"human Entrez → human symbol: {len(human_entrez_to_symbol)} genes")

# %%
# Save pickle files for backward compatibility
Mouse2Human_Genes = {}
Human2Mouse_Genes = {}
Homo_IDs = set(Human2MouseHom["DB Class Key"].values)

for ID in Homo_IDs:
    tmp_df = Human2MouseHom[Human2MouseHom["DB Class Key"] == ID]
    hum_genes, mou_genes = [], []
    for _, row in tmp_df.iterrows():
        TaxonID = row["NCBI Taxon ID"]
        Symbol = row["Symbol"]
        Entrez = row["EntrezGene ID"]
        if TaxonID == 9606:
            hum_genes.append((Symbol, Entrez))
        elif TaxonID == 10090:
            mou_genes.append((Symbol, Entrez))

    for (Symbol, Entrez) in hum_genes:
        if Entrez not in Human2Mouse_Genes:
            Human2Mouse_Genes[Entrez] = {"symbol": Symbol, "mouseHomo": mou_genes}
        else:
            Human2Mouse_Genes[Entrez]["mouseHomo"].extend(mou_genes)

    for (Symbol, Entrez) in mou_genes:
        if Entrez not in Mouse2Human_Genes:
            Mouse2Human_Genes[Entrez] = {"symbol": Symbol, "humanHomo": hum_genes}
        else:
            Mouse2Human_Genes[Entrez]["humanHomo"].extend(hum_genes)

Mouse2Human_Genes_2 = {}
for k, v in Mouse2Human_Genes.items():
    Mouse2Human_Genes_2[v["symbol"]] = {"Entrez": k, "humanHomo": v["humanHomo"]}

pk.dump(Mouse2Human_Genes_2, open(os.path.join(ProjDIR, "dat/Mouse2Human_Symbol.pk"), "wb"))
pk.dump(Mouse2Human_Genes, open(os.path.join(ProjDIR, "dat/Mouse2Human_Entrez.pk"), "wb"))
print(f"Human genes with mouse orthologs: {len(Human2Mouse_Genes)}")
print(f"Mouse genes with human orthologs: {len(Mouse2Human_Genes)}")
print("Saved Mouse2Human_Symbol.pk and Mouse2Human_Entrez.pk")

# %% [markdown]
# ## 6. Update Discontinued Mouse Entrez IDs in Allen Data
#
# Allen ISH experiments reference mouse Entrez IDs that may be discontinued.
# Before mapping, update these to current IDs (Jon: `download.R` + `human_gene.R`).
# This recovers ~414 sections across ~253 mouse genes that would otherwise be lost.

# %%
df_disc = pd.read_csv(os.path.join(ProjDIR, "dat/gene_history.human.mouse.tsv"), delimiter="\t")

# Build discontinued → current mapping for mouse genes (tax_id=10090)
disc_mouse = df_disc[(df_disc["#tax_id"] == 10090) & (df_disc["GeneID"] != "-")].copy()
disc_mouse["GeneID"] = disc_mouse["GeneID"].astype(int)
disc_mouse["Discontinued_GeneID"] = disc_mouse["Discontinued_GeneID"].astype(int)
disc_mouse_map = dict(zip(disc_mouse["Discontinued_GeneID"], disc_mouse["GeneID"]))

# Resolve chains: if A→B and B→C, then A→C (Jon's while-loop logic)
changed = True
while changed:
    changed = False
    for old_id, new_id in list(disc_mouse_map.items()):
        if new_id in disc_mouse_map:
            disc_mouse_map[old_id] = disc_mouse_map[new_id]
            changed = True

print(f"Discontinued mouse Entrez mappings: {len(disc_mouse_map)} (chain-resolved)")

# %%
# Update Allen experiment table with current Entrez IDs
allen_work = AllenMouseGenes[AllenMouseGenes["genes_entrez_id"].notna()].copy()
allen_work["genes_entrez_id"] = allen_work["genes_entrez_id"].astype(int)

n_updated = allen_work["genes_entrez_id"].isin(disc_mouse_map.keys()).sum()
allen_work["genes_entrez_id"] = allen_work["genes_entrez_id"].map(
    lambda x: disc_mouse_map.get(x, x)
)
print(f"Updated {n_updated} Allen section entries (discontinued → current Entrez)")

# %% [markdown]
# ## 7. Map Human Genes → Allen Section IDs (Union Strategy)
#
# Following Jon's `human_gene.R`, we map each human gene to Allen ISH
# section IDs via **two independent routes**, then **union** the results:
#
# 1. **Entrez route**: human Entrez → mouse Entrez → section IDs
# 2. **Symbol route**: human Entrez → human symbol → mouse symbol → section IDs
#
# This union approach recovers additional sections for genes where the Allen
# database uses different identifiers (e.g., deprecated gene symbols) across
# the two mapping paths.

# %%
# Build mouse gene → Allen section ID maps from updated Allen data
mouse_entrez_to_sections = {}
for eid, group in allen_work.groupby("genes_entrez_id"):
    mouse_entrez_to_sections[int(eid)] = sorted(set(group["section_data_set_id"]))

mouse_symbol_to_sections = {}
for sym, group in allen_work.groupby("gene_acronym"):
    mouse_symbol_to_sections[sym] = sorted(set(group["section_data_set_id"]))

print(f"Mouse Entrez → sections: {len(mouse_entrez_to_sections)} genes")
print(f"Mouse symbol → sections: {len(mouse_symbol_to_sections)} genes")

# %%
# Entrez route: human Entrez → mouse Entrez → sections
human_sections_entrez = {}
for hentrez, mouse_list in human_entrez_to_mouse_entrez.items():
    sections = []
    for mentrez in mouse_list:
        sections.extend(mouse_entrez_to_sections.get(mentrez, []))
    if sections:
        human_sections_entrez[hentrez] = sorted(set(sections))

# Symbol route: human Entrez → human symbol → mouse symbol → sections
# Step 1: human symbol → mouse symbol → sections
hsym_to_sections = {}
for hsym, mouse_syms in human_symbol_to_mouse_symbol.items():
    sections = []
    for msym in mouse_syms:
        sections.extend(mouse_symbol_to_sections.get(msym, []))
    if sections:
        hsym_to_sections[hsym] = sorted(set(sections))

# Step 2: human Entrez → human symbol → sections
human_sections_symbol = {}
for hentrez, symbols in human_entrez_to_symbol.items():
    sections = []
    for sym in symbols:
        sections.extend(hsym_to_sections.get(sym, []))
    if sections:
        human_sections_symbol[hentrez] = sorted(set(sections))

# Union both routes
all_human_genes = sorted(set(human_sections_entrez.keys()) | set(human_sections_symbol.keys()))
gene_sections = {}
for hentrez in all_human_genes:
    ent_secs = set(human_sections_entrez.get(hentrez, []))
    sym_secs = set(human_sections_symbol.get(hentrez, []))
    combined = sorted(ent_secs | sym_secs)
    if combined:
        gene_sections[hentrez] = combined

# Count contributions from each route
n_only_entrez = len(set(human_sections_entrez) - set(human_sections_symbol))
n_only_symbol = len(set(human_sections_symbol) - set(human_sections_entrez))
n_both = len(set(human_sections_entrez) & set(human_sections_symbol))
n_extra = sum(
    1 for g in set(human_sections_entrez) & set(human_sections_symbol)
    if set(human_sections_symbol.get(g, [])) - set(human_sections_entrez.get(g, []))
)

print(f"\nEntrez route: {len(human_sections_entrez)} genes")
print(f"Symbol route: {len(human_sections_symbol)} genes")
print(f"Union (final): {len(gene_sections)} genes")
print(f"  Only via Entrez: {n_only_entrez}")
print(f"  Only via symbol: {n_only_symbol}")
print(f"  Both routes: {n_both} ({n_extra} gain extra sections from union)")

# %%
# Save gene mapping
DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp/")

with open(os.path.join(DIR, "human2mouse.0420.json"), "w") as f:
    json.dump(Human2Mouse_Genes, f)

# Save human gene → section ID mapping (used by section 9)
with open(os.path.join(DIR, "human_gene_sections.json"), "w") as f:
    json.dump(gene_sections, f)

print(f"Saved human2mouse.0420.json ({len(Human2Mouse_Genes)} entries)")
print(f"Saved human_gene_sections.json ({len(gene_sections)} entries)")

# %% [markdown]
# ## 8. Structure Metadata

# %%
STR_Meta = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/allen_brain_atlas_structures.csv"))
STR_Meta.dropna(inplace=True, subset=["atlas_id"])
STR_Meta["atlas_id"] = STR_Meta["atlas_id"].astype(int)
STR_Meta = STR_Meta.set_index("atlas_id")

# Clean up structure names using modify_str
STR_Meta["Name2"] = STR_Meta["safe_name"].apply(modify_str)

# Filter to the 213 selected structures
Selected_STRs = LoadList(os.path.join(ProjDIR, "dat/allen-mouse-exp/Structures.txt"))
STR_Meta_2 = STR_Meta[STR_Meta["Name2"].isin(Selected_STRs)].sort_values("Name2")

out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/Selected_213_STRs.Meta.csv")
STR_Meta_2.to_csv(out_path)
print(f"Selected structures: {len(STR_Meta_2)} (expected 213)")
print(f"Saved to {out_path}")

# %% [markdown]
# ## 9. Build Expression Matrix from Raw ISH Files
#
# For each human gene with mapped Allen section IDs (from union strategy above),
# read the raw ISH expression energy from per-section CSV files and compute the
# **arithmetic mean** across ISH experiments. This produces a (genes × 213
# structures) matrix of raw expression energy values.
#
# This follows Jon's original aggregation method:
# `colMeans(expression_energy)` across all section datasets per gene.
# The log2 and QN transforms are applied in notebook 02.
#
# See `docs/Jon_Exp_Mat.md` for documentation of the aggregation method.

# %%
from multiprocessing import Pool
from functools import partial

ISH_DIR = config["data_files"]["ish_expression_dir"]
exp_mat_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/ExpressionMatrix_raw.parquet")

if os.path.exists(exp_mat_path):
    print(f"Expression matrix already exists: {exp_mat_path}")
    ExpMat = pd.read_parquet(exp_mat_path)
    print(f"  Shape: {ExpMat.shape}")
else:
    # Structure mapping: Allen structure 'id' → structure name
    structid2name = dict(zip(STR_Meta_2["id"], STR_Meta_2["Name2"]))
    selected_struct_ids = set(structid2name.keys())

    # Filter gene_sections to only include available ISH files
    existing_files = set(os.listdir(ISH_DIR))
    gene_sections_available = {}
    for hentrez, sids in gene_sections.items():
        available = [s for s in sids if f"{s}.csv" in existing_files]
        if available:
            gene_sections_available[hentrez] = available
    print(f"Genes with available ISH data: {len(gene_sections_available)}")

    # Function to process one gene: read ISH CSVs, compute mean expression energy
    def process_gene(args, ish_dir, selected_struct_ids, structid2name, structures):
        human_entrez, section_ids = args
        acc = {s: [] for s in structures}
        for sid in section_ids:
            csv_path = os.path.join(ish_dir, f"{sid}.csv")
            try:
                df = pd.read_csv(csv_path, usecols=["structure_id", "expression_energy"])
            except (FileNotFoundError, ValueError):
                continue
            for _, row in df.iterrows():
                struct_id = int(row["structure_id"])
                if struct_id in selected_struct_ids:
                    name = structid2name[struct_id]
                    ee = row["expression_energy"]
                    if pd.notna(ee):
                        acc[name].append(ee)
        result = {}
        for s in structures:
            result[s] = np.mean(acc[s]) if acc[s] else np.nan
        return human_entrez, result

    structures = sorted(Selected_STRs)
    gene_list = list(gene_sections_available.items())

    print(f"Processing {len(gene_list)} genes from {ISH_DIR} ...")
    worker = partial(
        process_gene,
        ish_dir=ISH_DIR,
        selected_struct_ids=selected_struct_ids,
        structid2name=structid2name,
        structures=structures,
    )
    with Pool(10) as pool:
        results = pool.map(worker, gene_list)

    rows = {}
    for human_entrez, row_data in results:
        rows[human_entrez] = row_data
    ExpMat = pd.DataFrame.from_dict(rows, orient="index", columns=structures)
    ExpMat.index.name = "ROW"
    ExpMat = ExpMat.sort_index()

    ExpMat.to_parquet(exp_mat_path)
    print(f"Saved: {exp_mat_path}")
    print(f"  Shape: {ExpMat.shape}")
    print(f"  NaN fraction: {ExpMat.isna().sum().sum() / ExpMat.size:.4f}")
    print(f"  Value range: [{np.nanmin(ExpMat.values):.4f}, {np.nanmax(ExpMat.values):.4f}]")

# %% [markdown]
# ## 10. Validate Against Jon's Raw Matrix

# %%
jon_raw_path = os.path.join(ProjDIR, config["data_files"]["jon_exp_raw"])
if os.path.exists(jon_raw_path):
    jon_raw = pd.read_csv(jon_raw_path, index_col="ROW")
    common = sorted(set(ExpMat.index) & set(jon_raw.index))
    jon_only = sorted(set(jon_raw.index) - set(ExpMat.index))
    our_only = sorted(set(ExpMat.index) - set(jon_raw.index))

    diff = (ExpMat.loc[common, jon_raw.columns] - jon_raw.loc[common]).abs()
    exact = (diff.max(axis=1) < 1e-10).sum()

    print(f"Jon: {len(jon_raw)} genes | Ours: {len(ExpMat)} genes | Common: {len(common)}")
    print(f"Jon-only: {len(jon_only)} (older HGNC version)")
    print(f"Ours-only: {len(our_only)}")
    print(f"Exact match: {exact}/{len(common)} ({100*exact/len(common):.1f}%)")
    print(f"Max diff: {diff.max().max():.2e}")
else:
    print(f"Jon's raw matrix not found at {jon_raw_path} — skipping validation")
