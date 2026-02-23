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
# # 01. Download ISH Data from Allen Brain Atlas
#
# This notebook downloads and preprocesses input data for the GENCIC pipeline:
#
# 1. **Allen ISH experiment metadata** — list of all mouse brain ISH experiments with gene info
# 2. **Allen gene catalog** — all genes with Entrez IDs in the Allen Mouse Brain Atlas
# 3. **ISH expression energy** — per-structure unionized expression for each experiment (bulk download)
# 4. **Human-Mouse gene homology** — ortholog mapping from MGI/JAX
# 5. **Gene mapping construction** — match human genes → mouse orthologs → Allen section IDs
# 6. **Structure metadata** — 213 selected brain structures
#
# All outputs are saved to `dat/allen-mouse-exp/`.
# Data downloads are guarded by `os.path.exists()` checks to avoid re-downloading.

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
# ## 4. Build Human-Mouse Gene Mapping

# %%
AllenMouseGenes = pd.read_csv(
    os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv")
)
Human2MouseHom = pd.read_csv(hom_path, delimiter="\t")
print(f"Allen ISH experiments: {len(AllenMouseGenes)}")
print(f"Homology entries: {len(Human2MouseHom)}")

# %%
# Build bidirectional Human <-> Mouse gene mappings using homology IDs
Homo_IDs = set(Human2MouseHom["DB Class Key"].values)
print(f"Unique homology groups: {len(Homo_IDs)}")

Human2Mouse_Genes = {}
Mouse2Human_Genes = {}

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
            Mouse2Human_Genes[Entrez] = {
                "symbol": Symbol,
                "humanHomo": hum_genes,
                "allen_section_data_set_id": [],
            }
        else:
            Mouse2Human_Genes[Entrez]["humanHomo"].extend(hum_genes)

print(f"Human genes with mouse orthologs: {len(Human2Mouse_Genes)}")
print(f"Mouse genes with human orthologs: {len(Mouse2Human_Genes)}")

# %%
# Create symbol-keyed version for convenience
Mouse2Human_Genes_2 = {}
for k, v in Mouse2Human_Genes.items():
    Mouse2Human_Genes_2[v["symbol"]] = {"Entrez": k, "humanHomo": v["humanHomo"]}

# Save pickle files
pk.dump(Mouse2Human_Genes_2, open(os.path.join(ProjDIR, "dat/Mouse2Human_Symbol.pk"), "wb"))
pk.dump(Mouse2Human_Genes, open(os.path.join(ProjDIR, "dat/Mouse2Human_Entrez.pk"), "wb"))
print("Saved Mouse2Human_Symbol.pk and Mouse2Human_Entrez.pk")

# %% [markdown]
# ## 5. Match Allen Section IDs to Mouse Genes

# %%
# Load discontinued Entrez IDs to handle gene ID changes
df_disc = pd.read_csv(os.path.join(ProjDIR, "dat/gene_history.human.mouse.tsv"), delimiter="\t")
Discontinued_ID = dict(zip(df_disc["Discontinued_GeneID"].values, df_disc["GeneID"].values))
print(f"Discontinued gene ID mappings: {len(Discontinued_ID)}")

# %%
# Match Allen section IDs to mouse genes via Entrez ID (primary) or discontinued ID (fallback)
Entrez_Failed_ID = []
for _, row in AllenMouseGenes.iterrows():
    allen_entrez = int(row["genes_entrez_id"]) if str(row["genes_entrez_id"]) != "nan" else None
    allen_section_id = row["section_data_set_id"]

    potential_ID = Discontinued_ID.get(allen_entrez, -1)
    if isinstance(potential_ID, (int, float)) and not np.isnan(potential_ID):
        potential_ID = int(potential_ID)
    else:
        potential_ID = -1

    if allen_entrez in Mouse2Human_Genes:
        Mouse2Human_Genes[allen_entrez]["allen_section_data_set_id"].append(allen_section_id)
    elif potential_ID in Mouse2Human_Genes:
        Mouse2Human_Genes[potential_ID]["allen_section_data_set_id"].append(allen_section_id)
    else:
        Entrez_Failed_ID.append(allen_section_id)

print(f"Section IDs matched by Entrez: {len(AllenMouseGenes) - len(Entrez_Failed_ID)}")
print(f"Section IDs unmatched: {len(Entrez_Failed_ID)}")

# %%
# Count genes still missing section IDs
Mouse_MissSectionID = [k for k, v in Mouse2Human_Genes.items() if len(v["allen_section_data_set_id"]) == 0]
print(f"Mouse genes with no Allen section IDs yet: {len(Mouse_MissSectionID)}")

# %%
# Second pass: match unlinked section IDs by gene symbol or alias
Unlinked_AllenMouseGenes = AllenMouseGenes[AllenMouseGenes["section_data_set_id"].isin(Entrez_Failed_ID)]
Symbol_Failed_ID = []
mapped_by_symbol = 0
mapped_by_alias = 0

for _, row in Unlinked_AllenMouseGenes.iterrows():
    allen_symbol = row["gene_acronym"] if str(row["gene_acronym"]) != "nan" else None
    allen_alias = row["gene_alias_tags"].split() if str(row["gene_alias_tags"]) != "nan" else []
    allen_section_id = row["section_data_set_id"]

    found = False
    for k in Mouse_MissSectionID:
        symbol = Mouse2Human_Genes[k]["symbol"]
        if allen_symbol and symbol.lower() == allen_symbol.lower():
            Mouse2Human_Genes[k]["allen_section_data_set_id"].append(allen_section_id)
            found = True
            mapped_by_symbol += 1
            break
        else:
            for alias in allen_alias:
                if alias.lower() == symbol.lower():
                    Mouse2Human_Genes[k]["allen_section_data_set_id"].append(allen_section_id)
                    found = True
                    mapped_by_alias += 1
                    break
            if found:
                break
    if not found:
        Symbol_Failed_ID.append(allen_section_id)

print(f"Matched by symbol: {mapped_by_symbol}")
print(f"Matched by alias: {mapped_by_alias}")
print(f"Still unmatched: {len(Symbol_Failed_ID)}")

# %%
# Final count of genes still missing
Mouse_MissSectionID = [k for k, v in Mouse2Human_Genes.items() if len(v["allen_section_data_set_id"]) == 0]
print(f"Mouse genes with no Allen section IDs (final): {len(Mouse_MissSectionID)}")

# %% [markdown]
# ## 6. Save Gene Mapping and Unlinked IDs

# %%
DIR = os.path.join(ProjDIR, "dat/allen-mouse-exp/")

with open(os.path.join(DIR, "human2mouse.0420.json"), "w") as f:
    json.dump(Human2Mouse_Genes, f)

with open(os.path.join(DIR, "mouse2sectionID.0420.json"), "w") as f:
    json.dump(Mouse2Human_Genes, f)

Unlinked_final = AllenMouseGenes[AllenMouseGenes["section_data_set_id"].isin(Symbol_Failed_ID)]
Unlinked_final.to_csv(os.path.join(DIR, "Allen_Unlinked_SectionIDs.0415.csv"), index=False)

print(f"Saved human2mouse.0420.json ({len(Human2Mouse_Genes)} entries)")
print(f"Saved mouse2sectionID.0420.json ({len(Mouse2Human_Genes)} entries)")
print(f"Saved Allen_Unlinked_SectionIDs.0415.csv ({len(Unlinked_final)} rows)")

# %% [markdown]
# ## 7. Structure Metadata

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
