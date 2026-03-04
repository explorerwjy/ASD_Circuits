#!/usr/bin/env python3
"""
Rebuild the expression matrix from Allen ISH files using Jon's mapping strategy.

Jon's pipeline (human_gene.R):
1. Update discontinued mouse Entrez IDs in Allen section data
2. Map human Entrez → mouse Entrez → section IDs (Entrez route)
3. Map human Entrez → human symbol → mouse symbol → section IDs (symbol route)
4. UNION both routes for each human gene
5. For each human gene, compute colMeans of expression_energy across sections

This script replicates that logic in Python to validate reproducibility.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

ProjDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ProjDIR, "src"))
from ASD_Circuits import modify_str, LoadList

# ──────────────────────────────────────────────────────────────────────
# 1. Load all data sources
# ──────────────────────────────────────────────────────────────────────

print("Loading data...")
allen = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/All_Mouse_Brain_ISH_experiments.csv"))
hom = pd.read_csv(os.path.join(ProjDIR, "dat/HOM_MouseHumanSequence.rpt"), delimiter="\t")
disc = pd.read_csv(os.path.join(ProjDIR, "dat/gene_history.human.mouse.tsv"), delimiter="\t")

# Structure metadata
STR_Meta = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/allen_brain_atlas_structures.csv"))
STR_Meta.dropna(inplace=True, subset=["atlas_id"])
STR_Meta["atlas_id"] = STR_Meta["atlas_id"].astype(int)
STR_Meta = STR_Meta.set_index("atlas_id")
STR_Meta["Name2"] = STR_Meta["safe_name"].apply(modify_str)
Selected_STRs = LoadList(os.path.join(ProjDIR, "dat/allen-mouse-exp/Structures.txt"))
STR_Meta_2 = STR_Meta[STR_Meta["Name2"].isin(Selected_STRs)].sort_values("Name2")
structid2name = dict(zip(STR_Meta_2["id"], STR_Meta_2["Name2"]))
selected_struct_ids = set(structid2name.keys())
structures = sorted(Selected_STRs)

# ──────────────────────────────────────────────────────────────────────
# 2. Build discontinued mouse Entrez → current Entrez mapping
#    (Jon: mapl$discontinued.mouse.entrez.to.mouse.entrez)
# ──────────────────────────────────────────────────────────────────────

print("Building discontinued mouse Entrez ID mapping...")
disc_mouse = disc[(disc["#tax_id"] == 10090) & (disc["GeneID"] != "-")].copy()
disc_mouse["GeneID"] = disc_mouse["GeneID"].astype(int)
disc_mouse["Discontinued_GeneID"] = disc_mouse["Discontinued_GeneID"].astype(int)
disc_mouse_map = dict(zip(disc_mouse["Discontinued_GeneID"], disc_mouse["GeneID"]))

# Resolve chains: if A→B and B→C, then A→C (Jon's while loop)
changed = True
while changed:
    changed = False
    for old_id, new_id in list(disc_mouse_map.items()):
        if new_id in disc_mouse_map:
            disc_mouse_map[old_id] = disc_mouse_map[new_id]
            changed = True
print(f"  Discontinued mouse Entrez mappings: {len(disc_mouse_map)}")

# ──────────────────────────────────────────────────────────────────────
# 3. Update Allen section data with current mouse Entrez IDs
#    (Jon: dat$gene_entrez_id[ind] <- mapl$discontinued...)
# ──────────────────────────────────────────────────────────────────────

print("Updating Allen section data with current Entrez IDs...")
allen_work = allen.copy()
allen_work = allen_work[allen_work["genes_entrez_id"].notna()].copy()
allen_work["genes_entrez_id"] = allen_work["genes_entrez_id"].astype(int)

# Count how many get updated
before_ids = set(allen_work["genes_entrez_id"].unique())
updated = allen_work["genes_entrez_id"].isin(disc_mouse_map.keys())
allen_work.loc[updated, "genes_entrez_id"] = allen_work.loc[updated, "genes_entrez_id"].map(
    lambda x: disc_mouse_map.get(x, x)
)
after_ids = set(allen_work["genes_entrez_id"].unique())
print(f"  Updated {updated.sum()} section entries ({len(before_ids - after_ids)} old IDs replaced)")

# ──────────────────────────────────────────────────────────────────────
# 4. Build HGNC ortholog mappings
# ──────────────────────────────────────────────────────────────────────

print("Building HGNC ortholog mappings...")
hum_dat = hom[hom["NCBI Taxon ID"] == 9606]
mus_dat = hom[hom["NCBI Taxon ID"] == 10090]

# human Entrez → mouse Entrez (via DB Class Key)
# Jon: mapl$human.entrez.to.mouse.entrez
human_entrez_to_mouse_entrez = {}
for hentrez in hum_dat["EntrezGene ID"].unique():
    # All homology group IDs for this human gene
    db_keys = set(hum_dat[hum_dat["EntrezGene ID"] == hentrez]["DB Class Key"])
    # Other human genes sharing these groups
    all_human = set(hum_dat[hum_dat["DB Class Key"].isin(db_keys)]["EntrezGene ID"])
    # Expand to all DB keys from those human genes
    all_db_keys = set(hum_dat[hum_dat["EntrezGene ID"].isin(all_human)]["DB Class Key"])
    # Mouse genes in those groups
    mouse_genes = set(mus_dat[mus_dat["DB Class Key"].isin(all_db_keys)]["EntrezGene ID"])
    if mouse_genes:
        human_entrez_to_mouse_entrez[int(hentrez)] = [int(m) for m in mouse_genes]

# human symbol → mouse symbol (via DB Class Key)
# Jon: mapl$human.symbol.to.mouse.symbol
human_symbol_to_mouse_symbol = {}
for hsym in hum_dat["Symbol"].unique():
    db_keys = set(hum_dat[hum_dat["Symbol"] == hsym]["DB Class Key"])
    all_human = set(hum_dat[hum_dat["DB Class Key"].isin(db_keys)]["Symbol"])
    all_db_keys = set(hum_dat[hum_dat["Symbol"].isin(all_human)]["DB Class Key"])
    mouse_syms = set(mus_dat[mus_dat["DB Class Key"].isin(all_db_keys)]["Symbol"])
    if mouse_syms:
        human_symbol_to_mouse_symbol[hsym] = list(mouse_syms)

# human Entrez → human symbol (Jon uses org.Hs.eg.db; we use HGNC)
human_entrez_to_symbol = {}
for _, row in hum_dat[["EntrezGene ID", "Symbol"]].drop_duplicates().iterrows():
    eid = int(row["EntrezGene ID"])
    sym = row["Symbol"]
    if eid not in human_entrez_to_symbol:
        human_entrez_to_symbol[eid] = []
    human_entrez_to_symbol[eid].append(sym)

print(f"  human Entrez → mouse Entrez: {len(human_entrez_to_mouse_entrez)} genes")
print(f"  human symbol → mouse symbol: {len(human_symbol_to_mouse_symbol)} genes")
print(f"  human Entrez → human symbol: {len(human_entrez_to_symbol)} genes")

# ──────────────────────────────────────────────────────────────────────
# 5. Map mouse genes → Allen section IDs
# ──────────────────────────────────────────────────────────────────────

print("Mapping mouse genes to Allen sections...")

# Route 1: mouse Entrez → section IDs
# (Jon: map.ent <- split(dat$id, dat$gene_entrez_id))
mouse_entrez_to_sections = {}
for eid, group in allen_work.groupby("genes_entrez_id"):
    mouse_entrez_to_sections[int(eid)] = sorted(set(group["section_data_set_id"]))

# Route 2: mouse symbol → section IDs
# (Jon: map.sym <- split(dat$id, dat$gene_acronym))
mouse_symbol_to_sections = {}
for sym, group in allen_work.groupby("gene_acronym"):
    mouse_symbol_to_sections[sym] = sorted(set(group["section_data_set_id"]))

print(f"  Mouse Entrez → sections: {len(mouse_entrez_to_sections)} genes")
print(f"  Mouse symbol → sections: {len(mouse_symbol_to_sections)} genes")

# ──────────────────────────────────────────────────────────────────────
# 6. Map human genes → Allen section IDs (UNION of Entrez + symbol routes)
# ──────────────────────────────────────────────────────────────────────

print("Building human gene → section ID mapping (union strategy)...")

# Entrez route: human Entrez → mouse Entrez → sections
# (Jon: map.ent <- lapply(mapl$human.entrez.to.mouse.entrez, function(x) unlist(map.ent[x])))
human_sections_entrez = {}
for hentrez, mouse_list in human_entrez_to_mouse_entrez.items():
    sections = []
    for mentrez in mouse_list:
        sections.extend(mouse_entrez_to_sections.get(mentrez, []))
    if sections:
        human_sections_entrez[hentrez] = sorted(set(sections))

# Symbol route: human symbol → mouse symbol → sections
# Then: human Entrez → human symbol → sections
# (Jon: map.sym <- lapply(mapl$human.symbol.to.mouse.symbol, ...)
#        map.sym <- lapply(mapl$human.entrez.to.human.symbol, ...))
human_sections_symbol = {}
# First: human symbol → sections
hsym_to_sections = {}
for hsym, mouse_syms in human_symbol_to_mouse_symbol.items():
    sections = []
    for msym in mouse_syms:
        sections.extend(mouse_symbol_to_sections.get(msym, []))
    if sections:
        hsym_to_sections[hsym] = sorted(set(sections))

# Then: human Entrez → human symbol → sections
for hentrez, symbols in human_entrez_to_symbol.items():
    sections = []
    for sym in symbols:
        sections.extend(hsym_to_sections.get(sym, []))
    if sections:
        human_sections_symbol[hentrez] = sorted(set(sections))

# UNION: combine Entrez and symbol routes
all_human_genes = sorted(set(human_sections_entrez.keys()) | set(human_sections_symbol.keys()))
gene_sections = {}
for hentrez in all_human_genes:
    ent_secs = set(human_sections_entrez.get(hentrez, []))
    sym_secs = set(human_sections_symbol.get(hentrez, []))
    combined = sorted(ent_secs | sym_secs)
    if combined:
        gene_sections[hentrez] = combined

print(f"  Entrez route only: {len(human_sections_entrez)} genes")
print(f"  Symbol route only: {len(human_sections_symbol)} genes")
print(f"  Union (final): {len(gene_sections)} genes")

# Count genes gained from each route
only_entrez = set(human_sections_entrez.keys()) - set(human_sections_symbol.keys())
only_symbol = set(human_sections_symbol.keys()) - set(human_sections_entrez.keys())
both = set(human_sections_entrez.keys()) & set(human_sections_symbol.keys())
# Genes where union gives MORE sections than either alone
extra_from_union = 0
for g in both:
    if set(human_sections_entrez[g]) != set(human_sections_symbol.get(g, [])):
        e_only = set(human_sections_entrez[g]) - set(human_sections_symbol.get(g, []))
        s_only = set(human_sections_symbol.get(g, [])) - set(human_sections_entrez[g])
        if s_only:
            extra_from_union += 1

print(f"  Genes only from Entrez route: {len(only_entrez)}")
print(f"  Genes only from symbol route: {len(only_symbol)}")
print(f"  Genes from both: {len(both)} ({extra_from_union} gain extra sections from union)")

# ──────────────────────────────────────────────────────────────────────
# 7. Check which ISH files exist
# ──────────────────────────────────────────────────────────────────────

import yaml
with open(os.path.join(ProjDIR, "config/config.yaml")) as f:
    config = yaml.safe_load(f)
ISH_DIR = config["data_files"]["ish_expression_dir"]

existing_files = set(os.listdir(ISH_DIR))
print(f"\nISH directory: {ISH_DIR}")
print(f"Available ISH files: {len(existing_files)}")

# Filter gene_sections to only include available section IDs
for hentrez in list(gene_sections.keys()):
    available = [s for s in gene_sections[hentrez] if f"{s}.csv" in existing_files]
    if available:
        gene_sections[hentrez] = available
    else:
        del gene_sections[hentrez]

print(f"Genes with available ISH data: {len(gene_sections)}")

# ──────────────────────────────────────────────────────────────────────
# 8. Build expression matrix (arithmetic mean of raw expression energy)
# ──────────────────────────────────────────────────────────────────────

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


gene_list = list(gene_sections.items())
print(f"\nBuilding expression matrix for {len(gene_list)} genes...")

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

out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/ExpressionMatrix_raw.jon_strategy.parquet")
ExpMat.to_parquet(out_path)
print(f"Saved: {out_path}")
print(f"  Shape: {ExpMat.shape}")
print(f"  NaN fraction: {ExpMat.isna().sum().sum() / ExpMat.size:.4f}")

# ──────────────────────────────────────────────────────────────────────
# 9. Compare with Jon's raw matrix
# ──────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("COMPARISON WITH JON'S RAW MATRIX")
print("=" * 60)

jon_raw = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/Jon_ExpMat.raw.csv"), index_col="ROW")

jon_genes = set(jon_raw.index)
our_genes = set(ExpMat.index)
common = sorted(jon_genes & our_genes)

print(f"\nJon: {len(jon_genes)} genes")
print(f"Ours (union): {len(our_genes)} genes")
print(f"Common: {len(common)}")
print(f"Jon only: {len(jon_genes - our_genes)}")
print(f"Ours only: {len(our_genes - jon_genes)}")

j = jon_raw.loc[common]
o = ExpMat.loc[common]
diff = (j - o).abs()
exact = (diff.max(axis=1) < 1e-10).sum()
close = (diff.max(axis=1) < 0.001).sum()

print(f"\nExact match (<1e-10): {exact}/{len(common)} ({100*exact/len(common):.1f}%)")
print(f"Close match (<0.001): {close}/{len(common)} ({100*close/len(common):.1f}%)")
print(f"Max diff: {diff.max().max():.6f}")
print(f"Mean diff: {diff.mean().mean():.6f}")

# Correlation
from scipy.stats import pearsonr
jf = j.values.flatten()
of = o.values.flatten()
mask = ~(np.isnan(jf) | np.isnan(of))
r, _ = pearsonr(jf[mask], of[mask])
print(f"Pearson r: {r:.8f}")

# Show Jon-only genes
jon_only_list = sorted(jon_genes - our_genes)
if jon_only_list:
    print(f"\nJon-only genes ({len(jon_only_list)}): {jon_only_list[:20]}...")
