#!/usr/bin/env python3
"""Rebuild expression matrix with correct structure ID mapping (Allen id, not atlas_id)."""
import sys, os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType"
sys.path.insert(1, os.path.join(ProjDIR, "src"))
from ASD_Circuits import modify_str, LoadList

import yaml
with open(os.path.join(ProjDIR, "config/config.yaml")) as f:
    config = yaml.safe_load(f)

ISH_DIR = config["data_files"]["ish_expression_dir"]

# Load structure metadata
STR_Meta = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/allen_brain_atlas_structures.csv"))
STR_Meta.dropna(inplace=True, subset=["atlas_id"])
STR_Meta["atlas_id"] = STR_Meta["atlas_id"].astype(int)
STR_Meta = STR_Meta.set_index("atlas_id")
STR_Meta["Name2"] = STR_Meta["safe_name"].apply(modify_str)

Selected_STRs = LoadList(os.path.join(ProjDIR, "dat/allen-mouse-exp/Structures.txt"))
STR_Meta_2 = STR_Meta[STR_Meta["Name2"].isin(Selected_STRs)].sort_values("Name2")

# FIX: Use Allen 'id' column (not atlas_id index) to match ISH structure_id
structid2name = dict(zip(STR_Meta_2["id"], STR_Meta_2["Name2"]))
selected_struct_ids = set(structid2name.keys())
print(f"Structure mapping: {len(selected_struct_ids)} Allen IDs → structure names")

# Load gene mapping
with open(os.path.join(ProjDIR, "dat/allen-mouse-exp/mouse2sectionID.0420.json")) as f:
    Mouse2Human = json.load(f)
with open(os.path.join(ProjDIR, "dat/allen-mouse-exp/human2mouse.0420.json")) as f:
    Human2Mouse = json.load(f)

gene_sections = {}
for human_entrez, info in Human2Mouse.items():
    section_ids = []
    for _, mouse_entrez in info["mouseHomo"]:
        mouse_key = str(int(mouse_entrez))
        if mouse_key in Mouse2Human:
            section_ids.extend(Mouse2Human[mouse_key]["allen_section_data_set_id"])
    if section_ids:
        gene_sections[int(human_entrez)] = [int(s) for s in set(section_ids)]

print(f"Genes with ISH data: {len(gene_sections)}")

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
gene_list = list(gene_sections.items())

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

out_path = os.path.join(ProjDIR, "dat/allen-mouse-exp/ExpressionMatrix_raw.parquet")
ExpMat.to_parquet(out_path)
print(f"\nSaved: {out_path}")
print(f"Shape: {ExpMat.shape}")
print(f"NaN count: {ExpMat.isna().sum().sum()} ({ExpMat.isna().sum().sum()/ExpMat.size*100:.1f}%)")
print(f"Range: [{np.nanmin(ExpMat.values):.4f}, {np.nanmax(ExpMat.values):.4f}]")

# Compare with Jon's
jon = pd.read_csv(os.path.join(ProjDIR, "dat/allen-mouse-exp/Jon_ExpMat.raw.csv"), index_col="ROW")
common = ExpMat.index.intersection(jon.index)
print(f"\nComparison with Jon's raw matrix:")
print(f"Common genes: {len(common)} / Jon={len(jon)} / Ours={len(ExpMat)}")

j = jon.loc[common].values.flatten()
o = ExpMat.loc[common].values.flatten()
valid = ~np.isnan(j) & ~np.isnan(o)
diffs = np.abs(j[valid] - o[valid])
r = np.corrcoef(j[valid], o[valid])[0, 1]
print(f"Pearson r:     {r:.10f}")
print(f"Max abs diff:  {diffs.max():.6f}")
print(f"Mean abs diff: {diffs.mean():.6f}")
print(f"Exact (diff=0): {(diffs == 0).mean()*100:.1f}%")
print(f"Jon NaN: {np.isnan(j).sum()}, Our NaN: {np.isnan(o).sum()}")
