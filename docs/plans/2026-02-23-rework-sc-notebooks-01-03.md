# Rework SC Notebooks 01 and 03 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `notebooks_mouse_sc/01.Preprocessing` to be purely cluster-level (Z1→Z2), and move all MERFISH content into `03.MERFISH_Preprocessing`.

**Architecture:** Notebook 01 reads `cluster_MeanLogUMI.csv`, computes Z1 (z-score per gene, clip ±3), then Z2 (ISH expression-matched z-score using match files), outputs parquet. Notebook 03 absorbs MERFISH Z1/Z2/quantile sections from old 01. Both validated against legacy outputs.

**Tech Stack:** pandas, numpy, joblib (parallel Z2), jupytext

**Reference files:**
- Old pipeline: `/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/`
- Legacy Z2 output: `/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv`
- ISH match dir: `/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/`
- Current notebook 01: `notebooks_mouse_sc/01.Preprocessing.py`
- Current notebook 03: `notebooks_mouse_sc/03.MERFISH_Preprocessing.py`
- Z2 script (reference): `scripts/build_celltype_z2_matrix.py`

---

### Task 1: Rewrite `01.Preprocessing.py` — cluster-level Z1→Z2 pipeline

**Files:**
- Rewrite: `notebooks_mouse_sc/01.Preprocessing.py`
- Reference: `scripts/build_celltype_z2_matrix.py` (stages 2-3 logic)
- Reference: `src/CellType_PSY.py` (`Z1Conversion` function)

**Step 1: Write the new `01.Preprocessing.py`**

The notebook has these sections:

**Section 0: Setup** — autoreload, imports, config, paths

```python
# %load_ext autoreload
# %autoreload 2

import sys
import os
import yaml
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ProjDIR = "/home/jw3514/Work/ASD_Circuits_CellType/"
sys.path.insert(1, f"{ProjDIR}/src/")
from CellType_PSY import *

os.chdir(f"{ProjDIR}/notebooks_mouse_sc/")

with open("../config/config.yaml") as f:
    config = yaml.safe_load(f)
```

**Section 1: Load cluster expression matrix**

```python
# Input: cluster_MeanLogUMI.csv (17938 genes x 5312 clusters)
# Produced by scripts/build_celltype_z2_matrix.py --stage1
ClusterExpDF = pd.read_csv(f"../{config['data_files']['cluster_mean_log_umi_csv']}", index_col=0)
print(f"Loaded cluster expression: {ClusterExpDF.shape}")
```

**Section 2: Z1 normalization**

Use `Z1Conversion` from `src/CellType_PSY.py` (calls `ZscoreConverting` per gene row). Then clip to ±3.

```python
ClusterZ1 = Z1Conversion(ClusterExpDF)
ClusterZ1_clip3 = ClusterZ1.clip(upper=3, lower=-3)
print(f"Z1 matrix: {ClusterZ1_clip3.shape}, range: [{ClusterZ1_clip3.min().min():.1f}, {ClusterZ1_clip3.max().max():.1f}]")
```

No CSV output — only keep in memory for Z2 step.

**Section 3: Z2 calculation (ISH expression-matched)**

Inline the Z2 logic from `build_celltype_z2_matrix.py` stage 3. Key details:
- ISH match dir: each file `{entrez_id}.csv` contains up to 1000+ ranked gene IDs
- For each gene: load match list → filter to genes in Z1 matrix → take top 1000
- Z2 = (Z1_gene - mean(Z1_matches)) / std(Z1_matches) per cluster
- Parallelize with joblib (n_jobs=10, chunk_size=500)

Use the helper functions `_load_match_genes` and `_z2_gene_chunk` from the script. Define them in-notebook.

ISH match directory path: `"/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"` — this is the same path used by the old pipeline. Add to config if not already present.

```python
ISH_MATCH_DIR = "/home/jw3514/Work/ASD_Circuits/dat/genes/ExpMatch_RootExp_uniform_kernal/"

def _load_match_genes(entrez_id, match_dir, max_genes=1000):
    fpath = os.path.join(match_dir, f"{entrez_id}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        genes = [int(df.columns[0])] + [int(x) for x in df.iloc[:, 0].values]
        return genes[:max_genes]
    except Exception:
        return None

def _z2_gene_chunk(gene_indices, z1_values, z1_index, z1_columns,
                   match_dir, index_to_entrez, entrez_to_row):
    result_indices = []
    result_rows = []
    for idx in gene_indices:
        if idx not in index_to_entrez:
            continue
        entrez = index_to_entrez[idx]
        match_genes = _load_match_genes(entrez, match_dir)
        if match_genes is None:
            continue
        match_rows = [entrez_to_row[g] for g in match_genes if g in entrez_to_row]
        if len(match_rows) < 2:
            continue
        gene_row = z1_values[entrez_to_row[entrez]]
        match_vals = z1_values[match_rows]
        match_mean = np.nanmean(match_vals, axis=0)
        match_std = np.nanstd(match_vals, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            z2_row = (gene_row - match_mean) / match_std
        z2_row[~np.isfinite(z2_row)] = np.nan
        result_indices.append(entrez)
        result_rows.append(z2_row)
    return result_indices, result_rows

# Build lookup tables
z1_values = ClusterZ1_clip3.values
z1_index = ClusterZ1_clip3.index.values
entrez_to_row = {int(z1_index[i]): i for i in range(len(z1_index))}
index_to_entrez = {i: int(z1_index[i]) for i in range(len(z1_index))}

# Split into chunks and run in parallel
chunk_size = 500
n_genes = len(z1_index)
chunks = [range(i, min(i + chunk_size, n_genes)) for i in range(0, n_genes, chunk_size)]

results = Parallel(n_jobs=10, verbose=5)(
    delayed(_z2_gene_chunk)(
        list(chunk), z1_values, z1_index, ClusterZ1_clip3.columns.values,
        ISH_MATCH_DIR, index_to_entrez, entrez_to_row
    )
    for chunk in chunks
)

# Assemble Z2 matrix
all_indices, all_rows = [], []
for indices, rows in results:
    all_indices.extend(indices)
    all_rows.extend(rows)

ClusterZ2 = pd.DataFrame(
    data=np.array(all_rows),
    index=np.array(all_indices),
    columns=ClusterZ1_clip3.columns
)
ClusterZ2.index.name = None
print(f"Z2 matrix: {ClusterZ2.shape}")
print(f"NaN count: {ClusterZ2.isna().sum().sum()}")
```

**Section 4: Save and deploy**

```python
OUT_PATH = "../dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet"
ClusterZ2.to_parquet(OUT_PATH)
print(f"Saved: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
```

**Step 2: Sync and run notebook**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType/notebooks_mouse_sc
conda run -n gencic jupytext --sync 01.Preprocessing.py
```

Then execute to verify it runs end-to-end and produces the parquet.

**Step 3: Validate against legacy**

Load the legacy CSV and compare with the new parquet:
```bash
conda run -n gencic python -c "
import pandas as pd, numpy as np
new = pd.read_parquet('dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet')
legacy = pd.read_csv('/mnt/data0/home_backup/Work/CellType_Psy/AllenBrainCellAtlas/dat/SC_UMI_Mats/Cluster_Z2Mat_ISHMatch.z1clip3.csv', index_col=0)
legacy.index = legacy.index.astype(int)
sg = new.index.intersection(legacy.index)
sc = new.columns.intersection(legacy.columns)
d = (new.loc[sg,sc] - legacy.loc[sg,sc]).abs()
mask = new.loc[sg,sc].notna() & legacy.loc[sg,sc].notna()
flat_new = new.loc[sg,sc].values[mask.values]
flat_leg = legacy.loc[sg,sc].values[mask.values]
r = np.corrcoef(flat_new, flat_leg)[0,1]
print(f'Shape: new={new.shape}, legacy={legacy.shape}')
print(f'Shared: {len(sg)} genes, {len(sc)} clusters')
print(f'Max diff: {d.max().max():.6e}')
print(f'Mean diff: {d.mean().mean():.6e}')
print(f'Correlation: {r:.10f}')
print(f'NaN new: {new.loc[sg,sc].isna().sum().sum()}, legacy: {legacy.loc[sg,sc].isna().sum().sum()}')
"
```

Expected: correlation ~0.99998, small differences from NaN handling (new pipeline: 0 NaN via `np.nan` → `np.nanmean`; legacy: 201K NaN from row-by-row iteration).

**Step 4: Commit notebook 01**

```bash
git add notebooks_mouse_sc/01.Preprocessing.py notebooks_mouse_sc/01.Preprocessing.ipynb
git commit -m "Rewrite SC notebook 01: cluster-only Z1→Z2 pipeline (no MERFISH)"
```

---

### Task 2: Update `03.MERFISH_Preprocessing.py` — absorb MERFISH Z1/Z2 from old 01

**Files:**
- Modify: `notebooks_mouse_sc/03.MERFISH_Preprocessing.py`
- Reference: old `notebooks_mouse_sc/01.Preprocessing.py` sections 2-5

**Step 1: Add MERFISH Z1/Z2 sections to notebook 03**

After the existing Section 4 (Save Parquet), add:

**Section 5: Allen MERFISH Z1 Matrices**

Read 4 raw UMI CSVs from config paths → Z1Conversion → clip ±5 → save CSVs.

```python
# Allen MERFISH — all cells
MERFISH_CellMeanExp = pd.read_csv(f"../{config['data_files']['merfish_cell_mean_umi']}", index_col=0)
MERFISH_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_vol_mean_umi']}", index_col=0)
print(f"Allen MERFISH Cell-mean: {MERFISH_CellMeanExp.shape}")
print(f"Allen MERFISH Vol-mean:  {MERFISH_VolMeanExp.shape}")

MERFISH_CellMean_Z1 = Z1Conversion(MERFISH_CellMeanExp, "../dat/MERFISH/STR_Cell_Mean_Z1Mat.csv")
MERFISH_VolMean_Z1 = Z1Conversion(MERFISH_VolMeanExp, "../dat/MERFISH/STR_Vol_Mean_Z1Mat.csv")

MERFISH_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Cell_Mean_Z1Mat.clip.csv")
MERFISH_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_Vol_Mean_Z1Mat.clip.csv")
```

Allen MERFISH neuron-only:
```python
MERFISH_NEU_MeanExp = pd.read_csv(f"../{config['data_files']['merfish_neur_mean_umi']}", index_col=0)
MERFISH_NEU_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_neur_vol_mean_umi']}", index_col=0)

MERFISH_NEU_Z1 = Z1Conversion(MERFISH_NEU_MeanExp, "../dat/MERFISH/STR_NEU_Mean_Z1Mat.csv")
MERFISH_NEU_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Mean_Z1Mat.clip.csv")

MERFISH_NEU_Vol_Z1 = Z1Conversion(MERFISH_NEU_VolMeanExp, "../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.csv")
MERFISH_NEU_Vol_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH/STR_NEU_Vol_Mean_Z1Mat.clip.csv")
```

**Section 6: Zhuang/MIT MERFISH Z1 Matrices**

```python
if os.path.exists(f"../{config['data_files']['merfish_zhuang_cell_mean_umi']}"):
    Zhuang_CellMeanExp = pd.read_csv(f"../{config['data_files']['merfish_zhuang_cell_mean_umi']}", index_col=0)
    Zhuang_VolMeanExp = pd.read_csv(f"../{config['data_files']['merfish_zhuang_vol_mean_umi']}", index_col=0)
    Zhuang_CellMean_Z1 = Z1Conversion(Zhuang_CellMeanExp, "../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.csv")
    Zhuang_VolMean_Z1 = Z1Conversion(Zhuang_VolMeanExp, "../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.csv")
    Zhuang_CellMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Cell_Mean_Z1Mat.clip.csv")
    Zhuang_VolMean_Z1.clip(upper=5, lower=-5).to_csv("../dat/MERFISH_Zhuang/STR_Vol_Mean_Z1Mat.clip.csv")
else:
    print("Zhuang MERFISH data not found — skipping")
```

**Section 7: MERFISH Z2 Matrices (from pre-computed splits)**

```python
Z2_SPLIT_BASE = config["data_files"]["z2_split_base"]

def assemble_z2_splits(split_dir, outpath):
    if not os.path.isdir(split_dir):
        print(f"  SKIP (not found): {split_dir}")
        return None
    dfs = []
    for f in sorted(os.listdir(split_dir)):
        dfs.append(pd.read_csv(os.path.join(split_dir, f), index_col=0))
    z2 = pd.concat(dfs)
    z2.to_csv(outpath)
    print(f"  {outpath}: {z2.shape}")
    return z2

# Allen MERFISH Z2
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_CellMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_Cell_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_VolMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_Vol_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_NEU_Mean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_NEUR_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_Allen_NEU_Vol_Mean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH/STR_NEUR_Vol_Mean_Z2Mat_ISHMatch.csv")

# Zhuang MERFISH Z2
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_MIT_CellMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH_Zhuang/STR_Cell_Mean_Z2Mat_ISHMatch.csv")
assemble_z2_splits(f"{Z2_SPLIT_BASE}/MERFISH_MIT_VolMean_UMI_ISHMatch_Z2",
                   "../dat/MERFISH_Zhuang/STR_Vol_Mean_Z2Mat_ISHMatch.csv")
```

**Section 8: MERFISH Expression Matching Quantiles**

```python
ClusterExpDF = pd.read_csv(f"../{config['data_files']['cluster_mean_log_umi_csv']}", index_col=0)
MERFISH_STRAnn = pd.read_csv(f"../{config['data_files']['merfish_annotation']}")

Total_Exp_Genes = np.zeros(ClusterExpDF.shape[0])
matched_clusters = 0
for _, row in MERFISH_STRAnn.iterrows():
    cluster = row.get("cluster")
    if cluster is not None and cluster in ClusterExpDF.columns:
        Total_Exp_Genes += ClusterExpDF[cluster].values
        matched_clusters += 1
print(f"Matched {matched_clusters} MERFISH entries to clusters")

WB_ExpDF = pd.DataFrame(Total_Exp_Genes, index=ClusterExpDF.index, columns=["TotalExp"])
WB_ExpDF = WB_ExpDF.sort_values("TotalExp")
WB_ExpDF["Rank"] = range(1, len(WB_ExpDF) + 1)
WB_ExpDF["quantile"] = WB_ExpDF["Rank"] / len(WB_ExpDF)
WB_ExpDF.to_csv("../dat/MERFISH/MouseMERFISHGeneMatchQuantile.csv")
print(f"Saved expression quantiles: {WB_ExpDF.shape}")
```

**Step 2: Update notebook title and imports**

Update the notebook header to reflect expanded scope:
```
# 03. MERFISH Preprocessing
#
# Map MERFISH cells to ISH structures, compute Z1/Z2 matrices,
# and prepare expression matching quantiles.
```

Add `from CellType_PSY import *` if not already imported (needed for `Z1Conversion`).

**Step 3: Sync and commit**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType/notebooks_mouse_sc
conda run -n gencic jupytext --sync 03.MERFISH_Preprocessing.py
git add notebooks_mouse_sc/03.MERFISH_Preprocessing.py notebooks_mouse_sc/03.MERFISH_Preprocessing.ipynb
git commit -m "Move MERFISH Z1/Z2/quantile sections from notebook 01 into notebook 03"
```

---

### Task 3: Update memory

**Step 1: Update MEMORY.md**

Update the notebook status entries for 01 and 03 to reflect the rework.
