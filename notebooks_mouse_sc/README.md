# Mouse Single-Cell Analysis Pipeline

Cell-type-level analysis of ASD mutation bias using the Allen Brain Cell Atlas
(ABC) and MERFISH spatial transcriptomics. Corresponds to **Figure 5** in the
manuscript.

## Pipeline Overview

```
01.Preprocessing        → Z1/Z2 expression matrices (cluster & MERFISH structure)
02.Cell_Type_Bias       → ASD/DDD cluster-level bias + residual analysis
03.MERFISH_Preprocessing → MERFISH cell annotation + ISH structure mapping
04.MERFISH_Structure_Bias → Structure-level bias from MERFISH (4 aggregation methods)
05.Bias_Controls        → Sibling-null p-values for cell-type bias (FDR correction)
06.Bias_Display         → Spatial visualization of cell-level bias on brain sections
07.Figures              → Figure 5 panels (5A, 5B, 5F, connectivity)
```

## Notebooks

### 01. Preprocessing
Compute normalized expression matrices from Allen Brain Cell Atlas cluster-level
UMI counts and MERFISH structure-level aggregates.

- **Outputs**:
  - `dat/SC_UMI_Mats/cluster_V3_Z1Mat.clip3.csv` — Z1 normalized (clipped ±3)
  - `dat/MERFISH/STR_*_Z2Mat_ISHMatch.csv` — Z2 (ISH-matched) MERFISH matrices
  - `dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet` — Cluster Z2 matrix (16,916 genes × 5,312 clusters)
- **External dependency**: Z2 split files on `/mnt/data0/` (see `config.yaml: z2_split_base`)

### 02. Cell-Type Bias
Compute weighted ASD and DDD bias per cell-type cluster. Includes residual
analysis (neuron vs non-neuron) and permutation testing.

- **Inputs**: Cluster Z2 matrix, gene weights (DN variants)
- **Outputs**: `dat/Bias/CT_Z2/ASD_Spark61_DN.csv`, `DDD_293_ExcludeASD_DN.csv`

### 03. MERFISH Preprocessing
Documents the MERFISH cell-to-ISH-structure mapping (`Add_ISH_STR` function)
and creates a fast-loading Parquet version of the annotation.

- **Input**: `dat/MERFISH/MERFISH.ISH_Annot.csv` (1.6 GB, 3.7M cells)
- **Output**: `dat/MERFISH/MERFISH.ISH_Annot.parquet` (518 MB)

### 04. MERFISH Structure Bias
Compute ASD mutation bias at the brain structure level using four MERFISH
aggregation methods (Cell Mean, Volume Mean, Neuron Mean, Neuron Volume Mean),
and validate against ISH-derived bias.

- **Outputs**: `dat/Bias/STR/ASD.MERFISH_Allen.{CM,VM,NM,NVM}.ISHMatch.Z2.csv`
- **Validation**: Pearson r vs ISH: CM=0.75, VM=0.63, NM=0.60, NVM=0.66

### 05. Bias Controls
Documents the sibling-null permutation methodology for computing cell-type
bias p-values. The raw null data (10K CSVs) is not stored locally; the notebook
validates the existing output.

- **Output**: `dat/Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.addP.csv`
  (5,312 clusters; 178 at FDR<0.05, 327 at FDR<0.10)

### 06. Bias Display
Maps cluster-level ASD bias onto individual MERFISH cells and creates spatial
scatter plots on representative brain sections.

- **Output**: `dat/MERFISH/MERFISH.cells.ASD.Bias.Anno.parquet` (annotated cells)
- **Sections shown**: Thalamus/hippocampus (38), striatum/cortex (51),
  amygdala (36), prefrontal cortex (56)

### 07. Figures (Figure 5)
Generates manuscript Figure 5 panels:

| Panel | Description |
|-------|-------------|
| 5A | ISH vs MERFISH structure bias scatter (by brain region) |
| 5B | QQ plot of cell-type p-values + class-level boxplot |
| 5F | Circuit structure × cell-type composition/bias heatmap |
| Supp | MERFISH connectivity scoring vs null (3 distance variants) |

## Data Dependencies

All paths are relative to the project root (`/home/jw3514/Work/ASD_Circuits_CellType/`)
and defined in `config/config.yaml` under `data_files`.

### Input Data (External)

| Config Key | Path | Description |
|------------|------|-------------|
| `cluster_mean_log_umi` | `dat/SC_UMI_Mats/cluster_MeanLogUMI.parquet` | ABC cluster UMI (17,938 × 5,312) |
| `cluster_expression_quantiles` | `dat/SC_UMI_Mats/ABC_LogUMI.Match.10xV3.parquet` | Expression quantiles |
| `merfish_annotation` | `dat/MERFISH/MERFISH.ISH_Annot.csv` | MERFISH cell metadata (3.7M cells) |
| `merfish_ccf_ontology` | `dat/MERFISH/CCF_V3_ISH_MERFISH.csv` | CCF structure mapping |
| `merfish_*_umi` | `dat/MERFISH/STR_*_Mean_DF.UMI.csv` | Raw MERFISH UMI matrices |
| `asd_gene_weights_v2` | `dat/Unionize_bias/Spark_Meta_EWS.GeneWeight.v2.csv` | ASD gene weights (61 genes) |
| `asd_gene_weights_dn` | `dat/Genetics/GeneWeights_DN/...` | DN gene weight variants |
| `str_bias_fdr` | `dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv` | ISH structure bias (FDR) |
| `str_bias_raw` | `dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.csv` | ISH structure bias (raw) |
| `infomat_ipsi*` | `dat/allen-mouse-conn/ConnectomeScoringMat/...` | Connectivity scoring matrices |
| `rankscore_ipsi*` | `dat/allen-mouse-conn/RankScores/...` | Null connectivity scores |
| `asd_circuit_size46` | `results/STR_ISH/ASD.SA.Circuits.Size46.csv` | SA circuit search results |
| `mouse_ct_annotation` | `dat/MouseCT_Cluster_Anno.csv` | Cluster annotations |
| `cell_type_hierarchy` | `dat/CellTypeHierarchy.csv` | Class/subclass/supertype hierarchy |

### Intermediate Outputs (Notebook-Local)

Stored in `notebooks_mouse_sc/dat/`:

| File | Produced By | Used By |
|------|-------------|---------|
| `Bias/CT_Z2/ASD_Spark61_DN.csv` | 02 | — |
| `Bias/STR/ASD.MERFISH_Allen.*.csv` | 04 | 07 |
| `Bias/ASD.ClusterV3.top60.UMI.Z2.z1clip3.addP.csv` | 05 | 06, 07 |
| `MERFISH/MERFISH.ISH_Annot.parquet` | 03 | 06 |
| `MERFISH/MERFISH.cells.ASD.Bias.Anno.parquet` | 06 | 07 |

## Running

```bash
conda activate gencic

# Execute all notebooks in order
for nb in 01 02 03 04 05 06 07; do
    jupyter nbconvert --to notebook --execute --inplace ${nb}.*.ipynb
done
```

Note: Notebooks 01 and 02 require the Z2 split files on `/mnt/data0/` and
the DN gene weights. Notebooks 03-07 can run independently if their input
files exist.

## Notebook Conventions

- Paired with jupytext (`.py:percent` format) — edit `.py` files, then `jupytext --sync`
- First cell: `%load_ext autoreload` / `%autoreload 2`
- All data paths loaded from `config/config.yaml` (via `config['data_files']`)
- Transparent figure backgrounds (`fig.patch.set_alpha(0)`)
- Conda env: `gencic` (Python 3.10)
