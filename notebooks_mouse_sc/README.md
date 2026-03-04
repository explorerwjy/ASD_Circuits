# Cell-Type Analysis Pipeline (`notebooks_mouse_sc/`)

Cell-type-level analysis of ASD mutation bias using the Allen Brain Cell Atlas
(ABC) single-cell RNA-seq and MERFISH spatial transcriptomics data.
Corresponds to **Figure 5** in the manuscript.

## Raw Data Required

| Source | Description | Used By |
|--------|-------------|---------|
| [Allen Brain Cell Atlas](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas) | 10x V2/V3 single-cell RNA-seq (5,312 clusters) | 01, 02 |
| [Allen MERFISH](https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas) | Whole-brain spatial transcriptomics (~3.7M cells) | 03, 04, 06 |
| Allen CCF v3 ontology | Structure mapping between MERFISH parcellation and ISH names | 03 |

## Intermediate Data (Provided)

These files can be produced from raw data but are provided in the download bundle.
Files from the structure-level pipeline (`notebooks_mouse_str/`) are also required.

| File | Description | Produced By |
|------|-------------|-------------|
| `dat/SC_UMI_Mats/cluster_MeanLogUMI.csv` | Cluster-level mean log2(UMI+1) expression (17,938 genes x 5,312 clusters) | `scripts/build_cluster_mean_log_umi.py` |
| `dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet` | Z2 expression specificity matrix (16,916 genes x 5,312 clusters) | Notebook 01 |
| `dat/ExpressionMats/Subclass_V{2,3}_ExpMat_CPM.csv` | Subclass-level V2/V3 CPM expression matrices | `scripts/build_subclass_expmat_cpm.py` |
| `dat/allen-mouse-exp/ExpressionMatrix_raw.parquet` | ISH raw expression matrix (17,210 genes x 213 structures) | `notebooks_mouse_str/01` |
| `dat/allen-mouse-exp/ExpMatch/` | ISH expression-matched gene lists (17,189 files) | Legacy pipeline |
| `dat/allen-mouse-exp/ExpMatchFeatures.csv` | Per-gene expression features for expression matching | `notebooks_mouse_str/02` |
| `dat/MERFISH/MERFISH.ISH_Annot.csv` | MERFISH cell metadata with ISH structure annotations (~3.7M cells) | `scripts/build_merfish_annotation.py` |
| `dat/MERFISH/STR_*_Mean_DF.UMI.csv` | MERFISH structure-level UMI matrices (4 aggregation methods) | `scripts/build_merfish_umi_matrices.py` |
| `dat/MERFISH/CCF_V3_ISH_MERFISH.csv` | CCF v3 structure mapping (MERFISH to ISH names) | External |
| `dat/MouseCT_Cluster_Anno.csv` | Cluster metadata (class, subclass, supertype) | External |
| `dat/CellTypeHierarchy.csv` | Cell type hierarchy mapping | External |
| `dat/Genetics/GeneWeights/Spark_Meta_EWS.gw` | ASD ISH gene weights (61 genes) | `notebooks_mouse_str/04` |
| `dat/Genetics/GeneWeights/DDD.top293.ExcludeASD.gw` | DDD ISH gene weights (293 genes) | `notebooks_mouse_str/04` |

## Running

```bash
conda activate gencic

# Step 1: Run preprocessing and bias computation (notebooks 01-04)
for nb in 01 02 03 04; do
    jupyter nbconvert --to notebook --execute --inplace ${nb}.*.ipynb
done

# Step 2: Run sibling null bias pipeline (produces p-values for cell-type bias)
cd ..
snakemake -s Snakefile.bias --configfile config/config.SC.DN.yaml --cores 10
cd notebooks_mouse_sc

# Step 3: Run visualization and figures (notebooks 05-07)
for nb in 05 06 07; do
    jupyter nbconvert --to notebook --execute --inplace ${nb}.*.ipynb
done
```

**Dependencies between steps:**
- Notebooks 01-04 can run independently if their input files exist
- Notebooks 05-07 require `Snakefile.bias` output (`results/CT_Z2/*_bias_addP_*.csv`)
- Notebook 06 requires notebook 03 output (MERFISH parquet)
- Notebook 07 requires outputs from notebooks 04, 06, and the Snakefile.bias pipeline

## Pipeline Overview

```
                  Allen Brain Cell Atlas
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
  h5ad files       MERFISH cells        V2/V3 chemistry
    │                    │                    │
build_cluster_       build_merfish_      build_subclass_
mean_log_umi.py     annotation.py       expmat_cpm.py
    │                    │                    │
    ▼                    ▼                    ▼
01.Preprocessing    03.MERFISH_Prep     V2/V3 CPM matrices
  (Z1 → Z2)          (Z1 → Z2)              │
    │                    │                    │
    ▼                    ▼                    ▼
02.Cell_Type_Bias   04.MERFISH_STR    V2-V3 Spearman corr
  (DN weights)        _Bias              → DN weights
    │                    │                    │
    └────────┬───────────┘                    │
             │                                │
      Snakefile.bias  ←───── DN gene weights ─┘
      (10K permutations)
             │
    ┌────────┼────────┐
    ▼        ▼        ▼
05.Bias   06.Bias   07.Figures
Controls  Display   (Fig 5)
```

## Notebooks

### 01. Preprocessing
Compute Z2 expression specificity from cluster-level UMI counts: load cluster
mean log UMI → Z1 conversion (clip +/-3) → Z2 via ISH expression matching
(parallel, joblib).

- **Input**: `dat/SC_UMI_Mats/cluster_MeanLogUMI.csv`, `dat/allen-mouse-exp/ExpMatch/`
- **Output**: `dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet`

### 02. Cell-Type Bias
Compute V2-V3 Spearman correlation per gene → derive DN (de novo) gene weights
(`weight_DN = weight_ISH * max(spearman_r, 0)^2`) → compute ASD and DDD bias
per cell-type cluster → residual analysis (neuron vs non-neuron) → permutation
testing (1,000 permutations).

- **Input**: Cluster Z2 matrix, V2/V3 CPM matrices, ISH gene weights
- **Output**: DN gene weights (`.gw`), bias CSVs, cached correlations

### 03. MERFISH Preprocessing
Map MERFISH cell parcellation to ISH structure names via CCF ontology. Compute
MERFISH structure-level Z1 matrices (4 aggregation methods x 2 datasets), then
Z2 via ISH expression matching with quantile alignment.

- **Input**: MERFISH UMI matrices, CCF mapping, expression match files
- **Output**: Z1 and Z2 matrices for Allen and Zhuang MERFISH datasets

### 04. MERFISH Structure Bias
Compute structure-level ASD bias from 4 MERFISH Z2 matrices (Cell Mean, Volume
Mean, Neuron Mean, Neuron Volume Mean). Validate against ISH-derived bias.

- **Input**: MERFISH Z2 matrices, DN gene weights
- **Output**: Structure bias CSVs (r=0.75 Cell Mean vs ISH)

### 05. Bias Controls
Visualize sibling-null permutation results for cell-type bias. QQ plot with
7 highlighted cell classes and FDR 0.05/0.10 threshold lines.

- **Input**: `results/CT_Z2/ASD_All_bias_addP_sibling.csv` (from Snakefile.bias)
- **Output**: Visualization only

### 06. Bias Display
Map cluster-level bias onto individual MERFISH cells. Spatial scatter plots on
4 representative brain sections plus regional violin plots.

- **Input**: MERFISH cell annotations, cell-type bias with p-values
- **Output**: `MERFISH.cells.ASD.Bias.Anno.parquet` (annotated cells, cached)

### 07. Figures (Figure 5)
Generate manuscript Figure 5 panels:

| Panel | Description |
|-------|-------------|
| 5A | ISH vs MERFISH structure bias scatter (region-colored, r=0.75) |
| 5B | QQ plot of cell-type p-values + class-level boxplot |
| 5C | Structure x cell-type dotplot (bias color + composition size) |
| Supp | Subclass-level bias boxplots, MERFISH connectivity scoring |

- **Input**: ISH bias, MERFISH bias, circuit structures, annotated cells, InfoMat
- **Output**: Figure PDFs in `results/figs/`

## Snakefile.bias Pipeline (CT_Z2)

Runs between notebooks 04 and 05. Computes sibling-null p-values for cell-type
bias via 10,000 permutations.

```bash
snakemake -s Snakefile.bias --configfile config/config.SC.DN.yaml --cores 10
```

**Config**: `config/config.SC.DN.yaml` (ASD_All + DDD_293_ExcludeASD, mutability null model)

**Outputs** (in `results/CT_Z2/`):
- `ASD_All_bias_addP_sibling.csv` — ASD bias with sibling-null p-values (149 at FDR<0.05)
- `ASD_All_bias_addP_random.csv` — ASD bias with random-null p-values
- `DDD_293_ExcludeASD_bias_addP_sibling.csv` — DDD bias with sibling-null p-values
- `DDD_293_ExcludeASD_bias_addP_random.csv` — DDD bias with random-null p-values

## Notebook Conventions

- All notebooks paired with `.py` files via [jupytext](https://jupytext.readthedocs.io/)
- Edit `.py` files, then sync: `jupytext --sync <notebook>.py`
- First cell: `%load_ext autoreload` / `%autoreload 2`
- Data paths loaded from `config/config.yaml` (not hardcoded)
- Transparent figure backgrounds for compositing
- Conda environment: `gencic` (Python 3.10)

## See Also

- `DATA_MANIFEST.yaml` (this directory) for detailed file-by-file documentation
- `config/config.yaml` for all data paths and gene set definitions
- `config/config.SC.DN.yaml` for cell-type bias pipeline config
- `src/CellType_PSY.py` for cell-type analysis functions
- `src/ASD_Circuits.py` for core analysis functions
