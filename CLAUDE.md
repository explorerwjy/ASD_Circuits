# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GENCIC (Genetics and Expression iNtegration with Connectome to Infer affected Circuits) is a computational framework for identifying brain circuits preferentially targeted by autism spectrum disorder (ASD) mutations. The project integrates genomic data from ASD patients with spatial transcriptomics and brain-wide connectome data to discover functionally cohesive neural circuits affected in autism.

## Development Environment

### Conda Environment
```bash
conda activate gencic    # Python 3.10
```
Always activate `gencic` before running any command (python, snakemake, pip, etc.).

### Core Dependencies
- Python 3.10 (conda env: `gencic`)
- Key packages: numpy, pandas, scipy, matplotlib, seaborn, statsmodels, igraph, anndata
- Snakemake for workflow management
- jupytext + nbstripout for notebook version control

### Notebook Version Control
This project uses **jupytext** to pair `.ipynb` notebooks with `.py` files and **nbstripout** to strip outputs from `.ipynb` before committing. Both files are tracked in git.

- `.gitattributes` enables nbstripout: `*.ipynb filter=nbstripout`
- Edit either the `.ipynb` in Jupyter or the `.py` in any editor
- Sync manually after editing `.py` outside Jupyter: `jupytext --sync <file>.py`
- **Never use NotebookEdit** to modify `.ipynb` cells — edit the paired `.py` file instead, then sync
- **Always include autoreload** as the first code cell in every notebook

### Git Configuration
- The `.gitignore` uses a whitelist approach: ignores everything except `*.py`, `*.ipynb`, `.gitignore`
- Data files, results, and configs are NOT tracked — only code files
- nbstripout ensures `.ipynb` outputs are stripped on commit

## Snakemake Workflows

### Bias Calculation (`Snakefile.bias`)
```bash
snakemake -s Snakefile.bias --cores 10
```
Pipeline: generate random gene weights → compute null distributions → calculate bias with p-values.

### Circuit Search (`Snakefile.circuit`)
```bash
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10
```
Pipeline: generate bias limits → run simulated annealing → create Pareto fronts → extract best circuits.

### Bootstrap Analysis (`Snakefile.circuit.bootstrap`)
```bash
snakemake -s Snakefile.circuit.bootstrap --configfile config/circuit_config_bootstrap.yaml --cores 10
```
Bootstrap resampling for circuit robustness assessment.

### Sibling Control Analysis (`Snakefile.circuit.sibling`)
```bash
snakemake -s Snakefile.circuit.sibling --configfile config/circuit_config_sibling_random.yaml --cores 10
```

## Architecture

### Source Modules (`src/`)
- **ASD_Circuits.py**: Main analysis library — bias calculation, gene weights, circuit scoring, bootstrap, neurotransmitter analysis. This is the primary module imported by all notebooks.
- **ASD_Circuits.bkup.py**: Legacy version of the library. If a function is referenced in notebooks but missing from ASD_Circuits.py, check here and move it over.
- **SA.py**: Simulated annealing base implementation for circuit optimization.
- **SA_optimized.py**: Numba-accelerated SA variant.
- **CellType_PSY.py**: Cell type-specific analysis functions (builds on ASD_Circuits.py). Used primarily by `notebooks_mouse_sc/`.
- **CellType_PSY.bkup.py**: Legacy version of CellType_PSY.
- **plot.py**: Shared visualization functions.

### Key Functions in `ASD_Circuits.py`
- `MouseSTR_AvgZ_Weighted(ExpZscoreMat, Gene2Weights)` — structure-level weighted bias
- `MouseCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights)` — cell type-level weighted bias
- `HumanCT_AvgZ_Weighted(ExpZscoreMat, Gene2Weights)` — human cell type bias
- `SPARK_Gene_Weights(MutFil)` — compute mutation-derived gene weights from SPARK data
- `bootstrap_gene_mutations(df, n_boot, weighted)` — bootstrap mutations preserving gene identity
- `GetPermutationP_vectorized(null_matrix, observed_vals)` — vectorized permutation p-values
- `ScoreCircuit_SI_Joint(STRs, InfoMat)` — circuit connectivity score using Shannon Information
- `Fil2Dict(fil_)` / `Dict2Fil(dict_, fil_)` — load/save gene weight files (.gw format)
- `LoadGeneINFO()` — load HGNC gene annotations, returns (HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol)
- `STR2Region()` — mapping from brain structure names to major brain divisions

### Scripts (`scripts/`)
- **script_bias_cal.py**: Calculate bias with p-values from null distributions
- **script_run_ctrl_sim.py**: Generate null distributions (sibling & random)
- **script_generate_geneweights.py**: Generate random gene weight permutations
- **script_circuit_search.SI.py**: Circuit search using simulated annealing
- **script_permutation_bias_comparison.py**: Permutation testing for dataset comparisons
- **scripts/workflow/**: Snakemake helper scripts (SA search, Pareto fronts, bias limits, bootstrap aggregation)

### Notebook Organization
- **notebooks_mouse_str/**: Brain structure-level analyses (preprocessing 01–04, circuit search 05, phenotype analysis 06–07)
- **notebooks_mouse_sc/**: Cell type-level analyses using Allen Brain Cell Atlas and MERFISH spatial transcriptomics
- **notebooks_figures/**: Publication figure and table generation (Figures_Tables.ipynb, Tables.ipynb)
- **notebook_rebuttal/**: Reviewer response analyses:
  - `DDD.ipynb` — DDD/NDD disease comparison with bootstrap
  - `NumberOfASDGenes.ipynb` — gene set size stability (expanding gene sets)
  - `Mut_Bootstrap.ipynb` — mutation bootstrapping analysis
  - `PositiveCircuits.ipynb` — positive control circuit analysis
  - `GeneClustering.ipynb` — gene clustering and mutation rate analysis
  - `Gencic_vs_Buch_et_al_CLEAN.ipynb` — comparison with Buch et al.
  - `Test_Vlidation_fMRI.ipynb` — fMRI validation
- **notebooks_other/**: Miscellaneous exploratory analyses

### Configuration (`config/`)
- **config.yaml**: Main config — defines gene sets (70+), expression matrices, output paths, null distribution parameters
- **circuit_config.yaml**: Circuit search parameters (sizes, SA runtime, steps, measures)
- **circuit_config_bootstrap.yaml**: Bootstrap circuit search config
- **circuit_config_sibling_*.yaml**: Sibling control circuit configs

### Data Flow
1. **Gene Weights**: `.gw` files mapping Entrez gene IDs to mutation-derived weights (`dat/Genetics/GeneWeights/`)
2. **Expression Matrices** (parquet):
   - `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` — structure-level Z2 bias (genes × 213 structures)
   - `dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet` — cell type Z2 bias
   - `dat/BiasMatrices/MouseBrain_MERIFHbias.parquet` — MERFISH spatial transcriptomics
3. **Connectivity**: `dat/allen-mouse-conn/ConnectomeScoringMat/` (WeightMat, InfoMat for ipsilateral connections)
4. **Analysis Pipeline**: Snakemake coordinates bias calculation → null distributions → p-values → circuit search
5. **Output**: `results/` directory (not tracked in git)

### Gene Set Types in `config.yaml`
- **ASD cohorts**: ASD_HIQ, ASD_LIQ, ASD_SPARK_159, Spark_top20, Fu_ASD_72/185, ASC_102
- **NDD**: DDD_61, DDD_293, DDD_293_ExcludeASD
- **Schizophrenia**: SCZ
- **Constraint**: Constraint_top25_LOEUF
- **Neurotransmitter systems**: Dopamine/Serotonin/Oxytocin/Acetylcholine (source/target/combined)
- **Negative controls**: HDL_C, IBD, T2D, hba1c (non-brain traits)
- **Neurodegeneration**: Parkinson, Alzheimer

## Reproducibility Rules (Pre-Submission Cleanup)

These rules govern the codebase cleanup for paper resubmission. Apply them when reworking any notebook or script.

### 1. Fixed Random Seeds
- **Every** use of randomness must have a deterministic seed. No unseeded `np.random` calls.
- Use `np.random.default_rng(seed)` (preferred) or `np.random.seed(seed)` at the top of any stochastic block.
- Default project seed: **`42`** for notebooks. Scripts that run many independent jobs use `base_seed + job_index`.
- Seeds must be visible in code (not buried in library defaults). If calling a function that uses randomness internally, the function must accept a `seed` parameter.

### 2. No Legacy Paths
- **All data must be read from `./dat/` or `./results/`** relative to the project root. Never reference:
  - `/home/jw3514/Work/ASD_Circuits/` (the old project)
  - `/mnt/data0/` (backup mounts)
  - Any other absolute path outside this project
- If data currently lives in a legacy location, copy it into `dat/` first, then update the reference.
- Use `ProjDIR` + relative paths or `../dat/` from notebook directories. Never hardcode absolute paths to data.

### 3. Config-Driven Data Loading
- All input data files must be registered in **`config/config.yaml`** (for pipeline inputs) or **`config/supplementary_tables.yaml`** (for publication tables).
- Notebooks should load data paths from config, not hardcode them. Standard pattern:
  ```python
  with open("../config/config.yaml") as f:
      config = yaml.safe_load(f)
  STR_BiasMat = pd.read_parquet(f"../{config['analysis_types']['STR_ISH']['expr_matrix']}")
  ```
- When adding a new data dependency, add it to the relevant config file first.

### 4. Results From `./results/` Only
- Any computed result used in a publication figure or table **must** come from `./results/`, produced by a reproducible pipeline (Snakefile or notebook).
- Never read intermediate results from `dat/Unionize_bias/` or other ad-hoc locations for publication outputs. Those files are acceptable as raw/input data only.
- The flow is: `dat/` (raw input) → pipeline → `results/` (computed output) → notebooks read from `results/`.
- Exception: static reference data (expression matrices, annotations, connectome) stays in `dat/` since it is not computed by this project.

### 5. Notebook Standards (Applied During Rework)
- **One notebook = one clear purpose.** Split exploratory/dead code into separate notebooks or remove it.
- **No dead cells**: remove commented-out blocks, empty cells, inspection-only cells (`df.head()`, `print(len(x))`).
- **No inline function definitions** that duplicate `src/` functions. Import from `src/plot.py` or `src/ASD_Circuits.py`.
- **No hardcoded magic numbers**: computed values (correlations, p-values, gene counts) must be derived from data, not typed as literals.
- **Markdown section headers** for every logical section (e.g., `# Section 1: ...`, `## 1.1 ...`).
- **Shared variables defined once**: cluster dictionaries, palettes, and pairwise test lists defined in one cell, reused throughout.
- **Cache expensive computations**: permutation loops, bootstrap runs, etc. should save/load from `results/cache/` with a clear cache key.
- **Figure saving**: all publication figures saved to `results/figures/` with `transparent=True, dpi=300, bbox_inches='tight'`.

### 6. Notebook Rework Checklist
When reworking a notebook, apply every item:
- [ ] Pair with jupytext (`.py:percent` format)
- [ ] First cell: `%load_ext autoreload` / `%autoreload 2`
- [ ] All data paths relative or from config (no legacy paths)
- [ ] All results read from `./results/`
- [ ] Random seeds set for all stochastic operations
- [ ] Dead cells removed
- [ ] Inline functions moved to `src/` if reusable
- [ ] Hardcoded values replaced with computed values
- [ ] Markdown section headers added
- [ ] Runs end-to-end via `jupyter nbconvert --execute`
- [ ] Committed (both `.py` and `.ipynb`)

### 7. Function Extraction Convention
- Reusable **plotting functions** → `src/plot.py`
- Reusable **analysis/data functions** → `src/ASD_Circuits.py`
- Don't extract one-off notebook-specific code — only functions used by 2+ notebooks or that encapsulate complex logic.

## Important Notes

- The pipeline uses parallel processing — default to `n_processes=10` to avoid hogging resources
- This machine has 40 CPU cores and 2 GPUs
- When plotting, use transparent backgrounds (`facecolor='none'`, `transparent=True` in savefig)
- Gene weight `.gw` files are CSV format: `EntrezID,weight` (no header)
- Structure bias DataFrames have columns: EFFECT, Rank, REGION (indexed by structure name)
- Cell type bias DataFrames have columns: EFFECT, Rank (indexed by cell type cluster ID)
