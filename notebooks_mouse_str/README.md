# Structure-Level Analysis Pipeline (`notebooks_mouse_str/`)

Brain structure-level analysis of ASD mutation bias using Allen Brain Atlas
ISH expression and connectivity data. Corresponds to **Figures 1-4** and
supplementary figures in the manuscript.

## Raw Data Required

| Source | Description | Used By |
|--------|-------------|---------|
| [Allen Mouse Brain ISH](https://mouse.brain-map.org/) | In-situ hybridization expression energy (~17K genes, 213 structures) | 01 |
| [Allen Mouse Connectivity](https://connectivity.brain-map.org/) | Structural connectivity atlas (Oh et al. 2014 Nature) | 03 |
| [SPARK/ASC exome data](https://base.sfari.org/) | De novo mutations from ASD whole-exome sequencing | 04 |
| [DDD/NDD mutations](https://doi.org/10.1038/s41586-020-2832-5) | Developmental disorder de novo mutations (Kaplanis et al. 2020) | 08 |
| [gnomAD v4.0](https://gnomad.broadinstitute.org/) | Gene constraint metrics (LOEUF) | 04 |

## Intermediate Data (Provided)

These files can be produced from raw data but are provided as a download bundle
at [vitkuplab.c2b2.columbia.edu/Gencic/](https://vitkuplab.c2b2.columbia.edu/Gencic/index.html).
Once placed in `dat/`, all notebooks can run without the raw data above.

| Directory | Contents | Produced By |
|-----------|----------|-------------|
| `dat/allen-mouse-exp/ExpressionMatrix_raw.parquet` | Raw ISH expression matrix (17,210 genes x 213 structures) | Notebook 01 |
| `dat/allen-mouse-exp/Jon_ExpMat.log2.qn.csv` | Log2 + quantile-normalized expression | External (R pipeline) |
| `dat/allen-mouse-exp/ExpMatch/` | ISH expression-matched gene lists (17,189 files) | Legacy pipeline |
| `dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet` | Z2 expression specificity matrix | Notebook 02 |
| `dat/allen-mouse-conn/ConnectomeScoringMat/` | WeightMat and InfoMat (ipsilateral, 3 distance variants) | Notebook 03 |
| `dat/allen-mouse-conn/RankScores/` | Null connectivity scores from rank permutations | Notebook 03 |
| `dat/Genetics/GeneWeights/` | Gene weight files for all gene sets (.gw format) | Notebook 04 |
| `dat/Genetics/TabS_DenovoWEST_Stage1.xlsx` | DenovoWEST Stage 1 mutation table | External |
| `dat/Unionize_bias/` | Pre-computed bias with FDR, null distributions | Snakefile.bias |
| `dat/structure2region.tsv` | Structure-to-brain-region mapping (213 structures) | External |

## Running

```bash
conda activate gencic

# Core pipeline (run in order)
for nb in 01 02 03 04 05 06 07; do
    jupyter nbconvert --to notebook --execute --inplace ${nb}.*.ipynb
done

# Bias significance (requires gene weights from notebook 04)
cd .. && snakemake -s Snakefile.bias --cores 10

# Circuit search (requires bias from Snakefile.bias)
snakemake -s Snakefile.circuit --configfile config/circuit_config.yaml --cores 10

# Supplementary analyses (independent, run in any order after core pipeline)
for nb in 08 09 10 11 12 13; do
    jupyter nbconvert --to notebook --execute --inplace ${nb}.*.ipynb
done
```

## Core Pipeline (Notebooks 01-07)

| # | Notebook | Description | Key Outputs | Manuscript |
|---|----------|-------------|-------------|------------|
| 01 | Download_ISH_data | Download ISH expression data from Allen Brain Atlas API. Build raw expression matrix (arithmetic mean across sections). | `ExpressionMatrix_raw.parquet` | Methods |
| 02 | Preprocessing_ISH_data | Log2 + quantile normalization, Z1 conversion (clip +/-3), Z2 expression specificity via ISH expression matching. | `AllenMouseBrain_Z2bias.parquet` | Methods |
| 03 | Preprocessing_Connectivity_data | Process Oh et al. connectivity data into Shannon Information scoring matrices. Three distance variants (full, short <3988um, long >3988um). | `InfoMat.Ipsi.*.csv`, `WeightMat.*.csv`, `RankScore.*.npy` | Methods |
| 04 | Weighted_ASD_bias | Compute mutation-weighted expression bias per brain structure for 5 ASD gene sets (SPARK 61, SPARK 159, ASC 102, Fu 72, Fu 185). Bias scatter comparisons and CCS profiles. | Gene weight `.gw` files | Fig 1 |
| 05 | circuit_search | Run simulated annealing to find circuits maximizing ASD bias + connectivity. Pareto front analysis across circuit sizes. | Circuit structure lists | Fig 2 |
| 06 | Phenotype_Analysis | Stratify ASD cases by IQ (HIQ vs LIQ) and sex. Bootstrap resampling and permutation tests for bias profile differences. | Bootstrap/permutation results in `results/` | Fig 3 |
| 07 | Stratified_distance_analysis | Compare distance-stratified connectivity of ASD circuits vs sibling controls for top-bias structures and SA-derived circuits. | Distance analysis NPZ files | Fig 3 |

## Supplementary Analyses (Notebooks 08-13)

| # | Notebook | Description | Manuscript |
|---|----------|-------------|------------|
| 08a | DDD_Comparison | Compare DDD/NDD mutation bias with ASD. Overlap analysis between disease circuits. | Fig 4 |
| 08b | Circuit_Network | Network visualization of ASD circuit connectivity (graph layout, community structure). | Fig 4 |
| 09 | Mutation_Bootstrap | Bootstrap resampling of ASD mutations to assess circuit membership robustness. | Fig S |
| 10 | Positive_Control_Circuits | Positive control analysis using neurotransmitter system circuits (dopamine, serotonin, oxytocin, acetylcholine). | Fig S |
| 11 | Gene_Clustering | Gene clustering analysis and mutation rate examination across ASD gene sets. | Fig S |
| 12 | Comparison_Buch_et_al | Cross-species validation comparing mouse ASD circuits with Buch et al. human fMRI connectivity. | Fig S |
| 13 | fMRI_Validation | Validation of circuit predictions using mouse model fMRI data. | Fig S |

## Additional Notebooks

| Notebook | Description |
|----------|-------------|
| circuit_size_robustness | Analysis of circuit robustness across different circuit sizes. |

## Data Flow

```
Allen ISH API ──→ 01.Download ──→ ExpressionMatrix_raw.parquet
                                         │
Jon's R pipeline ──→ Jon_ExpMat.log2.qn.csv
                           │
                     02.Preprocessing ──→ Z1 ──→ Z2 (AllenMouseBrain_Z2bias.parquet)
                                                    │
Oh et al. 2014 ──→ 03.Connectivity ──→ InfoMat     │
                                          │         │
SPARK/ASC mutations ──→ 04.Bias ──→ Gene weights + Bias per structure
                                          │
                    Snakefile.bias ──→ Null distributions + FDR
                                          │
                              05.Circuit_search ──→ Pareto circuits
                                                        │
                              06.Phenotype ──→ HIQ/LIQ stratification
                              07.Distance ──→ Distance-stratified connectivity
                              08-13 ──→ Supplementary analyses
```

## Notebook Conventions

- All notebooks paired with `.py` files via [jupytext](https://jupytext.readthedocs.io/)
- Edit `.py` files, then sync: `jupytext --sync <notebook>.py`
- First cell: `%load_ext autoreload` / `%autoreload 2`
- Data paths loaded from `config/config.yaml` (not hardcoded)
- Transparent figure backgrounds for compositing
- Conda environment: `gencic` (Python 3.10)

## See Also

- `DATA_MANIFEST.yaml` (project root) for detailed file documentation
- `config/config.yaml` for all data paths and gene set definitions
- `src/ASD_Circuits.py` for core analysis functions
- `src/plot.py` for shared plotting functions
