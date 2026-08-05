# Codebase Rework Plan — Pre-Submission Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the entire codebase reproducible with fixed seeds, no legacy paths, config-driven data loading, and clean notebooks — ready for paper resubmission.

**Architecture:** Work in pipeline order (raw data preprocessing → bias calculation → circuit search → cell types → cross-disorder → figures). Each notebook gets paired with jupytext, cleaned, and verified to run end-to-end. Shared functions extracted to `src/`. Legacy absolute paths replaced with relative paths.

**Tech Stack:** Python 3.10 (conda env: `gencic`), jupytext, nbstripout, numpy, pandas, matplotlib

---

## Current State Summary

| Category | Count |
|----------|-------|
| Total notebooks | 77 |
| Paired with .py | 10 |
| Legacy path occurrences | 17 in 9 files |
| Unseeded RNG calls | ~20+ across scripts/notebooks |

### Notebooks Contributing to the Paper

**Core Pipeline (notebooks_mouse_str/)** — Figures 2-4, Supp Figs 2-6, 8-10:
| Notebook | Lines | Paired | Legacy Paths | Figures |
|----------|-------|--------|--------------|---------|
| 01.Download_ISH_data | 514 | No | ? | Data prep |
| 02.Preprocessing_ISH_data | 553 | No | ? | Data prep |
| 03.Preprocessing_Connectivity_data | 1916 | No | ? | Data prep |
| 04.Weighted_ASD_bias | 1193 | No | ? | Fig 2, Supp Fig 2-4 |
| 05.circuit_search | 1465 | Yes | 2 | Fig 3-4, Supp Fig 6, 8-10 |
| 06.Phenotype_Analysis | 2314 | No | ? | Fig 6a, Supp Fig 15 |
| 07.Stratified_distance_analysis | 1409 | No | ? | Fig 3b-c, Supp Fig 5 |

**Cell Type (notebooks_mouse_sc/)** — Figure 5, Supp Figs 11-13:
| Notebook | Purpose | Figures |
|----------|---------|---------|
| Preprocessing.ipynb | ABC atlas preprocessing | Data prep |
| MakeClusterExpressionMat.ipynb | Build expression matrix | Data prep |
| ABC_Bias_Cal.ipynb | Cell type bias calculation | Fig 5a |
| MERFISH_PreProcessing.ipynb | MERFISH preprocessing | Supp Fig 13 |
| ASD_SC_MERFISH_Bias.ipynb | MERFISH bias | Fig 5b |
| STR_CellCompo.ipynb | Cell composition | Fig 5c, Table S4 |
| Figures.ipynb | Cell type figures | Fig 5, Supp Fig 11-12 |

**Rebuttal (notebook_rebuttal/)** — Supp Figs 7, 9, 14-18:
| Notebook | Paired | Legacy Paths | Figures |
|----------|--------|--------------|---------|
| DDD.ipynb | Yes | 2 | Fig 6b-c, Supp Fig 16-17 |
| Mut_Bootstrap.ipynb | Yes | 0 | Supp Fig 9 |
| PositiveCircuits.ipynb | Yes | 2 | Supp Fig 7, 18 |
| GeneClustering.ipynb | Yes | 0 | Supp Fig 15 |
| Test_Vlidation_fMRI.ipynb | Yes | 2 | Supp Fig 14 |
| NumberOfASDGenes.ipynb | No | ? | Supp Fig 4 |
| Gencic_vs_Buch_et_al_CLEAN.ipynb | No | ? | Supp Fig 14b |
| ConstraintPermutation.ipynb | Yes | 0 | Supp Fig 16c |

**Figures (notebooks_figures/)** — All main/supp figures:
| Notebook | Paired | Legacy Paths | Figures |
|----------|--------|--------------|---------|
| Figures_Tables.ipynb | Yes | 5 | Figs 2-4, Supp 2-6, 10 |
| Tables.ipynb | Yes | 1 | (was Table S1-S6) |
| SupplementaryTables.ipynb | Yes | 0 | Tables S1-S14 |

**Source Modules (src/):**
| File | Legacy Paths | Unseeded RNG |
|------|--------------|--------------|
| ASD_Circuits.py | 0 | 2 (SA functions) |
| SA.py | 0 | 2 (random.random) |
| SA_optimized.py | 0 | 2 (np.random.choice) |
| ASD_Circuits.bkup.py | 2 | - |
| plot.py | 0 | 0 |

---

## Phase 0: Source Modules & Scripts

Fix the foundation before touching notebooks.

### Task 0.1: Fix legacy paths in source modules

**Files:**
- Modify: `src/ASD_Circuits.bkup.py` (lines 38, 270)
- Modify: `scripts/script_cohesiveness_profile.py` (line 10)

**Steps:**
1. In `ASD_Circuits.bkup.py`, replace absolute paths with relative paths or paths derived from `__file__`
2. In `script_cohesiveness_profile.py`, replace old project path with current project path
3. Verify: `python -c "import sys; sys.path.insert(0,'src'); import ASD_Circuits"` runs without error

### Task 0.2: Add seed parameters to SA modules

**Files:**
- Modify: `src/SA.py` (lines 219, 268)
- Modify: `src/SA_optimized.py` (lines 121-122, 222-223)

**Steps:**
1. Add `seed` parameter to SA search functions
2. Initialize `rng = np.random.default_rng(seed)` at function entry
3. Replace `random.random()` → `rng.random()` and `np.random.choice()` → `rng.choice()`
4. Ensure backward compatibility (default `seed=None` preserves old behavior)
5. Verify: existing SA scripts still work with `seed=42`

### Task 0.3: Add seed parameters to ASD_Circuits.py SA functions

**Files:**
- Modify: `src/ASD_Circuits.py` (lines 837-838, 865-866)

**Steps:**
1. SA functions in ASD_Circuits.py that call `np.random.choice` — add `seed` parameter
2. Same pattern as Task 0.2

### Task 0.4: Seed script_generate_geneweights.py

**Files:**
- Modify: `scripts/script_generate_geneweights.py` (lines 116, 120, 156, 160)

**Steps:**
1. Accept `--seed` argument (default: use job index as seed component)
2. Call `np.random.seed(base_seed + job_index)` at start of each job
3. Verify: running with same seed produces identical output

---

## Phase 1: Core Pipeline (notebooks_mouse_str/ 01-07)

These notebooks form the data preparation and main analysis pipeline. Rework in order since each depends on outputs from previous ones.

### Task 1.1: Rework 01.Download_ISH_data.ipynb

**Files:**
- Modify: `notebooks_mouse_str/01.Download_ISH_data.ipynb`
- Create: `notebooks_mouse_str/01.Download_ISH_data.py` (jupytext pair)

**Steps:**
1. Create .py pair: `jupytext --set-formats ipynb,py:percent notebooks_mouse_str/01.Download_ISH_data.ipynb`
2. Apply rework checklist (CLAUDE.md §6): autoreload, relative paths, remove dead cells, add section headers
3. This notebook downloads data from Allen Brain Atlas API — ensure all download paths are relative to project
4. Verify: `jupytext --sync notebooks_mouse_str/01.Download_ISH_data.py`
5. Commit

### Task 1.2: Rework 02.Preprocessing_ISH_data.ipynb

**Steps:** Same as 1.1 but for notebook 02.
- Key: This produces expression matrices. Ensure output goes to `dat/` or `results/` with clear paths.
- Check for random operations (expression sampling, Z-score normalization) — add seeds.

### Task 1.3: Rework 03.Preprocessing_Connectivity_data.ipynb

**Steps:** Same pattern.
- Key: This produces connectivity matrices. Large notebook (1916 lines) — likely has dead cells to remove.
- Produces: `dat/allen-mouse-conn/ConnectomeScoringMat/` files

### Task 1.4: Rework 04.Weighted_ASD_bias.ipynb

**Steps:** Same pattern.
- Key: Main bias calculation notebook. Produces structure-level biases.
- Known duplicates: re-defines `plot_structure_bias_correlation`, `plot_circuit_connectivity_scores_multi` (from src/plot.py)
- Replace inline function definitions with imports from `src/plot.py`
- Check for sibling null generation — ensure seeds
- Verify output matches `dat/Unionize_bias/Spark_Meta_EWS.Z2.bias.FDR.csv`

### Task 1.5: Rework 05.circuit_search.ipynb (already paired)

**Files:**
- Modify: `notebooks_mouse_str/05.circuit_search.py`

**Steps:**
1. Fix 2 legacy paths (lines 478-479: SIB_SA_DIR, SIB_BIAS_DIR)
2. Copy sibling data to local `dat/` or `results/` if not already present
3. Add seeds to SA calls
4. Remove TODO items already resolved
5. Sync and verify

### Task 1.6: Rework 06.Phenotype_Analysis_seperating_HIQ_LIQ.ipynb

**Steps:** Same pairing pattern.
- Key: IQ stratification analysis. Produces Fig 6a, Supp Fig 15.
- Large notebook (2314 lines) — likely significant cleanup needed.
- Check for IQ permutation test — ensure seed.

### Task 1.7: Rework 07.Stratified_distance_analysis.ipynb

**Steps:** Same pairing pattern.
- Key: Distance analysis for Pareto fronts. Produces Fig 3b-c.
- Check for random operations in distance binning/significance tests.

---

## Phase 2: Cell Type Pipeline (notebooks_mouse_sc/)

### Task 2.0: Identify core vs. exploratory notebooks

**Steps:**
1. Read each notebook's first few cells to understand purpose
2. Classify as: CORE (needed for paper), ARCHIVE (legacy/exploratory), KEEP (useful but not for paper)
3. Move ARCHIVE notebooks to `notebooks_mouse_sc/archive/`
4. List CORE notebooks in order of dependency

**Expected core notebooks (based on figure mapping):**
1. `Preprocessing.ipynb` — ABC atlas preprocessing
2. `MakeClusterExpressionMat.ipynb` — Build cluster expression matrix
3. `Cal_AvgExp_Cluster.ipynb` — Calculate average expression per cluster
4. `ABC_Bias_Cal.ipynb` — Cell type bias calculation
5. `MERFISH_PreProcessing.ipynb` — MERFISH spatial data
6. `ASD_SC_MERFISH_Bias.ipynb` — MERFISH bias scores
7. `STR_CellCompo.ipynb` — Cell composition analysis
8. `Figures.ipynb` — Publication figures (Fig 5, Supp Fig 11-12)

### Tasks 2.1-2.8: Rework each core notebook

Same pattern as Phase 1: pair with jupytext, apply rework checklist, fix paths, add seeds, remove dead cells, commit.

---

## Phase 3: Rebuttal Notebooks (notebook_rebuttal/)

Most are already paired. Focus on legacy paths, seeds, and remaining cleanup.

### Task 3.1: Rework DDD.ipynb

**Files:** `notebook_rebuttal/DDD.py` (already paired, 670 lines)

**Steps:**
1. Fix 2 legacy paths (lines 53, 57)
2. Copy needed data: `ScoreingMat_jw_v3/`, `RankScores/` → local `dat/`
3. Apply existing DDD rework plan (docs/plans/harmonic-tumbling-scroll.md) if still relevant:
   - Move utility functions to src/
   - De-duplicate code
   - Remove dead cells
   - Cache permutation loop
4. Add seeds to permutation test
5. Verify end-to-end

### Task 3.2: Rework PositiveCircuits.ipynb

**Files:** `notebook_rebuttal/PositiveCircuits.py` (already paired, 458 lines)

**Steps:**
1. Fix 2 legacy paths (lines 170, 177)
2. Copy `ScoreingMat_jw_v3/` and `RankScores/` data locally
3. Add seeds to any shuffling/permutation operations
4. Verify end-to-end

### Task 3.3: Rework Test_Vlidation_fMRI.ipynb

**Files:** `notebook_rebuttal/Test_Vlidation_fMRI.py` (already paired, 1955 lines)

**Steps:**
1. Fix 2 legacy paths (lines 174, 1144)
2. Large notebook — identify dead/exploratory cells
3. Add seeds to permutation tests (lines 67-68: `np.random.choice` without seed)
4. Verify end-to-end

### Task 3.4: Rework NumberOfASDGenes.ipynb

**Files:** `notebook_rebuttal/NumberOfASDGenes.ipynb` (NOT paired)

**Steps:**
1. Pair with jupytext
2. Replace duplicated functions: `plot_structure_bias_correlation`, `plot_circuit_connectivity_scores_multi`, `plot_correlation_profile_together` → import from `src/plot.py`
3. Apply full rework checklist
4. Commit

### Task 3.5: Rework Gencic_vs_Buch_et_al_CLEAN.ipynb

**Files:** `notebook_rebuttal/Gencic_vs_Buch_et_al_CLEAN.ipynb` (NOT paired)

**Steps:**
1. Pair with jupytext
2. Check for legacy paths
3. Apply rework checklist
4. Commit

### Task 3.6: Review remaining rebuttal notebooks

- **Mut_Bootstrap.ipynb** — already trimmed (Feb 2026), has seeds. Quick review.
- **GeneClustering.ipynb** — already paired, has seeds (line 573-574). Quick review.
- **ConstraintPermutation.ipynb** — already paired. Quick review.

---

## Phase 4: Figure Notebooks (notebooks_figures/)

### Task 4.1: Rework Figures_Tables.ipynb

**Files:** `notebooks_figures/Figures_Tables.py` (paired, 1452 lines)

**Steps:**
1. Fix 5 legacy paths (lines 680, 978, 1399, 1401, 1404)
2. Copy needed data locally:
   - `RankScores/` → `dat/` or `results/`
   - `ASD.SA.Circuits.Size46.csv` → `results/`
   - `ScoreingMat_jw_v3/WeightMat.Ipsi.csv` → already in `dat/allen-mouse-conn/ConnectomeScoringMat/`?
   - `Dist_CartesianDistance.ipsi.csv` → `dat/allen-mouse-conn/`
3. Add seeds to bootstrap (lines 1038-1039, 1071)
4. Fix undefined variables in Fig5/supp cells (`adj_mat`, `graph`, `ExpZ2`, `ExpL`) — either add the loading code or split those cells to appropriate notebooks
5. Verify end-to-end (at least for cells that have all dependencies)

### Task 4.2: Review Tables.ipynb

**Files:** `notebooks_figures/Tables.py` (paired, 338 lines)

**Steps:**
1. Fix 1 legacy path (line 309)
2. Determine if this notebook is still needed now that SupplementaryTables.ipynb exists
3. If redundant, mark as deprecated

### Task 4.3: Review SupplementaryTables.ipynb

**Files:** `notebooks_figures/SupplementaryTables.py` (paired, 281 lines)
- Recently created, should be clean. Quick review only.

---

## Phase 5: Cleanup & Archive

### Task 5.1: Archive exploratory notebooks

**Steps:**
1. Move non-core notebooks in `notebooks_mouse_str/` to `notebooks_mouse_str/archive/`:
   - CircuitsInformationScore.ipynb, Connectome.ipynb, debug_SA.ipynb, GENCIC.ipynb, ipsi_vs_contra_weights.ipynb, NoteBook_*.ipynb (all 7), NoteBooks_topNvsCohesiveness.ipynb, Optimized_Circuits_*.ipynb, Phenotype_Graph_New.ipynb, Preprocessing.ipynb, SA.ipynb, Sibling_SI_score_significance.ipynb, Test_Vlidation_fMRI.ipynb, WeighedBias.ipynb
2. Same for non-core notebooks in `notebooks_mouse_sc/`
3. Commit

### Task 5.2: Delete backup source files

**Steps:**
1. Verify `src/ASD_Circuits.bkup.py` and `src/CellType_PSY.bkup.py` have no unique functions still needed
2. If all needed functions are in the main files, delete backups
3. Commit

### Task 5.3: Final verification

**Steps:**
1. Run `grep -r "/home/jw3514/Work/ASD_Circuits/" --include="*.py" .` — should return 0 results
2. Run `grep -rn "np\.random\.\(choice\|shuffle\|randint\|seed\)" --include="*.py" .` — verify all have seeds
3. Verify all paired notebooks sync: `jupytext --sync` on each
4. Run key notebooks end-to-end (at least Figures_Tables, SupplementaryTables, DDD)

---

## Rework Order (Recommended)

Start from Phase 0 (source modules), then work through each phase in order. Within each phase, process notebooks in dependency order. Each notebook rework is a self-contained session:

1. **Phase 0**: src/ and scripts (Tasks 0.1-0.4) — 1 session
2. **Phase 1**: Core pipeline 01-07 (Tasks 1.1-1.7) — 2-3 sessions
3. **Phase 2**: Cell type notebooks (Tasks 2.0-2.8) — 2-3 sessions
4. **Phase 3**: Rebuttal notebooks (Tasks 3.1-3.6) — 2 sessions
5. **Phase 4**: Figure notebooks (Tasks 4.1-4.3) — 1 session
6. **Phase 5**: Cleanup (Tasks 5.1-5.3) — 1 session

**Total: ~20 tasks across ~10 sessions**

---

## Common Data to Copy from Legacy Project

These files are referenced from `/home/jw3514/Work/ASD_Circuits/` and need to be copied to the current project:

| Legacy Path | Copy To | Used By |
|-------------|---------|---------|
| `scripts/RankScores/RankScore.Ipsi.*.npy` | `dat/RankScores/` or `results/RankScores/` | Figures_Tables, PositiveCircuits, DDD |
| `notebooks/ASD.SA.Circuits.Size46.csv` | `results/ASD.SA.Circuits.Size46.csv` | Figures_Tables, Tables, SupplementaryTables |
| `dat/allen-mouse-conn/ScoreingMat_jw_v3/WeightMat.Ipsi.csv` | Already in `dat/allen-mouse-conn/ConnectomeScoringMat/`? Verify | Figures_Tables, DDD, PositiveCircuits |
| `dat/allen-mouse-conn/Dist_CartesianDistance.ipsi.csv` | `dat/allen-mouse-conn/` | Figures_Tables |
| `dat/Other/ontology.csv` | `dat/Other/ontology.csv` | Test_Vlidation_fMRI |
| `dat/Unionize_bias/SubSampleSib/` | `dat/Unionize_bias/SubSampleSib/` or `results/` | 05.circuit_search, Figures_Tables |

**Action:** Before starting Phase 1, verify which files already exist locally and copy any missing ones.

---

## Standard Rework Procedure (for each notebook)

```
1. READ notebook to understand purpose and current state
2. PAIR with jupytext (if not already): jupytext --set-formats ipynb,py:percent <name>.ipynb
3. EDIT the .py file:
   a. First cell: %load_ext autoreload / %autoreload 2
   b. Replace all legacy absolute paths with relative paths
   c. Replace data loading with config-driven paths where appropriate
   d. Add np.random.seed(42) or rng = np.random.default_rng(42) before stochastic blocks
   e. Remove dead cells (empty, inspection-only, commented-out)
   f. Replace inline function definitions with imports from src/
   g. Replace hardcoded magic numbers with computed values
   h. Add markdown section headers
   i. Ensure all figure saves use transparent=True, dpi=300
4. SYNC: jupytext --sync <name>.py
5. VERIFY: jupyter nbconvert --to notebook --execute <name>.ipynb (or run key cells)
6. COMMIT both .py and .ipynb
```
