# GENCIC Webapp Pipeline Implementation Design

**Date:** 2026-03-04
**Status:** Approved

## Overview

Implement the full GENCIC pipeline as an interactive Streamlit webapp: gene input → bias computation → circuit search → circuit visualization. Users can select from preset gene sets (ASD SPARK, ASC, DDD) or upload custom gene lists, compute both structure-level and cell-type bias, run simulated annealing circuit search to generate a Pareto front, and visualize selected circuits as network graphs.

## Pipeline Flow

```
Page 1: Bias Explorer     →  Page 2: Permutation Test (existing)
    ↓                              ↓
Page 3: Circuit Search    →  Page 4: Circuit Viewer
```

Gene set selected once in sidebar; results flow via `st.session_state` across pages.

## Page 1: Bias Explorer (`pages/01_bias_explorer.py`)

**Input:** Gene set from sidebar selector (presets + custom upload)

**Computation:** Lightweight reimplementation of `MouseSTR_AvgZ_Weighted()` and `MouseCT_AvgZ_Weighted()` — same pattern as `2_Permutation.py` (no heavy imports like igraph/matplotlib).

**Output:**
- Two tabs: "Structure Bias (ISH)" and "Cell Type Bias (scRNA)"
- Each tab:
  - Plotly bar chart: ranked by EFFECT, colored by brain region
  - Sortable results table (EFFECT, Rank, Region)
  - CSV download
- Both bias DataFrames stored in `st.session_state["str_bias_results"]` and `st.session_state["ct_bias_results"]`

## Page 2: Permutation Testing (existing)

Already implemented in `2_Permutation.py`. No changes needed.

## Page 3: Circuit Search (`pages/02_circuit_search.py`)

**Prerequisites:** Structure bias from Page 1 in session state.

**Parameters (sidebar):**
- Circuit size slider (default 46, range 10–100)
- Number of Pareto points / bias limits (default 20)
- SA steps (default 50,000)
- SA runs per bias limit (default 5)

**Computation:**
1. Auto-generate bias limits via `BiasLim()` logic
2. For each bias limit, run 5 SA iterations using `CircuitSearch_SA_InfoContent_Numba` from `src/SA_optimized.py`
3. Keep best circuit per bias limit → Pareto front
4. Add baseline circuit (top N by bias, no optimization)
5. Background thread with progress bar; multiprocessing.Pool(10) for parallel SA runs

**Output:**
- Plotly scatter: Pareto front (x=circuit score, y=mean bias), points clickable
- Table of all Pareto circuits (score, mean_bias, n_structures, structures)
- Selected circuit stored in `st.session_state["selected_circuit"]`

## Page 4: Circuit Viewer (`pages/03_circuit_viewer.py`)

**Input:** Selected circuit from Page 3 or manual structure list.

**Visualization:** Plotly network graph (no Cytoscape.js dependency):
- Nodes = brain structures, sized by bias EFFECT, colored by brain region
- Edges = connections from InfoMat, width ∝ information content
- Layout: networkx `spring_layout()` rendered as Plotly scatter + lines

**Side panel:**
- Circuit stats: mean bias, connectivity score, edge count, region composition
- Structure table (EFFECT, Rank, Region for circuit members)

**Export:** CSV download of circuit structures + adjacency submatrix.

## Core Modules

| Module | Functions | Purpose |
|---|---|---|
| `core/bias.py` | `compute_structure_bias()`, `compute_celltype_bias()` | Weighted average z-scores, lightweight (numpy only) |
| `core/circuit_search.py` | `run_pareto_search()`, `run_single_sa()` | Orchestrates SA across bias limits with multiprocessing |
| `core/pareto.py` | `generate_bias_limits()`, `extract_pareto_front()`, `compute_baseline_circuit()` | Bias limit generation, Pareto filtering, baseline |

## Key Design Decisions

1. **Lightweight bias computation** — reimplements core math in numpy (no igraph/matplotlib imports) for fast Streamlit loading
2. **Numba-optimized SA** — uses `SA_optimized.py::CircuitSearch_SA_InfoContent_Numba` for ~15x speedup
3. **Plotly network graph** — simpler than Cytoscape.js, no custom component needed, sufficient for 46-node circuits
4. **20 Pareto points** — instead of 100 in paper; faster for interactive use, still shows trade-off clearly
5. **Multiprocessing (10 cores)** — SA runs are embarrassingly parallel; 100 total runs (20 limits × 5 runs) finish in reasonable time
6. **Session state pipeline** — bias results flow to circuit search via `st.session_state`, avoiding recomputation

## Session State Keys

```python
st.session_state["gene_weights"]          # dict[int, float] — from gene_set_selector
st.session_state["gene_set_label"]        # str — preset name or "Custom"
st.session_state["str_bias_results"]      # DataFrame — structure bias (EFFECT, Rank, REGION)
st.session_state["ct_bias_results"]       # DataFrame — cell type bias (EFFECT, Rank)
st.session_state["pareto_results"]        # DataFrame — Pareto front
st.session_state["selected_circuit"]      # dict — {structures: list, score: float, mean_bias: float}
```

## Data Dependencies

All loaded via existing `core/data_loader.py`:
- `AllenMouseBrain_Z2bias.parquet` — structure expression matrix
- `Cluster_Z2Mat_ISHMatch.z1clip3.parquet` — cell type expression matrix
- `WeightMat.Ipsi.csv` — connectome weight matrix (213×213)
- `InfoMat.Ipsi.csv` — Shannon information matrix (213×213)
- `structure2region.tsv` — structure → brain region mapping
- `protein-coding_gene.txt` — HGNC gene annotations
