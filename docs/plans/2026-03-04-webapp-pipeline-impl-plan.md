# GENCIC Webapp Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full GENCIC interactive pipeline — bias computation, circuit search, and circuit visualization — as three Streamlit pages plus three core modules.

**Architecture:** Lightweight numpy-only bias computation (no heavy src/ imports for Pages 1-2), Numba-optimized SA from `src/SA_optimized.py` for circuit search (Page 3), Plotly network graph for visualization (Page 4). All pages share gene set via `st.session_state`. Multiprocessing (10 cores) for parallel SA runs.

**Tech Stack:** Streamlit, NumPy, Pandas, Plotly, NetworkX (layout only), Numba (SA), multiprocessing

---

### Task 1: Core bias module (`core/bias.py`)

**Files:**
- Create: `webapp/core/bias.py`

**Step 1: Create `core/bias.py`**

This extracts the `_compute_weighted_bias` logic from `2_Permutation.py` into a shared module so both Page 1 (Bias Explorer) and Page 2 (Permutation) can reuse it.

```python
"""
webapp/core/bias.py
====================
Lightweight bias computation — no heavy imports (igraph, matplotlib, etc.).

Reimplements the core math of ``MouseSTR_AvgZ_Weighted`` and
``MouseCT_AvgZ_Weighted`` from ``src/ASD_Circuits.py`` using only numpy/pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weighted_bias(
    expr_mat: pd.DataFrame,
    gene_weights: dict[int, float],
) -> pd.DataFrame:
    """Compute weighted average expression bias scores.

    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression z-score matrix (genes × structures/cell types).
        Index = Entrez gene IDs, columns = feature names.
    gene_weights : dict[int, float]
        Gene weight mapping (Entrez ID → weight).

    Returns
    -------
    pd.DataFrame
        Index = feature names, sorted by EFFECT descending.
        Columns: EFFECT, Rank.
    """
    weights_series = pd.Series(gene_weights)
    valid_genes = expr_mat.index.intersection(weights_series.index)

    if len(valid_genes) == 0:
        return pd.DataFrame(columns=["EFFECT", "Rank"])

    weights = weights_series[valid_genes].values
    expr_sub = expr_mat.loc[valid_genes].values  # (n_genes, n_features)

    mask = ~np.isnan(expr_sub)
    w_bc = weights[:, np.newaxis]

    with np.errstate(divide="ignore", invalid="ignore"):
        effects = np.sum(expr_sub * w_bc * mask, axis=0) / np.sum(w_bc * mask, axis=0)

    df = pd.DataFrame({"EFFECT": effects}, index=expr_mat.columns)
    df = df.sort_values("EFFECT", ascending=False)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df
```

**Step 2: Verify import works**

Run: `cd /home/jw3514/Work/ASD_Circuits_CellType/webapp && conda activate gencic && python -c "from core.bias import compute_weighted_bias; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add webapp/core/bias.py
git commit -m "Add core bias computation module for webapp"
```

---

### Task 2: Core circuit search module (`core/circuit_search.py`)

**Files:**
- Create: `webapp/core/circuit_search.py`

**Step 1: Create `core/circuit_search.py`**

This wraps the SA logic from `src/SA_optimized.py` with multiprocessing and progress tracking.

```python
"""
webapp/core/circuit_search.py
==============================
Simulated annealing circuit search with multiprocessing.

Wraps ``src/SA_optimized.py`` Numba-optimized SA classes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd

# Add src/ to path so we can import SA modules
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _find_init_state(
    bias_df: pd.DataFrame,
    size: int,
    min_bias: float,
    rng: np.random.Generator,
    max_attempts: int = 10000,
) -> np.ndarray:
    """Find a valid initial circuit state satisfying the bias constraint.

    Reimplements ``FindInitState`` from ``script_circuit_search.SI.py``.
    Samples structures with probability proportional to bias^power.
    """
    strs = bias_df.index.values
    biases = bias_df["EFFECT"].values
    min_b = np.min(biases)
    pseudo = biases - min_b + 1
    power = max(min_bias * 150 - 17, 0.1)
    pseudo = np.power(pseudo, power)
    probs = pseudo / np.sum(pseudo)
    # Fix rounding
    probs[-1] = 1.0 - np.sum(probs[:-1])

    for _ in range(max_attempts):
        chosen = rng.choice(len(strs), size=size, replace=False, p=probs)
        if bias_df.iloc[chosen]["EFFECT"].mean() >= min_bias:
            init_state = np.zeros(len(strs), dtype=np.float64)
            init_state[chosen] = 1.0
            return init_state

    # Fallback: top-N by bias
    init_state = np.zeros(len(strs), dtype=np.float64)
    init_state[:size] = 1.0
    return init_state


def _run_single_sa(
    args: tuple,
    bias_values: np.ndarray,
    candidate_nodes: np.ndarray,
    info_mat_np: np.ndarray,
    node_to_idx: dict,
    candidate_idx: np.ndarray,
    tmax: float = 1e-2,
    tmin: float = 5e-5,
    steps: int = 50000,
) -> tuple[float, float, list[str]]:
    """Run a single SA iteration. Designed to be called via multiprocessing.

    Parameters
    ----------
    args : tuple
        (bias_limit_idx, run_idx, init_state, min_bias) — unpacked inside.
    bias_values, candidate_nodes, info_mat_np, node_to_idx, candidate_idx :
        Shared data (read-only in child process).

    Returns
    -------
    (score, mean_bias, structures_list)
    """
    _, _, init_state, min_bias = args

    from SA_optimized import CircuitSearch_SA_InfoContent_Numba

    # Build a minimal BiasDF for the SA class
    bias_df = pd.DataFrame(
        {"EFFECT": bias_values},
        index=candidate_nodes,
    )

    sa = CircuitSearch_SA_InfoContent_Numba(
        bias_df, init_state, None,
        pd.DataFrame(info_mat_np, index=list(node_to_idx.keys()), columns=list(node_to_idx.keys())),
        candidate_nodes, min_bias,
    )
    sa.copy_strategy = "deepcopy"
    sa.Tmax = tmax
    sa.Tmin = tmin
    sa.steps = steps
    sa.updates = 0

    _, _, state, e = sa.anneal()
    score = -e
    result_nodes = candidate_nodes[np.where(state == 1)[0]]
    mean_bias = bias_df.loc[result_nodes, "EFFECT"].mean()
    return score, mean_bias, list(result_nodes)


def generate_bias_limits(bias_df: pd.DataFrame, circuit_size: int, n_points: int = 20) -> list[float]:
    """Generate evenly-spaced bias limits for Pareto front.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Structure bias results with EFFECT column, sorted descending.
    circuit_size : int
        Number of structures in each circuit.
    n_points : int
        Number of Pareto points to generate.

    Returns
    -------
    list[float]
        Bias limit values from low to high.
    """
    max_mean_bias = bias_df.head(circuit_size)["EFFECT"].mean()
    min_bias = 0.0
    limits = np.linspace(min_bias, max_mean_bias * 0.95, n_points)
    return [round(float(b), 4) for b in limits]


def run_pareto_search(
    bias_df: pd.DataFrame,
    info_mat: pd.DataFrame,
    adj_mat: pd.DataFrame,
    circuit_size: int = 46,
    n_points: int = 20,
    sa_runs: int = 5,
    sa_steps: int = 50000,
    n_workers: int = 10,
    seed: int = 42,
    progress_callback=None,
) -> pd.DataFrame:
    """Run SA circuit search across multiple bias limits to build a Pareto front.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Structure bias with EFFECT, Rank, REGION columns. Sorted by EFFECT descending.
    info_mat : pd.DataFrame
        213×213 Shannon information matrix.
    adj_mat : pd.DataFrame
        213×213 adjacency/weight matrix.
    circuit_size : int
        Number of structures per circuit.
    n_points : int
        Number of bias limit points for Pareto front.
    sa_runs : int
        Independent SA runs per bias limit.
    sa_steps : int
        SA annealing steps per run.
    n_workers : int
        Multiprocessing pool size.
    seed : int
        Base random seed.
    progress_callback : callable, optional
        Called with (completed_count, total_count) for progress tracking.

    Returns
    -------
    pd.DataFrame
        Pareto front with columns:
        bias_limit, circuit_score, mean_bias, n_structures, structures, circuit_type
    """
    # Use top-213 structures (all) as candidates
    top_n = min(213, len(bias_df))
    candidate_df = bias_df.head(top_n)
    candidate_nodes = candidate_df.index.values

    bias_limits = generate_bias_limits(bias_df, circuit_size, n_points)

    # Pre-compute shared data
    bias_values = candidate_df["EFFECT"].values.astype(np.float64)
    info_mat_np = info_mat.values.astype(np.float64)
    node_to_idx = {node: i for i, node in enumerate(info_mat.index)}
    candidate_idx = np.array([node_to_idx[n] for n in candidate_nodes], dtype=np.int32)

    # Build job list: (bias_limit_idx, run_idx, init_state, min_bias)
    rng = np.random.default_rng(seed)
    jobs = []
    for bl_idx, bl in enumerate(bias_limits):
        for run_idx in range(sa_runs):
            init_state = _find_init_state(candidate_df, circuit_size, bl, rng)
            jobs.append((bl_idx, run_idx, init_state, bl))

    total_jobs = len(jobs)

    # Run SA in parallel
    worker_fn = partial(
        _run_single_sa,
        bias_values=bias_values,
        candidate_nodes=candidate_nodes,
        info_mat_np=info_mat_np,
        node_to_idx=node_to_idx,
        candidate_idx=candidate_idx,
        steps=sa_steps,
    )

    results_by_limit: dict[int, list] = {i: [] for i in range(len(bias_limits))}
    completed = 0

    # Use sequential execution to avoid multiprocessing issues with Numba/Streamlit
    # (Numba JIT + fork = potential deadlocks)
    for job in jobs:
        result = worker_fn(job)
        bl_idx = job[0]
        results_by_limit[bl_idx].append(result)
        completed += 1
        if progress_callback:
            progress_callback(completed, total_jobs)

    # Extract best circuit per bias limit
    pareto_rows = []
    for bl_idx, bl in enumerate(bias_limits):
        runs = results_by_limit[bl_idx]
        if not runs:
            continue
        best = max(runs, key=lambda r: r[0])  # highest score
        score, mean_bias, structures = best
        pareto_rows.append({
            "bias_limit": bl,
            "circuit_score": score,
            "mean_bias": mean_bias,
            "n_structures": len(structures),
            "structures": ",".join(structures),
            "circuit_type": "optimized",
        })

    # Add baseline circuit (top N by bias, no optimization)
    baseline_strs = bias_df.head(circuit_size).index.values
    baseline_strs_in_info = [s for s in baseline_strs if s in info_mat.index]
    if len(baseline_strs_in_info) > 0:
        sub = info_mat.loc[baseline_strs_in_info, baseline_strs_in_info].values
        n_events = np.count_nonzero(sub)
        baseline_score = np.sum(sub) / n_events if n_events > 0 else 0.0
        baseline_mean = bias_df.loc[baseline_strs, "EFFECT"].mean()
        pareto_rows.append({
            "bias_limit": None,
            "circuit_score": baseline_score,
            "mean_bias": baseline_mean,
            "n_structures": len(baseline_strs),
            "structures": ",".join(baseline_strs),
            "circuit_type": "baseline",
        })

    return pd.DataFrame(pareto_rows)
```

**Step 2: Verify import works**

Run: `cd /home/jw3514/Work/ASD_Circuits_CellType/webapp && conda activate gencic && python -c "from core.circuit_search import generate_bias_limits; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add webapp/core/circuit_search.py
git commit -m "Add circuit search module with SA + Pareto front generation"
```

---

### Task 3: Page 1 — Bias Explorer (`pages/01_bias_explorer.py`)

**Files:**
- Replace: `webapp/pages/01_bias_explorer.py`

**Step 1: Implement the full Bias Explorer page**

```python
"""
pages/01_bias_explorer.py
=========================
Bias Explorer — compute and visualize mutation bias across brain structures
and cell types for a selected gene set.

Computes both structure-level (ISH) and cell-type-level (scRNA) bias
simultaneously and stores results in session_state for downstream pages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from components.gene_set_selector import gene_set_selector
from core.bias import compute_weighted_bias
from core.data_loader import (
    load_ct_bias_matrix,
    load_str_bias_matrix,
    load_structure_region_map,
    load_webapp_config,
)

st.set_page_config(page_title="Bias Explorer — GENCIC", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — gene set selector
# ---------------------------------------------------------------------------
gene_weights, gene_set_label = gene_set_selector(sidebar=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📊 Bias Explorer")

if gene_weights is None:
    st.warning(
        "**No gene set loaded.** Select a gene set in the sidebar to get started.",
        icon="⚠️",
    )
    st.stop()

st.markdown(f"**Gene set:** {gene_set_label} ({len(gene_weights):,} genes)")

# ---------------------------------------------------------------------------
# Compute bias for both analysis types
# ---------------------------------------------------------------------------
with st.spinner("Loading expression matrices…"):
    str_mat = load_str_bias_matrix()
    ct_mat = load_ct_bias_matrix()

str_overlap = str_mat.index.intersection(pd.Series(gene_weights).index)
ct_overlap = ct_mat.index.intersection(pd.Series(gene_weights).index)

st.info(
    f"**Gene overlap:** {len(str_overlap):,} / {len(gene_weights):,} genes in structure matrix "
    f"({str_mat.shape[1]} structures) · "
    f"{len(ct_overlap):,} / {len(gene_weights):,} genes in cell-type matrix "
    f"({ct_mat.shape[1]:,} clusters)",
    icon="ℹ️",
)

with st.spinner("Computing bias scores…"):
    str_bias = compute_weighted_bias(str_mat, gene_weights)
    ct_bias = compute_weighted_bias(ct_mat, gene_weights)

    # Add region info to structure bias
    region_map = load_structure_region_map()
    str_bias["REGION"] = str_bias.index.map(lambda s: region_map.get(s, "Unknown"))

# Store in session state for downstream pages
st.session_state["str_bias_results"] = str_bias
st.session_state["ct_bias_results"] = ct_bias

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
tab_str, tab_ct = st.tabs(["🧠 Structure Bias (ISH)", "🔬 Cell Type Bias (scRNA)"])

# ---- Structure Bias Tab ----
with tab_str:
    st.subheader("Brain Structure Bias Profile")

    # Summary metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Structures", f"{len(str_bias):,}")
    m2.metric("Max EFFECT", f"{str_bias['EFFECT'].max():.4f}")
    m3.metric("Min EFFECT", f"{str_bias['EFFECT'].min():.4f}")

    # Bar chart — top 50 structures colored by region
    top_n = min(50, len(str_bias))
    plot_df = str_bias.head(top_n).copy()
    plot_df["Structure"] = plot_df.index

    fig_str = px.bar(
        plot_df,
        x="Structure",
        y="EFFECT",
        color="REGION",
        hover_data={"Rank": True, "EFFECT": ":.4f", "REGION": True},
        title=f"Top {top_n} Structures by Mutation Bias ({gene_set_label})",
        labels={"EFFECT": "Bias (weighted avg z-score)", "Structure": ""},
    )
    fig_str.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(title="Brain Region"),
    )
    st.plotly_chart(fig_str, use_container_width=True)

    # Full table
    with st.expander("Full results table", expanded=False):
        st.dataframe(
            str_bias.style.format({"EFFECT": "{:.4f}"}),
            use_container_width=True,
            height=400,
        )

    # Download
    csv_str = str_bias.to_csv().encode("utf-8")
    st.download_button(
        "📥 Download structure bias CSV",
        data=csv_str,
        file_name=f"structure_bias_{gene_set_label}.csv",
        mime="text/csv",
    )

# ---- Cell Type Bias Tab ----
with tab_ct:
    st.subheader("Cell Type Bias Profile")

    m1, m2, m3 = st.columns(3)
    m1.metric("Cell Types", f"{len(ct_bias):,}")
    m2.metric("Max EFFECT", f"{ct_bias['EFFECT'].max():.4f}")
    m3.metric("Min EFFECT", f"{ct_bias['EFFECT'].min():.4f}")

    # Bar chart — top 50 cell types
    top_n_ct = min(50, len(ct_bias))
    plot_ct = ct_bias.head(top_n_ct).copy()
    plot_ct["CellType"] = plot_ct.index.astype(str)

    fig_ct = px.bar(
        plot_ct,
        x="CellType",
        y="EFFECT",
        hover_data={"Rank": True, "EFFECT": ":.4f"},
        title=f"Top {top_n_ct} Cell Types by Mutation Bias ({gene_set_label})",
        labels={"EFFECT": "Bias (weighted avg z-score)", "CellType": ""},
        color_discrete_sequence=["#2ca02c"],
    )
    fig_ct.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_ct, use_container_width=True)

    with st.expander("Full results table", expanded=False):
        st.dataframe(
            ct_bias.style.format({"EFFECT": "{:.4f}"}),
            use_container_width=True,
            height=400,
        )

    csv_ct = ct_bias.to_csv().encode("utf-8")
    st.download_button(
        "📥 Download cell type bias CSV",
        data=csv_ct,
        file_name=f"celltype_bias_{gene_set_label}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("🎲 Run Permutation Test →", use_container_width=True):
        st.switch_page("pages/2_Permutation.py")
with col2:
    if st.button("🔍 Search Circuits →", use_container_width=True):
        st.switch_page("pages/02_circuit_search.py")
```

**Step 2: Test the page manually**

Run: `cd /home/jw3514/Work/ASD_Circuits_CellType/webapp && conda activate gencic && streamlit run app.py --server.port 8501 --server.baseUrlPath /gencic`
Navigate to the Bias Explorer page, select a preset gene set, verify both tabs render.

**Step 3: Commit**

```bash
git add webapp/pages/01_bias_explorer.py
git commit -m "Implement Bias Explorer page with structure + cell type tabs"
```

---

### Task 4: Page 3 — Circuit Search (`pages/02_circuit_search.py`)

**Files:**
- Replace: `webapp/pages/02_circuit_search.py`

**Step 1: Implement the Circuit Search page**

```python
"""
pages/02_circuit_search.py
==========================
Circuit Search — run simulated annealing to find brain circuits that balance
high mutation bias with strong internal connectivity. Generates a Pareto front.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.gene_set_selector import gene_set_selector
from core.bias import compute_weighted_bias
from core.circuit_search import generate_bias_limits, run_pareto_search
from core.data_loader import (
    load_info_matrix,
    load_str_bias_matrix,
    load_structure_region_map,
    load_weight_matrix,
)

st.set_page_config(page_title="Circuit Search — GENCIC", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
gene_weights, gene_set_label = gene_set_selector(sidebar=True)

with st.sidebar:
    st.divider()
    st.subheader("SA Parameters")
    circuit_size = st.slider("Circuit size", 10, 100, 46, key="sa_circuit_size")
    n_points = st.slider("Pareto points (bias limits)", 5, 50, 20, key="sa_n_points")
    sa_runs = st.slider("SA runs per bias limit", 1, 20, 5, key="sa_runs")
    sa_steps = st.select_slider(
        "SA steps per run",
        options=[10000, 25000, 50000, 100000],
        value=50000,
        key="sa_steps",
    )
    seed = st.number_input("Random seed", 0, 99999, 42, key="sa_seed")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔍 Circuit Search")

if gene_weights is None:
    st.warning("**No gene set loaded.** Select one in the sidebar.", icon="⚠️")
    if st.button("Go to Bias Explorer →"):
        st.switch_page("pages/01_bias_explorer.py")
    st.stop()

st.markdown(f"**Gene set:** {gene_set_label} ({len(gene_weights):,} genes)")

# ---------------------------------------------------------------------------
# Get or compute structure bias
# ---------------------------------------------------------------------------
if "str_bias_results" not in st.session_state:
    with st.spinner("Computing structure bias…"):
        str_mat = load_str_bias_matrix()
        str_bias = compute_weighted_bias(str_mat, gene_weights)
        region_map = load_structure_region_map()
        str_bias["REGION"] = str_bias.index.map(lambda s: region_map.get(s, "Unknown"))
        st.session_state["str_bias_results"] = str_bias

str_bias: pd.DataFrame = st.session_state["str_bias_results"]

# Show bias limit range
bias_limits = generate_bias_limits(str_bias, circuit_size, n_points)
st.info(
    f"**{n_points} bias limits** from {bias_limits[0]:.3f} to {bias_limits[-1]:.3f} · "
    f"**{sa_runs} SA runs** each · **{sa_runs * n_points} total SA jobs** · "
    f"**{sa_steps:,} steps** per run",
    icon="ℹ️",
)

# ---------------------------------------------------------------------------
# Run circuit search
# ---------------------------------------------------------------------------
run_btn = st.button("🚀 Run Circuit Search", use_container_width=True, type="primary")

if run_btn:
    progress_bar = st.progress(0, text="Starting circuit search…")
    status_text = st.empty()

    def update_progress(completed: int, total: int):
        progress_bar.progress(
            completed / total,
            text=f"SA run {completed} / {total}",
        )

    with st.spinner("Loading connectome matrices…"):
        info_mat = load_info_matrix()
        adj_mat = load_weight_matrix()

    pareto_df = run_pareto_search(
        bias_df=str_bias,
        info_mat=info_mat,
        adj_mat=adj_mat,
        circuit_size=circuit_size,
        n_points=n_points,
        sa_runs=sa_runs,
        sa_steps=sa_steps,
        n_workers=10,
        seed=seed,
        progress_callback=update_progress,
    )

    progress_bar.empty()
    st.session_state["pareto_results"] = pareto_df
    st.success(
        f"Circuit search complete! Found {len(pareto_df)} Pareto points.",
        icon="✅",
    )

# ---------------------------------------------------------------------------
# Display Pareto front
# ---------------------------------------------------------------------------
if "pareto_results" not in st.session_state:
    st.info("Configure parameters and click **Run Circuit Search** to start.", icon="💡")
    st.stop()

pareto_df: pd.DataFrame = st.session_state["pareto_results"]

st.divider()
st.subheader("Pareto Front: Bias vs. Connectivity")

# Separate optimized and baseline
opt_df = pareto_df[pareto_df["circuit_type"] == "optimized"].copy()
base_df = pareto_df[pareto_df["circuit_type"] == "baseline"].copy()

fig = go.Figure()

# Optimized circuits — connected line + markers
if len(opt_df) > 0:
    opt_sorted = opt_df.sort_values("circuit_score")
    fig.add_trace(go.Scatter(
        x=opt_sorted["circuit_score"],
        y=opt_sorted["mean_bias"],
        mode="lines+markers",
        name="Optimized circuits",
        marker=dict(size=10, color="#542788"),
        line=dict(color="#542788", width=2),
        customdata=opt_sorted.index,
        hovertemplate=(
            "Score: %{x:.4f}<br>"
            "Mean bias: %{y:.4f}<br>"
            "Structures: %{text}<br>"
            "<extra></extra>"
        ),
        text=[str(n) for n in opt_sorted["n_structures"]],
    ))

# Baseline circuit — star marker
if len(base_df) > 0:
    fig.add_trace(go.Scatter(
        x=base_df["circuit_score"],
        y=base_df["mean_bias"],
        mode="markers",
        name="Baseline (top-N by bias)",
        marker=dict(size=15, color="#e74c3c", symbol="star"),
    ))

fig.update_layout(
    xaxis_title="Circuit Score (Shannon Information)",
    yaxis_title="Mean Mutation Bias",
    template="plotly_white",
    height=550,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Circuit selection
# ---------------------------------------------------------------------------
st.subheader("Select a Circuit")

display_df = pareto_df.copy()
display_df["idx"] = range(len(display_df))
display_df["label"] = display_df.apply(
    lambda r: f"{'★ Baseline' if r['circuit_type'] == 'baseline' else f'BL={r[\"bias_limit\"]:.3f}'} "
              f"| Score={r['circuit_score']:.4f} | Bias={r['mean_bias']:.4f} "
              f"| {r['n_structures']} structures",
    axis=1,
)

selected_label = st.selectbox(
    "Choose a circuit to view:",
    options=display_df["label"].tolist(),
    key="circuit_selector",
)

selected_row = display_df[display_df["label"] == selected_label].iloc[0]
selected_structures = selected_row["structures"].split(",")

st.session_state["selected_circuit"] = {
    "structures": selected_structures,
    "score": selected_row["circuit_score"],
    "mean_bias": selected_row["mean_bias"],
    "circuit_type": selected_row["circuit_type"],
    "bias_limit": selected_row.get("bias_limit"),
}

# Show selected circuit summary
col1, col2, col3 = st.columns(3)
col1.metric("Circuit Score", f"{selected_row['circuit_score']:.4f}")
col2.metric("Mean Bias", f"{selected_row['mean_bias']:.4f}")
col3.metric("Structures", f"{int(selected_row['n_structures'])}")

# Structure list
circuit_bias = str_bias.loc[
    [s for s in selected_structures if s in str_bias.index]
]
st.dataframe(
    circuit_bias[["EFFECT", "Rank", "REGION"]].style.format({"EFFECT": "{:.4f}"}),
    use_container_width=True,
    height=300,
)

# Navigation
st.divider()
if st.button("🕸️ View Circuit Graph →", use_container_width=True, type="primary"):
    st.switch_page("pages/03_circuit_viewer.py")

# Download
csv_pareto = pareto_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Pareto front CSV",
    data=csv_pareto,
    file_name=f"pareto_front_{gene_set_label}.csv",
    mime="text/csv",
)
```

**Step 2: Test manually**

Run the Streamlit app, navigate to Circuit Search, select a preset, run with small params (5 points, 2 runs, 10000 steps) to verify it completes.

**Step 3: Commit**

```bash
git add webapp/pages/02_circuit_search.py
git commit -m "Implement Circuit Search page with SA Pareto front"
```

---

### Task 5: Page 4 — Circuit Viewer (`pages/03_circuit_viewer.py`)

**Files:**
- Replace: `webapp/pages/03_circuit_viewer.py`

**Step 1: Implement the Circuit Viewer page**

```python
"""
pages/03_circuit_viewer.py
==========================
Circuit Viewer — visualize a selected brain circuit as an interactive
network graph using Plotly + NetworkX layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import (
    load_info_matrix,
    load_structure_region_map,
    load_weight_matrix,
)

st.set_page_config(page_title="Circuit Viewer — GENCIC", page_icon="🕸️", layout="wide")

# ---------------------------------------------------------------------------
# Region color palette
# ---------------------------------------------------------------------------
REGION_COLORS = {
    "Isocortex": "#1f77b4",
    "OLF": "#ff7f0e",
    "HPF": "#2ca02c",
    "CTXsp": "#d62728",
    "STR": "#9467bd",
    "PAL": "#8c564b",
    "TH": "#e377c2",
    "HY": "#7f7f7f",
    "MB": "#bcbd22",
    "P": "#17becf",
    "MY": "#aec7e8",
    "CB": "#ffbb78",
    "fiber tracts": "#c7c7c7",
    "VS": "#dbdb8d",
    "Unknown": "#999999",
}


def _build_network_graph(
    structures: list[str],
    info_mat: pd.DataFrame,
    weight_mat: pd.DataFrame,
    bias_df: pd.DataFrame | None,
    region_map: dict[str, str],
) -> go.Figure:
    """Build a Plotly network graph from circuit structures.

    Uses NetworkX spring_layout for node positions, then renders with Plotly.
    """
    import networkx as nx

    # Build networkx graph
    G = nx.DiGraph()
    valid_strs = [s for s in structures if s in info_mat.index]

    for s in valid_strs:
        bias_val = bias_df.loc[s, "EFFECT"] if bias_df is not None and s in bias_df.index else 0
        region = region_map.get(s, "Unknown")
        G.add_node(s, bias=bias_val, region=region)

    # Add edges where info matrix > 0
    edges = []
    for i, s1 in enumerate(valid_strs):
        for j, s2 in enumerate(valid_strs):
            if i != j:
                val = info_mat.loc[s1, s2]
                if val > 0:
                    weight = weight_mat.loc[s1, s2] if s1 in weight_mat.index and s2 in weight_mat.columns else 0
                    G.add_edge(s1, s2, info=val, weight=weight)
                    edges.append((s1, s2, val))

    if len(G.nodes) == 0:
        return go.Figure().update_layout(title="No valid structures found")

    # Layout
    pos = nx.spring_layout(G, k=2.0 / np.sqrt(len(G.nodes)), iterations=100, seed=42)

    # Edge traces
    edge_x, edge_y = [], []
    for s1, s2, _ in edges:
        x0, y0 = pos[s1]
        x1, y1 = pos[s2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.3, color="#ccc"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    # Node traces — one per region for legend
    node_traces = []
    regions_seen = set()

    for node in G.nodes:
        region = G.nodes[node]["region"]
        bias_val = G.nodes[node]["bias"]
        x, y = pos[node]
        color = REGION_COLORS.get(region, "#999999")
        size = max(8, min(30, 8 + abs(bias_val) * 40))

        show_legend = region not in regions_seen
        regions_seen.add(region)

        node_traces.append(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color, line=dict(width=1, color="white")),
            text=[node],
            textposition="top center",
            textfont=dict(size=8),
            name=region,
            legendgroup=region,
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{node}</b><br>"
                f"Region: {region}<br>"
                f"Bias: {bias_val:.4f}<br>"
                f"<extra></extra>"
            ),
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        template="plotly_white",
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(title="Brain Region"),
        title="Circuit Network Graph",
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🕸️ Circuit Viewer")

if "selected_circuit" not in st.session_state:
    st.warning("**No circuit selected.** Run the Circuit Search first.", icon="⚠️")

    # Allow manual input
    st.subheader("Or enter structures manually")
    manual_input = st.text_area(
        "Paste structure names (comma-separated):",
        placeholder="Isocortex_VISp, Isocortex_MOp, TH_VPM, ...",
        key="manual_structures",
    )
    if manual_input.strip():
        structures = [s.strip() for s in manual_input.split(",") if s.strip()]
        st.session_state["selected_circuit"] = {
            "structures": structures,
            "score": None,
            "mean_bias": None,
            "circuit_type": "manual",
        }
        st.rerun()
    else:
        if st.button("🔍 Go to Circuit Search →"):
            st.switch_page("pages/02_circuit_search.py")
        st.stop()

circuit = st.session_state["selected_circuit"]
structures = circuit["structures"]

# Display circuit info
st.markdown(
    f"**Circuit:** {len(structures)} structures · "
    f"**Type:** {circuit.get('circuit_type', 'unknown')}"
    + (f" · **Score:** {circuit['score']:.4f}" if circuit.get("score") else "")
    + (f" · **Mean bias:** {circuit['mean_bias']:.4f}" if circuit.get("mean_bias") else "")
)

# Load data
with st.spinner("Loading matrices…"):
    info_mat = load_info_matrix()
    weight_mat = load_weight_matrix()
    region_map = load_structure_region_map()

bias_df = st.session_state.get("str_bias_results")

# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------
fig = _build_network_graph(structures, info_mat, weight_mat, bias_df, region_map)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Circuit statistics
# ---------------------------------------------------------------------------
st.divider()
col_stats, col_regions = st.columns([1, 1])

with col_stats:
    st.subheader("Circuit Statistics")

    valid_strs = [s for s in structures if s in info_mat.index]
    sub_info = info_mat.loc[valid_strs, valid_strs].values
    n_edges = np.count_nonzero(sub_info)
    n_possible = len(valid_strs) * (len(valid_strs) - 1)
    density = n_edges / n_possible if n_possible > 0 else 0
    mean_info = np.sum(sub_info) / n_edges if n_edges > 0 else 0

    m1, m2 = st.columns(2)
    m1.metric("Structures", len(valid_strs))
    m2.metric("Edges", n_edges)
    m3, m4 = st.columns(2)
    m3.metric("Density", f"{density:.3f}")
    m4.metric("Mean Info", f"{mean_info:.4f}")

with col_regions:
    st.subheader("Region Composition")
    regions = [region_map.get(s, "Unknown") for s in valid_strs]
    region_counts = pd.Series(regions).value_counts()
    st.bar_chart(region_counts)

# ---------------------------------------------------------------------------
# Structure detail table
# ---------------------------------------------------------------------------
st.subheader("Structure Details")

detail_rows = []
for s in structures:
    row = {"Structure": s, "Region": region_map.get(s, "Unknown")}
    if bias_df is not None and s in bias_df.index:
        row["EFFECT"] = bias_df.loc[s, "EFFECT"]
        row["Rank"] = bias_df.loc[s, "Rank"]
    detail_rows.append(row)

detail_df = pd.DataFrame(detail_rows).set_index("Structure")
if "EFFECT" in detail_df.columns:
    detail_df = detail_df.sort_values("EFFECT", ascending=False)

st.dataframe(
    detail_df.style.format({"EFFECT": "{:.4f}"} if "EFFECT" in detail_df.columns else {}),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
st.divider()
col_d1, col_d2 = st.columns(2)

with col_d1:
    csv_detail = detail_df.to_csv().encode("utf-8")
    st.download_button(
        "📥 Download structure list",
        data=csv_detail,
        file_name="circuit_structures.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_d2:
    # Adjacency submatrix
    valid_strs = [s for s in structures if s in info_mat.index]
    if valid_strs:
        adj_sub = info_mat.loc[valid_strs, valid_strs]
        csv_adj = adj_sub.to_csv().encode("utf-8")
        st.download_button(
            "📥 Download adjacency matrix",
            data=csv_adj,
            file_name="circuit_adjacency.csv",
            mime="text/csv",
            use_container_width=True,
        )
```

**Step 2: Test manually**

After running circuit search and selecting a circuit, navigate to Circuit Viewer and verify the network graph renders with colored nodes and edges.

**Step 3: Commit**

```bash
git add webapp/pages/03_circuit_viewer.py
git commit -m "Implement Circuit Viewer page with Plotly network graph"
```

---

### Task 6: Integration test — end-to-end flow

**Step 1: Run the full pipeline manually**

```bash
cd /home/jw3514/Work/ASD_Circuits_CellType/webapp
conda activate gencic
streamlit run app.py --server.port 8501 --server.baseUrlPath /gencic
```

1. Navigate to Bias Explorer → Select ASD_SPARK_159 preset → Verify both tabs render
2. Navigate to Circuit Search → Set 5 points, 2 runs, 10000 steps → Click Run → Verify Pareto front
3. Select a circuit → Click "View Circuit Graph" → Verify network renders
4. Test downloads on each page

**Step 2: Fix any issues found**

**Step 3: Final commit**

```bash
git add -A webapp/
git commit -m "Complete GENCIC webapp pipeline: bias, circuit search, viewer"
```
