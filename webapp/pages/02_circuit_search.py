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
from core.result_cache import load_result, save_result

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
    cache_params = {
        "gene_set_label": gene_set_label,
        "circuit_size": circuit_size,
        "n_points": n_points,
        "sa_runs": sa_runs,
        "sa_steps": sa_steps,
        "seed": seed,
    }

    # Check cache first
    cached_df = load_result("circuit_search", cache_params)
    if cached_df is not None:
        pareto_df = cached_df
        st.session_state["pareto_results"] = pareto_df
        st.success(
            f"Loaded from cache — {len(pareto_df)} Pareto points.",
            icon="✅",
        )
    else:
        progress_bar = st.progress(0, text="Starting circuit search…")

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

        # Save to cache
        save_result("circuit_search", cache_params, pareto_df)
        st.success(
            f"Circuit search complete! Found {len(pareto_df)} Pareto points. (Cached for reuse)",
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
def _make_label(r):
    prefix = "★ Baseline" if r["circuit_type"] == "baseline" else f"BL={r['bias_limit']:.3f}"
    return f"{prefix} | Score={r['circuit_score']:.4f} | Bias={r['mean_bias']:.4f} | {r['n_structures']} structures"

display_df["label"] = display_df.apply(_make_label, axis=1)

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
