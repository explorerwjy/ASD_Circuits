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
