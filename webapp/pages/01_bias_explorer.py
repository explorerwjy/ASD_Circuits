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
    load_ct_cluster_annotation,
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

    # Load annotation and join with bias
    anno = load_ct_cluster_annotation()
    ct_bias_anno = ct_bias.join(anno[["class_label", "class_id_label",
                                      "subclass_label", "subclass_id_label",
                                      "nt_type_label", "CCF_broad.freq"]])

    m1, m2, m3 = st.columns(3)
    m1.metric("Cell Types", f"{len(ct_bias):,}")
    m2.metric("Max EFFECT", f"{ct_bias['EFFECT'].max():.4f}")
    m3.metric("Min EFFECT", f"{ct_bias['EFFECT'].min():.4f}")

    # --- Color assignment by neurotransmitter type ---
    def _assign_class_colors(classes: pd.Series, anno_df: pd.DataFrame) -> dict:
        """Assign distinct colors to cell type classes based on NT type."""
        # NT color families
        nt_palettes = {
            "Glut": ["#1f77b4", "#4a90d9", "#6baed6", "#9ecae1", "#c6dbef",
                      "#2171b5", "#08519c", "#08306b", "#3182bd", "#6baed6",
                      "#4292c6", "#2166ac", "#053061", "#0570b0", "#034e7b"],
            "GABA": ["#d62728", "#e45756", "#ff6b6b", "#ff9999", "#ffcccc",
                      "#cb181d", "#a50f15", "#67000d", "#ef3b2c", "#fb6a4a",
                      "#fc9272", "#e31a1c", "#b2182b", "#d6604d", "#f4a582"],
            "Dopa": ["#9467bd", "#b07cd8", "#c49ce8"],
            "Sero": ["#bcbd22", "#dbdb8d", "#e5c100"],
            "Nora": ["#8c564b", "#a0785c"],
            "Chol": ["#17becf", "#76d7e8"],
            "Hist": ["#e377c2", "#f1a7d6"],
        }
        gray_palette = ["#7f7f7f", "#999999", "#aaaaaa", "#bbbbbb", "#cccccc"]

        # Map each class to its dominant NT type
        class_nt = anno_df.groupby("class_label")["nt_type_label"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Other"
        )

        color_map = {}
        nt_counters = {}
        for cls in classes:
            nt = class_nt.get(cls, "Other")
            # Map compound types to primary
            primary_nt = nt.split("-")[0] if "-" in nt else nt
            palette = nt_palettes.get(primary_nt, gray_palette)
            idx = nt_counters.get(primary_nt, 0)
            color_map[cls] = palette[idx % len(palette)]
            nt_counters[primary_nt] = idx + 1
        return color_map

    # --- 3a. Class-level overview ---
    class_bias = (
        ct_bias_anno
        .groupby("class_label")["EFFECT"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    class_colors = _assign_class_colors(class_bias["class_label"], anno)

    fig_class = px.bar(
        class_bias,
        x="class_label",
        y="EFFECT",
        color="class_label",
        color_discrete_map=class_colors,
        title=f"Class-Level Mean Bias ({gene_set_label})",
        labels={"EFFECT": "Mean bias (weighted avg z-score)", "class_label": ""},
    )
    fig_class.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig_class, use_container_width=True)

    # --- 3b. Hierarchy drill-down ---
    st.markdown("---")
    st.subheader("Hierarchy Drill-Down")

    all_classes = sorted(ct_bias_anno["class_label"].dropna().unique())
    selected_class = st.selectbox("Select a cell type class:", all_classes,
                                  index=0, key="ct_class_select")

    if selected_class:
        class_mask = ct_bias_anno["class_label"] == selected_class
        class_color = class_colors.get(selected_class, "#2ca02c")

        # Subclass-level bar chart
        subclass_bias = (
            ct_bias_anno.loc[class_mask]
            .groupby("subclass_label")["EFFECT"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig_sub = px.bar(
            subclass_bias,
            x="subclass_label",
            y="EFFECT",
            title=f"Subclass Bias within {selected_class}",
            labels={"EFFECT": "Mean bias", "subclass_label": ""},
            color_discrete_sequence=[class_color],
        )
        fig_sub.update_layout(
            xaxis_tickangle=-45, template="plotly_white", height=400,
        )
        st.plotly_chart(fig_sub, use_container_width=True)

        # Subclass selector → cluster-level
        subclasses = sorted(
            ct_bias_anno.loc[class_mask, "subclass_label"].dropna().unique()
        )
        selected_subclass = st.selectbox(
            "Drill into subclass:", subclasses, index=0, key="ct_subclass_select"
        )

        if selected_subclass:
            sub_mask = class_mask & (ct_bias_anno["subclass_label"] == selected_subclass)
            cluster_df = (
                ct_bias_anno.loc[sub_mask, ["EFFECT"]]
                .sort_values("EFFECT", ascending=False)
                .copy()
            )
            cluster_df["cluster"] = cluster_df.index.astype(str)

            fig_clust = px.bar(
                cluster_df,
                x="cluster",
                y="EFFECT",
                title=f"Cluster Bias within {selected_subclass}",
                labels={"EFFECT": "Bias", "cluster": ""},
                color_discrete_sequence=[class_color],
            )
            fig_clust.update_layout(
                xaxis_tickangle=-45, template="plotly_white", height=400,
            )
            st.plotly_chart(fig_clust, use_container_width=True)

    # --- 3c. Region × Cell Type heatmap ---
    st.markdown("---")
    st.subheader("Region × Cell Type Spatial Heatmap")

    # Parse CCF_broad.freq into (clusters × regions) composition matrix
    def _parse_ccf_composition(anno_df: pd.DataFrame) -> pd.DataFrame:
        """Parse CCF_broad.freq strings into a clusters × regions matrix."""
        records = {}
        for cluster_id, row in anno_df.iterrows():
            freq_str = row.get("CCF_broad.freq", "")
            if pd.isna(freq_str) or not freq_str:
                continue
            parts = {}
            for pair in str(freq_str).split(","):
                pair = pair.strip()
                if ":" in pair:
                    region, val = pair.rsplit(":", 1)
                    try:
                        parts[region] = float(val)
                    except ValueError:
                        pass
            if parts:
                records[cluster_id] = parts
        return pd.DataFrame.from_dict(records, orient="index").fillna(0.0)

    ccf_mat = _parse_ccf_composition(ct_bias_anno)

    if not ccf_mat.empty:
        # Aggregate to class level
        ccf_mat_with_class = ccf_mat.join(
            ct_bias_anno[["class_label"]], how="inner"
        )
        class_comp = ccf_mat_with_class.groupby("class_label").mean()

        # Sort by class bias order
        class_order = class_bias["class_label"].tolist()
        class_comp = class_comp.reindex(
            [c for c in class_order if c in class_comp.index]
        )

        heatmap_mode = st.radio(
            "Heatmap mode:",
            ["Spatial composition", "Bias-weighted composition"],
            horizontal=True,
            key="heatmap_mode",
        )

        if heatmap_mode == "Bias-weighted composition":
            # Multiply each class row by its mean bias
            class_mean_bias = class_bias.set_index("class_label")["EFFECT"]
            plot_heatmap = class_comp.multiply(
                class_mean_bias.reindex(class_comp.index), axis=0
            )
            colorscale = "RdBu_r"
            hm_title = f"Bias-Weighted Spatial Composition ({gene_set_label})"
        else:
            plot_heatmap = class_comp
            colorscale = "Viridis"
            hm_title = "Spatial Composition by Cell Type Class (CCF regions)"

        fig_hm = px.imshow(
            plot_heatmap.T,
            aspect="auto",
            color_continuous_scale=colorscale,
            title=hm_title,
            labels={"x": "Cell Type Class", "y": "Brain Region", "color": "Value"},
        )
        fig_hm.update_layout(
            height=500,
            template="plotly_white",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.warning("No spatial composition data available.")

    # Full table and download
    with st.expander("Full results table", expanded=False):
        st.dataframe(
            ct_bias_anno[["EFFECT", "Rank", "class_label", "subclass_label",
                          "nt_type_label"]].style.format({"EFFECT": "{:.4f}"}),
            use_container_width=True,
            height=400,
        )

    csv_ct = ct_bias_anno.to_csv().encode("utf-8")
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
