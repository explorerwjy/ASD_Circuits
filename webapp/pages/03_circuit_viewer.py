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
