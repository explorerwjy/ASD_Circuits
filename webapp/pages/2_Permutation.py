"""
pages/2_Permutation.py
======================
Permutation Testing — compute null bias distributions and empirical p-values.

Takes gene weights from session_state (redirects to the Bias Explorer if no
gene set has been loaded yet).  The user selects the number of permutations
and the analysis type; the page then:

1. Computes observed bias scores using the selected gene weights.
2. Generates null distributions by randomly sampling genes (uniform, same count).
3. Calculates empirical p-values and z-scores (GetPermutationP_vectorized logic).
4. Applies Benjamini-Hochberg FDR correction.
5. Displays an enhanced results table, a Plotly volcano plot, and a CSV download.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.gene_set_selector import gene_set_selector
from core.data_loader import (
    load_ct_bias_matrix,
    load_str_bias_matrix,
    load_structure_region_map,
    load_webapp_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Permutation Testing — GENCIC",
    page_icon="🎲",
    layout="wide",
)


# ===========================================================================
# Computation helpers
# ===========================================================================
# Lightweight reimplementations of ASD_Circuits functions to avoid pulling in
# igraph, matplotlib, and other heavy dependencies at import time.
# ---------------------------------------------------------------------------


def _compute_weighted_bias(
    expr_mat: pd.DataFrame,
    gene_weights: dict[int, float],
) -> pd.DataFrame:
    """Compute weighted average expression bias scores.

    Reimplements the core logic of ``MouseSTR_AvgZ_Weighted`` /
    ``MouseCT_AvgZ_Weighted`` without importing ``src.ASD_Circuits``.

    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression z-score matrix (genes x structures/cell types).
        Index = Entrez gene IDs, columns = feature names.
    gene_weights : dict[int, float]
        Gene weight mapping (Entrez ID -> weight).

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
    weighted_vals = expr_sub * w_bc * mask
    weight_sums = w_bc * mask

    with np.errstate(divide="ignore", invalid="ignore"):
        effects = np.sum(weighted_vals, axis=0) / np.sum(weight_sums, axis=0)

    df = pd.DataFrame({"EFFECT": effects}, index=expr_mat.columns)
    df = df.sort_values("EFFECT", ascending=False)
    df["Rank"] = np.arange(1, len(df) + 1)
    return df


def _permutation_p_values(
    null_matrix: np.ndarray,
    observed_vals: np.ndarray,
    greater_than: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized empirical permutation p-values.

    Reimplements ``GetPermutationP_vectorized`` from ``ASD_Circuits.py``.

    Parameters
    ----------
    null_matrix : np.ndarray, shape (n_perms, n_features)
    observed_vals : np.ndarray, shape (n_features,)
    greater_than : bool
        If True, right-tail test (count null >= observed).

    Returns
    -------
    z_scores, p_values, obs_adj : tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    mask = ~np.isnan(null_matrix)
    n_valid = np.sum(mask, axis=0)

    means = np.nanmean(null_matrix, axis=0)
    stds = np.nanstd(null_matrix, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = (observed_vals - means) / stds

    if greater_than:
        counts = np.nansum(null_matrix >= observed_vals[None, :], axis=0)
    else:
        counts = np.nansum(null_matrix <= observed_vals[None, :], axis=0)

    # Phipson-Smyth pseudo-count correction to avoid zero p-values
    p_values = (counts + 1) / (n_valid + 1)
    obs_adj = observed_vals - means

    return z_scores, p_values, obs_adj


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return p_values

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity (largest to smallest so adjusted values never decrease)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty_like(p_values)
    result[sorted_idx] = adjusted
    return result


def _run_permutation_test(
    expr_mat: pd.DataFrame,
    gene_weights: dict[int, float],
    n_perms: int,
    seed: int = 42,
    progress_bar=None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Run a full permutation test.

    For each permutation, randomly sample *n_genes* genes (same count as the
    user's gene set) uniformly from the expression matrix, assign the original
    weights (preserving the weight distribution), and compute bias scores.

    Parameters
    ----------
    expr_mat : pd.DataFrame
        Expression z-score matrix (genes x structures/cell types).
    gene_weights : dict[int, float]
        Observed gene weight mapping.
    n_perms : int
        Number of permutations.
    seed : int
        Random seed for reproducibility (project default: 42).
    progress_bar : st.progress, optional
        Streamlit progress bar to update during the loop.

    Returns
    -------
    result_df : pd.DataFrame
        Enhanced bias table with columns: EFFECT, Rank, Z-score, P-value,
        FDR, Bias_adjusted.  Index = feature names.
    null_matrix : np.ndarray, shape (n_perms, n_features)
        Full null distribution matrix (stored for optional downstream use).
    """
    rng = np.random.default_rng(seed)

    # ---- Overlap between gene set and expression matrix --------------------
    weights_series = pd.Series(gene_weights)
    valid_genes = expr_mat.index.intersection(weights_series.index)
    original_weights = weights_series[valid_genes].values
    n_genes = len(original_weights)

    if n_genes == 0:
        raise ValueError(
            "No genes in the selected gene set overlap with the expression "
            "matrix.  Cannot compute bias."
        )

    if n_genes > expr_mat.shape[0]:
        raise ValueError(
            f"Gene set ({n_genes} overlapping genes) is larger than the "
            f"expression matrix ({expr_mat.shape[0]} genes).  Cannot sample "
            f"without replacement."
        )

    n_features = expr_mat.shape[1]

    # ---- Observed bias (aligned to expr_mat.columns order) -----------------
    expr_valid = expr_mat.loc[valid_genes].values  # (n_genes, n_features)
    mask_obs = ~np.isnan(expr_valid)
    w_bc = original_weights[:, np.newaxis]

    with np.errstate(divide="ignore", invalid="ignore"):
        observed_effects = np.sum(expr_valid * w_bc * mask_obs, axis=0) / np.sum(
            w_bc * mask_obs, axis=0
        )

    # ---- Null distribution -------------------------------------------------
    expr_values = expr_mat.values  # (n_genes_total, n_features)
    n_total_genes = expr_values.shape[0]
    null_matrix = np.empty((n_perms, n_features), dtype=np.float64)

    # Update progress bar at most ~200 times to avoid Streamlit overhead
    update_interval = max(1, n_perms // 200)

    for i in range(n_perms):
        rand_idx = rng.choice(n_total_genes, size=n_genes, replace=False)
        expr_sub = expr_values[rand_idx]  # (n_genes, n_features)
        mask = ~np.isnan(expr_sub)
        weighted = expr_sub * w_bc * mask
        weight_total = w_bc * mask

        with np.errstate(divide="ignore", invalid="ignore"):
            null_matrix[i] = np.sum(weighted, axis=0) / np.sum(weight_total, axis=0)

        if progress_bar is not None and (i + 1) % update_interval == 0:
            progress_bar.progress(
                (i + 1) / n_perms,
                text=f"Permutation {i + 1:,} / {n_perms:,}",
            )

    # Final progress update
    if progress_bar is not None:
        progress_bar.progress(1.0, text=f"Permutation {n_perms:,} / {n_perms:,}")

    # ---- P-values & FDR ----------------------------------------------------
    z_scores, p_values, obs_adj = _permutation_p_values(
        null_matrix, observed_effects, greater_than=True
    )
    fdr = _benjamini_hochberg(p_values)

    # ---- Build result DataFrame --------------------------------------------
    result_df = pd.DataFrame(
        {
            "EFFECT": observed_effects,
            "Z-score": z_scores,
            "P-value": p_values,
            "FDR": fdr,
            "Bias_adjusted": obs_adj,
        },
        index=expr_mat.columns,
    )
    result_df = result_df.sort_values("EFFECT", ascending=False)
    result_df["Rank"] = np.arange(1, len(result_df) + 1)

    return result_df, null_matrix


# ===========================================================================
# Sidebar — gene set selector & analysis type
# ===========================================================================

# The gene_set_selector renders itself inside st.sidebar when sidebar=True.
gene_weights, gene_set_label = gene_set_selector(sidebar=True)

cfg = load_webapp_config()
analysis_types = cfg.get("analysis_types", {})
analysis_labels = {k: v["label"] for k, v in analysis_types.items()}

with st.sidebar:
    st.divider()
    selected_analysis = st.selectbox(
        "Analysis Type",
        options=list(analysis_labels.keys()),
        format_func=lambda k: analysis_labels[k],
        index=0,
        key="perm_analysis_type",
    )


# ===========================================================================
# Main content
# ===========================================================================

st.title("🎲 Permutation Testing")

# ---------------------------------------------------------------------------
# Guard: redirect to Bias Explorer if no gene set is loaded
# ---------------------------------------------------------------------------
if gene_weights is None:
    st.warning(
        "**No gene set loaded.** Please select a gene set in the sidebar, "
        "or go to the Bias Explorer page to load one first.",
        icon="⚠️",
    )
    if st.button("Go to Bias Explorer →"):
        st.switch_page("pages/01_bias_explorer.py")
    st.stop()

# ---------------------------------------------------------------------------
# Gene set & expression matrix summary
# ---------------------------------------------------------------------------
st.markdown(
    f"**Gene set:** {gene_set_label} ({len(gene_weights):,} genes)  \u2022  "
    f"**Analysis:** {analysis_labels.get(selected_analysis, selected_analysis)}"
)

with st.spinner("Loading expression matrix\u2026"):
    if selected_analysis == "STR_ISH":
        expr_mat = load_str_bias_matrix()
    else:
        expr_mat = load_ct_bias_matrix()

# Show overlap info
overlap_genes = expr_mat.index.intersection(pd.Series(gene_weights).index)
feature_label = "structures" if selected_analysis == "STR_ISH" else "cell types"

st.info(
    f"**{len(overlap_genes):,}** of {len(gene_weights):,} genes overlap with "
    f"the expression matrix ({expr_mat.shape[0]:,} genes \u00d7 "
    f"{expr_mat.shape[1]:,} {feature_label})",
    icon="\u2139\ufe0f",
)

# ---------------------------------------------------------------------------
# Permutation settings
# ---------------------------------------------------------------------------
st.subheader("Settings")

col_perms, col_seed, col_fdr = st.columns([2, 1, 1])

with col_perms:
    n_perms = st.slider(
        "Number of permutations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="n_perms_slider",
        help="More permutations = more accurate p-values but slower computation.",
    )

with col_seed:
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=42,
        key="perm_seed",
        help="Seed for reproducibility (project default: 42).",
    )

with col_fdr:
    fdr_threshold = st.number_input(
        "FDR threshold",
        min_value=0.001,
        max_value=0.500,
        value=0.050,
        step=0.010,
        format="%.3f",
        key="fdr_threshold",
        help="Significance threshold after Benjamini-Hochberg FDR correction.",
    )

run_btn = st.button(
    "\U0001f680 Run Permutations",
    use_container_width=True,
    type="primary",
)

# ---------------------------------------------------------------------------
# Run permutation test
# ---------------------------------------------------------------------------
if run_btn:
    st.divider()
    progress_bar = st.progress(0, text="Starting permutation test\u2026")

    try:
        result_df, null_matrix = _run_permutation_test(
            expr_mat=expr_mat,
            gene_weights=gene_weights,
            n_perms=n_perms,
            seed=seed,
            progress_bar=progress_bar,
        )
    except ValueError as exc:
        progress_bar.empty()
        st.error(str(exc), icon="\u274c")
        st.stop()

    progress_bar.empty()

    # Add region info for structure-level analysis
    if selected_analysis == "STR_ISH":
        region_map = load_structure_region_map()
        result_df["Region"] = result_df.index.map(
            lambda s: region_map.get(s, "Unknown")
        )

    # Persist results in session_state
    st.session_state["permutation_results"] = result_df
    st.session_state["permutation_null_matrix"] = null_matrix
    st.session_state["perm_gene_set_label"] = gene_set_label
    st.session_state["perm_analysis_type_used"] = selected_analysis
    st.session_state["perm_n_perms"] = n_perms

    st.success(
        f"Permutation test complete ({n_perms:,} permutations, seed={seed}).",
        icon="\u2705",
    )

# ---------------------------------------------------------------------------
# Display results (if available in session_state)
# ---------------------------------------------------------------------------
if "permutation_results" not in st.session_state:
    st.info(
        "Configure settings above and click **Run Permutations** to start.",
        icon="\U0001f4a1",
    )
    st.stop()

result_df: pd.DataFrame = st.session_state["permutation_results"]
analysis_used = st.session_state.get("perm_analysis_type_used", selected_analysis)
is_str = analysis_used == "STR_ISH"

# Mark significance using the *current* FDR threshold (reactive to slider)
result_df["Significant"] = result_df["FDR"] < fdr_threshold

st.divider()

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
n_sig = int(result_df["Significant"].sum())
n_total = len(result_df)

met1, met2, met3, met4 = st.columns(4)
met1.metric("Total features", f"{n_total:,}")
met2.metric(f"Significant (FDR < {fdr_threshold:.2f})", f"{n_sig:,}")
met3.metric("Min p-value", f"{result_df['P-value'].min():.2e}")
met4.metric(
    "Max |z-score|",
    f"{np.nanmax(np.abs(result_df['Z-score'].values)):.2f}",
)

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
st.subheader("Bias Table with Permutation Statistics")

display_df = result_df.sort_values("P-value").copy()

display_cols = ["EFFECT", "Rank"]
if "Region" in display_df.columns:
    display_cols.append("Region")
display_cols += ["Z-score", "P-value", "FDR", "Significant"]

format_dict = {
    "EFFECT": "{:.4f}",
    "Z-score": "{:.2f}",
    "P-value": "{:.2e}",
    "FDR": "{:.2e}",
}


def _highlight_significant(row: pd.Series) -> list[str]:
    """Highlight significant rows with a golden background."""
    style = "background-color: rgba(255, 215, 0, 0.15)" if row["Significant"] else ""
    return [style] * len(row)


st.dataframe(
    display_df[display_cols]
    .style.format(format_dict)
    .apply(_highlight_significant, axis=1),
    use_container_width=True,
    height=500,
)

# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------
st.subheader("Volcano Plot")

plot_df = result_df.copy()
plot_df["-log10(p)"] = -np.log10(plot_df["P-value"].clip(lower=1e-300))
plot_df["Name"] = plot_df.index.astype(str)
sig_label = f"Significant (FDR < {fdr_threshold})"
plot_df["Status"] = np.where(plot_df["Significant"], sig_label, "Not significant")

fig = px.scatter(
    plot_df,
    x="EFFECT",
    y="-log10(p)",
    color="Status",
    hover_name="Name",
    hover_data={
        "EFFECT": ":.4f",
        "Z-score": ":.2f",
        "P-value": ":.2e",
        "FDR": ":.2e",
        "-log10(p)": False,
        "Status": False,
    },
    color_discrete_map={
        sig_label: "#e74c3c",
        "Not significant": "#95a5a6",
    },
    labels={
        "EFFECT": "Bias (weighted average z-score)",
        "-log10(p)": "-log\u2081\u2080(p-value)",
    },
    title="Bias vs. Statistical Significance",
)

# Add a horizontal dashed line at the FDR threshold boundary
sig_rows = plot_df.loc[plot_df["Significant"]]
if len(sig_rows) > 0:
    # The threshold line sits at the -log10(p) of the largest p-value that
    # still passes the FDR threshold.
    max_sig_p = sig_rows["P-value"].max()
    fig.add_hline(
        y=-np.log10(max_sig_p),
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text=f"FDR = {fdr_threshold}",
        annotation_position="top right",
    )

# Label the top significant features
if len(sig_rows) > 0:
    top_sig = sig_rows.nsmallest(10, "P-value")
    fig.add_trace(
        go.Scatter(
            x=top_sig["EFFECT"],
            y=-np.log10(top_sig["P-value"].clip(lower=1e-300)),
            mode="text",
            text=top_sig.index.astype(str),
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

fig.update_layout(
    template="plotly_white",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Significant features detail
# ---------------------------------------------------------------------------
if n_sig > 0:
    feature_type = "Structures" if is_str else "Cell Types"
    st.subheader(f"Significant {feature_type} (FDR < {fdr_threshold})")

    sig_df = result_df[result_df["Significant"]].sort_values("P-value")
    sig_cols = ["EFFECT", "Rank"]
    if "Region" in sig_df.columns:
        sig_cols.append("Region")
    sig_cols += ["Z-score", "P-value", "FDR"]

    st.dataframe(
        sig_df[sig_cols].style.format(format_dict),
        use_container_width=True,
    )

    # Region breakdown (structure analysis only)
    if is_str and "Region" in sig_df.columns:
        st.markdown("**Region breakdown of significant structures:**")
        region_counts = sig_df["Region"].value_counts()
        col_chart, col_table = st.columns([2, 1])
        with col_chart:
            st.bar_chart(region_counts)
        with col_table:
            st.dataframe(
                region_counts.rename("Count").to_frame(),
                use_container_width=True,
            )

# ---------------------------------------------------------------------------
# Download enriched CSV
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Download Results")

csv_bytes = result_df.to_csv().encode("utf-8")
gs_label = st.session_state.get("perm_gene_set_label", "unknown")
n_p = st.session_state.get("perm_n_perms", n_perms)
filename = f"permutation_results_{gs_label}_{n_p}perms.csv"

st.download_button(
    label="\U0001f4e5 Download enriched CSV",
    data=csv_bytes,
    file_name=filename,
    mime="text/csv",
    use_container_width=True,
)
