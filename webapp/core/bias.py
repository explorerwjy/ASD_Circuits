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
