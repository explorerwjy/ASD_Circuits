"""
webapp/core/bias.py
===================
Lightweight bias computation utilities for the GENCIC webapp.

This module provides a standalone implementation of the weighted-average
brain-structure bias algorithm, equivalent to ``ASD_Circuits.MouseSTR_AvgZ_Weighted``,
without importing the full ``src/ASD_Circuits.py`` module (which drags in
igraph, matplotlib, and other heavy dependencies).

The primary entry point is :func:`compute_str_bias`.

Typical usage
-------------
::

    import pandas as pd
    from webapp.core.data_loader import load_str_bias_matrix, load_gene_info
    from webapp.core.gene_mapping import build_gene_weight_dict
    from webapp.core.bias import compute_str_bias, load_str2region

    # Load data
    expr_matrix = load_str_bias_matrix()
    _, ENSID2Entrez, GeneSymbol2Entrez, _ = load_gene_info()

    # Build gene weights
    gene2weight, unmatched, _ = build_gene_weight_dict(
        gene_ids=["SHANK3", "SYNGAP1", "CHD8"],
        ENSID2Entrez=ENSID2Entrez,
        GeneSymbol2Entrez=GeneSymbol2Entrez,
    )

    # Load region annotations and compute bias
    str2region = load_str2region()
    bias_df = compute_str_bias(expr_matrix, gene2weight, str2region=str2region)
    # Returns DataFrame: index=structure name, columns=[EFFECT, Rank, REGION]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers (mirror data_loader.py convention)
# ---------------------------------------------------------------------------

_WEBAPP_DIR = Path(__file__).resolve().parent.parent   # .../webapp/
_PROJECT_ROOT = _WEBAPP_DIR.parent                     # .../ASD_Circuits_CellType/
_CONFIG_PATH = _WEBAPP_DIR / "config" / "webapp_config.yaml"


def _resolve(relative_to_webapp: str) -> Path:
    """Convert a '../dat/...' style path (relative to webapp/) to absolute."""
    return (_WEBAPP_DIR / relative_to_webapp).resolve()


def _load_webapp_config() -> dict:
    """Load webapp config (non-cached version for use outside Streamlit)."""
    with open(_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Structure-to-region mapping
# ---------------------------------------------------------------------------


def load_str2region(config: Optional[dict] = None) -> Dict[str, str]:
    """Load the brain-structure → major brain division mapping.

    Reads the ``major_brain_divisions`` file referenced in
    ``webapp_config.yaml``.  The TSV file is expected to have a header row
    with columns ``STR`` (structure name) and ``REG`` (region name),
    matching the format consumed by ``ASD_Circuits.STR2Region()``.

    Parameters
    ----------
    config : dict or None, optional
        Pre-loaded webapp config dict.  If ``None``, the config is loaded
        from ``webapp/config/webapp_config.yaml`` automatically.

    Returns
    -------
    dict[str, str]
        Mapping ``{structure_name: region_name}`` for all structures in the
        reference file.  Returns an empty dict if the file is not found,
        allowing downstream callers to continue without region annotation.

    Notes
    -----
    The REGION column in the bias DataFrame will be ``NaN`` for structures not
    present in the mapping (e.g., structures newly added to the expression
    matrix but not yet annotated in the reference file).
    """
    if config is None:
        config = _load_webapp_config()

    rel_path = config["data_files"].get("major_brain_divisions", "")
    if not rel_path:
        logger.warning("major_brain_divisions path not set in webapp_config.yaml")
        return {}

    abs_path = _resolve(rel_path)
    if not abs_path.exists():
        logger.warning("Structure-region map not found: %s", abs_path)
        return {}

    try:
        df = pd.read_csv(abs_path, delimiter="\t")
        # Canonical columns from ASD_Circuits.STR2Region()
        if "STR" in df.columns and "REG" in df.columns:
            str2reg = dict(zip(df["STR"].values, df["REG"].values))
        elif "structure" in df.columns and "region" in df.columns:
            # Fallback for alternative column names
            str2reg = dict(zip(df["structure"].values, df["region"].values))
        else:
            # Fall back to positional columns (no header assumed)
            df = pd.read_csv(abs_path, delimiter="\t", header=None)
            str2reg = dict(zip(df.iloc[:, 0].values, df.iloc[:, 1].values))
        logger.info(
            "Structure-region map loaded: %d structures", len(str2reg)
        )
        return str2reg
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load structure-region map: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Core bias computation
# ---------------------------------------------------------------------------


def compute_str_bias(
    expr_matrix: pd.DataFrame,
    gene2weight: Dict[int, float],
    str2region: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute weighted average bias scores for mouse brain structures.

    This is a lightweight, self-contained implementation of
    ``ASD_Circuits.MouseSTR_AvgZ_Weighted`` designed for use in the webapp
    without importing the full analysis library.

    The algorithm:

    1. Intersect the expression matrix genes (row index = Entrez IDs) with
       the keys of ``gene2weight``.
    2. For each brain structure (column), compute the weighted average of
       gene expression Z-scores, skipping ``NaN`` values.
    3. Rank structures by descending EFFECT score.
    4. Optionally annotate structures with major brain region labels.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Gene × structure expression Z-score matrix.  Index must be Entrez
        gene IDs (``int`` or ``Int64``); columns are brain-structure names.
        This is the matrix returned by ``load_str_bias_matrix()``.
    gene2weight : dict[int, float]
        Gene weights: mapping from Entrez ID (``int``) to weight (``float``).
        Can be produced by ``build_gene_weight_dict()``.
    str2region : dict[str, str] or None, optional
        Structure name → major brain division (e.g. ``"Isocortex"``).
        If provided, a ``REGION`` column is added to the output.  Structures
        not found in the mapping receive ``NaN``.

    Returns
    -------
    pd.DataFrame
        Indexed by brain-structure name.  Columns:

        * ``EFFECT`` (float) — weighted average Z-score across input genes
        * ``Rank`` (int) — rank by descending EFFECT (1 = highest bias)
        * ``REGION`` (str) — major brain division (only if ``str2region`` is
          provided and non-empty)

    Raises
    ------
    ValueError
        If ``gene2weight`` is empty, or if no genes from ``gene2weight``
        appear in the expression matrix index.

    Examples
    --------
    ::

        bias_df = compute_str_bias(expr_matrix, gene2weight, str2region)
        top10 = bias_df.head(10)
        print(top10[["EFFECT", "REGION"]])
    """
    if not gene2weight:
        raise ValueError(
            "gene2weight is empty — no genes to compute bias from. "
            "Check that gene identifiers were mapped successfully."
        )

    # Align gene weights with expression matrix index
    weights_series = pd.Series(gene2weight, dtype=float)

    # Ensure both indices are comparable ints
    try:
        weights_series.index = weights_series.index.astype(int)
    except (ValueError, TypeError):
        pass

    expr_index_int = pd.Index(expr_matrix.index).astype(int)

    valid_genes = expr_index_int.intersection(weights_series.index)

    if len(valid_genes) == 0:
        raise ValueError(
            f"No overlap between gene2weight ({len(gene2weight)} genes) and "
            f"expression matrix ({len(expr_matrix)} genes). "
            "Verify that gene IDs are Entrez IDs matching the expression matrix index."
        )

    logger.info(
        "compute_str_bias: %d/%d input genes found in expression matrix (%d structures)",
        len(valid_genes),
        len(gene2weight),
        len(expr_matrix.columns),
    )

    # Extract aligned weights and expression values
    weights = weights_series.loc[valid_genes].values  # shape (n_genes,)

    # Re-index expr_matrix by integer index to align with valid_genes
    expr_matrix_int_idx = expr_matrix.copy()
    expr_matrix_int_idx.index = expr_index_int
    expr_sub = expr_matrix_int_idx.loc[valid_genes]   # shape (n_genes, n_structures)

    # Vectorised weighted average, NaN-safe
    #   mask[i, j] = 1 if expr[i, j] is not NaN, else 0
    expr_vals = expr_sub.values                         # (n_genes, n_structures)
    nan_mask = ~np.isnan(expr_vals)                     # (n_genes, n_structures)

    weights_col = weights[:, np.newaxis]                # (n_genes, 1) for broadcasting

    weighted_sum = np.nansum(expr_vals * weights_col * nan_mask, axis=0)  # (n_structures,)
    weight_sum = np.sum(weights_col * nan_mask, axis=0)                   # (n_structures,)

    # Avoid division by zero (structures with no valid gene data)
    with np.errstate(invalid="ignore", divide="ignore"):
        effects = np.where(weight_sum > 0, weighted_sum / weight_sum, np.nan)

    # Build result DataFrame
    bias_df = pd.DataFrame(
        {"EFFECT": effects},
        index=expr_matrix.columns,
    )
    bias_df.index.name = "Structure"

    # Sort by descending bias
    bias_df = bias_df.sort_values("EFFECT", ascending=False)
    bias_df["Rank"] = np.arange(1, len(bias_df) + 1, dtype=int)

    # Add region annotation if provided
    if str2region:
        bias_df["REGION"] = bias_df.index.map(str2region)

    return bias_df


# ---------------------------------------------------------------------------
# Convenience: compute bias from a preset gene weight file
# ---------------------------------------------------------------------------


def compute_str_bias_from_gw_file(
    gw_path: str | Path,
    expr_matrix: pd.DataFrame,
    str2region: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Convenience wrapper: load a .gw file and compute structure bias.

    Parameters
    ----------
    gw_path : str or Path
        Path to a headerless CSV file with columns ``EntrezID,weight``.
    expr_matrix : pd.DataFrame
        Gene × structure expression matrix (from ``load_str_bias_matrix()``).
    str2region : dict[str, str] or None, optional
        Structure → region mapping (from ``load_str2region()``).

    Returns
    -------
    pd.DataFrame
        Same as :func:`compute_str_bias`.
    """
    path = Path(gw_path)
    if not path.exists():
        raise FileNotFoundError(f"Gene weight file not found: {path}")

    gw_df = pd.read_csv(path, header=None, names=["entrez_id", "weight"])
    gw_df["entrez_id"] = pd.to_numeric(gw_df["entrez_id"], errors="coerce")
    gw_df = gw_df.dropna(subset=["entrez_id"])
    gw_df["entrez_id"] = gw_df["entrez_id"].astype(int)

    gene2weight: Dict[int, float] = dict(
        zip(gw_df["entrez_id"].values, gw_df["weight"].values)
    )
    return compute_str_bias(expr_matrix, gene2weight, str2region=str2region)
