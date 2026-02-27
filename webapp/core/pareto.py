"""
webapp/core/pareto.py
======================
Extract Pareto-optimal fronts from SA circuit-search results.

A solution is **Pareto-optimal** (non-dominated) when no other solution is
simultaneously at-least-as-good in *both* objectives and strictly better in
at least one.

Objectives
----------
1. **mean_bias** — higher is better (biological relevance).
2. **circuit_score** — higher is better (connectivity coherence).

The trade-off: forcing the circuit toward higher-bias structures may
reduce connectivity, and vice versa.  The Pareto front captures the
boundary of achievable (bias, connectivity) combinations.

Usage
-----
::

    from webapp.core.pareto import extract_pareto_front

    # df has columns: bias_limit, circuit_score, mean_bias, structures
    pareto_df = extract_pareto_front(df)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Ensure src/ is importable for scoring functions
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ASD_Circuits import ScoreCircuit_SI_Joint, ScoreCircuit_NEdges

logger = logging.getLogger(__name__)

# Required columns in the input DataFrame
_REQUIRED_COLS = {"bias_limit", "circuit_score", "mean_bias", "structures"}


# ---------------------------------------------------------------------------
# Core Pareto logic
# ---------------------------------------------------------------------------

def is_dominated(
    point: np.ndarray,
    others: np.ndarray,
) -> bool:
    """Check whether *point* is dominated by any row in *others*.

    Both *point* and each row of *others* are vectors of objectives where
    **higher is better**.

    Parameters
    ----------
    point : np.ndarray, shape (n_objectives,)
    others : np.ndarray, shape (n_points, n_objectives)

    Returns
    -------
    bool
        True if at least one row in *others* dominates *point*.
    """
    if len(others) == 0:
        return False
    # A row dominates `point` if it is >= in all objectives AND > in at least one
    at_least_as_good = np.all(others >= point, axis=1)
    strictly_better = np.any(others > point, axis=1)
    return bool(np.any(at_least_as_good & strictly_better))


def pareto_front_indices(values: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated rows (higher is better in all cols).

    Parameters
    ----------
    values : np.ndarray, shape (n, k)
        Each row is a k-objective vector.  Higher values are preferred.

    Returns
    -------
    np.ndarray
        Sorted indices of non-dominated rows.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=int)

    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # Check if any other unmasked point dominates i
        others_mask = mask.copy()
        others_mask[i] = False
        if is_dominated(values[i], values[others_mask]):
            mask[i] = False

    return np.where(mask)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pareto_front(
    results_df: pd.DataFrame,
    include_baseline: bool = True,
    bias_df: Optional[pd.DataFrame] = None,
    info_mat: Optional[pd.DataFrame] = None,
    adj_mat: Optional[pd.DataFrame] = None,
    circuit_size: Optional[int] = None,
    measure: str = "SI",
) -> pd.DataFrame:
    """Extract the Pareto-optimal front from SA search results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Best-per-bias-limit results.  Must contain columns:
        ``bias_limit``, ``circuit_score``, ``mean_bias``, ``structures``.
    include_baseline : bool
        If True, compute and prepend a "baseline" circuit (top *N*
        structures by bias, no SA optimisation).  Requires *bias_df*,
        *info_mat*/*adj_mat*, and *circuit_size*.
    bias_df : pd.DataFrame, optional
        Bias DataFrame (needed for baseline calculation).
    info_mat : pd.DataFrame, optional
        Information matrix (needed if measure == "SI").
    adj_mat : pd.DataFrame, optional
        Adjacency matrix (needed if measure == "Connectivity").
    circuit_size : int, optional
        Circuit size (needed for baseline).
    measure : str
        ``"SI"`` or ``"Connectivity"``.

    Returns
    -------
    pd.DataFrame
        Pareto front with columns:
        ``bias_limit``, ``circuit_score``, ``mean_bias``,
        ``structures``, ``n_structures``, ``circuit_type``.
        Rows are sorted by ``mean_bias`` ascending.
    """
    missing = _REQUIRED_COLS - set(results_df.columns)
    if missing:
        raise ValueError(
            f"results_df is missing required columns: {missing}"
        )

    df = results_df.copy()
    df["circuit_type"] = "optimized"

    # Optionally add baseline (unoptimized top-N by bias)
    if include_baseline and bias_df is not None and circuit_size is not None:
        baseline_row = _compute_baseline(
            bias_df, info_mat, adj_mat, circuit_size, measure,
        )
        if baseline_row is not None:
            baseline_df = pd.DataFrame([baseline_row])
            # Align dtypes to avoid FutureWarning on concat with NA columns
            for col in df.columns:
                if col in baseline_df.columns and col != "bias_limit":
                    try:
                        baseline_df[col] = baseline_df[col].astype(
                            df[col].dtype
                        )
                    except (TypeError, ValueError):
                        pass
            df = pd.concat([baseline_df, df], ignore_index=True)

    if df.empty:
        return df

    # Ensure n_structures column
    if "n_structures" not in df.columns:
        df["n_structures"] = df["structures"].apply(
            lambda s: len(s.split(",")) if isinstance(s, str) else 0
        )

    # Extract Pareto front (higher mean_bias and higher circuit_score
    # are both preferred)
    objectives = df[["mean_bias", "circuit_score"]].values
    front_idx = pareto_front_indices(objectives)

    pareto_df = df.iloc[front_idx].copy()
    pareto_df = pareto_df.sort_values("mean_bias").reset_index(drop=True)

    logger.info(
        "Pareto front: %d / %d solutions retained",
        len(pareto_df), len(df),
    )
    return pareto_df


def compute_full_pareto(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convenience: extract Pareto front without baseline.

    For use when you only have SA results and no access to raw data
    matrices (e.g., from cached results).

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain ``bias_limit``, ``circuit_score``, ``mean_bias``,
        ``structures``.

    Returns
    -------
    pd.DataFrame
        Pareto front sorted by mean_bias ascending.
    """
    return extract_pareto_front(results_df, include_baseline=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_baseline(
    bias_df: pd.DataFrame,
    info_mat: Optional[pd.DataFrame],
    adj_mat: Optional[pd.DataFrame],
    circuit_size: int,
    measure: str,
) -> Optional[dict]:
    """Compute the baseline (unoptimized) circuit from top-N structures.

    Returns
    -------
    dict or None
        Row dict compatible with the results DataFrame, or None on failure.
    """
    try:
        top_strs = bias_df.head(circuit_size).index.values
        mean_bias = bias_df.head(circuit_size)["EFFECT"].mean()

        if measure == "SI":
            if info_mat is None:
                logger.warning("info_mat required for SI baseline")
                return None
            score = ScoreCircuit_SI_Joint(top_strs, info_mat)
        elif measure == "Connectivity":
            if adj_mat is None:
                logger.warning("adj_mat required for Connectivity baseline")
                return None
            score = ScoreCircuit_NEdges(top_strs, adj_mat)
        else:
            logger.warning("Unknown measure '%s' for baseline", measure)
            return None

        return {
            "bias_limit": np.nan,  # No bias constraint for baseline
            "circuit_score": score,
            "mean_bias": mean_bias,
            "structures": ",".join(top_strs),
            "n_structures": len(top_strs),
            "circuit_type": "baseline",
        }
    except Exception:
        logger.warning("Baseline calculation failed", exc_info=True)
        return None
