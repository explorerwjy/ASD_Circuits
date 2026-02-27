"""
webapp/core/circuit_search.py
==============================
Threaded wrapper for simulated-annealing circuit search.

The :class:`CircuitSearchRunner` runs the full SA search pipeline in a
background :class:`threading.Thread` so that the Streamlit UI thread can
poll a shared ``progress`` dict without blocking.

Typical usage from a Streamlit page::

    runner = CircuitSearchRunner(
        bias_df=bias_df,
        circuit_size=46,
        info_mat=info_mat,
        adj_mat=adj_mat,
    )
    runner.start()

    while runner.is_alive():
        st.write(runner.progress)
        time.sleep(0.5)

    results_df = runner.results_df
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ is importable (same strategy as scripts/workflow/*.py)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Import SA classes — prefer optimized, fall back to original
try:
    from SA_optimized import (
        CircuitSearch_SA_InfoContent_Optimized as _SA_InfoContent,
        CircuitSearch_SA_Connectivity_Optimized as _SA_Connectivity,
    )
except ImportError:
    from ASD_Circuits import (  # type: ignore[assignment]
        CircuitSearch_SA_InfoContent as _SA_InfoContent,
        CircuitSearch_SA_Connectivity as _SA_Connectivity,
    )

from ASD_Circuits import BiasLim, ScoreCircuit_SI_Joint, ScoreCircuit_NEdges

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Webapp-tuned defaults (faster than the Snakemake pipeline defaults)
# ---------------------------------------------------------------------------
WEBAPP_SA_STEPS = 10_000      # pipeline uses 50_000
WEBAPP_SA_RUNTIMES = 3        # pipeline uses 10
WEBAPP_TMAX = 1e-2
WEBAPP_TMIN = 5e-5
WEBAPP_TOP_N = 213
WEBAPP_MEASURE = "SI"


# ---------------------------------------------------------------------------
# Helpers (adapted from scripts/workflow/run_sa_search.py)
# ---------------------------------------------------------------------------

def find_init_state(
    bias_df: pd.DataFrame,
    size: int,
    min_bias: float,
    rng: np.random.Generator,
    max_attempts: int = 10_000,
) -> np.ndarray:
    """Find an initial binary state satisfying the bias constraint.

    Uses bias-weighted sampling so that higher-bias structures are more
    likely to be selected, making it much easier to satisfy the constraint.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Candidate structures, sorted by EFFECT (descending).  Must have an
        ``EFFECT`` column.
    size : int
        Number of structures to include in the circuit.
    min_bias : float
        Minimum mean-bias constraint.
    rng : np.random.Generator
        NumPy random generator for reproducibility.
    max_attempts : int
        Safety limit on attempts before raising.

    Returns
    -------
    np.ndarray
        Array of structure names in the initial circuit.

    Raises
    ------
    ValueError
        If no valid initial state is found within *max_attempts*.
    """
    strs = bias_df.index.values
    biases = bias_df["EFFECT"].values
    min_b = biases.min()
    pseudo = biases - min_b + 1.0
    # Sharpen the distribution based on constraint strength
    exponent = max(min_bias * 150 - 17, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        pseudo = np.power(pseudo, exponent)
    # Guard against overflow / NaN — fall back to uniform
    if not np.all(np.isfinite(pseudo)) or pseudo.sum() == 0:
        probs = np.ones(len(strs)) / len(strs)
    else:
        probs = pseudo / pseudo.sum()
    # Fix floating-point rounding so probabilities sum to exactly 1
    probs[-1] = 1.0 - probs[:-1].sum()

    for _ in range(max_attempts):
        chosen = rng.choice(strs, size=size, replace=False, p=probs)
        if bias_df.loc[chosen, "EFFECT"].mean() >= min_bias:
            return chosen

    raise ValueError(
        f"Cannot find initial state with mean bias >= {min_bias} "
        f"after {max_attempts} attempts"
    )


def run_single_sa(
    bias_df: pd.DataFrame,
    adj_mat: pd.DataFrame,
    info_mat: pd.DataFrame,
    top_n: int,
    circuit_size: int,
    min_bias: float,
    measure: str,
    steps: int,
    Tmax: float,
    Tmin: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Execute a single SA optimisation run.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Full bias DataFrame (all structures), sorted by EFFECT descending.
    adj_mat, info_mat : pd.DataFrame
        Connectome matrices (213 x 213).
    top_n : int
        Number of top-ranked structures to use as candidates.
    circuit_size : int
        Target circuit size.
    min_bias : float
        Minimum mean-bias constraint.
    measure : str
        ``"SI"`` for Shannon information, ``"Connectivity"`` for edge count.
    steps, Tmax, Tmin : SA temperature schedule parameters.
    rng : np.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    dict
        Keys: ``structures`` (np.ndarray), ``score`` (float),
        ``mean_bias`` (float).
    """
    # Candidate pool = top_n structures by bias
    candidate_df = bias_df.head(top_n)
    candidate_nodes = candidate_df.index.values

    # Build initial state
    init_strs = find_init_state(candidate_df, circuit_size, min_bias, rng)
    init_state = np.zeros(top_n, dtype=np.float64)
    for i, node in enumerate(candidate_nodes):
        if node in init_strs:
            init_state[i] = 1

    # Seed numpy global RNG for SA internals (SA uses np.random.choice)
    np.random.seed(int(rng.integers(0, 2**31)))

    # Instantiate SA
    if measure == "SI":
        sa = _SA_InfoContent(
            candidate_df, init_state, adj_mat, info_mat,
            candidate_nodes, minbias=min_bias,
        )
    elif measure == "Connectivity":
        sa = _SA_Connectivity(
            candidate_df, init_state, adj_mat, info_mat,
            candidate_nodes, minbias=min_bias,
        )
    else:
        raise ValueError(f"Unknown measure: {measure!r}")

    sa.copy_strategy = "method"
    sa.Tmax = Tmax
    sa.Tmin = Tmin
    sa.steps = steps
    sa.updates = 0  # no console output

    _, _, best_state, best_energy = sa.anneal()

    result_strs = candidate_nodes[np.where(best_state == 1)[0]]
    score = -best_energy
    mean_bias = bias_df.loc[result_strs, "EFFECT"].mean()

    return {
        "structures": result_strs,
        "score": score,
        "mean_bias": mean_bias,
    }


def generate_bias_limits(
    bias_df: pd.DataFrame,
    circuit_size: int,
    min_bias_rank: int = 50,
) -> List[float]:
    """Generate filtered bias limits for a given circuit size.

    Wraps :func:`ASD_Circuits.BiasLim` and filters to keep only limits
    above the bias value at *min_bias_rank*.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Bias DataFrame sorted by EFFECT descending.
    circuit_size : int
        Target circuit size.
    min_bias_rank : int
        Rank used to set the minimum bias threshold.

    Returns
    -------
    list[float]
        Sorted list of bias limit values.
    """
    raw = BiasLim(bias_df, circuit_size)
    # BiasLim returns list of (size, bias) tuples
    all_limits = sorted(set(b for _, b in raw))

    # Filter by min_bias_rank threshold
    if min_bias_rank <= len(bias_df):
        threshold = bias_df.iloc[min_bias_rank - 1]["EFFECT"]
    else:
        threshold = bias_df.iloc[-1]["EFFECT"]

    return [b for b in all_limits if b >= threshold]


# ---------------------------------------------------------------------------
# Threaded runner
# ---------------------------------------------------------------------------

@dataclass
class CircuitSearchResult:
    """Result from a single SA run at a particular bias limit."""

    bias_limit: float
    runtime_idx: int
    score: float
    mean_bias: float
    structures: np.ndarray


class CircuitSearchRunner:
    """Run SA circuit search in a background thread.

    Parameters
    ----------
    bias_df : pd.DataFrame
        Structure bias DataFrame (sorted by EFFECT descending).  Must
        contain an ``EFFECT`` column.
    circuit_size : int
        Number of structures in the target circuit.
    info_mat : pd.DataFrame
        Shannon-information matrix (213 x 213).
    adj_mat : pd.DataFrame
        Connectome weight matrix (213 x 213).
    steps : int
        SA steps per run (default: webapp-tuned 10 000).
    runtimes : int
        Independent SA runs per bias limit (default: 3).
    Tmax, Tmin : float
        SA temperature bounds.
    measure : str
        ``"SI"`` or ``"Connectivity"``.
    top_n : int
        Number of top candidate structures.
    min_bias_rank : int
        Rank for bias-limit filtering.
    seed : int
        Base random seed for reproducibility.

    Attributes
    ----------
    progress : dict
        Live progress dict polled by the Streamlit thread.  Keys:

        - ``status`` : str — ``"pending"`` | ``"running"`` | ``"done"`` | ``"error"`` | ``"cancelled"``
        - ``current_bias_limit`` : float | None
        - ``current_bias_limit_idx`` : int
        - ``total_bias_limits`` : int
        - ``current_runtime`` : int
        - ``total_runtimes`` : int
        - ``best_score`` : float
        - ``percent_complete`` : float (0–100)
        - ``elapsed_seconds`` : float
        - ``error`` : str | None

    results : list[CircuitSearchResult]
        Populated when the search finishes.

    results_df : pd.DataFrame | None
        Best circuit per bias limit as a DataFrame.
    """

    def __init__(
        self,
        bias_df: pd.DataFrame,
        circuit_size: int,
        info_mat: pd.DataFrame,
        adj_mat: pd.DataFrame,
        *,
        steps: int = WEBAPP_SA_STEPS,
        runtimes: int = WEBAPP_SA_RUNTIMES,
        Tmax: float = WEBAPP_TMAX,
        Tmin: float = WEBAPP_TMIN,
        measure: str = WEBAPP_MEASURE,
        top_n: int = WEBAPP_TOP_N,
        min_bias_rank: int = 50,
        seed: int = 42,
    ) -> None:
        # Data
        self.bias_df = bias_df
        self.circuit_size = circuit_size
        self.info_mat = info_mat
        self.adj_mat = adj_mat

        # SA parameters
        self.steps = steps
        self.runtimes = runtimes
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.measure = measure
        self.top_n = min(top_n, len(bias_df))
        self.min_bias_rank = min_bias_rank
        self.seed = seed

        # State
        self.results: List[CircuitSearchResult] = []
        self._thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        # Shared progress dict (read by Streamlit main thread)
        self.progress: Dict[str, Any] = {
            "status": "pending",
            "current_bias_limit": None,
            "current_bias_limit_idx": 0,
            "total_bias_limits": 0,
            "current_runtime": 0,
            "total_runtimes": runtimes,
            "best_score": 0.0,
            "percent_complete": 0.0,
            "elapsed_seconds": 0.0,
            "error": None,
        }

    # -- public API ---------------------------------------------------------

    def start(self) -> None:
        """Launch the search in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Search is already running")
        self._cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="circuit-search"
        )
        self._thread.start()

    def is_alive(self) -> bool:
        """Return True if the background thread is still running."""
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        """Signal the background thread to stop as soon as possible."""
        self._cancel_event.set()

    @property
    def results_df(self) -> Optional[pd.DataFrame]:
        """Best circuit per bias limit as a tidy DataFrame.

        Returns None if no results are available yet.
        """
        if not self.results:
            return None
        return best_per_bias_limit(self.results)

    # -- internal -----------------------------------------------------------

    def _run(self) -> None:
        """Main loop executed in the background thread."""
        t0 = time.monotonic()
        try:
            self.progress["status"] = "running"

            # 1. Generate bias limits
            bias_limits = generate_bias_limits(
                self.bias_df, self.circuit_size, self.min_bias_rank
            )
            if not bias_limits:
                raise ValueError(
                    "No valid bias limits generated.  Check circuit_size "
                    "and min_bias_rank."
                )

            n_limits = len(bias_limits)
            self.progress["total_bias_limits"] = n_limits
            total_work = n_limits * self.runtimes

            rng = np.random.default_rng(self.seed)

            # 2. Iterate over bias limits
            for blim_idx, blim in enumerate(bias_limits):
                if self._cancel_event.is_set():
                    self.progress["status"] = "cancelled"
                    return

                self.progress["current_bias_limit"] = blim
                self.progress["current_bias_limit_idx"] = blim_idx

                # 3. Multiple SA runs per bias limit
                for rt in range(self.runtimes):
                    if self._cancel_event.is_set():
                        self.progress["status"] = "cancelled"
                        return

                    self.progress["current_runtime"] = rt + 1

                    try:
                        result = run_single_sa(
                            bias_df=self.bias_df,
                            adj_mat=self.adj_mat,
                            info_mat=self.info_mat,
                            top_n=self.top_n,
                            circuit_size=self.circuit_size,
                            min_bias=blim,
                            measure=self.measure,
                            steps=self.steps,
                            Tmax=self.Tmax,
                            Tmin=self.Tmin,
                            rng=rng,
                        )

                        self.results.append(
                            CircuitSearchResult(
                                bias_limit=blim,
                                runtime_idx=rt,
                                score=result["score"],
                                mean_bias=result["mean_bias"],
                                structures=result["structures"],
                            )
                        )

                        if result["score"] > self.progress["best_score"]:
                            self.progress["best_score"] = result["score"]

                    except Exception:
                        logger.warning(
                            "SA run failed (bias=%.3f, rt=%d): %s",
                            blim, rt, traceback.format_exc(),
                        )

                    # Update progress
                    done = blim_idx * self.runtimes + (rt + 1)
                    self.progress["percent_complete"] = (
                        100.0 * done / total_work
                    )
                    self.progress["elapsed_seconds"] = (
                        time.monotonic() - t0
                    )

            self.progress["status"] = "done"
            self.progress["percent_complete"] = 100.0

        except Exception as exc:
            self.progress["status"] = "error"
            self.progress["error"] = str(exc)
            logger.error("Circuit search failed: %s", traceback.format_exc())

        finally:
            self.progress["elapsed_seconds"] = time.monotonic() - t0


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def best_per_bias_limit(
    results: List[CircuitSearchResult],
) -> pd.DataFrame:
    """Select the best SA run (highest score) per bias limit.

    Parameters
    ----------
    results : list[CircuitSearchResult]
        All SA run results.

    Returns
    -------
    pd.DataFrame
        Columns: ``bias_limit``, ``circuit_score``, ``mean_bias``,
        ``structures`` (comma-separated string), ``n_structures``.
        Sorted by ``bias_limit`` ascending.
    """
    if not results:
        return pd.DataFrame(
            columns=["bias_limit", "circuit_score", "mean_bias",
                     "structures", "n_structures"]
        )

    # Group by bias limit, keep best score
    best: Dict[float, CircuitSearchResult] = {}
    for r in results:
        if r.bias_limit not in best or r.score > best[r.bias_limit].score:
            best[r.bias_limit] = r

    rows = []
    for blim in sorted(best):
        r = best[blim]
        rows.append({
            "bias_limit": r.bias_limit,
            "circuit_score": r.score,
            "mean_bias": r.mean_bias,
            "structures": ",".join(r.structures),
            "n_structures": len(r.structures),
        })

    return pd.DataFrame(rows)
