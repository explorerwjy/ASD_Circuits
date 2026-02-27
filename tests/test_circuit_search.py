"""
tests/test_circuit_search.py
==============================
Tests for the webapp SA circuit search wrapper and Pareto front extraction.

Unit tests use synthetic 30-structure matrices for speed.
Integration tests use real 213-structure data (skipped if unavailable).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from webapp.core.circuit_search import (
    CircuitSearchResult,
    CircuitSearchRunner,
    best_per_bias_limit,
    find_init_state,
    generate_bias_limits,
    run_single_sa,
    WEBAPP_SA_STEPS,
    WEBAPP_SA_RUNTIMES,
)
from webapp.core.pareto import (
    compute_full_pareto,
    extract_pareto_front,
    is_dominated,
    pareto_front_indices,
)
from tests.conftest import skip_without_real_data


# ============================================================================
# Unit tests — find_init_state
# ============================================================================

class TestFindInitState:
    """Tests for the initial state finder."""

    def test_returns_correct_size(self, synthetic_bias_df, rng):
        """Initial state should have exactly circuit_size structures."""
        size = 5
        min_bias = synthetic_bias_df.head(size)["EFFECT"].mean() * 0.5
        result = find_init_state(synthetic_bias_df, size, min_bias, rng)
        assert len(result) == size

    def test_satisfies_bias_constraint(self, synthetic_bias_df, rng):
        """Mean bias of initial state should be >= min_bias."""
        size = 5
        min_bias = 0.05  # low threshold, should be easy
        result = find_init_state(synthetic_bias_df, size, min_bias, rng)
        actual_bias = synthetic_bias_df.loc[result, "EFFECT"].mean()
        assert actual_bias >= min_bias

    def test_no_duplicate_structures(self, synthetic_bias_df, rng):
        """All structures in initial state should be unique."""
        size = 8
        min_bias = 0.01
        result = find_init_state(synthetic_bias_df, size, min_bias, rng)
        assert len(result) == len(set(result))

    def test_raises_for_impossible_constraint(self, synthetic_bias_df, rng):
        """Should raise ValueError if bias constraint is impossibly high."""
        with pytest.raises(ValueError, match="Cannot find initial state"):
            find_init_state(synthetic_bias_df, 5, 999.0, rng, max_attempts=100)

    def test_reproducible_with_same_seed(self, synthetic_bias_df):
        """Same seed should produce same initial state."""
        size = 5
        min_bias = 0.05
        r1 = find_init_state(synthetic_bias_df, size, min_bias,
                             np.random.default_rng(123))
        r2 = find_init_state(synthetic_bias_df, size, min_bias,
                             np.random.default_rng(123))
        np.testing.assert_array_equal(r1, r2)


# ============================================================================
# Unit tests — generate_bias_limits
# ============================================================================

class TestGenerateBiasLimits:
    """Tests for bias limit generation."""

    def test_returns_sorted_floats(self, synthetic_bias_df):
        limits = generate_bias_limits(synthetic_bias_df, circuit_size=5)
        assert len(limits) > 0
        assert all(isinstance(b, float) for b in limits)
        assert limits == sorted(limits)

    def test_no_duplicates(self, synthetic_bias_df):
        limits = generate_bias_limits(synthetic_bias_df, circuit_size=5)
        assert len(limits) == len(set(limits))

    def test_filtered_by_min_bias_rank(self, synthetic_bias_df):
        """All limits should be >= the EFFECT at min_bias_rank."""
        min_rank = 10
        limits = generate_bias_limits(
            synthetic_bias_df, circuit_size=5, min_bias_rank=min_rank,
        )
        threshold = synthetic_bias_df.iloc[min_rank - 1]["EFFECT"]
        assert all(b >= threshold for b in limits)

    def test_higher_rank_gives_fewer_limits(self, synthetic_bias_df):
        """A stricter threshold (lower rank = higher EFFECT) gives fewer limits."""
        lims_lenient = generate_bias_limits(
            synthetic_bias_df, circuit_size=5, min_bias_rank=25,
        )
        lims_strict = generate_bias_limits(
            synthetic_bias_df, circuit_size=5, min_bias_rank=5,
        )
        # Lower rank means higher EFFECT threshold → more limits pass
        # Higher rank means lower EFFECT threshold → fewer limits pass
        # So lenient (rank 25, low threshold) should have fewer limits
        assert len(lims_strict) <= len(lims_lenient)


# ============================================================================
# Unit tests — run_single_sa
# ============================================================================

class TestRunSingleSA:
    """Tests for a single SA optimization run."""

    def test_returns_valid_result(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """SA should return a dict with structures, score, mean_bias."""
        rng = np.random.default_rng(42)
        circuit_size = 5
        # Use a very low bias limit for easy satisfaction
        min_bias = 0.01
        result = run_single_sa(
            bias_df=synthetic_bias_df,
            adj_mat=synthetic_adj_mat,
            info_mat=synthetic_info_mat,
            top_n=20,
            circuit_size=circuit_size,
            min_bias=min_bias,
            measure="SI",
            steps=500,  # very short for testing
            Tmax=1e-2,
            Tmin=5e-5,
            rng=rng,
        )
        assert "structures" in result
        assert "score" in result
        assert "mean_bias" in result
        assert len(result["structures"]) == circuit_size
        assert result["score"] > 0
        assert result["mean_bias"] >= min_bias

    def test_connectivity_measure(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """SA with Connectivity measure should also work."""
        rng = np.random.default_rng(42)
        result = run_single_sa(
            bias_df=synthetic_bias_df,
            adj_mat=synthetic_adj_mat,
            info_mat=synthetic_info_mat,
            top_n=20,
            circuit_size=5,
            min_bias=0.01,
            measure="Connectivity",
            steps=500,
            Tmax=1e-2,
            Tmin=5e-5,
            rng=rng,
        )
        assert len(result["structures"]) == 5

    def test_invalid_measure_raises(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Unknown measure should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown measure"):
            run_single_sa(
                bias_df=synthetic_bias_df,
                adj_mat=synthetic_adj_mat,
                info_mat=synthetic_info_mat,
                top_n=20,
                circuit_size=5,
                min_bias=0.01,
                measure="INVALID",
                steps=100,
                Tmax=1e-2,
                Tmin=5e-5,
                rng=rng,
            )


# ============================================================================
# Unit tests — CircuitSearchRunner (threaded)
# ============================================================================

class TestCircuitSearchRunner:
    """Tests for the threaded CircuitSearchRunner."""

    def test_progress_dict_initial_state(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Progress dict should start with 'pending' status."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
        )
        assert runner.progress["status"] == "pending"
        assert runner.progress["percent_complete"] == 0.0

    def test_search_completes(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Runner should complete and produce results."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
            steps=200,
            runtimes=1,
            top_n=20,
            min_bias_rank=20,
            seed=42,
        )
        runner.start()

        # Wait for completion (generous timeout for CI)
        timeout = 120
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                runner.cancel()
                pytest.fail(f"Runner did not complete within {timeout}s")
            time.sleep(0.1)

        assert runner.progress["status"] == "done"
        assert runner.progress["percent_complete"] == 100.0
        assert len(runner.results) > 0

    def test_results_df_property(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """results_df should be a DataFrame with expected columns."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
            steps=200,
            runtimes=1,
            top_n=20,
            min_bias_rank=20,
            seed=42,
        )
        runner.start()

        timeout = 120
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                runner.cancel()
                pytest.fail("Timeout")
            time.sleep(0.1)

        df = runner.results_df
        assert df is not None
        expected_cols = {"bias_limit", "circuit_score", "mean_bias",
                        "structures", "n_structures"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) > 0

    def test_cancel_stops_search(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Cancellation should stop the runner."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
            steps=5000,  # longer so we can cancel mid-run
            runtimes=5,
            top_n=20,
            min_bias_rank=25,
            seed=42,
        )
        runner.start()
        time.sleep(0.5)  # Let it start
        runner.cancel()

        timeout = 30
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                pytest.fail("Runner did not stop after cancel")
            time.sleep(0.1)

        assert runner.progress["status"] == "cancelled"

    def test_results_df_none_before_start(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """results_df should be None before any results."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
        )
        assert runner.results_df is None


# ============================================================================
# Unit tests — best_per_bias_limit
# ============================================================================

class TestBestPerBiasLimit:
    """Tests for the best-per-bias-limit post-processing."""

    def test_selects_highest_score(self):
        """Should keep the run with the highest score per bias limit."""
        results = [
            CircuitSearchResult(0.1, 0, 0.5, 0.12, np.array(["A", "B"])),
            CircuitSearchResult(0.1, 1, 0.8, 0.13, np.array(["C", "D"])),
            CircuitSearchResult(0.2, 0, 0.6, 0.22, np.array(["E", "F"])),
        ]
        df = best_per_bias_limit(results)
        assert len(df) == 2  # two bias limits

        row_01 = df[df["bias_limit"] == 0.1].iloc[0]
        assert row_01["circuit_score"] == 0.8  # second run was better

    def test_empty_results(self):
        """Empty results should return empty DataFrame."""
        df = best_per_bias_limit([])
        assert len(df) == 0
        assert "bias_limit" in df.columns

    def test_sorted_by_bias_limit(self):
        """Results should be sorted by bias_limit ascending."""
        results = [
            CircuitSearchResult(0.3, 0, 0.5, 0.3, np.array(["A"])),
            CircuitSearchResult(0.1, 0, 0.4, 0.1, np.array(["B"])),
            CircuitSearchResult(0.2, 0, 0.6, 0.2, np.array(["C"])),
        ]
        df = best_per_bias_limit(results)
        assert list(df["bias_limit"]) == [0.1, 0.2, 0.3]


# ============================================================================
# Unit tests — Pareto front
# ============================================================================

class TestParetoFront:
    """Tests for Pareto front extraction."""

    def test_is_dominated_basic(self):
        """Point (1, 1) is dominated by (2, 2)."""
        assert is_dominated(
            np.array([1.0, 1.0]),
            np.array([[2.0, 2.0]]),
        )

    def test_not_dominated_by_worse(self):
        """Point (2, 2) is not dominated by (1, 1)."""
        assert not is_dominated(
            np.array([2.0, 2.0]),
            np.array([[1.0, 1.0]]),
        )

    def test_not_dominated_by_tradeoff(self):
        """Points on different sides of the tradeoff are non-dominated."""
        # (3, 1) vs (1, 3): neither dominates the other
        assert not is_dominated(
            np.array([3.0, 1.0]),
            np.array([[1.0, 3.0]]),
        )

    def test_equal_points_not_dominated(self):
        """Equal points do not dominate each other."""
        assert not is_dominated(
            np.array([1.0, 1.0]),
            np.array([[1.0, 1.0]]),
        )

    def test_pareto_front_indices_simple(self):
        """Classic 2D Pareto test: only non-dominated survive."""
        values = np.array([
            [3.0, 1.0],  # non-dominated
            [1.0, 3.0],  # non-dominated
            [2.0, 2.0],  # non-dominated
            [1.5, 1.5],  # dominated by (2, 2)
            [0.5, 0.5],  # dominated by all above
        ])
        idx = pareto_front_indices(values)
        assert set(idx) == {0, 1, 2}

    def test_pareto_front_all_nondominated(self):
        """When all points are on the front, all should be returned."""
        values = np.array([
            [3.0, 1.0],
            [2.0, 2.0],
            [1.0, 3.0],
        ])
        idx = pareto_front_indices(values)
        assert set(idx) == {0, 1, 2}

    def test_pareto_front_single_point(self):
        """Single point should always be on the front."""
        idx = pareto_front_indices(np.array([[5.0, 5.0]]))
        assert list(idx) == [0]

    def test_pareto_front_empty(self):
        """Empty input should return empty array."""
        idx = pareto_front_indices(np.array([]).reshape(0, 2))
        assert len(idx) == 0


class TestExtractParetoFront:
    """Tests for the DataFrame-level extract_pareto_front function."""

    def test_basic_extraction(self):
        """Should keep only non-dominated rows."""
        df = pd.DataFrame({
            "bias_limit": [0.1, 0.2, 0.3, 0.4],
            "circuit_score": [0.8, 0.7, 0.6, 0.5],
            "mean_bias": [0.12, 0.22, 0.20, 0.42],
            "structures": ["A,B", "C,D", "E,F", "G,H"],
        })
        pareto = extract_pareto_front(df, include_baseline=False)
        # Row 0: (bias=0.12, score=0.8) — non-dominated (highest score)
        # Row 1: (bias=0.22, score=0.7) — non-dominated
        # Row 2: (bias=0.20, score=0.6) — dominated by row 1 (0.22>0.20, 0.7>0.6)
        # Row 3: (bias=0.42, score=0.5) — non-dominated (highest bias)
        assert len(pareto) == 3
        assert "circuit_type" in pareto.columns
        # Row 2 should be excluded
        assert 0.20 not in pareto["mean_bias"].values

    def test_sorted_by_mean_bias(self):
        """Pareto front should be sorted by mean_bias ascending."""
        df = pd.DataFrame({
            "bias_limit": [0.3, 0.1, 0.2],
            "circuit_score": [0.5, 0.9, 0.7],
            "mean_bias": [0.35, 0.12, 0.22],
            "structures": ["A", "B", "C"],
        })
        pareto = extract_pareto_front(df, include_baseline=False)
        assert list(pareto["mean_bias"]) == sorted(pareto["mean_bias"])

    def test_missing_columns_raises(self):
        """Should raise ValueError for missing required columns."""
        df = pd.DataFrame({"bias_limit": [0.1], "score": [0.5]})
        with pytest.raises(ValueError, match="missing required columns"):
            extract_pareto_front(df, include_baseline=False)

    def test_with_baseline(
        self, synthetic_bias_df, synthetic_info_mat,
    ):
        """Should include a baseline row when data is provided."""
        df = pd.DataFrame({
            "bias_limit": [0.1],
            "circuit_score": [0.5],
            "mean_bias": [0.12],
            "structures": ["STR_000,STR_001,STR_002,STR_003,STR_004"],
        })
        pareto = extract_pareto_front(
            df,
            include_baseline=True,
            bias_df=synthetic_bias_df,
            info_mat=synthetic_info_mat,
            circuit_size=5,
            measure="SI",
        )
        # Should have baseline + at least 1 optimized
        assert any(pareto["circuit_type"] == "baseline")

    def test_compute_full_pareto_convenience(self):
        """compute_full_pareto should work without baseline."""
        df = pd.DataFrame({
            "bias_limit": [0.1, 0.2],
            "circuit_score": [0.8, 0.6],
            "mean_bias": [0.12, 0.22],
            "structures": ["A,B", "C,D"],
        })
        pareto = compute_full_pareto(df)
        assert len(pareto) == 2  # both non-dominated


# ============================================================================
# Integration test — webapp vs pipeline output format
# ============================================================================

class TestWebappPipelineCompatibility:
    """Compare webapp SA output format to full pipeline format.

    The pipeline produces TSV lines: score\\tmean_bias\\tstructures
    The webapp produces a DataFrame with the same information.
    This test verifies structural compatibility.
    """

    def test_results_df_matches_pipeline_columns(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Webapp results_df should contain the same fields as pipeline output."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
            steps=200,
            runtimes=1,
            top_n=20,
            min_bias_rank=20,
            seed=42,
        )
        runner.start()

        timeout = 120
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                runner.cancel()
                pytest.fail("Timeout")
            time.sleep(0.1)

        df = runner.results_df
        assert df is not None

        # Pipeline format: score, mean_bias, comma-separated structures
        # Webapp format: bias_limit, circuit_score, mean_bias, structures, n_structures
        for _, row in df.iterrows():
            # circuit_score should be positive float
            assert row["circuit_score"] > 0
            # mean_bias should be positive float
            assert row["mean_bias"] > 0
            # structures should be comma-separated string
            strs = row["structures"].split(",")
            assert len(strs) == row["n_structures"]
            assert len(strs) == 5  # circuit_size

    def test_pareto_front_from_runner_results(
        self, synthetic_bias_df, synthetic_adj_mat, synthetic_info_mat,
    ):
        """Pareto front extraction should work on runner results."""
        runner = CircuitSearchRunner(
            bias_df=synthetic_bias_df,
            circuit_size=5,
            info_mat=synthetic_info_mat,
            adj_mat=synthetic_adj_mat,
            steps=200,
            runtimes=2,
            top_n=20,
            min_bias_rank=20,
            seed=42,
        )
        runner.start()

        timeout = 120
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                runner.cancel()
                pytest.fail("Timeout")
            time.sleep(0.1)

        results_df = runner.results_df
        assert results_df is not None

        pareto = extract_pareto_front(
            results_df,
            include_baseline=True,
            bias_df=synthetic_bias_df,
            info_mat=synthetic_info_mat,
            circuit_size=5,
            measure="SI",
        )
        assert len(pareto) >= 1
        assert all(pareto["circuit_score"] > 0)


# ============================================================================
# Integration test — real data (skipped if unavailable)
# ============================================================================

@skip_without_real_data
class TestWithRealData:
    """Integration tests using real Allen Brain Atlas data."""

    def test_real_data_sa_run(self, real_info_mat, real_adj_mat):
        """Run SA on real data with small circuit size and few steps."""
        from ASD_Circuits import MouseSTR_AvgZ_Weighted

        # Load a real gene weight file
        gw_dir = Path("/home/jw3514/Work/ASD_Circuits_CellType/dat/Genetics/GeneWeights")
        gw_path = gw_dir / "Spark_top20.gw"
        if not gw_path.exists():
            pytest.skip(f"Gene weight file not found: {gw_path}")

        # Load expression matrix and compute bias
        str_bias = pd.read_parquet(
            "/home/jw3514/Work/ASD_Circuits_CellType/dat/BiasMatrices/"
            "AllenMouseBrain_Z2bias.parquet"
        )
        gene2w = dict(
            pd.read_csv(gw_path, header=None, names=["gid", "w"]).values
        )
        bias_df = MouseSTR_AvgZ_Weighted(str_bias, gene2w)

        rng = np.random.default_rng(42)
        result = run_single_sa(
            bias_df=bias_df,
            adj_mat=real_adj_mat,
            info_mat=real_info_mat,
            top_n=100,
            circuit_size=20,
            min_bias=0.05,
            measure="SI",
            steps=1000,  # fast
            Tmax=1e-2,
            Tmin=5e-5,
            rng=rng,
        )
        assert len(result["structures"]) == 20
        assert result["score"] > 0
        assert result["mean_bias"] >= 0.05

    def test_real_data_full_runner(self, real_info_mat, real_adj_mat):
        """Full threaded search on real data with reduced parameters."""
        from ASD_Circuits import MouseSTR_AvgZ_Weighted

        str_bias = pd.read_parquet(
            "/home/jw3514/Work/ASD_Circuits_CellType/dat/BiasMatrices/"
            "AllenMouseBrain_Z2bias.parquet"
        )
        gw_path = Path(
            "/home/jw3514/Work/ASD_Circuits_CellType/dat/Genetics/"
            "GeneWeights/Spark_top20.gw"
        )
        if not gw_path.exists():
            pytest.skip(f"Gene weight file not found: {gw_path}")

        gene2w = dict(
            pd.read_csv(gw_path, header=None, names=["gid", "w"]).values
        )
        bias_df = MouseSTR_AvgZ_Weighted(str_bias, gene2w)

        runner = CircuitSearchRunner(
            bias_df=bias_df,
            circuit_size=20,
            info_mat=real_info_mat,
            adj_mat=real_adj_mat,
            steps=1000,
            runtimes=2,
            top_n=100,
            min_bias_rank=30,
            seed=42,
        )
        runner.start()

        timeout = 300  # real data needs more time
        start = time.monotonic()
        while runner.is_alive():
            if time.monotonic() - start > timeout:
                runner.cancel()
                pytest.fail("Real data runner timed out")
            time.sleep(0.5)

        assert runner.progress["status"] == "done"
        df = runner.results_df
        assert df is not None
        assert len(df) > 0

        # Verify Pareto front
        pareto = extract_pareto_front(
            df,
            include_baseline=True,
            bias_df=bias_df,
            info_mat=real_info_mat,
            circuit_size=20,
            measure="SI",
        )
        assert len(pareto) >= 1
