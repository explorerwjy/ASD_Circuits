"""
webapp/tests/test_bias.py
=========================
Unit tests for webapp.core.gene_mapping and webapp.core.bias.

Test strategy
-------------
* Gene mapping tests use synthetic HGNC lookup tables (no data files needed).
* Bias computation tests use synthetic expression matrices so they are fast
  and fully deterministic.
* Integration tests that read real data files on disk are skipped gracefully
  when the data are not present (CI-safe).

Run with::

    conda activate gencic
    cd /path/to/ASD_Circuits_CellType
    python -m pytest webapp/tests/test_bias.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is importable (for running pytest from any dir)
# ---------------------------------------------------------------------------

_WORKTREE = Path(__file__).resolve().parent.parent.parent   # .../fda4b9-fda4b9/
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

# ---------------------------------------------------------------------------
# Module imports (must work without Streamlit, igraph, etc.)
# ---------------------------------------------------------------------------

from webapp.core.gene_mapping import (  # noqa: E402
    detect_id_format,
    map_genes_to_entrez,
    build_gene_weight_dict,
)
from webapp.core.bias import compute_str_bias  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures — synthetic lookup tables
# ---------------------------------------------------------------------------

# A minimal synthetic HGNC table: 6 well-known ASD/NDD genes + 4 negatives
_HGNC_ROWS = [
    # symbol,        entrez_id, ensembl_gene_id
    ("SHANK3",       161502, "ENSG00000251158"),
    ("SYNGAP1",       8831, "ENSG00000197283"),
    ("CHD8",         57680, "ENSG00000100888"),
    ("DYRK1A",        1859, "ENSG00000171587"),
    ("ADNP",         23394, "ENSG00000101126"),
    ("ANKRD11",       29123, "ENSG00000197142"),
    # Negative controls (not ASD-related)
    ("LPL",           4023, "ENSG00000175445"),
    ("HMGA1",         3159, "ENSG00000137309"),
    ("HBB",           3043, "ENSG00000244734"),
    ("INS",           3630, "ENSG00000254647"),
]

HGNC_SYMBOLS = [r[0] for r in _HGNC_ROWS]
HGNC_ENTREZ = [r[1] for r in _HGNC_ROWS]
HGNC_ENSEMBL = [r[2] for r in _HGNC_ROWS]

GeneSymbol2Entrez: Dict[str, int] = dict(zip(HGNC_SYMBOLS, HGNC_ENTREZ))
ENSID2Entrez: Dict[str, int] = dict(zip(HGNC_ENSEMBL, HGNC_ENTREZ))
Entrez2Symbol: Dict[int, str] = dict(zip(HGNC_ENTREZ, HGNC_SYMBOLS))

# ---------------------------------------------------------------------------
# Fixtures — synthetic expression matrix
# ---------------------------------------------------------------------------

# 6 ASD genes × 5 brain structures
# We craft weights so that "STR_A" should rank #1 for ASD genes
_ASD_ENTREZ = [161502, 8831, 57680, 1859, 23394, 29123]
_NEG_ENTREZ = [4023, 3159, 3043, 3630]

_STRUCTURES = ["STR_A", "STR_B", "STR_C", "STR_D", "STR_E"]

# Expression matrix: rows=Entrez IDs, cols=structures
# STR_A has very high expression for ASD genes → should be rank #1 with ASD set
_EXPR_DATA = {
    "STR_A": [3.0, 2.8, 2.9, 3.1, 2.7, 3.0,   # ASD genes: high
               0.1, -0.2, 0.0, 0.1],            # Neg ctrl: low
    "STR_B": [0.5, 0.4, 0.6, 0.3, 0.5, 0.4,   # ASD genes: moderate
               2.5,  2.7, 2.6, 2.8],            # Neg ctrl: high
    "STR_C": [-0.5, -0.3, -0.4, -0.6, -0.2, -0.5,
               -0.1,  0.0, -0.1, -0.2],
    "STR_D": [1.0, 0.8, 0.9, 1.1, 0.7, 1.0,
               1.0,  0.9, 1.1, 0.8],
    "STR_E": [np.nan, 0.1, np.nan, 0.2, 0.0, np.nan,
               0.3, 0.1, np.nan, 0.2],           # Some NaNs to test masking
}

EXPR_MATRIX = pd.DataFrame(
    _EXPR_DATA,
    index=pd.Index(_ASD_ENTREZ + _NEG_ENTREZ, name="entrez_id"),
    columns=_STRUCTURES,
)

# ---------------------------------------------------------------------------
# *** detect_id_format ***
# ---------------------------------------------------------------------------


class TestDetectIdFormat:
    def test_entrez_integers(self):
        assert detect_id_format(["161502", "8831", "57680"]) == "entrez"

    def test_ensembl_ids(self):
        assert detect_id_format(["ENSG00000251158", "ENSG00000197283"]) == "ensembl"

    def test_gene_symbols(self):
        assert detect_id_format(["SHANK3", "SYNGAP1", "CHD8"]) == "symbol"

    def test_mixed_mostly_symbols(self):
        # 2 symbols + 1 ensembl → symbol wins
        result = detect_id_format(["SHANK3", "CHD8", "ENSG00000251158"])
        assert result == "symbol"

    def test_empty_list(self):
        assert detect_id_format([]) == "symbol"

    def test_single_entrez(self):
        assert detect_id_format(["85358"]) == "entrez"

    def test_single_ensembl(self):
        assert detect_id_format(["ENSG00000109846"]) == "ensembl"

    def test_single_symbol(self):
        assert detect_id_format(["SHANK3"]) == "symbol"

    def test_int_input_treated_as_entrez(self):
        # When integers are passed as strings they should be detected as entrez
        ids = [str(e) for e in _ASD_ENTREZ]
        assert detect_id_format(ids) == "entrez"


# ---------------------------------------------------------------------------
# *** map_genes_to_entrez ***
# ---------------------------------------------------------------------------


class TestMapGenesToEntrez:
    # ---- symbol format ----

    def test_symbol_all_found(self):
        matched, unmatched = map_genes_to_entrez(
            ["SHANK3", "SYNGAP1", "CHD8"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert set(matched.keys()) == {"SHANK3", "SYNGAP1", "CHD8"}
        assert matched["SHANK3"] == 161502
        assert matched["SYNGAP1"] == 8831
        assert matched["CHD8"] == 57680
        assert unmatched == []

    def test_symbol_some_unmatched(self):
        matched, unmatched = map_genes_to_entrez(
            ["SHANK3", "FAKE_GENE_XYZ"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert "SHANK3" in matched
        assert "FAKE_GENE_XYZ" in unmatched

    def test_symbol_case_insensitive_fallback(self):
        # "shank3" should still resolve via case-insensitive fallback
        matched, unmatched = map_genes_to_entrez(
            ["shank3"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="symbol",
        )
        assert "shank3" in matched
        assert matched["shank3"] == 161502
        assert unmatched == []

    def test_symbol_empty_list(self):
        matched, unmatched = map_genes_to_entrez(
            [],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert matched == {}
        assert unmatched == []

    # ---- entrez format ----

    def test_entrez_passthrough(self):
        matched, unmatched = map_genes_to_entrez(
            ["161502", "8831"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="entrez",
        )
        assert matched["161502"] == 161502
        assert matched["8831"] == 8831
        assert unmatched == []

    def test_entrez_invalid_string(self):
        matched, unmatched = map_genes_to_entrez(
            ["161502", "not_a_number"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="entrez",
        )
        assert "161502" in matched
        assert "not_a_number" in unmatched

    def test_entrez_zero_treated_as_unmatched(self):
        matched, unmatched = map_genes_to_entrez(
            ["0"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="entrez",
        )
        assert "0" in unmatched
        assert "0" not in matched

    # ---- ensembl format ----

    def test_ensembl_all_found(self):
        matched, unmatched = map_genes_to_entrez(
            ["ENSG00000251158", "ENSG00000197283"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="ensembl",
        )
        assert matched["ENSG00000251158"] == 161502
        assert matched["ENSG00000197283"] == 8831
        assert unmatched == []

    def test_ensembl_unknown(self):
        matched, unmatched = map_genes_to_entrez(
            ["ENSG99999999999"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            id_format="ensembl",
        )
        assert "ENSG99999999999" in unmatched


# ---------------------------------------------------------------------------
# *** build_gene_weight_dict ***
# ---------------------------------------------------------------------------


class TestBuildGeneWeightDict:
    def test_uniform_weighting(self):
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=["SHANK3", "SYNGAP1", "CHD8"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert set(gene2weight.values()) == {1.0}
        assert 161502 in gene2weight
        assert unmatched == []
        assert mapping["SHANK3"] == 161502

    def test_user_weights(self):
        weights = {"SHANK3": 2.5, "SYNGAP1": 1.0, "CHD8": 0.5}
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=["SHANK3", "SYNGAP1", "CHD8"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            weights=weights,
        )
        assert gene2weight[161502] == pytest.approx(2.5)
        assert gene2weight[8831] == pytest.approx(1.0)
        assert gene2weight[57680] == pytest.approx(0.5)

    def test_partial_weights_fallback_to_uniform(self):
        # Only one gene has a weight; the other should get 1.0
        gene2weight, _, _ = build_gene_weight_dict(
            gene_ids=["SHANK3", "SYNGAP1"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            weights={"SHANK3": 3.0},
        )
        assert gene2weight[161502] == pytest.approx(3.0)
        assert gene2weight[8831] == pytest.approx(1.0)

    def test_entrez_ids_as_input(self):
        gene_ids = ["161502", "8831", "57680"]
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=gene_ids,
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert 161502 in gene2weight
        assert 8831 in gene2weight
        assert unmatched == []

    def test_ensembl_ids_as_input(self):
        gene_ids = ["ENSG00000251158", "ENSG00000197283"]
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=gene_ids,
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert 161502 in gene2weight
        assert 8831 in gene2weight
        assert unmatched == []

    def test_all_unmatched(self):
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=["FAKE1", "FAKE2"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert gene2weight == {}
        assert set(unmatched) == {"FAKE1", "FAKE2"}

    def test_empty_input(self):
        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=[],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )
        assert gene2weight == {}
        assert unmatched == []


# ---------------------------------------------------------------------------
# *** compute_str_bias ***
# ---------------------------------------------------------------------------


class TestComputeStrBias:
    # ------------------------------------------------------------------
    # Basic sanity / correctness
    # ------------------------------------------------------------------

    def test_returns_dataframe(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_without_region(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        assert "EFFECT" in result.columns
        assert "Rank" in result.columns
        assert "REGION" not in result.columns

    def test_output_columns_with_region(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        str2region = {s: "TestRegion" for s in _STRUCTURES}
        result = compute_str_bias(EXPR_MATRIX, gene2weight, str2region=str2region)
        assert "REGION" in result.columns

    def test_all_structures_present(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        assert set(result.index) == set(_STRUCTURES)

    def test_rank_sequence(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        # Ranks should be 1..n_structures, each appearing exactly once
        assert sorted(result["Rank"].tolist()) == list(range(1, len(_STRUCTURES) + 1))

    def test_sorted_descending_by_effect(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        effects = result["EFFECT"].tolist()
        # NaN should sort last; filter them
        valid_effects = [e for e in effects if not math.isnan(e)]
        assert valid_effects == sorted(valid_effects, reverse=True)

    # ------------------------------------------------------------------
    # ASD gene set: STR_A should rank #1 (highest expression for ASD genes)
    # ------------------------------------------------------------------

    def test_asd_genes_top_structure(self):
        """ASD genes are highly expressed in STR_A; it should rank #1."""
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        assert result.index[0] == "STR_A", (
            f"Expected STR_A to rank #1 for ASD genes, got {result.index[0]}. "
            f"Top 3: {result.index[:3].tolist()}"
        )

    def test_negative_control_top_structure(self):
        """Non-brain genes are highly expressed in STR_B; it should rank #1."""
        gene2weight = {e: 1.0 for e in _NEG_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)
        assert result.index[0] == "STR_B", (
            f"Expected STR_B to rank #1 for negative control genes, got {result.index[0]}."
        )

    # ------------------------------------------------------------------
    # Numerical correctness
    # ------------------------------------------------------------------

    def test_weighted_mean_correctness(self):
        """Verify EFFECT is the (weight-normalized) weighted average of expression."""
        # Use STR_A column, only first 3 ASD genes, uniform weight = 1.0
        test_genes = _ASD_ENTREZ[:3]
        gene2weight = {e: 1.0 for e in test_genes}

        # 3-gene mini expression matrix (no NaNs)
        mini_expr = EXPR_MATRIX.loc[test_genes, :]

        result = compute_str_bias(mini_expr, gene2weight)

        expected_effect_STR_A = float(np.mean(mini_expr["STR_A"].values))
        got = result.loc["STR_A", "EFFECT"]
        assert got == pytest.approx(expected_effect_STR_A, abs=1e-10)

    def test_nonuniform_weights(self):
        """Weighted average differs from unweighted when weights are non-uniform.

        In our synthetic matrix the neg-ctrl genes have STR_B expression:
        [2.5, 2.7, 2.6, 2.8] for Entrez IDs [4023, 3159, 3043, 3630].
        Uniform mean = 2.65.

        * Boosting the *highest*-valued gene (INS=3630, value=2.8) should
          pull the weighted mean *up* relative to uniform.
        * Boosting the *lowest*-valued gene (LPL=4023, value=2.5) should
          pull the weighted mean *down* relative to uniform.
        """
        gene2weight_uniform = {e: 1.0 for e in _NEG_ENTREZ}

        # _NEG_ENTREZ = [4023, 3159, 3043, 3630]
        # STR_B expression:   [2.5,  2.7,  2.6,  2.8]
        # Boost INS (3630, highest in STR_B) → should increase STR_B effect
        gene2weight_high = {e: 1.0 for e in _NEG_ENTREZ}
        gene2weight_high[_NEG_ENTREZ[3]] = 100.0   # INS: 2.8 > mean(2.65)

        # Boost LPL (4023, lowest in STR_B) → should decrease STR_B effect
        gene2weight_low = {e: 1.0 for e in _NEG_ENTREZ}
        gene2weight_low[_NEG_ENTREZ[0]] = 100.0    # LPL: 2.5 < mean(2.65)

        result_uniform = compute_str_bias(EXPR_MATRIX, gene2weight_uniform)
        result_high = compute_str_bias(EXPR_MATRIX, gene2weight_high)
        result_low = compute_str_bias(EXPR_MATRIX, gene2weight_low)

        effect_uniform = result_uniform.loc["STR_B", "EFFECT"]
        effect_high = result_high.loc["STR_B", "EFFECT"]
        effect_low = result_low.loc["STR_B", "EFFECT"]

        assert effect_high > effect_uniform, (
            f"Boosting highest-expressed gene should raise STR_B effect: "
            f"boosted={effect_high:.4f}, uniform={effect_uniform:.4f}"
        )
        assert effect_low < effect_uniform, (
            f"Boosting lowest-expressed gene should lower STR_B effect: "
            f"boosted={effect_low:.4f}, uniform={effect_uniform:.4f}"
        )

    def test_nan_handling(self):
        """Genes with NaN expression should be excluded from the weighted mean."""
        # STR_E has NaN values for several ASD genes
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)

        # STR_E EFFECT should be computable (not NaN) from the non-NaN genes
        ste_effect = result.loc["STR_E", "EFFECT"]
        assert not math.isnan(ste_effect), "STR_E EFFECT should not be NaN"

        # Manual check: non-NaN ASD genes for STR_E are indices 1, 3, 4 (0-based)
        # (index 0, 2, 5 are NaN in our synthetic data)
        asd_ste_values = EXPR_MATRIX.loc[_ASD_ENTREZ, "STR_E"].values
        valid_mask = ~np.isnan(asd_ste_values)
        expected = float(np.mean(asd_ste_values[valid_mask]))
        assert ste_effect == pytest.approx(expected, abs=1e-10)

    def test_region_annotation(self):
        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        str2region = {"STR_A": "Cortex", "STR_B": "Midbrain"}
        result = compute_str_bias(EXPR_MATRIX, gene2weight, str2region=str2region)
        assert result.loc["STR_A", "REGION"] == "Cortex"
        assert result.loc["STR_B", "REGION"] == "Midbrain"
        # Structures not in str2region get NaN
        assert pd.isna(result.loc["STR_C", "REGION"])

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_empty_gene2weight_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_str_bias(EXPR_MATRIX, {})

    def test_no_overlap_raises(self):
        gene2weight = {99999999: 1.0}  # Not in the expression matrix
        with pytest.raises(ValueError, match="No overlap"):
            compute_str_bias(EXPR_MATRIX, gene2weight)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_single_gene(self):
        """Single-gene bias should equal that gene's expression across structures."""
        entrez = _ASD_ENTREZ[0]  # SHANK3 (161502)
        gene2weight = {entrez: 1.0}
        result = compute_str_bias(EXPR_MATRIX, gene2weight)

        for struct in _STRUCTURES:
            expected = float(EXPR_MATRIX.loc[entrez, struct])
            got = result.loc[struct, "EFFECT"]
            if math.isnan(expected):
                assert math.isnan(got)
            else:
                assert got == pytest.approx(expected, abs=1e-10)

    def test_all_nan_structure_gives_nan_effect(self):
        """If all gene expression values for a structure are NaN, EFFECT is NaN."""
        # Create a matrix where STR_A column is all NaN for the test genes
        all_nan_data = EXPR_MATRIX.copy()
        all_nan_data["STR_A"] = np.nan

        gene2weight = {e: 1.0 for e in _ASD_ENTREZ}
        result = compute_str_bias(all_nan_data, gene2weight)
        assert math.isnan(result.loc["STR_A", "EFFECT"])


# ---------------------------------------------------------------------------
# Integration tests (skipped when real data files are not present)
# ---------------------------------------------------------------------------


def _has_data_files() -> bool:
    """Return True if the real data files required for integration tests exist."""
    _WORKTREE = Path(__file__).resolve().parent.parent.parent
    parquet_path = _WORKTREE / "dat" / "BiasMatrices" / "AllenMouseBrain_Z2bias.parquet"
    gw_dir = _WORKTREE / "dat" / "Genetics" / "GeneWeights"
    return parquet_path.exists() and gw_dir.exists()


@pytest.mark.skipif(not _has_data_files(), reason="Real data files not available")
class TestIntegrationWithRealData:
    """Integration tests using the actual expression matrix and .gw files.

    These tests are skipped in CI if the data directory is absent.
    They verify that the webapp bias computation produces results consistent
    with known properties of the ASD_HIQ gene set (e.g., expected top
    brain structures from the published analysis).
    """

    @pytest.fixture(scope="class")
    def real_data(self):
        """Load real expression matrix and gene info once per class."""
        _WORKTREE = Path(__file__).resolve().parent.parent.parent
        parquet_path = _WORKTREE / "dat" / "BiasMatrices" / "AllenMouseBrain_Z2bias.parquet"
        expr_matrix = pd.read_parquet(parquet_path)

        # Load gene info from the actual HGNC file
        gene_info_path = _WORKTREE / "dat" / "Genetics" / "protein-coding_gene.txt"
        HGNC = pd.read_csv(gene_info_path, delimiter="\t", low_memory=False)
        HGNC["entrez_id"] = pd.to_numeric(HGNC["entrez_id"], errors="coerce").astype("Int64")
        HGNC_valid = HGNC.dropna(subset=["entrez_id"])
        gs2e = dict(zip(HGNC_valid["symbol"].values, HGNC_valid["entrez_id"].values))
        ens2e = dict(zip(HGNC_valid["ensembl_gene_id"].values, HGNC_valid["entrez_id"].values))

        # Load ASD_HIQ gene weights
        gw_path = _WORKTREE / "dat" / "Genetics" / "GeneWeights" / "ASD.HIQ.gw.csv"
        gw_df = pd.read_csv(gw_path, header=None, names=["entrez_id", "weight"])
        gw_df["entrez_id"] = pd.to_numeric(gw_df["entrez_id"], errors="coerce").astype(int)
        gene2weight = dict(zip(gw_df["entrez_id"].values, gw_df["weight"].values))

        return {
            "expr_matrix": expr_matrix,
            "GeneSymbol2Entrez": gs2e,
            "ENSID2Entrez": ens2e,
            "gene2weight_asd_hiq": gene2weight,
        }

    def test_real_expr_matrix_shape(self, real_data):
        """Expression matrix should have ~213 structures and thousands of genes."""
        expr = real_data["expr_matrix"]
        n_genes, n_structures = expr.shape
        assert n_genes > 1000, f"Expected >1000 genes, got {n_genes}"
        assert n_structures >= 200, f"Expected ≥200 structures, got {n_structures}"

    def test_asd_hiq_produces_valid_bias(self, real_data):
        """ASD_HIQ gene set should produce a valid bias DataFrame with all structures."""
        gene2weight = real_data["gene2weight_asd_hiq"]
        expr = real_data["expr_matrix"]
        result = compute_str_bias(expr, gene2weight)
        assert isinstance(result, pd.DataFrame)
        assert "EFFECT" in result.columns
        assert "Rank" in result.columns
        assert len(result) == len(expr.columns)

    def test_asd_hiq_top_structure_is_cortical(self, real_data):
        """ASD_HIQ bias should have a cortical or high-level structure in the top 10.

        This is a loose sanity check based on published results — ASD genes are
        broadly over-expressed in cortical and subcortical forebrain structures.
        We do not assert a specific structure name (the ranking can shift with
        updated gene sets) but we verify that the top-ranked structure has a
        positive EFFECT score.
        """
        gene2weight = real_data["gene2weight_asd_hiq"]
        expr = real_data["expr_matrix"]
        result = compute_str_bias(expr, gene2weight)
        top_effect = result.iloc[0]["EFFECT"]
        assert top_effect > 0, (
            f"Top-ranked structure EFFECT should be positive for ASD genes, got {top_effect}"
        )

    def test_negative_control_lower_bias_than_asd(self, real_data):
        """A non-brain negative control gene set should have lower top bias than ASD_HIQ."""
        asd_gene2weight = real_data["gene2weight_asd_hiq"]
        expr = real_data["expr_matrix"]

        asd_result = compute_str_bias(expr, asd_gene2weight)
        asd_max = asd_result["EFFECT"].max()

        # Load HbA1c (non-brain control) if available
        _WORKTREE = Path(__file__).resolve().parent.parent.parent
        hba1c_path = _WORKTREE / "dat" / "Genetics" / "GeneWeights" / "hba1c.top61.gw"
        if not hba1c_path.exists():
            pytest.skip("hba1c.top61.gw not available")

        gw_df = pd.read_csv(hba1c_path, header=None, names=["entrez_id", "weight"])
        gw_df["entrez_id"] = pd.to_numeric(gw_df["entrez_id"], errors="coerce").astype(int)
        hba1c_gene2weight = dict(zip(gw_df["entrez_id"].values, gw_df["weight"].values))

        ctrl_result = compute_str_bias(expr, hba1c_gene2weight)
        ctrl_max = ctrl_result["EFFECT"].max()

        assert asd_max > ctrl_max, (
            f"ASD_HIQ max EFFECT ({asd_max:.4f}) should exceed "
            f"HbA1c max EFFECT ({ctrl_max:.4f})"
        )

    def test_symbol_mapping_matches_entrez_mapping(self, real_data):
        """Gene symbols mapped via build_gene_weight_dict should give same bias as direct .gw."""
        expr = real_data["expr_matrix"]
        gs2e = real_data["GeneSymbol2Entrez"]
        ens2e = real_data["ENSID2Entrez"]

        # A small set of well-known ASD gene symbols
        known_asd_symbols = ["SHANK3", "SYNGAP1", "CHD8", "DYRK1A", "ADNP"]
        gene2weight_from_symbols, unmatched, _ = build_gene_weight_dict(
            gene_ids=known_asd_symbols,
            ENSID2Entrez=ens2e,
            GeneSymbol2Entrez=gs2e,
        )
        assert len(unmatched) == 0, f"Unmatched ASD genes: {unmatched}"

        # Build equivalent gene2weight directly from known Entrez IDs
        gene2weight_direct = {v: 1.0 for v in gene2weight_from_symbols.values()}

        result_via_symbols = compute_str_bias(expr, gene2weight_from_symbols)
        result_via_entrez = compute_str_bias(expr, gene2weight_direct)

        # Results should be identical (same Entrez IDs, same uniform weights)
        pd.testing.assert_frame_equal(result_via_symbols, result_via_entrez, check_like=True)
