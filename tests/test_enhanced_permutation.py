# Author: jywang explorerwjy@gmail.com
# ========================================================================================================
# test_enhanced_permutation.py
# Unit tests for enhanced permutation framework
# ========================================================================================================

import pytest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gene_matching import (
    GenePropertyMatcher,
    MatchingConfig,
    load_gene_annotations
)


class TestMatchingConfig:
    """Tests for MatchingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MatchingConfig()
        assert config.match_length is True
        assert config.match_conservation is True
        assert config.match_expression is True
        assert config.length_bins == 10
        assert config.tolerance == 1.0
        assert config.min_candidates == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = MatchingConfig(
            match_length=True,
            match_conservation=False,
            match_expression=True,
            length_bins=5,
            tolerance=2.0
        )
        assert config.match_conservation is False
        assert config.length_bins == 5
        assert config.tolerance == 2.0


class TestGenePropertyMatcher:
    """Tests for GenePropertyMatcher class."""

    @pytest.fixture
    def sample_annotations(self):
        """Create sample gene annotations for testing."""
        np.random.seed(42)
        n_genes = 100

        return pd.DataFrame({
            'cds_length': np.random.lognormal(7, 1, n_genes),
            'phastcons': np.random.beta(2, 5, n_genes),
        }, index=range(1, n_genes + 1))

    @pytest.fixture
    def sample_expression(self):
        """Create sample expression matrix for testing."""
        np.random.seed(42)
        n_genes = 100
        n_cell_types = 10

        return pd.DataFrame(
            np.random.randn(n_genes, n_cell_types),
            index=range(1, n_genes + 1),
            columns=[f'CT{i}' for i in range(n_cell_types)]
        )

    def test_matcher_initialization(self, sample_annotations, sample_expression):
        """Test matcher initialization."""
        matcher = GenePropertyMatcher(
            gene_annotations=sample_annotations,
            expression_matrix=sample_expression
        )
        assert matcher is not None
        assert len(matcher._binned_genes) == 100

    def test_get_matched_candidates(self, sample_annotations, sample_expression):
        """Test getting matched candidates for a gene."""
        matcher = GenePropertyMatcher(
            gene_annotations=sample_annotations,
            expression_matrix=sample_expression,
            config=MatchingConfig(tolerance=2.0, min_candidates=5)
        )

        # Get candidates for gene 1
        candidates = matcher.get_matched_candidates(1, exclude_genes=[1])
        assert len(candidates) > 0
        assert 1 not in candidates

    def test_generate_matched_null(self, sample_annotations, sample_expression):
        """Test generating matched null gene sets."""
        matcher = GenePropertyMatcher(
            gene_annotations=sample_annotations,
            expression_matrix=sample_expression,
            config=MatchingConfig(tolerance=2.0, min_candidates=5)
        )

        query_genes = [1, 2, 3, 4, 5]
        gene_weights = {g: 1.0 for g in query_genes}

        null_df = matcher.generate_matched_null(
            query_genes=query_genes,
            gene_weights=gene_weights,
            n_permutations=10,
            random_state=42
        )

        # Check output shape
        assert null_df.shape[0] == len(query_genes)
        assert 'GeneWeight' in null_df.columns
        # 10 permutations + 1 GeneWeight column
        assert null_df.shape[1] == 11

    def test_matching_excludes_query_genes(self, sample_annotations, sample_expression):
        """Test that matched nulls don't include query genes."""
        matcher = GenePropertyMatcher(
            gene_annotations=sample_annotations,
            expression_matrix=sample_expression,
            config=MatchingConfig(tolerance=3.0, min_candidates=5)
        )

        query_genes = [1, 2, 3, 4, 5]
        gene_weights = {g: 1.0 for g in query_genes}

        null_df = matcher.generate_matched_null(
            query_genes=query_genes,
            gene_weights=gene_weights,
            n_permutations=100,
            random_state=42
        )

        # Check that query genes don't appear in null sets
        null_cols = [c for c in null_df.columns if c != 'GeneWeight']
        for col in null_cols:
            null_genes = set(null_df[col].values)
            for qg in query_genes:
                assert qg not in null_genes

    def test_matching_statistics(self, sample_annotations, sample_expression):
        """Test matching statistics computation."""
        matcher = GenePropertyMatcher(
            gene_annotations=sample_annotations,
            expression_matrix=sample_expression
        )

        query_genes = [1, 2, 3]
        stats = matcher.get_matching_statistics(query_genes)

        assert len(stats) == len(query_genes)
        assert 'gene' in stats.columns
        assert 'n_candidates' in stats.columns
        assert 'length_bin' in stats.columns


class TestSpecificityCapping:
    """Tests for specificity score capping functionality."""

    def test_apply_cap(self):
        """Test applying cap to expression matrix."""
        from scripts.script_specificity_sensitivity import apply_specificity_cap

        # Create test matrix with extreme values
        expr_mat = pd.DataFrame({
            'CT1': [-5.0, -2.0, 0.0, 2.0, 5.0],
            'CT2': [-3.0, -1.0, 0.5, 1.0, 4.0]
        })

        capped = apply_specificity_cap(expr_mat, cap_value=3.0)

        assert capped['CT1'].max() <= 3.0
        assert capped['CT1'].min() >= -3.0
        assert capped['CT2'].max() <= 3.0


class TestBGMRCorrection:
    """Tests for BGMR (background mutation rate) correction."""

    def test_ndd_bgmr_correction(self):
        """Test NDD gene weights with BGMR correction."""
        # Import from the local src directory (worktree)
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "ASD_Circuits_local",
            os.path.join(os.path.dirname(__file__), '..', 'src', 'ASD_Circuits.py')
        )
        asd_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(asd_module)
        Aggregate_Gene_Weights_NDD_BGMR = asd_module.Aggregate_Gene_Weights_NDD_BGMR

        # Create mock mutation data
        mut_df = pd.DataFrame({
            'EntrezID': [1, 2, 3],
            'frameshift_variant': [1, 0, 2],
            'splice_acceptor_variant': [0, 1, 0],
            'splice_donor_variant': [0, 0, 1],
            'stop_gained': [0, 0, 0],
            'stop_lost': [0, 0, 0],
            'missense_variant': [2, 1, 3]
        })

        # Test without BGMR (should work)
        weights = Aggregate_Gene_Weights_NDD_BGMR(mut_df, BGMR=None)
        assert len(weights) == 3
        assert all(isinstance(v, (int, float)) for v in weights.values())

        # Test with mock BGMR
        bgmr_df = pd.DataFrame({
            'p_LGD': [1e-5, 2e-5, 1.5e-5],
            'prevel_0.5': [1e-4, 1.5e-4, 2e-4],
            'p_misense': [5e-5, 7e-5, 8e-5]
        }, index=[1, 2, 3])

        weights_bgmr = Aggregate_Gene_Weights_NDD_BGMR(
            mut_df, BGMR=bgmr_df, Nproband=30000
        )
        assert len(weights_bgmr) == 3

        # With BGMR correction, weights should generally be lower
        # (subtracting expected counts)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
