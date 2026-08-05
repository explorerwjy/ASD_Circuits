import sys, os
import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))
from disease_validation import expression_decile_map, sample_expression_matched


@pytest.fixture
def toy_exp():
    # 100 genes with strictly increasing expression -> 10 clean deciles of 10
    return pd.DataFrame({"EXP": np.arange(100, dtype=float)}, index=np.arange(1000, 1100))


def test_decile_map_assigns_ten_equal_bins(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    assert dm.nunique() == 10
    assert dm.value_counts().unique().tolist() == [10]


def test_sampled_genes_preserve_decile_composition(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1001, 1050, 1099]          # deciles 0, 0, 5, 9
    rng = np.random.default_rng(42)
    draws = sample_expression_matched(target, dm, n_sims=50, rng=rng)
    assert draws.shape == (4, 50)
    want = dm.loc[target].value_counts().sort_index()
    for j in range(draws.shape[1]):
        got = dm.loc[draws[:, j]].value_counts().sort_index()
        assert got.equals(want)


def test_sampling_is_without_replacement_within_a_sim(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1001, 1002]                 # three genes, same decile
    rng = np.random.default_rng(42)
    draws = sample_expression_matched(target, dm, n_sims=100, rng=rng)
    for j in range(draws.shape[1]):
        assert len(set(draws[:, j])) == 3


def test_sampling_is_reproducible_under_a_fixed_seed(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1000, 1050, 1099]
    a = sample_expression_matched(target, dm, 20, np.random.default_rng(42))
    b = sample_expression_matched(target, dm, 20, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_single_gene_set_is_supported(toy_exp):
    """HD_HTT is one gene; the null must still be drawable."""
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    draws = sample_expression_matched([1050], dm, 30, np.random.default_rng(42))
    assert draws.shape == (1, 30)
    assert (dm.loc[draws[0]] == dm.loc[1050]).all()


def test_genes_absent_from_the_map_are_dropped(toy_exp):
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    draws = sample_expression_matched([1000, 999999], dm, 5, np.random.default_rng(42))
    assert draws.shape == (1, 5)
