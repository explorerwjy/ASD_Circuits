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


def test_sampled_rows_stay_paired_with_their_target_gene(toy_exp):
    """Row i must be decile-matched to target[i] specifically, not merely to
    the same *multiset* of deciles in some other order. A regression here
    (e.g. assembling draws decile-group by decile-group and reassigning them
    to output rows in group order) is invisible to
    test_sampled_genes_preserve_decile_composition, which only checks the
    aggregate composition per simulation — it would still pass even if rows
    were shuffled across genes within the same simulation."""
    dm = expression_decile_map(toy_exp, valid_genes=toy_exp.index)
    target = [1099, 1000, 1098, 1001]          # deciles 9, 0, 9, 0 (interleaved)
    rng = np.random.default_rng(42)
    draws = sample_expression_matched(target, dm, n_sims=30, rng=rng)
    want_deciles = dm.loc[target].values
    for j in range(draws.shape[1]):
        got_deciles = dm.loc[draws[:, j]].values
        np.testing.assert_array_equal(got_deciles, want_deciles)


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


import subprocess


def test_generate_geneweights_expmatched_cli(tmp_path):
    """The script must emit an entrez x n_sims table whose first column is the
    original weight, matching the existing sibling/random output contract."""
    gw = tmp_path / "toy.gw"
    gw.write_text("2629,1\n120892,1\n6622,1\n")     # GBA, LRRK2, SNCA
    # NOTE: filename deliberately ends in "_random.csv" so main()'s existing
    # (preserved) output-path derivation writes here directly rather than to
    # a further-suffixed "..._random.csv" — this matches how the real
    # Snakefile always names the --outfile it passes (weights_random target).
    out = tmp_path / "null_random.csv"
    r = subprocess.run(
        [sys.executable, "scripts/script_generate_geneweights.py",
         "--WeightDF", str(gw),
         "--SpecMat", "dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet",
         "--n_sims", "25",
         "--GeneProb", "None",
         "--null_mode", "expmatched",
         "--ExpMatch", "dat/allen-mouse-exp/ExpMatchFeatures.csv",
         "--outfile", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    df = pd.read_csv(out, index_col=0)
    assert df.columns[0] == "GeneWeight"
    assert df.shape == (3, 26)
    assert set(df.index) == {2629, 120892, 6622}


from disease_validation import recovery_stats, recovery_permutation_p


def _bias(order):
    """Build a bias frame whose EFFECT decreases down the given structure order."""
    return pd.DataFrame(
        {"EFFECT": np.linspace(1.0, 0.0, len(order))}, index=list(order)
    )


def test_perfect_recovery_gives_auroc_one():
    # 3 positives over 5 negatives: exact one-sided p = 1/C(8,3) = 0.0179.
    # With 2-over-4 the best achievable p is 1/C(6,2) = 0.0667, which would
    # make a p<0.05 assertion mathematically impossible.
    df = _bias(["A", "B", "C", "D", "E", "F", "G", "H"])
    s = recovery_stats(df, ["A", "B", "C"])
    assert s["auroc"] == 1.0
    assert s["p_mannwhitney"] < 0.05
    assert s["n_ground_truth"] == 3


def test_worst_recovery_gives_auroc_zero():
    df = _bias(["A", "B", "C", "D", "E", "F"])
    s = recovery_stats(df, ["E", "F"])
    assert s["auroc"] == 0.0
    assert s["p_mannwhitney"] > 0.5


def test_missing_ground_truth_structures_are_counted_not_fatal():
    df = _bias(["A", "B", "C"])
    s = recovery_stats(df, ["A", "NOT_IN_ATLAS"])
    assert s["n_ground_truth"] == 1
    assert s["n_missing"] == 1


def test_precision_at_20_counts_hits_in_the_top_20():
    order = [f"S{i:03d}" for i in range(100)]
    df = _bias(order)
    s = recovery_stats(df, ["S000", "S001", "S050"])
    assert s["precision_at_20"] == pytest.approx(2 / 20)


def test_permutation_p_is_small_for_perfect_recovery():
    order = [f"S{i:03d}" for i in range(213)]
    df = _bias(order)
    p = recovery_permutation_p(df, order[:13], n_perm=2000, seed=42)
    assert p < 0.01


def test_permutation_p_is_reproducible():
    order = [f"S{i:03d}" for i in range(213)]
    df = _bias(order)
    a = recovery_permutation_p(df, order[:13], n_perm=500, seed=42)
    b = recovery_permutation_p(df, order[:13], n_perm=500, seed=42)
    assert a == b


def test_all_ground_truth_missing_raises():
    df = _bias(["A", "B"])
    with pytest.raises(ValueError):
        recovery_stats(df, ["X", "Y"])


# --- Gene-set null: the test that actually differs between null models ---
from disease_validation import recovery_null_aurocs, empirical_p


def test_null_aurocs_one_per_simulation():
    order = [f"S{i:03d}" for i in range(20)]
    rng = np.random.default_rng(42)
    null = pd.DataFrame(rng.normal(size=(20, 50)), index=order,
                        columns=[str(i) for i in range(50)])
    aurocs = recovery_null_aurocs(null, order[:5])
    assert aurocs.shape == (50,)
    assert ((aurocs >= 0) & (aurocs <= 1)).all()


def test_null_auroc_distribution_is_centred_on_half():
    order = [f"S{i:03d}" for i in range(50)]
    rng = np.random.default_rng(42)
    null = pd.DataFrame(rng.normal(size=(50, 400)), index=order,
                        columns=[str(i) for i in range(400)])
    aurocs = recovery_null_aurocs(null, order[:10])
    assert 0.42 < np.median(aurocs) < 0.58


def test_null_auroc_handles_ties_like_scipy():
    """Regression: a double-argsort gives 0.5 here; the correct answer is 0.75."""
    null = pd.DataFrame({"0": [1.0, 1.0, 1.0, 0.0]}, index=list("ABCD"))
    got = recovery_null_aurocs(null, ["A", "B"])
    pos, neg = null.loc[["A", "B"], "0"], null.loc[["C", "D"], "0"]
    u, _ = mannwhitneyu(pos, neg, alternative="greater")
    assert got[0] == pytest.approx(u / (len(pos) * len(neg))) == pytest.approx(0.75)


def test_empirical_p_is_add_one_smoothed():
    assert empirical_p(1.0, np.zeros(99)) == pytest.approx(1 / 100)
    assert empirical_p(-1.0, np.zeros(99)) == pytest.approx(100 / 100)
