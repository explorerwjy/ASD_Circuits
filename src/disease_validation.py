# src/disease_validation.py
"""Pure functions for the PD / striatal-degeneration validation of GENCIC.

Spec: docs/plans/2026-08-05-pd-hd-circuit-validation-design.md
Every function here is side-effect free apart from reading files it is handed.
"""
import numpy as np
import pandas as pd
import yaml

DEFAULT_SEED = 42


def load_gene_sets(path):
    """Expand the pooled YAML curation into {set_name: [record, ...]}.

    Each record is the raw dict from the YAML (symbol/syndrome/justification)
    with a 'tier' key naming the pool it came from.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    pools = {
        name: [dict(r, tier=name) for r in recs]
        for name, recs in cfg["gene_pools"].items()
    }
    by_symbol = {r["symbol"]: r for recs in pools.values() for r in recs}
    out = {}
    for name, spec in cfg["gene_sets"].items():
        records, seen = [], set()
        for pool in spec.get("include", []):
            for r in pools[pool]:
                if r["symbol"] not in seen:
                    seen.add(r["symbol"])
                    records.append(r)
        for sym in spec.get("explicit", []):
            if sym not in seen:
                seen.add(sym)
                records.append(by_symbol.get(sym, {"symbol": sym, "tier": "explicit"}))
        out[name] = records
    return out


def load_ground_truth(path):
    with open(path) as f:
        return yaml.safe_load(f)


def lookup_symbol(record):
    """The key to look up in GeneSymbol2Entrez.

    LoadGeneINFO() indexes approved HGNC symbols only - not alias_symbol, not
    prev_symbol. A record whose display symbol is an alias (e.g. GBA1, current
    symbol for what this repo's HGNC table still calls GBA) must declare
    hgnc_symbol or it will be silently dropped from the gene set.
    """
    return record.get("hgnc_symbol", record["symbol"])


def gene_set_weights(records, symbol2entrez, valid_genes, weight=1.0):
    """Map curation records to {entrez: weight}, dropping genes absent from the matrix.

    valid_genes is any container of entrez ids (the expression matrix index).
    """
    valid = set(int(g) for g in valid_genes)
    out = {}
    for r in records:
        e = symbol2entrez.get(lookup_symbol(r))
        if e is None:
            continue
        e = int(e)
        if e in valid:
            out[e] = weight
    return out


def gene_set_report(records, symbol2entrez, valid_genes):
    """Per-gene resolution status, for the methods table. Returns a DataFrame."""
    valid = set(int(g) for g in valid_genes)
    rows = []
    for r in records:
        e = symbol2entrez.get(lookup_symbol(r))
        rows.append({
            "symbol": r["symbol"],
            "hgnc_symbol": lookup_symbol(r),
            "entrez": int(e) if e is not None else None,
            "tier": r.get("tier", ""),
            "syndrome": r.get("syndrome", ""),
            "justification": r.get("justification", ""),
            "resolved": e is not None,
            "in_matrix": e is not None and int(e) in valid,
        })
    return pd.DataFrame(rows)


def expression_decile_map(exp_df, valid_genes, n_bins=10, exp_col="EXP"):
    """Bin genes into equal-count expression deciles.

    Ranks before binning so ties cannot collapse a bin. Returns a Series
    entrez -> bin index, restricted to genes present in both inputs.
    """
    idx = pd.Index([int(g) for g in valid_genes])
    exp = exp_df.copy()
    exp.index = exp.index.astype(int)
    shared = exp.index.intersection(idx)
    vals = exp.loc[shared, exp_col].rank(method="first")
    bins = pd.qcut(vals, n_bins, labels=False)
    return pd.Series(bins.values, index=shared, name="decile")


def sample_expression_matched(target, decile_map, n_sims, rng):
    """Draw n_sims gene sets matching target's per-decile composition.

    Sampling is without replacement within a simulation. Genes in target that
    are absent from decile_map are dropped. Returns shape (n_kept, n_sims),
    with row i decile-matched to kept[i] (target, in its original order,
    filtered to genes present in decile_map) — not merely to the same
    multiset of deciles in some other order. Building the per-sim draws
    decile-group by decile-group and then reassembling in kept's original
    order (rather than concatenating group by group) is what keeps rows
    correctly paired to their target gene when deciles are interleaved,
    e.g. target deciles [9, 0, 9, 0].
    """
    kept = [int(g) for g in target if int(g) in decile_map.index]
    if not kept:
        raise ValueError("no target genes present in decile_map")
    kept_deciles = decile_map.loc[kept]
    counts = kept_deciles.value_counts()
    pools = {d: decile_map.index[decile_map == d].to_numpy() for d in counts.index}
    for d, k in counts.items():
        if len(pools[d]) < k:
            raise ValueError(f"decile {d} has {len(pools[d])} genes, need {k}")
    out = np.empty((len(kept), n_sims), dtype=np.int64)
    for j in range(n_sims):
        draws_by_decile = {d: iter(rng.choice(pools[d], size=k, replace=False))
                           for d, k in counts.items()}
        out[:, j] = [next(draws_by_decile[d]) for d in kept_deciles.values]
    return out


from scipy.stats import mannwhitneyu, rankdata


def recovery_stats(bias_df, ground_truth, effect_col="EFFECT"):
    """Do the pre-registered structures rank above the rest on bias?

    One-sided Mann-Whitney U (ground truth greater). AUROC is derived from U,
    so it is the exact rank-based area, not a threshold sweep.
    """
    scores = bias_df[effect_col].dropna()
    present = [s for s in ground_truth if s in scores.index]
    missing = [s for s in ground_truth if s not in scores.index]
    if not present:
        raise ValueError("no ground-truth structures present in bias_df")
    mask = scores.index.isin(present)
    pos, neg = scores[mask], scores[~mask]
    if len(neg) == 0:
        raise ValueError("no background structures to compare against")
    u, p = mannwhitneyu(pos, neg, alternative="greater")
    ranks = bias_df[effect_col].rank(ascending=False)
    top20 = set(scores.nlargest(20).index)
    return {
        "n_ground_truth": len(present),
        "n_missing": len(missing),
        "missing": missing,
        "u_stat": float(u),
        "p_mannwhitney": float(p),
        "auroc": float(u) / (len(pos) * len(neg)),
        "precision_at_20": len(top20 & set(present)) / 20.0,
        "median_rank": float(ranks[present].median()),
    }


def recovery_null_aurocs(null_bias_df, ground_truth):
    """AUROC of the ground-truth set under every null GENE SET simulation.

    null_bias_df is the (n_structures x n_sims) matrix written by the bias
    pipeline to results/{analysis}/null_bias/{geneset}_null_bias_{null}.parquet.
    Each column is the structure bias profile of one null gene set.

    THIS is the statistic that distinguishes null models. The observed EFFECT
    column is computed from the real gene set and is byte-identical regardless
    of which null was configured, so comparing recovery_stats() across nulls
    compares nothing. Confirmed empirically: for ASD_All the uniform and
    sibling bias files differ in P-value but max|dEFFECT| is exactly 0.0.

    Raises ValueError if any column contains NaN, naming the offending
    columns. rankdata(axis=0) turns an ENTIRE column NaN if even a single
    cell in it is NaN - confirmed empirically, and true regardless of
    whether the NaN cell is a ground-truth row or a background row - so
    without this guard a NaN would silently poison that simulation's AUROC
    and then empirical_p, with no error or warning. This is not
    hypothetical: 15 of 61 files in results/STR_ISH/null_bias/ (the NT_*
    source/target neurotransmitter sets) already contain NaN cells (1 to
    25,861 per file). A caller that needs to tolerate NaN (e.g. drop the
    affected columns first) must do so explicitly before calling; this
    function will not guess at a policy.
    """
    present = [s for s in ground_truth if s in null_bias_df.index]
    if not present:
        raise ValueError("no ground-truth structures present in null_bias_df")
    mask = null_bias_df.index.isin(present)
    n_pos, n_neg = int(mask.sum()), int((~mask).sum())
    if n_neg == 0:
        raise ValueError("no background structures to compare against")
    nan_cols = null_bias_df.columns[null_bias_df.isna().any(axis=0)]
    if len(nan_cols) > 0:
        preview = ", ".join(map(str, nan_cols[:10]))
        more = f" (+{len(nan_cols) - 10} more)" if len(nan_cols) > 10 else ""
        raise ValueError(
            f"null_bias_df contains NaN in {len(nan_cols)} column(s): "
            f"{preview}{more}. recovery_null_aurocs refuses to silently "
            "propagate NaN into an AUROC - drop or otherwise handle the "
            "affected columns/rows before calling."
        )
    vals = null_bias_df.to_numpy(dtype=float)
    # AUROC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg),
    # the vectorised Mann-Whitney U formulation.
    # MUST use rankdata(method="average"): a double-argsort assigns arbitrary
    # distinct ranks to tied values and silently gives the wrong answer. For
    # scores [1,1,1,0] with positives {A,B}, argsort yields 0.5 where the
    # correct (and scipy) answer is 0.75.
    ranks = rankdata(vals, axis=0, method="average")
    pos_rank_sum = ranks[mask, :].sum(axis=0)
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def empirical_p(observed, null_values):
    """Add-one-smoothed one-sided p: P(null >= observed)."""
    null_values = np.asarray(null_values, dtype=float)
    return (int((null_values >= observed).sum()) + 1) / (len(null_values) + 1)


def recovery_permutation_p(bias_df, ground_truth, n_perm=10000,
                           seed=DEFAULT_SEED, effect_col="EFFECT"):
    """Permutation p for the AUROC, drawing random STRUCTURE sets of matched size.

    Distinct from recovery_null_aurocs: this asks whether these particular
    structures sit unusually high in this one ranking (a structure-label
    permutation), while recovery_null_aurocs asks whether this gene set beats
    null gene sets. Both are reported; they are not interchangeable.
    """
    scores = bias_df[effect_col].dropna()
    present = [s for s in ground_truth if s in scores.index]
    if not present:
        raise ValueError("no ground-truth structures present in bias_df")
    observed = recovery_stats(bias_df, ground_truth, effect_col)["auroc"]
    rng = np.random.default_rng(seed)
    all_structs = scores.index.to_numpy()
    hits = 0
    for _ in range(n_perm):
        draw = rng.choice(all_structs, size=len(present), replace=False)
        mask = scores.index.isin(draw)
        u, _ = mannwhitneyu(scores[mask], scores[~mask], alternative="greater")
        if float(u) / (mask.sum() * (~mask).sum()) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def nested_subset_recovery(expr_mat, ordered_entrez, ground_truth, sizes, bias_fn,
                           weight=1.0, effect_col="EFFECT"):
    """Recovery as a function of gene-set size, over nested prefixes.

    ordered_entrez must already be ordered by evidence tier (strongest first).
    """
    rows = []
    for n in sizes:
        subset = list(ordered_entrez)[:n]
        if not subset:
            continue
        weights = {int(g): weight for g in subset}
        bias = bias_fn(expr_mat, weights)
        try:
            s = recovery_stats(bias, ground_truth, effect_col)
        except ValueError:
            continue
        rows.append({"n_genes": n, "auroc": s["auroc"],
                     "p_mannwhitney": s["p_mannwhitney"],
                     "precision_at_20": s["precision_at_20"]})
    return pd.DataFrame(rows)


def leave_one_out_recovery(expr_mat, entrez_weights, ground_truth, entrez2symbol,
                           bias_fn, effect_col="EFFECT"):
    """Drop each gene in turn; report the change in AUROC.

    A large negative delta_auroc means that gene was carrying the result.
    """
    full = recovery_stats(bias_fn(expr_mat, dict(entrez_weights)),
                          ground_truth, effect_col)["auroc"]
    rows = []
    for g in list(entrez_weights):
        reduced = {k: v for k, v in entrez_weights.items() if k != g}
        if not reduced:
            continue
        s = recovery_stats(bias_fn(expr_mat, reduced), ground_truth, effect_col)
        rows.append({"dropped_entrez": g,
                     "dropped_symbol": entrez2symbol.get(g, str(g)),
                     "auroc": s["auroc"],
                     "delta_auroc": s["auroc"] - full})
    return pd.DataFrame(rows).sort_values("delta_auroc").reset_index(drop=True)
