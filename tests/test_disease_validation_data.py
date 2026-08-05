# tests/test_disease_validation_data.py
import sys, os
import pandas as pd
import pytest

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "src"))
from disease_validation import (load_gene_sets, load_ground_truth,
                                gene_set_weights, gene_set_report)
from ASD_Circuits import LoadGeneINFO, STR2Region

GENESETS = "config/disease_validation_genesets.yaml"
GROUND_TRUTH = "config/disease_validation_ground_truth.yaml"
Z2 = "dat/BiasMatrices/AllenMouseBrain_Z2bias.parquet"

EXPECTED_SIZES = {
    "PD_Primary": 15,
    "PD_Sens_DA": 20,
    "PD_Sens_Atypical": 24,
    "PD_GWAS_L2G": 41,
    "HD_HTT": 1,
    "StriatalDegeneration": 9,
}
# FTL is absent from the Z2 matrix (spec section 3.6).
# Verified 2026-08-05: all 41 PD_GWAS_L2G symbols resolve; exactly one
# (FAM47E) is absent from the Z2 matrix, so 40 are usable.
EXPECTED_IN_MATRIX = {**EXPECTED_SIZES, "StriatalDegeneration": 8, "PD_GWAS_L2G": 40}
EXPECTED_DROPPED = {"StriatalDegeneration": {"FTL"}, "PD_GWAS_L2G": {"FAM47E"}}


@pytest.fixture(scope="module")
def gene_info():
    return LoadGeneINFO()


@pytest.fixture(scope="module")
def z2_genes():
    return set(pd.read_parquet(Z2).index)


def test_gene_set_sizes_match_spec():
    sets = load_gene_sets(GENESETS)
    assert {k: len(v) for k, v in sets.items()} == EXPECTED_SIZES


def test_every_symbol_resolves_to_entrez(gene_info):
    """LoadGeneINFO maps approved HGNC symbols only - aliases must carry
    an explicit hgnc_symbol override or they silently vanish from the set."""
    _, _, sym2entrez, _ = gene_info
    sets = load_gene_sets(GENESETS)
    unresolved = {
        name: [r["symbol"] for r in recs
               if sym2entrez.get(r.get("hgnc_symbol", r["symbol"])) is None]
        for name, recs in sets.items()
    }
    assert all(not v for v in unresolved.values()), unresolved


def test_gba1_alias_resolves_to_entrez_2629(gene_info):
    """Regression guard: GeneSymbol2Entrez['GBA1'] is None in this repo."""
    _, _, sym2entrez, _ = gene_info
    assert sym2entrez.get("GBA1") is None, "repo changed; revisit the alias override"
    rec = [r for r in load_gene_sets(GENESETS)["PD_Primary"] if r["symbol"] == "GBA1"]
    assert len(rec) == 1 and rec[0]["hgnc_symbol"] == "GBA"
    assert int(sym2entrez[rec[0]["hgnc_symbol"]]) == 2629


def test_matrix_coverage_matches_expectation(gene_info, z2_genes):
    """Exact counts and exact dropped sets - a floor would let genes vanish silently.

    Dropped symbols come from gene_set_report(), which already tracks resolution
    and matrix membership per gene. Recomputing them inline would raise on an
    unresolved symbol instead of reporting it, which is the opposite of what a
    data-contract test should do.
    """
    _, _, sym2entrez, _ = gene_info
    sets = load_gene_sets(GENESETS)
    for name, recs in sets.items():
        w = gene_set_weights(recs, sym2entrez, z2_genes)
        assert len(w) == EXPECTED_IN_MATRIX[name], (name, len(w))
        rep = gene_set_report(recs, sym2entrez, z2_genes)
        dropped = set(rep.loc[~rep["in_matrix"], "symbol"])
        assert dropped == EXPECTED_DROPPED.get(name, set()), (name, dropped)
        assert rep["resolved"].all(), (name, set(rep.loc[~rep["resolved"], "symbol"]))


def test_ground_truth_structures_exist_in_infomat():
    """Circuit search consumes InfoMat.Ipsi.csv; ground truth must be in it too."""
    im = pd.read_csv("dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv",
                     index_col=0)
    gt = load_ground_truth(GROUND_TRUTH)
    for disease, groups in gt["structures"].items():
        for group, names in groups.items():
            missing = [n for n in names if n not in im.index]
            assert not missing, f"{disease}/{group} missing from InfoMat: {missing}"


def test_frozen_cell_type_subclasses_map_to_clusters():
    """A renamed subclass would silently yield an empty ground-truth set."""
    import re
    ct = pd.read_parquet("dat/BiasMatrices/Cluster_Z2Mat_ISHMatch.z1clip3.parquet")
    subclasses = {re.sub(r"^\d+\s+", "", c).rsplit("_", 1)[0] for c in ct.columns}
    gt = load_ground_truth(GROUND_TRUTH)
    wanted = set()
    for disease, groups in gt["cell_type_subclasses"].items():
        wanted |= set(groups["core"]) if isinstance(groups, dict) else set(groups)
    missing = wanted - subclasses
    assert not missing, f"subclasses absent from the CT matrix: {missing}"


def test_pd_gwas_excludes_artifact_loci():
    syms = {r["symbol"] for r in load_gene_sets(GENESETS)["PD_GWAS_L2G"]}
    assert not (syms & {"FLG", "HRNR", "MUC19"})


def test_ground_truth_structures_exist_in_atlas(z2_genes):
    gt = load_ground_truth(GROUND_TRUTH)
    atlas = set(pd.read_parquet(Z2).columns) & set(STR2Region())
    for disease, groups in gt["structures"].items():
        for group, names in groups.items():
            missing = [n for n in names if n not in atlas]
            assert not missing, f"{disease}/{group}: {missing}"


def test_no_circular_marker_in_any_primary_set():
    circular = {"TH", "SLC6A3", "DDC", "GCH1", "SPR",
                "PDE10A", "GPR88", "ADCY5", "RASD2", "DRD2"}
    sets = load_gene_sets(GENESETS)
    for name in ("PD_Primary", "HD_HTT", "StriatalDegeneration"):
        syms = {r["symbol"] for r in sets[name]}
        assert not (syms & circular), f"{name} contains circular markers"
