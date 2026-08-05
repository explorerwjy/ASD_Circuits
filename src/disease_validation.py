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
