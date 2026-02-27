"""
webapp/core/gene_mapping.py
============================
Gene identifier mapping utilities for the GENCIC webapp.

Accepts gene identifiers in three formats and maps them to Entrez IDs:

* **Gene symbols** (HGNC) — e.g. "SHANK3", "SYNGAP1"
* **Entrez IDs** — e.g. "85358", 85358
* **Ensembl gene IDs** — e.g. "ENSG00000109846"

The module auto-detects the format from the identifier strings and uses the
HGNC annotation table (loaded by ``webapp.core.data_loader.load_gene_info``)
to perform the conversion.

Typical usage
-------------
::

    from webapp.core.data_loader import load_gene_info
    from webapp.core.gene_mapping import build_gene_weight_dict

    _, ENSID2Entrez, GeneSymbol2Entrez, _ = load_gene_info()

    gene2weight, unmatched, matched_map = build_gene_weight_dict(
        gene_ids=["SHANK3", "SYNGAP1", "CHD8"],
        ENSID2Entrez=ENSID2Entrez,
        GeneSymbol2Entrez=GeneSymbol2Entrez,
    )
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_ENSEMBL_PATTERN = re.compile(r"^ENSG\d{11}$")
_ENTREZ_PATTERN = re.compile(r"^\d+$")


def detect_id_format(gene_ids: List[str]) -> str:
    """Auto-detect the identifier format of a list of gene IDs.

    Voting strategy: the format with the most votes among the first 20 tokens
    wins.  Ties break in this priority: entrez > ensembl > symbol.

    Parameters
    ----------
    gene_ids : list[str]
        List of gene identifier strings (stripped of whitespace).

    Returns
    -------
    str
        One of ``"entrez"``, ``"ensembl"``, or ``"symbol"``.

    Examples
    --------
    >>> detect_id_format(["85358", "8503"])
    'entrez'
    >>> detect_id_format(["ENSG00000109846"])
    'ensembl'
    >>> detect_id_format(["SHANK3", "CHD8"])
    'symbol'
    """
    if not gene_ids:
        return "symbol"

    # Sample up to 20 tokens for speed
    sample = [str(g).strip() for g in gene_ids[:20] if str(g).strip()]

    entrez_votes = sum(1 for g in sample if _ENTREZ_PATTERN.match(g))
    ensembl_votes = sum(1 for g in sample if _ENSEMBL_PATTERN.match(g))
    symbol_votes = len(sample) - entrez_votes - ensembl_votes

    if entrez_votes >= ensembl_votes and entrez_votes >= symbol_votes:
        return "entrez"
    if ensembl_votes > symbol_votes:
        return "ensembl"
    return "symbol"


# ---------------------------------------------------------------------------
# Core mapping function
# ---------------------------------------------------------------------------


def map_genes_to_entrez(
    gene_ids: List[str],
    ENSID2Entrez: Dict[str, int],
    GeneSymbol2Entrez: Dict[str, int],
    id_format: Optional[str] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """Map a list of gene identifiers to Entrez IDs.

    Parameters
    ----------
    gene_ids : list[str]
        Input gene identifiers (symbols, Entrez IDs, or Ensembl IDs).
    ENSID2Entrez : dict[str, int]
        Ensembl gene ID → Entrez ID mapping from ``load_gene_info()``.
    GeneSymbol2Entrez : dict[str, int]
        HGNC symbol → Entrez ID mapping from ``load_gene_info()``.
    id_format : str or None, optional
        One of ``"entrez"``, ``"ensembl"``, ``"symbol"``, or ``None`` for
        auto-detection.  When ``None`` the format is inferred from the data.

    Returns
    -------
    matched : dict[str, int]
        Mapping from the original gene identifier string to its Entrez ID for
        every gene that was successfully resolved.
    unmatched : list[str]
        Identifiers that could not be mapped to any Entrez ID.

    Notes
    -----
    * Entrez-format inputs are passed through directly (after int conversion).
    * Symbol lookup is attempted **case-sensitively** first; if that fails, a
      case-insensitive fallback is attempted using a pre-built upper-case index.
    * Genes mapping to Entrez 0 (a sentinel used in some legacy pipelines) are
      treated as unmatched.
    """
    if not gene_ids:
        return {}, []

    clean_ids = [str(g).strip() for g in gene_ids]
    fmt = id_format if id_format is not None else detect_id_format(clean_ids)

    matched: Dict[str, int] = {}
    unmatched: List[str] = []

    if fmt == "entrez":
        for gid in clean_ids:
            try:
                entrez = int(gid)
            except (ValueError, TypeError):
                unmatched.append(gid)
                continue
            if entrez == 0:
                unmatched.append(gid)
            else:
                matched[gid] = entrez

    elif fmt == "ensembl":
        for gid in clean_ids:
            entrez = ENSID2Entrez.get(gid)
            if entrez is None or int(entrez) == 0:
                unmatched.append(gid)
            else:
                matched[gid] = int(entrez)

    else:  # "symbol"
        # Build a case-insensitive fallback index on first call
        _upper_index: Dict[str, int] = {
            k.upper(): v for k, v in GeneSymbol2Entrez.items()
        }

        for gid in clean_ids:
            entrez = GeneSymbol2Entrez.get(gid)
            if entrez is None:
                # Case-insensitive fallback
                entrez = _upper_index.get(gid.upper())
            if entrez is None or int(entrez) == 0:
                unmatched.append(gid)
            else:
                matched[gid] = int(entrez)

    logger.debug(
        "map_genes_to_entrez [%s]: %d matched, %d unmatched",
        fmt,
        len(matched),
        len(unmatched),
    )
    return matched, unmatched


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def build_gene_weight_dict(
    gene_ids: List[str],
    ENSID2Entrez: Dict[str, int],
    GeneSymbol2Entrez: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
    id_format: Optional[str] = None,
) -> Tuple[Dict[int, float], List[str], Dict[str, int]]:
    """Build a gene-to-weight dictionary suitable for bias computation.

    This is the primary entry point for converting user-provided gene lists
    (from text input, file upload, or a preset .gw file) into the
    ``gene2weight`` dict expected by ``webapp.core.bias.compute_str_bias``.

    Parameters
    ----------
    gene_ids : list[str]
        Input gene identifiers (symbols, Entrez IDs, or Ensembl IDs).
    ENSID2Entrez : dict[str, int]
        Ensembl gene ID → Entrez ID mapping.
    GeneSymbol2Entrez : dict[str, int]
        HGNC symbol → Entrez ID mapping.
    weights : dict[str, float] or None, optional
        Per-gene weights keyed by the **same** identifiers as ``gene_ids``.
        When ``None``, all matched genes receive a uniform weight of ``1.0``.
    id_format : str or None, optional
        Force a specific format (``"entrez"``, ``"ensembl"``, ``"symbol"``).
        If ``None`` the format is auto-detected.

    Returns
    -------
    gene2weight : dict[int, float]
        Mapping from Entrez gene ID to weight.  Keys are ``int``.
    unmatched : list[str]
        Gene identifiers that could not be resolved to an Entrez ID.
    matched_mapping : dict[str, int]
        Mapping from the original identifier to its Entrez ID for genes that
        were successfully resolved.

    Examples
    --------
    Uniform weighting from gene symbols::

        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=["SHANK3", "SYNGAP1", "CHD8"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
        )

    User-provided weights::

        gene2weight, unmatched, mapping = build_gene_weight_dict(
            gene_ids=["SHANK3", "SYNGAP1"],
            ENSID2Entrez=ENSID2Entrez,
            GeneSymbol2Entrez=GeneSymbol2Entrez,
            weights={"SHANK3": 2.5, "SYNGAP1": 1.0},
        )
    """
    matched_mapping, unmatched = map_genes_to_entrez(
        gene_ids=gene_ids,
        ENSID2Entrez=ENSID2Entrez,
        GeneSymbol2Entrez=GeneSymbol2Entrez,
        id_format=id_format,
    )

    if weights is None:
        # Uniform weighting
        gene2weight: Dict[int, float] = {
            entrez: 1.0 for entrez in matched_mapping.values()
        }
    else:
        gene2weight = {}
        for original_id, entrez in matched_mapping.items():
            w = weights.get(original_id)
            if w is not None:
                try:
                    gene2weight[entrez] = float(w)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid weight for %s (Entrez %d): %r — defaulting to 1.0",
                        original_id,
                        entrez,
                        w,
                    )
                    gene2weight[entrez] = 1.0
            else:
                # Gene matched but no weight provided — fall back to uniform
                gene2weight[entrez] = 1.0

    logger.info(
        "build_gene_weight_dict: %d genes mapped, %d unmatched",
        len(gene2weight),
        len(unmatched),
    )
    return gene2weight, unmatched, matched_mapping
