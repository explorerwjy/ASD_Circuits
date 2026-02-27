"""
webapp/core/data_loader.py
==========================
Cached data loaders for the GENCIC webapp.

All heavy I/O is wrapped with @st.cache_resource so each object is loaded
once per server process and shared across all user sessions.

Path resolution strategy
------------------------
All paths are resolved relative to the **project root**, which is the parent
of the webapp/ directory.  From within any file in webapp/core/, the project
root is two levels up:

    project_root = Path(__file__).resolve().parent.parent.parent

This matches the convention in webapp_config.yaml, where data paths are
written as "../dat/..." (i.e., relative to the webapp/ directory).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Resolve stable roots at import time so they're available everywhere.
_WEBAPP_DIR = Path(__file__).resolve().parent.parent   # .../webapp/
_PROJECT_ROOT = _WEBAPP_DIR.parent                     # .../ASD_Circuits_CellType/
_CONFIG_PATH = _WEBAPP_DIR / "config" / "webapp_config.yaml"


def _resolve(relative_to_webapp: str) -> Path:
    """Convert a '../dat/...' style path (relative to webapp/) to absolute."""
    return (_WEBAPP_DIR / relative_to_webapp).resolve()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_webapp_config() -> dict:
    """Load and cache the webapp YAML configuration.

    Returns
    -------
    dict
        Parsed webapp_config.yaml contents.
    """
    logger.info("Loading webapp config from %s", _CONFIG_PATH)
    with open(_CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


# ---------------------------------------------------------------------------
# Expression matrix loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading brain-structure expression matrix…")
def load_str_bias_matrix() -> pd.DataFrame:
    """Load the Allen Mouse Brain ISH Z2-bias expression matrix.

    Rows   : Entrez gene IDs (int)
    Columns: 213 brain-structure names (str)

    Returns
    -------
    pd.DataFrame
        Shape (n_genes, 213).  Index is Entrez gene ID.

    Raises
    ------
    FileNotFoundError
        If the parquet file cannot be found at the configured path.
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["str_bias_matrix"]
    abs_path = _resolve(rel_path)

    logger.info("Loading STR bias matrix from %s", abs_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Brain-structure bias matrix not found: {abs_path}\n"
            f"Expected at: {rel_path} (relative to webapp/)"
        )

    df = pd.read_parquet(abs_path)
    logger.info("STR bias matrix loaded — shape: %s", df.shape)
    return df


@st.cache_resource(show_spinner="Loading cell-type expression matrix…")
def load_ct_bias_matrix() -> pd.DataFrame:
    """Load the Allen Brain Cell Atlas Z2-bias cell-type expression matrix.

    Rows   : Entrez gene IDs (int)
    Columns: Cell-type cluster IDs (str)

    Returns
    -------
    pd.DataFrame
        Shape (n_genes, n_clusters).
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["ct_bias_matrix"]
    abs_path = _resolve(rel_path)

    logger.info("Loading CT bias matrix from %s", abs_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Cell-type bias matrix not found: {abs_path}\n"
            f"Expected at: {rel_path} (relative to webapp/)"
        )

    df = pd.read_parquet(abs_path)
    logger.info("CT bias matrix loaded — shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Connectome matrix loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading connectome weight matrix…")
def load_weight_matrix() -> pd.DataFrame:
    """Load the Allen Mouse Brain ipsilateral connectome weight matrix.

    A symmetric (213 × 213) matrix of inter-structure connection strengths.
    Both row and column indices are brain-structure names.

    Returns
    -------
    pd.DataFrame
        Shape (213, 213).  Index and columns are structure names.
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["weight_matrix"]
    abs_path = _resolve(rel_path)

    logger.info("Loading weight matrix from %s", abs_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Weight matrix not found: {abs_path}\n"
            f"Expected at: {rel_path} (relative to webapp/)"
        )

    df = pd.read_csv(abs_path, index_col=0)
    logger.info("Weight matrix loaded — shape: %s", df.shape)
    return df


@st.cache_resource(show_spinner="Loading connectome information matrix…")
def load_info_matrix() -> pd.DataFrame:
    """Load the Allen Mouse Brain ipsilateral Shannon-information matrix.

    A (213 × 213) matrix of inter-structure Shannon information scores
    used as the objective function in circuit search.  Both row and column
    indices are brain-structure names.

    Returns
    -------
    pd.DataFrame
        Shape (213, 213).  Index and columns are structure names.
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["info_matrix"]
    abs_path = _resolve(rel_path)

    logger.info("Loading info matrix from %s", abs_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Info matrix not found: {abs_path}\n"
            f"Expected at: {rel_path} (relative to webapp/)"
        )

    df = pd.read_csv(abs_path, index_col=0)
    logger.info("Info matrix loaded — shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Gene info loader (replaces ASD_Circuits.LoadGeneINFO)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading gene annotations…")
def load_gene_info() -> Tuple[
    pd.DataFrame,
    Dict[str, int],
    Dict[str, int],
    Dict[int, str],
]:
    """Load HGNC protein-coding gene annotations.

    This is a lightweight reimplementation of ``ASD_Circuits.LoadGeneINFO``
    that avoids importing the full ASD_Circuits module (which pulls in igraph,
    matplotlib, etc.).

    Returns
    -------
    HGNC : pd.DataFrame
        Full HGNC table of protein-coding genes.
    ENSID2Entrez : dict[str, int]
        Ensembl gene ID → Entrez ID mapping.
    GeneSymbol2Entrez : dict[str, int]
        HGNC symbol → Entrez ID mapping.
    Entrez2Symbol : dict[int, str]
        Entrez ID → HGNC symbol mapping.

    Raises
    ------
    FileNotFoundError
        If the protein-coding gene annotation file cannot be found.
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["protein_coding_genes"]
    abs_path = _resolve(rel_path)

    logger.info("Loading gene info from %s", abs_path)
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Protein-coding gene annotation file not found: {abs_path}\n"
            f"Expected at: {rel_path} (relative to webapp/)"
        )

    HGNC = pd.read_csv(abs_path, delimiter="\t", low_memory=False)
    HGNC["entrez_id"] = pd.to_numeric(HGNC["entrez_id"], errors="coerce").astype(
        "Int64"
    )
    HGNC_valid = HGNC.dropna(subset=["entrez_id"])

    ENSID2Entrez: Dict[str, int] = dict(
        zip(HGNC_valid["ensembl_gene_id"].values, HGNC_valid["entrez_id"].values)
    )
    GeneSymbol2Entrez: Dict[str, int] = dict(
        zip(HGNC_valid["symbol"].values, HGNC_valid["entrez_id"].values)
    )
    Entrez2Symbol: Dict[int, str] = dict(
        zip(HGNC_valid["entrez_id"].values, HGNC_valid["symbol"].values)
    )

    logger.info(
        "Gene info loaded — %d protein-coding genes with Entrez IDs",
        len(HGNC_valid),
    )
    return HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol


# ---------------------------------------------------------------------------
# Gene weight helpers
# ---------------------------------------------------------------------------

def load_gene_weights(file_path: str | Path) -> Dict[int, float]:
    """Load a gene weight file (.gw or .csv) into a dictionary.

    Gene weight files are headerless CSV files with two columns:
    ``EntrezID,weight``.

    Parameters
    ----------
    file_path : str | Path
        Absolute path to the .gw or .csv file.

    Returns
    -------
    dict[int, float]
        Mapping from Entrez gene ID (int) to mutation-derived weight (float).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Gene weight file not found: {path}")

    df = pd.read_csv(path, header=None, names=["entrez_id", "weight"])
    df["entrez_id"] = pd.to_numeric(df["entrez_id"], errors="coerce")
    df = df.dropna(subset=["entrez_id"])
    df["entrez_id"] = df["entrez_id"].astype(int)
    return dict(zip(df["entrez_id"].values, df["weight"].values))


def load_preset_gene_weights(preset_name: str) -> Dict[int, float]:
    """Load gene weights for a named preset from webapp_config.yaml.

    Parameters
    ----------
    preset_name : str
        The ``name`` field of a gene set defined in ``gene_set_presets``.

    Returns
    -------
    dict[int, float]
        Mapping from Entrez gene ID to weight.

    Raises
    ------
    ValueError
        If ``preset_name`` is not found in the config.
    FileNotFoundError
        If the corresponding .gw file is missing from disk.
    """
    cfg = load_webapp_config()
    gw_dir = _resolve(cfg["data_files"]["gene_weights_dir"])

    preset = next(
        (p for p in cfg["gene_set_presets"] if p["name"] == preset_name), None
    )
    if preset is None:
        available = [p["name"] for p in cfg["gene_set_presets"]]
        raise ValueError(
            f"Gene set preset '{preset_name}' not found. "
            f"Available: {available}"
        )

    gw_path = gw_dir / preset["file"]
    return load_gene_weights(gw_path)


# ---------------------------------------------------------------------------
# Structure-to-region mapping loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_structure_region_map() -> Dict[str, str]:
    """Load the brain-structure → major-region mapping.

    Returns
    -------
    dict[str, str]
        Mapping from structure name to major brain division (e.g., 'Isocortex').
    """
    cfg = load_webapp_config()
    rel_path = cfg["data_files"]["major_brain_divisions"]
    abs_path = _resolve(rel_path)

    if not abs_path.exists():
        logger.warning("Structure-region map not found: %s", abs_path)
        return {}

    df = pd.read_csv(abs_path, sep="\t", header=None, names=["structure", "region"])
    return dict(zip(df["structure"].values, df["region"].values))


# ---------------------------------------------------------------------------
# Convenience: verify all critical data files are accessible
# ---------------------------------------------------------------------------

def verify_data_files() -> Dict[str, bool]:
    """Check that all critical data files listed in webapp_config.yaml exist.

    Returns
    -------
    dict[str, bool]
        Mapping from data file key → True if accessible, False otherwise.
    """
    cfg = load_webapp_config()
    results: Dict[str, bool] = {}

    critical_keys = [
        "str_bias_matrix",
        "weight_matrix",
        "info_matrix",
        "protein_coding_genes",
    ]

    for key in critical_keys:
        rel_path = cfg["data_files"].get(key, "")
        if not rel_path:
            results[key] = False
            continue
        abs_path = _resolve(rel_path)
        results[key] = abs_path.exists()

    return results
