"""
webapp/verify_data.py
=====================
Standalone data verification script (no Streamlit dependency).

Run from the webapp/ directory:
    conda activate gencic
    python verify_data.py

Checks that all critical data files are accessible and that matrices
load correctly with the expected shapes and dtypes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup (mirrors data_loader.py logic, without streamlit)
# ---------------------------------------------------------------------------
_WEBAPP_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _WEBAPP_DIR / "config" / "webapp_config.yaml"


def _resolve(rel_path: str) -> Path:
    return (_WEBAPP_DIR / rel_path).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ok(msg): print(f"  ✅  {msg}")
def fail(msg): print(f"  ❌  {msg}")
def info(msg): print(f"  ℹ️   {msg}")
def header(msg): print(f"\n{'='*60}\n{msg}\n{'='*60}")


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------
def main():
    print("GENCIC Webapp Data Verification")
    print(f"Webapp dir : {_WEBAPP_DIR}")
    print(f"Config     : {_CONFIG_PATH}")

    # Load config
    header("1. Config Loading")
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        ok(f"webapp_config.yaml loaded ({len(cfg)} top-level keys)")
        ok(f"  Gene set presets: {len(cfg.get('gene_set_presets', []))}")
        ok(f"  SA params: {list(cfg.get('sa_params', {}).keys())}")
        ok(f"  Analysis types: {list(cfg.get('analysis_types', {}).keys())}")
    except Exception as exc:
        fail(f"Config load failed: {exc}")
        sys.exit(1)

    # File accessibility
    header("2. Data File Accessibility")
    data_files = cfg["data_files"]
    critical = [
        "str_bias_matrix",
        "weight_matrix",
        "info_matrix",
        "protein_coding_genes",
    ]
    all_ok = True
    for key in critical:
        rel = data_files.get(key, "")
        path = _resolve(rel)
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            ok(f"{key}: {size_mb:.1f} MB")
        else:
            fail(f"{key}: NOT FOUND at {path}")
            all_ok = False

    if not all_ok:
        print("\n⚠️  Some critical files missing — matrix loading skipped.")
        sys.exit(1)

    # Matrix loading
    header("3. Expression Matrix (STR Bias)")
    t0 = time.time()
    try:
        str_mat = pd.read_parquet(_resolve(data_files["str_bias_matrix"]))
        elapsed = time.time() - t0
        ok(f"Loaded in {elapsed:.1f}s — shape: {str_mat.shape}")
        ok(f"Index dtype: {str_mat.index.dtype}, sample index: {list(str_mat.index[:3])}")
        ok(f"Columns (first 5): {list(str_mat.columns[:5])}")
        ok(f"Value range: [{str_mat.values.min():.3f}, {str_mat.values.max():.3f}]")
        nan_count = str_mat.isna().sum().sum()
        info(f"NaN count: {nan_count:,}")
    except Exception as exc:
        fail(f"Failed: {exc}")

    header("4. Connectome Weight Matrix")
    t0 = time.time()
    try:
        w_mat = pd.read_csv(_resolve(data_files["weight_matrix"]), index_col=0)
        elapsed = time.time() - t0
        ok(f"Loaded in {elapsed:.1f}s — shape: {w_mat.shape}")
        ok(f"Index (first 3): {list(w_mat.index[:3])}")
        ok(f"Value range: [{w_mat.values.min():.4f}, {w_mat.values.max():.4f}]")
        is_sym = (w_mat.values == w_mat.values.T).all()
        info(f"Symmetric: {is_sym}")
    except Exception as exc:
        fail(f"Failed: {exc}")

    header("5. Connectome Info Matrix")
    t0 = time.time()
    try:
        i_mat = pd.read_csv(_resolve(data_files["info_matrix"]), index_col=0)
        elapsed = time.time() - t0
        ok(f"Loaded in {elapsed:.1f}s — shape: {i_mat.shape}")
        ok(f"Value range: [{i_mat.values.min():.4f}, {i_mat.values.max():.4f}]")
    except Exception as exc:
        fail(f"Failed: {exc}")

    header("6. Gene Annotations (LoadGeneINFO)")
    t0 = time.time()
    try:
        HGNC = pd.read_csv(
            _resolve(data_files["protein_coding_genes"]),
            delimiter="\t",
            low_memory=False,
        )
        HGNC["entrez_id"] = pd.to_numeric(
            HGNC["entrez_id"], errors="coerce"
        ).astype("Int64")
        HGNC_valid = HGNC.dropna(subset=["entrez_id"])
        elapsed = time.time() - t0
        ok(f"Loaded in {elapsed:.1f}s — {len(HGNC):,} rows total")
        ok(f"Genes with Entrez ID: {len(HGNC_valid):,}")
        ENSID2Entrez = dict(
            zip(HGNC_valid["ensembl_gene_id"].values, HGNC_valid["entrez_id"].values)
        )
        ok(f"ENSID2Entrez entries: {len(ENSID2Entrez):,}")
        GeneSymbol2Entrez = dict(
            zip(HGNC_valid["symbol"].values, HGNC_valid["entrez_id"].values)
        )
        ok(f"GeneSymbol2Entrez entries: {len(GeneSymbol2Entrez):,}")
        # Check a known ASD gene
        for gene in ["SHANK3", "CHD8", "SYNGAP1"]:
            eid = GeneSymbol2Entrez.get(gene)
            if eid:
                ok(f"  {gene} → Entrez {eid}")
            else:
                info(f"  {gene} not found in mapping")
    except Exception as exc:
        fail(f"Failed: {exc}")

    header("7. Sample Gene Weight File")
    gw_dir = _resolve(data_files["gene_weights_dir"])
    sample_gw = gw_dir / "ASD.HIQ.gw.csv"
    if sample_gw.exists():
        try:
            df = pd.read_csv(sample_gw, header=None, names=["entrez_id", "weight"])
            df["entrez_id"] = pd.to_numeric(df["entrez_id"], errors="coerce")
            df = df.dropna(subset=["entrez_id"])
            df["entrez_id"] = df["entrez_id"].astype(int)
            gw = dict(zip(df["entrez_id"].values, df["weight"].values))
            ok(f"ASD_HIQ gene weights: {len(gw):,} genes")
            info(f"Weight range: [{min(gw.values()):.4f}, {max(gw.values()):.4f}]")
        except Exception as exc:
            fail(f"Failed: {exc}")
    else:
        info(f"ASD_HIQ gene weight file not found at {sample_gw}")

    print(f"\n{'='*60}")
    print("Verification complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
