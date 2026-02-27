"""
pages/04_data_diagnostics.py
============================
Data Diagnostics page.

Verifies that all required data files are accessible and reports basic
statistics about each loaded matrix.  Useful for debugging deployment issues.
"""

import streamlit as st
import pandas as pd

from core.data_loader import (
    load_webapp_config,
    verify_data_files,
    load_str_bias_matrix,
    load_weight_matrix,
    load_info_matrix,
    load_gene_info,
)

st.set_page_config(
    page_title="Data Diagnostics — GENCIC", page_icon="🔧", layout="wide"
)

st.title("🔧 Data Diagnostics")
st.markdown(
    "This page verifies that all data files are accessible and reports "
    "basic statistics for each matrix."
)

# ---------------------------------------------------------------------------
# File existence check
# ---------------------------------------------------------------------------
st.header("File Accessibility")

statuses = verify_data_files()
rows = []
for key, ok in statuses.items():
    rows.append({"Data File": key.replace("_", " ").title(), "Accessible": "✅ Yes" if ok else "❌ No"})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if not all(statuses.values()):
    st.error(
        "Some critical files are missing. Please check the data paths in "
        "`webapp/config/webapp_config.yaml` and ensure `dat/` is present "
        "in the project root.",
        icon="❌",
    )
    st.stop()

st.success("All critical data files found.", icon="✅")

# ---------------------------------------------------------------------------
# Matrix statistics
# ---------------------------------------------------------------------------
st.header("Matrix Statistics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Brain-Structure Bias Matrix")
    with st.spinner("Loading…"):
        try:
            str_mat = load_str_bias_matrix()
            st.write(f"**Shape**: {str_mat.shape[0]:,} genes × {str_mat.shape[1]} structures")
            st.write(f"**Index dtype**: {str_mat.index.dtype}")
            st.write(f"**Value range**: [{str_mat.values.min():.3f}, {str_mat.values.max():.3f}]")
            st.write(f"**NaN count**: {str_mat.isna().sum().sum():,}")
            with st.expander("Column names (first 20)"):
                st.write(list(str_mat.columns[:20]))
        except Exception as exc:
            st.error(f"Failed to load: {exc}", icon="❌")

with col2:
    st.subheader("Connectome Weight Matrix")
    with st.spinner("Loading…"):
        try:
            w_mat = load_weight_matrix()
            st.write(f"**Shape**: {w_mat.shape[0]} × {w_mat.shape[1]}")
            st.write(f"**Index dtype**: {w_mat.index.dtype}")
            st.write(f"**Value range**: [{w_mat.values.min():.4f}, {w_mat.values.max():.4f}]")
            st.write(f"**NaN count**: {w_mat.isna().sum().sum():,}")
            st.write(f"**Is symmetric**: {(w_mat.values == w_mat.values.T).all()}")
        except Exception as exc:
            st.error(f"Failed to load: {exc}", icon="❌")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Connectome Info Matrix")
    with st.spinner("Loading…"):
        try:
            i_mat = load_info_matrix()
            st.write(f"**Shape**: {i_mat.shape[0]} × {i_mat.shape[1]}")
            st.write(f"**Value range**: [{i_mat.values.min():.4f}, {i_mat.values.max():.4f}]")
            st.write(f"**NaN count**: {i_mat.isna().sum().sum():,}")
        except Exception as exc:
            st.error(f"Failed to load: {exc}", icon="❌")

with col4:
    st.subheader("Gene Annotations")
    with st.spinner("Loading…"):
        try:
            HGNC, ENSID2Entrez, GeneSymbol2Entrez, Entrez2Symbol = load_gene_info()
            st.write(f"**Total rows**: {len(HGNC):,}")
            st.write(f"**Genes with Entrez ID**: {len(Entrez2Symbol):,}")
            st.write(f"**Genes with Ensembl ID**: {len(ENSID2Entrez):,}")
            with st.expander("Sample: first 5 gene rows"):
                st.dataframe(HGNC[["symbol", "entrez_id", "ensembl_gene_id"]].head(5))
        except Exception as exc:
            st.error(f"Failed to load: {exc}", icon="❌")

# ---------------------------------------------------------------------------
# Config dump
# ---------------------------------------------------------------------------
st.header("Webapp Configuration")
with st.expander("Show webapp_config.yaml contents"):
    cfg = load_webapp_config()
    import yaml
    st.code(yaml.dump(cfg, default_flow_style=False), language="yaml")
