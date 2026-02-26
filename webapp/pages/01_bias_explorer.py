"""
pages/01_bias_explorer.py
=========================
Bias Explorer page — placeholder.

This page will allow users to:
- Select a gene set (preset or upload)
- Select an analysis type (STR_ISH or CT_Z2)
- Visualise mutation bias scores across brain structures / cell types
- Inspect top-ranked structures and their brain-region membership

TODO: Implement full bias computation and visualisation.
"""

import streamlit as st

st.set_page_config(page_title="Bias Explorer — GENCIC", page_icon="📊", layout="wide")

st.title("📊 Bias Explorer")
st.info(
    "**Coming soon.** This page will display mutation bias scores across "
    "brain structures and cell types for the selected gene set.",
    icon="🚧",
)
