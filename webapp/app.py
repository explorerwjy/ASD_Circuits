"""
webapp/app.py
=============
GENCIC — Brain Circuit Explorer
Main Streamlit application entry point.

Usage
-----
    cd webapp/
    streamlit run app.py

Or from the project root:
    streamlit run webapp/app.py

Multi-page structure
--------------------
Streamlit automatically discovers pages in the ``pages/`` subdirectory.
This file serves as the landing page / home page.
"""

import logging

import streamlit as st

from core.data_loader import (
    load_webapp_config,
    verify_data_files,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GENCIC — Brain Circuit Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": (
            "**GENCIC** — Genetics and Expression iNtegration with Connectome "
            "to Infer affected Circuits.\n\n"
            "A computational framework for identifying brain circuits "
            "preferentially targeted by ASD mutations."
        ),
    },
)

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
cfg = load_webapp_config()
ui = cfg.get("ui", {})

# ---------------------------------------------------------------------------
# Sidebar — navigation & status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Stylised_Lithium_Atom.svg/240px-Stylised_Lithium_Atom.svg.png",
        width=60,
    )
    st.title("GENCIC")
    st.caption(ui.get("app_subtitle", "Brain Circuit Explorer"))
    st.divider()

    st.header("Navigation")
    st.page_link("app.py", label="🏠 Home", icon=None)
    st.page_link("pages/01_bias_explorer.py", label="📊 Bias Explorer")
    st.page_link("pages/2_Permutation.py", label="🎲 Permutation Testing")
    st.page_link("pages/02_circuit_search.py", label="🔍 Circuit Search")
    st.page_link("pages/03_circuit_viewer.py", label="🕸️ Circuit Viewer")
    st.page_link("pages/04_data_diagnostics.py", label="🔧 Data Diagnostics")

    st.divider()

    # Data status panel
    st.subheader("Data Status")
    statuses = verify_data_files()
    for key, ok in statuses.items():
        icon = "✅" if ok else "❌"
        label = key.replace("_", " ").title()
        st.write(f"{icon} {label}")

# ---------------------------------------------------------------------------
# Main content — Home page
# ---------------------------------------------------------------------------
st.title("🧠 GENCIC — Brain Circuit Explorer")
st.subheader(
    "Genetics and Expression iNtegration with Connectome to Infer affected Circuits"
)

st.markdown(
    """
GENCIC is a computational framework for identifying brain circuits
preferentially targeted by autism spectrum disorder (ASD) mutations.
It integrates:

- **Genomic data** from ASD patient cohorts (SPARK, ASC, Fu et al.)
- **Spatial transcriptomics** from the Allen Brain Atlas (ISH & scRNA-seq)
- **Brain-wide connectome data** from the Allen Mouse Brain Connectivity Atlas

---
"""
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Bias Explorer")
    st.markdown(
        "Visualise how strongly each brain structure or cell type is "
        "enriched for expression of ASD-associated genes."
    )
    if st.button("Open Bias Explorer", use_container_width=True):
        st.switch_page("pages/01_bias_explorer.py")

with col2:
    st.markdown("### 🔍 Circuit Search")
    st.markdown(
        "Run simulated annealing to find the set of connected brain structures "
        "with the highest combined mutation bias and connectome score."
    )
    if st.button("Open Circuit Search", use_container_width=True):
        st.switch_page("pages/02_circuit_search.py")

with col3:
    st.markdown("### 🕸️ Circuit Viewer")
    st.markdown(
        "Interactively explore identified circuits as a network graph, "
        "colour-coded by brain region and bias score."
    )
    if st.button("Open Circuit Viewer", use_container_width=True):
        st.switch_page("pages/03_circuit_viewer.py")

st.divider()

# ---------------------------------------------------------------------------
# Data availability summary
# ---------------------------------------------------------------------------
st.subheader("Data Availability")

all_ok = all(statuses.values())
if all_ok:
    st.success(
        "All critical data files are accessible.  "
        "The webapp is ready for analysis.",
        icon="✅",
    )
else:
    missing = [k for k, v in statuses.items() if not v]
    st.error(
        f"**{len(missing)} critical data file(s) not found:** "
        + ", ".join(missing)
        + "\n\nPlease ensure the `dat/` directory is present in the project root "
        "and contains the required files.  See DATA_MANIFEST.yaml for details.",
        icon="❌",
    )

with st.expander("Show data file paths"):
    data_cfg = cfg.get("data_files", {})
    rows = []
    for key, rel_path in data_cfg.items():
        status = statuses.get(key, None)
        if status is True:
            icon = "✅"
        elif status is False:
            icon = "❌"
        else:
            icon = "ℹ️"
        rows.append({"Key": key, "Path (relative to webapp/)": rel_path, "Status": icon})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Quick-start guide
# ---------------------------------------------------------------------------
with st.expander("Quick Start Guide"):
    st.markdown(
        """
**1. Select a gene set** — Choose from pre-defined ASD, NDD, or neurotransmitter
gene sets in the Bias Explorer, or upload your own `.gw` file.

**2. Compute bias** — The bias score measures how strongly each brain structure
or cell type is enriched for high expression of the selected mutation-associated
genes (Z2 method, controlling for baseline expression level).

**3. Search for circuits** — Use simulated annealing (SA) to find a set of
interconnected brain structures that jointly maximises both mutation bias and
connectome information score.

**4. Explore circuits** — Visualise the top circuit as a network graph,
inspect which brain regions are included, and compare across gene sets.

---

**Key parameters (Circuit Search)**

| Parameter | Default | Description |
|-----------|---------|-------------|
| Circuit size | 46 | Number of structures in the circuit |
| SA steps | 50 000 | Iterations per SA run |
| SA runtimes | 10 | Independent SA runs (more = better) |
| Measure | SI | Connectome score: Shannon Information |
"""
    )
