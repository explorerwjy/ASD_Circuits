"""
pages/03_circuit_viewer.py
==========================
Circuit Viewer page — placeholder.

This page will allow users to:
- Load a circuit result (from search or a saved file)
- Visualise the circuit as an interactive network graph (Plotly)
- Colour nodes by brain region, bias score, or neurotransmitter type
- Inspect individual structure details

TODO: Implement Plotly-based network visualisation.
"""

import streamlit as st

st.set_page_config(page_title="Circuit Viewer — GENCIC", page_icon="🕸️", layout="wide")

st.title("🕸️ Circuit Viewer")
st.info(
    "**Coming soon.** This page will display identified brain circuits "
    "as interactive network graphs coloured by brain region and bias score.",
    icon="🚧",
)
