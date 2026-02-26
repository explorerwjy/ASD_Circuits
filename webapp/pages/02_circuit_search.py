"""
pages/02_circuit_search.py
==========================
Circuit Search page — placeholder.

This page will allow users to:
- Configure simulated annealing parameters (size, steps, runtimes)
- Select a bias input (gene set or pre-computed bias file)
- Run the SA circuit search interactively
- View the Pareto front of bias vs. connectivity score

TODO: Implement SA integration and interactive search UI.
"""

import streamlit as st

st.set_page_config(page_title="Circuit Search — GENCIC", page_icon="🔍", layout="wide")

st.title("🔍 Circuit Search")
st.info(
    "**Coming soon.** This page will run simulated annealing circuit search "
    "with configurable parameters and display the Pareto front.",
    icon="🚧",
)
