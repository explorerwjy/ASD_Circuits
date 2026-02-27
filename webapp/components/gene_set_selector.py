"""
components/gene_set_selector.py
================================
Reusable sidebar widget for selecting a gene set.

Renders a grouped selectbox (by category) for pre-defined gene set presets,
plus an optional file-upload path for custom .gw files.

Usage
-----
    from components.gene_set_selector import gene_set_selector

    gene_weights, preset_name = gene_set_selector(sidebar=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from core.data_loader import load_webapp_config, load_preset_gene_weights, load_gene_weights


def gene_set_selector(
    sidebar: bool = True,
    allow_upload: bool = True,
) -> Tuple[Optional[Dict[int, float]], Optional[str]]:
    """Render a gene set selection widget and return the loaded gene weights.

    Parameters
    ----------
    sidebar : bool
        If True, render inside ``st.sidebar``; otherwise in the main area.
    allow_upload : bool
        If True, include a file-upload widget for custom .gw files.

    Returns
    -------
    gene_weights : dict[int, float] | None
        Loaded gene weight mapping (Entrez ID → weight), or None if not loaded.
    preset_name : str | None
        Name of the selected preset, or "custom" for uploads, or None.
    """
    cfg = load_webapp_config()
    presets = cfg.get("gene_set_presets", [])
    categories = cfg.get("ui", {}).get("gene_set_categories", [])

    # Build grouped options
    grouped: Dict[str, list] = {cat: [] for cat in categories}
    for p in presets:
        cat = p.get("category", "Other")
        grouped.setdefault(cat, [])
        grouped[cat].append(p["name"])

    container = st.sidebar if sidebar else st

    with container:
        st.subheader("Gene Set")

        source = st.radio(
            "Source",
            options=["Preset", "Upload .gw file"] if allow_upload else ["Preset"],
            horizontal=True,
            key="gene_set_source",
        )

        if source == "Preset":
            # Category filter
            selected_cat = st.selectbox(
                "Category",
                options=["All"] + [c for c in categories if grouped.get(c)],
                key="gene_set_category",
            )
            if selected_cat == "All":
                options = [p["name"] for p in presets]
            else:
                options = grouped.get(selected_cat, [])

            selected_name = st.selectbox(
                "Gene Set",
                options=options,
                key="gene_set_name",
            )

            # Show description
            preset = next((p for p in presets if p["name"] == selected_name), None)
            if preset:
                st.caption(preset.get("description", ""))

            if st.button("Load gene set", key="load_preset_btn"):
                with st.spinner(f"Loading {selected_name}…"):
                    try:
                        gw = load_preset_gene_weights(selected_name)
                        st.session_state["gene_weights"] = gw
                        st.session_state["gene_set_label"] = selected_name
                        st.success(f"Loaded {len(gw):,} genes", icon="✅")
                    except Exception as exc:
                        st.error(str(exc), icon="❌")

            gw = st.session_state.get("gene_weights")
            name = st.session_state.get("gene_set_label")
            return gw, name

        else:  # Upload
            uploaded = st.file_uploader(
                "Upload gene weight file (.gw or .csv)",
                type=["gw", "csv"],
                key="gw_upload",
                help="Headerless CSV: EntrezID,weight (one gene per line)",
            )
            if uploaded is not None:
                import tempfile, os
                with tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False
                ) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                try:
                    gw = load_gene_weights(tmp_path)
                    st.session_state["gene_weights"] = gw
                    st.session_state["gene_set_label"] = "custom"
                    st.success(f"Loaded {len(gw):,} genes", icon="✅")
                except Exception as exc:
                    st.error(str(exc), icon="❌")
                finally:
                    os.unlink(tmp_path)

            gw = st.session_state.get("gene_weights")
            name = st.session_state.get("gene_set_label")
            return gw, name
