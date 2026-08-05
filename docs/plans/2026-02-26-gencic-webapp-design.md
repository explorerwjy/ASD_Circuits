# GENCIC Webapp Design

**Date**: 2026-02-26
**Goal**: Public webapp for running the GENCIC framework interactively — gene list input, bias profiling, circuit search with Pareto front, and circuit visualization.

## Audience

Public community tool for the broader neuroscience community. Must be polished, documented, and robust.

## Tech Stack

- **Frontend/Framework**: Streamlit (multi-page app)
- **Backend**: Python, reusing existing `src/ASD_Circuits.py` functions directly
- **Visualization**: Plotly (charts) + custom Cytoscape.js embed (circuit network graphs)
- **Compute model**: Hybrid — instant bias calculation, async SA circuit search via threading
- **Architecture**: Single Streamlit app with background threading for SA
- **Location**: `webapp/` directory inside this repository

## User Flow — 4 Pages

### Page 1: Gene Input & Bias Profile
1. User provides genes via:
   - Text area: paste gene symbols (one per line or comma-separated)
   - File upload: CSV/TSV with gene symbols or Entrez IDs
   - Preset dropdown: ~70 pre-configured gene sets from config.yaml
2. App converts gene symbols → Entrez IDs (via HGNC mapping from `LoadGeneINFO()`)
3. Shows matched/unmatched genes summary
4. Weighting mode:
   - Uniform (default): all genes weight = 1
   - Upload weights: CSV with gene,weight columns
   - Mutation-derived (SPARK): if user uploads mutation data
5. Computes bias instantly via `MouseSTR_AvgZ_Weighted()`
6. Displays:
   - Plotly bar chart of structure bias (ranked, colored by brain region)
   - Table of all structures with EFFECT, Rank, Region
   - Download button for bias CSV

### Page 2: Permutation Testing
1. After bias is computed, user runs permutation p-values
2. Parameters: number of permutations (default 1000, max 10000)
3. Progress bar during computation
4. Results: bias with p-values, volcano-style plot, significant structures highlighted

### Page 3: Circuit Search (Pareto Front)
1. User selects circuit size (slider, e.g., 5–50)
2. SA parameters with sensible defaults (steps=10k, runtimes=5)
3. "Start Search" button → SA runs in background thread
4. Progress bar + live best-score display
5. Pareto front plot (mean bias vs. connectivity score, Plotly)
6. Interactive: click points on Pareto front to select circuits

### Page 4: Circuit Visualization
1. Select a circuit from the Pareto front or manually specify structures
2. Custom Cytoscape.js network graph:
   - Nodes = brain structures, sized by bias, colored by brain region
   - Edges = connections, width ∝ InfoMat value
   - Force-directed layout (cose-bilkent)
   - Hover/click interactivity
3. Side panel: circuit statistics (mean bias, connectivity score, edges, region composition)
4. Table of structures in circuit
5. Export: PNG, CSV, adjacency matrix

## Module Structure

```
webapp/
├── app.py                    # Main entry point
├── pages/
│   ├── 1_Gene_Input.py       # Gene input & bias
│   ├── 2_Permutation.py      # Permutation testing
│   ├── 3_Circuit_Search.py   # SA search & Pareto
│   └── 4_Visualization.py    # Cytoscape viewer
├── core/
│   ├── bias.py               # Bias computation wrapper
│   ├── circuit_search.py     # SA with threading & progress
│   ├── pareto.py             # Pareto front generation
│   ├── gene_mapping.py       # Gene symbol ↔ Entrez ID
│   └── data_loader.py        # Cached matrix loading
├── components/
│   ├── cytoscape_viewer.py   # Cytoscape.js custom component
│   ├── pareto_plot.py        # Plotly Pareto front
│   └── bias_chart.py         # Plotly bias bar chart
├── static/
│   └── cytoscape.min.js      # Cytoscape.js library
├── config/
│   └── webapp_config.yaml    # Webapp settings
└── requirements.txt
```

## Key Design Decisions

1. **Reuse src/ directly** — import from `src/ASD_Circuits.py`, no rewriting core math
2. **@st.cache_data** for expression matrices and connectome (loaded once per session)
3. **threading.Thread** for SA — shared dict for progress, Streamlit polls every 2 seconds
4. **Uniform weighting default** — most public users won't have mutation data
5. **Auto-detect gene ID format** — symbols, Entrez, or Ensembl
6. **Custom Cytoscape.js** via `streamlit.components.v1.html` for maximum flexibility
7. **Pre-computed examples** bundled for instant demo (ASD_HIQ, DDD gene sets)

## Verification Strategy

- **Automated tests**: bias computation, SA correctness, gene mapping, Pareto extraction
- **Human review**: Streamlit UI layout, Cytoscape rendering, UX flow, responsiveness
