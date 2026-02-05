"""
Patch-seq Transcriptome Mapping Package

A framework for mapping Patch-seq neurons onto high-resolution scRNA-seq reference
atlases and computing canonical expression profiles for downstream biophysical modeling.

This package supports both mouse and human datasets, with configurable pipelines for:
- Data loading and preprocessing
- Reference integration (scVI, scANVI, Harmony)
- Hierarchical atlas mapping
- Canonical expression computation
"""

__version__ = "0.2.0"

# Data loading
from .data import (
    load_reference_atlas,
    load_patchseq_v1,
    load_patchseq_m1,
    load_ion_channel_genes,
)

# Preprocessing
from .preprocessing import (
    intersect_genes,
    normalize_expression,
    select_hvgs,
    select_union_hvgs,
    add_ion_channel_genes,
    compute_scaling_params,
    scale_expression,
    preprocess_pipeline,
)

# Integration
from .integration import (
    train_scvi,
    train_scanvi,
    map_query_scarches,
    integrate_pca_harmony,
)

# Atlas mapping
from .mapping import (
    compute_cluster_centroids,
    find_candidate_clusters,
    knn_label_transfer,
    hierarchical_assignment,
)

# Canonical expression
from .expression import (
    compute_canonical_profiles,
    compute_query_effective_expression,
    get_canonical_expression_for_query,
    compute_canonical_pipeline,
)

# Configuration utilities
from .utils import (
    load_species_config,
    load_dataset_config,
    get_qc_thresholds,
)

__all__ = [
    # Version
    "__version__",
    # Data loading
    "load_reference_atlas",
    "load_patchseq_v1",
    "load_patchseq_m1",
    "load_ion_channel_genes",
    # Preprocessing
    "intersect_genes",
    "normalize_expression",
    "select_hvgs",
    "select_union_hvgs",
    "add_ion_channel_genes",
    "compute_scaling_params",
    "scale_expression",
    "preprocess_pipeline",
    # Integration
    "train_scvi",
    "train_scanvi",
    "map_query_scarches",
    "integrate_pca_harmony",
    # Atlas mapping
    "compute_cluster_centroids",
    "find_candidate_clusters",
    "knn_label_transfer",
    "hierarchical_assignment",
    # Canonical expression
    "compute_canonical_profiles",
    "compute_query_effective_expression",
    "get_canonical_expression_for_query",
    "compute_canonical_pipeline",
    # Configuration
    "load_species_config",
    "load_dataset_config",
    "get_qc_thresholds",
]
