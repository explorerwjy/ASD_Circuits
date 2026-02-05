"""
Expression preprocessing for atlas-grounded biophysical modeling.

This module implements:
- Gene intersection between reference and query
- Normalization (CPM + log transform)
- Highly variable gene (HVG) selection
- Ion channel gene inclusion
- Z-score scaling using reference statistics
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from typing import List, Tuple, Optional
from pathlib import Path


def intersect_genes(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    case_sensitive: bool = False
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Find intersection of genes between reference and query datasets.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data
    adata_query : ad.AnnData
        Query Patch-seq expression data
    case_sensitive : bool
        If False, convert gene names to lowercase for matching
        
    Returns
    -------
    ad.AnnData
        Subset reference with common genes
    ad.AnnData
        Subset query with common genes
    """
    # Get gene names
    genes_ref = list(adata_ref.var_names)
    genes_query = list(adata_query.var_names)
    
    # Show sample gene names for debugging
    print(f"Gene intersection:")
    print(f"  Reference genes: {len(genes_ref)}")
    print(f"  Query genes: {len(genes_query)}")
    print(f"  Sample reference genes: {genes_ref[:5]}")
    print(f"  Sample query genes: {genes_query[:5]}")
    
    # First try exact matching
    genes_ref_set = set(genes_ref)
    genes_query_set = set(genes_query)
    common_genes = sorted(list(genes_ref_set & genes_query_set))
    
    # If no exact match, try case-insensitive matching
    if len(common_genes) == 0 and not case_sensitive:
        print("  No exact matches found, trying case-insensitive matching...")
        # Convert to lowercase for matching
        genes_ref_lower = {g.lower(): g for g in genes_ref}
        genes_query_lower = {g.lower(): g for g in genes_query}
        common_lower = set(genes_ref_lower.keys()) & set(genes_query_lower.keys())
        
        if len(common_lower) > 0:
            # Map back to original case (prefer reference case)
            common_genes = sorted([genes_ref_lower[g] for g in common_lower])
            print(f"  Common genes (case-insensitive): {len(common_genes)}")
            
            # Create mapping from query genes to reference gene names
            query_to_ref = {
                genes_query_lower[g.lower()]: genes_ref_lower[g.lower()]
                for g in common_lower
            }
            
            # Subset reference
            adata_ref_subset = adata_ref[:, common_genes].copy()
            
            # Subset query and rename genes to match reference case
            query_common_original = sorted(query_to_ref.keys())
            adata_query_subset = adata_query[:, query_common_original].copy()
            # Rename query genes to match reference case
            adata_query_subset.var_names = [query_to_ref[g] for g in adata_query_subset.var_names]
            
            return adata_ref_subset, adata_query_subset
    
    print(f"  Common genes: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print("  ERROR: No common genes found!")
        print("  Possible causes:")
        print("    1. Different gene identifier formats:")
        print("       - Reference might use Ensembl IDs (e.g., ENSMUSG00000000001)")
        print("       - Query might use gene symbols (e.g., Gapdh)")
        print("    2. Case sensitivity: Gene symbols differ in case")
        print("    3. Gene name formatting: Different conventions or suffixes")
        print("  ")
        print("  Diagnostic information:")
        # Check if reference looks like Ensembl IDs
        ref_sample = genes_ref[0] if len(genes_ref) > 0 else ""
        query_sample = genes_query[0] if len(genes_query) > 0 else ""
        print(f"    Reference gene format: '{ref_sample}'")
        print(f"    Query gene format: '{query_sample}'")
        if ref_sample.startswith('ENSMUS') or ref_sample.startswith('ENSG'):
            print("    → Reference appears to use Ensembl IDs")
        if any(c.islower() for c in query_sample) and any(c.isupper() for c in query_sample):
            print("    → Query appears to use gene symbols")
        raise ValueError(
            "No common genes found between reference and query datasets. "
            "Gene identifier formats may be incompatible. "
            "Please check gene name formats and consider adding conversion logic."
        )
    
    # Subset to common genes
    adata_ref_subset = adata_ref[:, common_genes].copy()
    adata_query_subset = adata_query[:, common_genes].copy()
    
    return adata_ref_subset, adata_query_subset


def normalize_expression(
    adata: ad.AnnData,
    log_transform: bool = True,
    target_sum: float = 1e6
) -> ad.AnnData:
    """
    Normalize expression data: library-size normalize to CPM, then log-transform.
    
    Parameters
    ----------
    adata : ad.AnnData
        Expression data (raw counts)
    log_transform : bool
        If True, apply log(CPM + 1) transform
    target_sum : float
        Target sum for library-size normalization (default 1e6 for CPM)
        
    Returns
    -------
    ad.AnnData
        Normalized expression data
    """
    adata = adata.copy()
    
    # Library-size normalize to CPM
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Log-transform
    if log_transform:
        sc.pp.log1p(adata)
    
    return adata


def select_hvgs(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    flavor: str = 'seurat_v3',
    subset: bool = False
) -> List[str]:
    """
    Select highly variable genes (HVGs) from a dataset.
    
    Parameters
    ----------
    adata : ad.AnnData
        Expression data (normalized)
    n_top_genes : int
        Number of top HVGs to select
    flavor : str
        Method for HVG selection ('seurat_v3', 'seurat', or 'cell_ranger')
    subset : bool
        If True, subset adata to HVGs (modifies in place)
        
    Returns
    -------
    list
        List of HVG gene names
    """
    print(f"Selecting {n_top_genes} highly variable genes using {flavor} method...")
    
    # Compute HVG statistics
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        subset=subset
    )
    
    # Get HVG names
    hvg_mask = adata.var['highly_variable']
    hvg_genes = adata.var_names[hvg_mask].tolist()
    
    print(f"  Selected {len(hvg_genes)} HVGs")
    
    return hvg_genes


def select_union_hvgs(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    n_hvgs_ref: int = 3000,
    n_hvgs_query: int = 500,
    flavor: str = 'seurat_v3'
) -> List[str]:
    """
    Union HVG strategy: Select top HVGs from reference and query, then take union.
    
    This ensures rare Patch-seq cell types aren't lost by including query-specific
    highly variable genes.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (normalized)
    adata_query : ad.AnnData
        Query Patch-seq expression data (normalized)
    n_hvgs_ref : int
        Number of top HVGs to select from reference (default: 3000)
    n_hvgs_query : int
        Number of top HVGs to select from query (default: 500)
    flavor : str
        Method for HVG selection
        
    Returns
    -------
    list
        Union of HVG gene names from reference and query
    """
    print(f"\nUnion HVG Strategy:")
    print(f"  Selecting top {n_hvgs_ref} HVGs from reference...")
    hvg_ref = select_hvgs(adata_ref, n_top_genes=n_hvgs_ref, flavor=flavor, subset=False)
    
    print(f"  Selecting top {n_hvgs_query} HVGs from query...")
    hvg_query = select_hvgs(adata_query, n_top_genes=n_hvgs_query, flavor=flavor, subset=False)
    
    # Take union
    union_hvgs = sorted(list(set(hvg_ref) | set(hvg_query)))
    print(f"  Union: {len(union_hvgs)} unique HVGs ({len(hvg_ref)} ref + {len(hvg_query)} query)")
    
    return union_hvgs


def add_ion_channel_genes(
    hvg_genes: List[str],
    ion_channel_genes: List[str],
    all_genes: List[str]
) -> List[str]:
    """
    Force include ion channel genes in final gene set.
    
    Parameters
    ----------
    hvg_genes : list
        List of HVG gene names
    ion_channel_genes : list
        List of ion channel gene names to include
    all_genes : list
        All available genes (for filtering)
        
    Returns
    -------
    list
        Combined gene set: HVGs ∪ ion_channel_genes (intersected with available genes)
    """
    # Filter ion channel genes to those present in dataset
    available_ion_channels = [g for g in ion_channel_genes if g in all_genes]
    
    print(f"Ion channel gene inclusion:")
    print(f"  Requested: {len(ion_channel_genes)}")
    print(f"  Available in dataset: {len(available_ion_channels)}")
    
    # Combine: HVGs ∪ ion channel genes
    final_genes = list(set(hvg_genes) | set(available_ion_channels))
    
    added_count = len(final_genes) - len(hvg_genes)
    print(f"  Added {added_count} ion channel genes not in HVG set")
    print(f"  Final gene set size: {len(final_genes)}")
    
    return final_genes


def compute_scaling_params(
    adata_ref: ad.AnnData,
    genes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation for each gene from reference.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (normalized)
    genes : list
        List of genes to compute statistics for
        
    Returns
    -------
    np.ndarray
        Mean expression per gene (shape: n_genes,)
    np.ndarray
        Standard deviation per gene (shape: n_genes,)
    """
    # Subset to requested genes
    adata_subset = adata_ref[:, genes].copy()

    # Compute mean and std (handle sparse matrices)
    X = adata_subset.X
    gene_means = np.array(X.mean(axis=0)).flatten()

    # For sparse matrices, compute std using variance formula: std = sqrt(E[X^2] - E[X]^2)
    if hasattr(X, 'toarray'):
        # Sparse matrix - use variance formula
        X_squared = X.multiply(X)  # Element-wise square
        mean_of_squares = np.array(X_squared.mean(axis=0)).flatten()
        variance = mean_of_squares - gene_means ** 2
        # Handle numerical issues (small negative values)
        variance = np.maximum(variance, 0)
        gene_stds = np.sqrt(variance)
    else:
        # Dense matrix
        gene_stds = np.array(X.std(axis=0)).flatten()

    # Avoid division by zero
    gene_stds = np.maximum(gene_stds, 1e-6)
    
    return gene_means, gene_stds


def scale_expression(
    adata: ad.AnnData,
    gene_means: np.ndarray,
    gene_stds: np.ndarray,
    genes: List[str]
) -> ad.AnnData:
    """
    Z-score normalize expression using reference-derived statistics.
    
    Parameters
    ----------
    adata : ad.AnnData
        Expression data to scale (normalized)
    gene_means : np.ndarray
        Mean expression per gene from reference
    gene_stds : np.ndarray
        Standard deviation per gene from reference
    genes : list
        List of genes (must match order of gene_means/gene_stds)
        
    Returns
    -------
    ad.AnnData
        Z-score normalized expression data
    """
    adata = adata.copy()
    
    # Subset to requested genes
    adata_subset = adata[:, genes].copy()
    
    # Z-score normalize: (X - mean) / std
    X = adata_subset.X
    if hasattr(X, 'toarray'):  # sparse matrix
        X = X.toarray()
    
    X_scaled = (X - gene_means) / gene_stds
    
    # Update AnnData
    adata_subset.X = X_scaled
    
    return adata_subset


def preprocess_pipeline(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    ion_channel_genes: List[str],
    n_hvgs: int = 3000,
    n_hvgs_query: int = 500,
    hvg_flavor: str = 'seurat_v3',
    ref_is_log2: bool = True,
    return_log_normalized: bool = False
) -> Tuple[ad.AnnData, ad.AnnData, List[str], np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline: intersection → normalization → Union HVG selection → scaling.
    
    CRITICAL: Preserves raw counts in layers["counts"] for scVI modeling with Negative Binomial likelihood.
    .X contains log-normalized data for distance-based operations.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (should have layers["counts"] with raw UMI counts)
    adata_query : ad.AnnData
        Query Patch-seq expression data (should have layers["counts"] with raw UMI counts)
    ion_channel_genes : list
        List of ion channel genes to force include
    n_hvgs : int
        Number of HVGs to select from reference (default: 3000)
    n_hvgs_query : int
        Number of HVGs to select from query for union (default: 500)
    hvg_flavor : str
        HVG selection method
    ref_is_log2 : bool
        If True, reference .X is already log2(CPM+1) transformed
    return_log_normalized : bool
        If True, also return log-normalized (non-Z-scored) data for scVI training

    Returns
    -------
    ad.AnnData
        Preprocessed reference data (Z-scored in .X, raw counts in layers["counts"])
    ad.AnnData
        Preprocessed query data (Z-scored in .X, raw counts in layers["counts"])
    list
        Final gene set (Union HVGs + ion channels)
    np.ndarray
        Gene means from reference
    np.ndarray
        Gene standard deviations from reference

    If return_log_normalized=True, also returns:
    ad.AnnData
        Log-normalized reference data (for scVI training, with counts layer)
    ad.AnnData
        Log-normalized query data (for scVI training, with counts layer)
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Gene intersection
    print("\n[1/6] Gene intersection...")
    adata_ref, adata_query = intersect_genes(adata_ref, adata_query)
    
    # Note: AnnData subsetting in intersect_genes already preserves layers
    # CRITICAL: Explicitly cast counts layer to int32 after intersection
    # scvi-tools requires strictly discrete integers for Negative Binomial likelihood
    if "counts" in adata_ref.layers:
        counts_ref = adata_ref.layers["counts"]
        if hasattr(counts_ref, 'toarray'):
            counts_dense = counts_ref.toarray().astype(np.int32)
            adata_ref.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_ref.layers["counts"] = counts_ref.astype(np.int32)
    
    if "counts" in adata_query.layers:
        counts_query = adata_query.layers["counts"]
        if hasattr(counts_query, 'toarray'):
            counts_dense = counts_query.toarray().astype(np.int32)
            adata_query.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_query.layers["counts"] = counts_query.astype(np.int32)

    # Step 2: Normalization (only .X, preserve counts layer)
    print("\n[2/6] Normalization...")
    if not ref_is_log2:
        print("  Normalizing reference .X (preserving counts layer)...")
        # Normalize .X but keep counts layer intact
        adata_ref_norm = normalize_expression(adata_ref.copy(), log_transform=True)
        adata_ref.X = adata_ref_norm.X
    else:
        print("  Reference .X already log2(CPM+1) transformed, skipping normalization")

    print("  Normalizing query .X (preserving counts layer)...")
    adata_query_norm = normalize_expression(adata_query.copy(), log_transform=True)
    adata_query.X = adata_query_norm.X

    # Step 3: Union HVG selection
    print("\n[3/6] Union HVG selection...")
    union_hvg_genes = select_union_hvgs(
        adata_ref, 
        adata_query, 
        n_hvgs_ref=n_hvgs, 
        n_hvgs_query=n_hvgs_query, 
        flavor=hvg_flavor
    )

    # Step 4: Add ion channel genes
    print("\n[4/6] Adding ion channel genes...")
    final_genes = add_ion_channel_genes(
        union_hvg_genes,
        ion_channel_genes,
        adata_ref.var_names.tolist()
    )

    # Step 5: Subset to final genes (preserve counts layer)
    print("\n[5/6] Subsetting to final gene set (preserving counts layer)...")
    adata_ref_subset = adata_ref[:, final_genes].copy()
    adata_query_subset = adata_query[:, final_genes].copy()
    
    # CRITICAL: Ensure counts layer is preserved and explicitly cast to int32 after subsetting
    # scvi-tools requires strictly discrete integers for Negative Binomial likelihood
    # Floating-point counts (even if they end in .0) can cause training instability
    if "counts" in adata_ref_subset.layers:
        counts_ref = adata_ref_subset.layers["counts"]
        if hasattr(counts_ref, 'toarray'):
            # Sparse matrix - convert to dense, explicitly cast to int32, convert back
            counts_dense = counts_ref.toarray().astype(np.int32)
            adata_ref_subset.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            # Dense matrix - explicitly cast to int32
            adata_ref_subset.layers["counts"] = counts_ref.astype(np.int32)
    
    if "counts" in adata_query_subset.layers:
        counts_query = adata_query_subset.layers["counts"]
        if hasattr(counts_query, 'toarray'):
            counts_dense = counts_query.toarray().astype(np.int32)
            adata_query_subset.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_query_subset.layers["counts"] = counts_query.astype(np.int32)
    
    print(f"  Preserved counts layer: ref={adata_ref_subset.layers.get('counts') is not None}, query={adata_query_subset.layers.get('counts') is not None}")
    print(f"  Enforced integer type (int32) for counts layers after subsetting")

    # Keep log-normalized versions (subset to final genes) for scVI training
    adata_ref_log = adata_ref_subset.copy()
    adata_query_log = adata_query_subset.copy()

    # Step 6: Compute scaling parameters from reference
    print("\n[6/6] Computing scaling parameters...")
    gene_means, gene_stds = compute_scaling_params(adata_ref_subset, final_genes)

    # Step 7: Scale both datasets (.X only, keep counts layer)
    print("\n[7/7] Z-score scaling (.X only, preserving counts layer)...")
    adata_ref_scaled = scale_expression(adata_ref_subset, gene_means, gene_stds, final_genes)
    adata_query_scaled = scale_expression(adata_query_subset, gene_means, gene_stds, final_genes)
    
    # Copy counts layer to scaled versions and enforce integer type
    # CRITICAL: scvi-tools requires strictly discrete integers for Negative Binomial likelihood
    if "counts" in adata_ref_subset.layers:
        counts_ref = adata_ref_subset.layers["counts"].copy()
        # Explicitly cast to int32 after copying
        if hasattr(counts_ref, 'toarray'):
            counts_dense = counts_ref.toarray().astype(np.int32)
            adata_ref_scaled.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_ref_scaled.layers["counts"] = counts_ref.astype(np.int32)
    
    if "counts" in adata_query_subset.layers:
        counts_query = adata_query_subset.layers["counts"].copy()
        # Explicitly cast to int32 after copying
        if hasattr(counts_query, 'toarray'):
            counts_dense = counts_query.toarray().astype(np.int32)
            adata_query_scaled.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_query_scaled.layers["counts"] = counts_query.astype(np.int32)
    
    # Also ensure log-normalized versions have integer counts layer
    if "counts" in adata_ref_log.layers:
        counts_ref_log = adata_ref_log.layers["counts"].copy()
        if hasattr(counts_ref_log, 'toarray'):
            counts_dense = counts_ref_log.toarray().astype(np.int32)
            adata_ref_log.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_ref_log.layers["counts"] = counts_ref_log.astype(np.int32)
    
    if "counts" in adata_query_log.layers:
        counts_query_log = adata_query_log.layers["counts"].copy()
        if hasattr(counts_query_log, 'toarray'):
            counts_dense = counts_query_log.toarray().astype(np.int32)
            adata_query_log.layers["counts"] = sp.csr_matrix(counts_dense)
        else:
            adata_query_log.layers["counts"] = counts_query_log.astype(np.int32)
    
    print(f"  Enforced integer type (int32) for all counts layers")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final gene set: {len(final_genes)} genes")
    print(f"Reference shape: {adata_ref_scaled.shape}")
    print(f"Query shape: {adata_query_scaled.shape}")
    print(f"Counts layer preserved: ref={adata_ref_scaled.layers.get('counts') is not None}, query={adata_query_scaled.layers.get('counts') is not None}")

    if return_log_normalized:
        return adata_ref_scaled, adata_query_scaled, final_genes, gene_means, gene_stds, adata_ref_log, adata_query_log

    return adata_ref_scaled, adata_query_scaled, final_genes, gene_means, gene_stds
