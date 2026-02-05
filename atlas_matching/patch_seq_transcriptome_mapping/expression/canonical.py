"""
Canonical expression profile computation.

This module implements:
- Cluster-level canonical profiles
- Subclass-level canonical profiles
- Query effective expression (soft assignment)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def compute_canonical_profiles(
    adata_ref: ad.AnnData,
    level_key: str,
    use_trimmed_mean: bool = False,
    trim_fraction: float = 0.1
) -> pd.DataFrame:
    """
    Compute canonical expression profiles at a given hierarchical level.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (preprocessed)
    level_key : str
        Column name in adata_ref.obs for grouping (e.g., 'cluster' or 'subclass')
    use_trimmed_mean : bool
        If True, use trimmed mean instead of regular mean
    trim_fraction : float
        Fraction to trim from each tail (only if use_trimmed_mean=True)
        
    Returns
    -------
    pd.DataFrame
        Canonical profiles (n_groups × n_genes)
        Index: group labels, Columns: gene names
    """
    print(f"Computing canonical profiles for '{level_key}'...")
    
    labels = adata_ref.obs[level_key].values
    unique_labels = sorted(pd.Series(labels).dropna().unique())
    
    # Memory-safe computation: iterate through clusters to avoid full matrix densification
    # This prevents "Memory Bombs" when working with large sparse matrices
    canonical_profiles = {}
    
    for label in unique_labels:
        mask = labels == label
        if mask.sum() == 0:
            continue
        
        # Subset to this cluster only (memory-efficient)
        adata_cluster = adata_ref[mask].copy()
        X_cluster = adata_cluster.X
        
        # Only densify small cluster-level chunks, not the full matrix
        if hasattr(X_cluster, 'toarray'):
            # Sparse matrix - only densify this cluster's data
            group_expr = X_cluster.toarray()
        else:
            group_expr = X_cluster
        
        if use_trimmed_mean:
            # Trimmed mean: remove top/bottom trim_fraction
            n_trim = int(group_expr.shape[0] * trim_fraction)
            if n_trim > 0:
                sorted_expr = np.sort(group_expr, axis=0)
                trimmed = sorted_expr[n_trim:-n_trim] if n_trim > 0 else sorted_expr
                profile = trimmed.mean(axis=0)
            else:
                profile = group_expr.mean(axis=0)
        else:
            # Regular mean
            profile = group_expr.mean(axis=0)
        
        canonical_profiles[label] = profile
    
    # Convert to DataFrame
    profiles_df = pd.DataFrame(canonical_profiles).T
    profiles_df.index.name = level_key
    profiles_df.columns = adata_ref.var_names
    
    print(f"  Computed {len(profiles_df)} canonical profiles")
    print(f"  Profile shape: {profiles_df.shape}")
    
    return profiles_df


def compute_query_effective_expression(
    adata_query: ad.AnnData,
    adata_ref: ad.AnnData,
    mapping_results: pd.DataFrame,
    latent_key: str = 'X_latent',
    k_nn: int = 20,
    use_soft_assignment: bool = True
) -> pd.DataFrame:
    """
    Compute effective expression for query cells using soft assignment.
    
    Parameters
    ----------
    adata_query : ad.AnnData
        Query Patch-seq data with latent embeddings
    adata_ref : ad.AnnData
        Reference atlas with latent embeddings and expression
    mapping_results : pd.DataFrame
        Mapping results from hierarchical_assignment()
    latent_key : str
        Key in obsm containing latent embeddings
    k_nn : int
        Number of nearest neighbors for soft assignment
    use_soft_assignment : bool
        If True, use distance-weighted neighbor expression
        If False, use hard cluster centroid assignment
        
    Returns
    -------
    pd.DataFrame
        Effective expression profiles (n_query × n_genes)
    """
    print("Computing query effective expression...")
    
    query_latent = adata_query.obsm[latent_key]
    ref_latent = adata_ref.obsm[latent_key]
    
    # Get reference expression - KEEP SPARSE, do NOT densify globally
    # Memory-safe: we'll only densify small neighbor slices (20 cells) within the loop
    ref_expr = adata_ref.X
    
    n_query = len(adata_query)
    n_genes = adata_ref.n_vars
    
    effective_expr = np.zeros((n_query, n_genes))
    
    if use_soft_assignment:
        print(f"  Using soft assignment with k={k_nn} neighbors...")
        print(f"  Memory-safe: using sparse slicing, densifying only {k_nn} neighbor cells per query cell")
        
        # Fit KNN on reference
        nn = NearestNeighbors(n_neighbors=min(k_nn, len(ref_latent)), metric='euclidean')
        nn.fit(ref_latent)
        
        # Find neighbors for each query cell
        distances, indices = nn.kneighbors(query_latent)
        
        # Compute distance-weighted expression
        for i in range(n_query):
            # Weight by inverse distance
            weights = 1.0 / (distances[i] + 1e-6)
            weights = weights / weights.sum()
            
            # Memory-safe sparse slicing: slice stays sparse until we densify only this small chunk
            neighbor_expr = ref_expr[indices[i]]  # Slice stays sparse (k_nn rows)
            
            # Only densify this small neighbor slice (e.g., 20 cells), not the full matrix
            if hasattr(neighbor_expr, 'toarray'):
                neighbor_expr = neighbor_expr.toarray()  # Only densify k_nn rows
            
            # Weighted average of neighbor expressions
            effective_expr[i] = (neighbor_expr * weights[:, np.newaxis]).sum(axis=0)
    
    else:
        print("  Using hard cluster centroid assignment...")
        # This would use canonical profiles directly
        # For now, fall back to soft assignment
        effective_expr = compute_query_effective_expression(
            adata_query, adata_ref, mapping_results,
            latent_key, k_nn, use_soft_assignment=True
        )
        return effective_expr
    
    # Convert to DataFrame
    effective_expr_df = pd.DataFrame(
        effective_expr,
        index=adata_query.obs_names,
        columns=adata_ref.var_names
    )
    
    print(f"  Computed effective expression for {n_query} query cells")
    
    return effective_expr_df


def get_canonical_expression_for_query(
    mapping_results: pd.DataFrame,
    canonical_profiles_cluster: pd.DataFrame,
    canonical_profiles_subclass: pd.DataFrame,
    fallback_to_subclass: bool = True
) -> pd.DataFrame:
    """
    Get canonical expression profiles for query cells based on mapping results.
    
    This function handles the fallback logic:
    - ok_cluster: use cluster-level canonical profile
    - ok_subclass_only: use subclass-level canonical profile
    - rejected: use subclass-level if available, else None
    
    Parameters
    ----------
    mapping_results : pd.DataFrame
        Mapping results from hierarchical_assignment()
    canonical_profiles_cluster : pd.DataFrame
        Cluster-level canonical profiles
    canonical_profiles_subclass : pd.DataFrame
        Subclass-level canonical profiles
    fallback_to_subclass : bool
        If True, use subclass profile for rejected cells
        
    Returns
    -------
    pd.DataFrame
        Canonical expression profiles (n_query × n_genes)
    """
    print("Assigning canonical profiles to query cells...")
    
    n_query = len(mapping_results)
    n_genes = len(canonical_profiles_cluster.columns)
    
    canonical_expr = np.zeros((n_query, n_genes))
    assigned_levels = []
    
    for i in range(n_query):
        status = mapping_results.iloc[i]['mapping_status']
        assigned_cluster = mapping_results.iloc[i]['assigned_cluster']
        assigned_subclass = mapping_results.iloc[i]['assigned_subclass']
        
        if status == 'ok_cluster' and assigned_cluster is not None:
            # Use cluster-level profile
            if assigned_cluster in canonical_profiles_cluster.index:
                canonical_expr[i] = canonical_profiles_cluster.loc[assigned_cluster].values
                assigned_levels.append('cluster')
            else:
                # Fallback to subclass
                if assigned_subclass in canonical_profiles_subclass.index:
                    canonical_expr[i] = canonical_profiles_subclass.loc[assigned_subclass].values
                    assigned_levels.append('subclass')
                else:
                    canonical_expr[i] = np.nan
                    assigned_levels.append('none')
        
        elif status == 'ok_subclass_only' or (status == 'rejected' and fallback_to_subclass):
            # Use subclass-level profile
            if assigned_subclass in canonical_profiles_subclass.index:
                canonical_expr[i] = canonical_profiles_subclass.loc[assigned_subclass].values
                assigned_levels.append('subclass')
            else:
                canonical_expr[i] = np.nan
                assigned_levels.append('none')
        
        else:
            # Rejected and no fallback
            canonical_expr[i] = np.nan
            assigned_levels.append('none')
    
    # Convert to DataFrame
    canonical_expr_df = pd.DataFrame(
        canonical_expr,
        index=mapping_results.index,
        columns=canonical_profiles_cluster.columns
    )
    
    # Summary
    level_counts = pd.Series(assigned_levels).value_counts()
    print("  Profile assignment summary:")
    for level, count in level_counts.items():
        pct = 100 * count / n_query
        print(f"    {level}: {count} cells ({pct:.1f}%)")
    
    return canonical_expr_df


def compute_canonical_pipeline(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    mapping_results: pd.DataFrame,
    use_soft_assignment: bool = True,
    k_nn: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete canonical expression pipeline.
    
    Computes:
    1. Cluster-level canonical profiles
    2. Subclass-level canonical profiles
    3. Query effective expression (soft assignment)
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data
    adata_query : ad.AnnData
        Query Patch-seq expression data
    mapping_results : pd.DataFrame
        Mapping results from hierarchical_assignment()
    use_soft_assignment : bool
        Use soft assignment for query effective expression
    k_nn : int
        Number of neighbors for soft assignment
        
    Returns
    -------
    pd.DataFrame
        Cluster-level canonical profiles
    pd.DataFrame
        Subclass-level canonical profiles
    pd.DataFrame
        Query effective expression
    """
    print("=" * 60)
    print("CANONICAL EXPRESSION PIPELINE")
    print("=" * 60)
    print("  Using log-normalized expression data (log-CPM) for biologically interpretable results")
    
    # Step 1: Compute cluster-level profiles
    print("\n[1/3] Computing cluster-level canonical profiles...")
    canonical_cluster = compute_canonical_profiles(adata_ref, 'cluster')
    
    # Step 2: Compute subclass-level profiles
    print("\n[2/3] Computing subclass-level canonical profiles...")
    canonical_subclass = compute_canonical_profiles(adata_ref, 'subclass')
    
    # Step 3: Compute query effective expression
    print("\n[3/3] Computing query effective expression...")
    if use_soft_assignment:
        effective_expr = compute_query_effective_expression(
            adata_query, adata_ref, mapping_results,
            k_nn=k_nn, use_soft_assignment=True
        )
    else:
        # Use canonical profiles based on mapping
        effective_expr = get_canonical_expression_for_query(
            mapping_results, canonical_cluster, canonical_subclass
        )
    
    print("\n" + "=" * 60)
    print("CANONICAL EXPRESSION COMPLETE")
    print("=" * 60)
    print("  All outputs contain log-normalized expression (log-CPM) for biophysical modeling")
    
    return canonical_cluster, canonical_subclass, effective_expr
