"""
Atlas mapping (label transfer) using two-stage hybrid approach.

This module implements:
- Two-stage hybrid mapping (cluster candidate selection + KNN)
- Two-level hierarchical assignment (subclass + cluster)
- Confidence scoring and rejection handling
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Tuple, Optional, Dict
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import warnings


def compute_cluster_centroids(
    adata_ref: ad.AnnData,
    cluster_key: str,
    latent_key: str = 'X_latent'
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Precompute cluster centroids in latent space.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas with latent embeddings
    cluster_key : str
        Column name in adata_ref.obs containing cluster labels
    latent_key : str
        Key in adata_ref.obsm containing latent embeddings
        
    Returns
    -------
    np.ndarray
        Cluster centroids (n_clusters × n_latent)
    dict
        Mapping from cluster label to centroid index
    """
    print(f"Computing cluster centroids for '{cluster_key}'...")
    
    latent = adata_ref.obsm[latent_key]
    clusters = adata_ref.obs[cluster_key].values
    
    # Get unique clusters
    unique_clusters = sorted(pd.Series(clusters).dropna().unique())
    n_clusters = len(unique_clusters)
    
    # Compute centroids
    centroids = np.zeros((n_clusters, latent.shape[1]))
    cluster_to_idx = {}
    
    for idx, cluster in enumerate(unique_clusters):
        mask = clusters == cluster
        centroids[idx] = latent[mask].mean(axis=0)
        cluster_to_idx[cluster] = idx
    
    print(f"  Computed {n_clusters} cluster centroids")
    
    return centroids, cluster_to_idx


def find_candidate_clusters(
    query_latent: np.ndarray,
    cluster_centroids: np.ndarray,
    k: int = 15
) -> np.ndarray:
    """
    Stage 1: Find top-K candidate clusters for each query cell.
    
    Parameters
    ----------
    query_latent : np.ndarray
        Query cell latent embeddings (n_query × n_latent)
    cluster_centroids : np.ndarray
        Cluster centroids (n_clusters × n_latent)
    k : int
        Number of candidate clusters to select (default: 10-20)
        
    Returns
    -------
    np.ndarray
        Candidate cluster indices for each query cell (n_query × k)
    """
    # Compute distances from query cells to cluster centroids
    distances = cdist(query_latent, cluster_centroids, metric='euclidean')
    
    # Find top-K nearest clusters for each query cell
    candidate_indices = np.argsort(distances, axis=1)[:, :k]
    
    return candidate_indices


def knn_label_transfer(
    query_latent: np.ndarray,
    ref_latent: np.ndarray,
    ref_labels: np.ndarray,
    candidate_mask: Optional[np.ndarray] = None,
    k: int = 20,
    distance_weighted: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2: KNN label transfer within candidate clusters.
    
    Parameters
    ----------
    query_latent : np.ndarray
        Query cell latent embeddings (n_query × n_latent)
    ref_latent : np.ndarray
        Reference cell latent embeddings (n_ref × n_latent)
    ref_labels : np.ndarray
        Reference cell labels (n_ref,)
    candidate_mask : np.ndarray, optional
        Boolean mask indicating candidate reference cells (n_ref,)
        If None, use all reference cells
    k : int
        Number of nearest neighbors (default: 15-30)
    distance_weighted : bool
        If True, weight votes by inverse distance
        
    Returns
    -------
    np.ndarray
        Predicted labels for query cells (n_query,)
    np.ndarray
        Confidence scores (max posterior probability) (n_query,)
    np.ndarray
        Label probability matrix (n_query × n_labels)
    np.ndarray
        Unique label array (n_labels,)
    """
    if candidate_mask is None:
        candidate_mask = np.ones(len(ref_latent), dtype=bool)
    
    # Subset reference to candidates
    ref_latent_candidates = ref_latent[candidate_mask]
    ref_labels_candidates = ref_labels[candidate_mask]
    
    # Fit KNN on candidate cells
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_latent_candidates)), metric='euclidean')
    nn.fit(ref_latent_candidates)
    
    # Find neighbors for query cells
    distances, indices = nn.kneighbors(query_latent)
    
    # Get labels of neighbors
    neighbor_labels = ref_labels_candidates[indices]  # (n_query × k)
    
    # Compute label probabilities
    n_query = len(query_latent)
    unique_labels = np.unique(ref_labels_candidates)
    label_probs = np.zeros((n_query, len(unique_labels)))
    
    for i in range(n_query):
        if distance_weighted:
            # Weight by inverse distance
            weights = 1.0 / (distances[i] + 1e-6)
            weights = weights / weights.sum()
        else:
            # Uniform weights
            weights = np.ones(len(indices[i])) / len(indices[i])
        
        # Aggregate votes
        for j, label in enumerate(unique_labels):
            mask = neighbor_labels[i] == label
            label_probs[i, j] = weights[mask].sum()
    
    # Get predicted labels and confidence
    pred_indices = label_probs.argmax(axis=1)
    pred_labels = unique_labels[pred_indices]
    confidences = label_probs.max(axis=1)
    
    return pred_labels, confidences, label_probs, unique_labels


def hierarchical_assignment(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    latent_key: str = 'X_latent',
    subclass_key: str = 'subclass',
    cluster_key: str = 'cluster',
    k_candidates: int = 15,
    k_nn: int = 20,
    conf_threshold_subclass: float = 0.7,
    conf_threshold_cluster: float = 0.7
) -> pd.DataFrame:
    """
    Two-level hierarchical assignment: subclass (coarse) + cluster (fine).
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas with latent embeddings and labels
    adata_query : ad.AnnData
        Query Patch-seq data with latent embeddings
    latent_key : str
        Key in obsm containing latent embeddings
    subclass_key : str
        Column name for subclass labels in reference
    cluster_key : str
        Column name for cluster labels in reference
    k_candidates : int
        Number of candidate clusters in Stage 1
    k_nn : int
        Number of nearest neighbors in Stage 2
    conf_threshold_subclass : float
        Confidence threshold for subclass assignment
    conf_threshold_cluster : float
        Confidence threshold for cluster assignment
        
    Returns
    -------
    pd.DataFrame
        Mapping results with columns:
        - assigned_subclass: predicted subclass
        - assigned_cluster: predicted cluster
        - conf_subclass: subclass confidence
        - conf_cluster: cluster confidence
        - mapping_status: ok_cluster / ok_subclass_only / rejected
    """
    print("=" * 60)
    print("HIERARCHICAL ATLAS MAPPING")
    print("=" * 60)
    
    ref_latent = adata_ref.obsm[latent_key]
    query_latent = adata_query.obsm[latent_key]
    
    # Get reference labels
    ref_subclasses = adata_ref.obs[subclass_key].values
    ref_clusters = adata_ref.obs[cluster_key].values
    
    # Step 1: Compute cluster centroids
    print("\n[Stage 1] Computing cluster centroids...")
    cluster_centroids, cluster_to_idx = compute_cluster_centroids(
        adata_ref, cluster_key, latent_key
    )
    
    # Create reverse mapping: cluster label -> subclass
    cluster_to_subclass = {}
    for cluster in np.unique(ref_clusters):
        mask = ref_clusters == cluster
        subclasses = ref_subclasses[mask]
        # Get most common subclass for this cluster
        cluster_to_subclass[cluster] = pd.Series(subclasses).mode()[0]
    
    # Step 2: Find candidate clusters for each query cell
    print(f"\n[Stage 2] Finding top-{k_candidates} candidate clusters...")
    candidate_cluster_indices = find_candidate_clusters(
        query_latent, cluster_centroids, k=k_candidates
    )
    
    # Convert cluster indices back to labels
    idx_to_cluster = {v: k for k, v in cluster_to_idx.items()}
    candidate_clusters = np.array([
        [idx_to_cluster[idx] for idx in row]
        for row in candidate_cluster_indices
    ])
    
    # Step 3: KNN label transfer for subclass
    print(f"\n[Stage 3] KNN label transfer for subclass (k={k_nn})...")
    pred_subclass, conf_subclass, _, _ = knn_label_transfer(
        query_latent, ref_latent, ref_subclasses,
        k=k_nn, distance_weighted=True
    )
    
    # Step 4: KNN label transfer for cluster (within candidates)
    print(f"\n[Stage 4] KNN label transfer for cluster (k={k_nn})...")
    print(f"  OPTIMIZED: Processing all {len(query_latent)} query cells with vectorized operations...")

    n_query = len(query_latent)
    pred_cluster = np.full(n_query, None, dtype=object)
    conf_cluster = np.zeros(n_query)

    # VECTORIZED OPTIMIZATION: Process all query cells at once
    # Create a mapping from cluster label to indices
    cluster_labels_unique = np.unique(ref_clusters)
    cluster_to_ref_indices = {}
    for cluster_label in cluster_labels_unique:
        cluster_to_ref_indices[cluster_label] = np.where(ref_clusters == cluster_label)[0]

    # Process in batches for memory efficiency
    batch_size = 100  # Process 100 query cells at a time
    n_batches = int(np.ceil(n_query / batch_size))

    for batch_idx in range(n_batches):
        if batch_idx % 10 == 0:
            print(f"  Processing batch {batch_idx + 1}/{n_batches}...")

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_query)
        batch_size_actual = end_idx - start_idx

        # Get candidate clusters for this batch
        batch_candidate_clusters = candidate_clusters[start_idx:end_idx]

        # For each cell in batch, collect all reference indices in candidate clusters
        for i_batch in range(batch_size_actual):
            i_global = start_idx + i_batch

            # Get all reference indices for this cell's candidate clusters
            candidate_ref_indices = []
            for cluster_label in batch_candidate_clusters[i_batch]:
                if cluster_label in cluster_to_ref_indices:
                    candidate_ref_indices.extend(cluster_to_ref_indices[cluster_label])

            if len(candidate_ref_indices) > 0:
                # Create boolean mask (vectorized)
                candidate_mask = np.zeros(len(ref_latent), dtype=bool)
                candidate_mask[candidate_ref_indices] = True

                # Perform KNN within candidates
                pred, conf, _, _ = knn_label_transfer(
                    query_latent[i_global:i_global+1],
                    ref_latent,
                    ref_clusters,
                    candidate_mask=candidate_mask,
                    k=k_nn,
                    distance_weighted=True
                )
                pred_cluster[i_global] = pred[0]
                conf_cluster[i_global] = conf[0]
            else:
                pred_cluster[i_global] = None
                conf_cluster[i_global] = 0.0

    print(f"  ✓ Completed cluster assignment for all {n_query} cells")
    
    # Step 5: Consistency check and confidence-based rejection
    print("\n[Stage 5] Consistency check and confidence scoring...")
    
    # Create DataFrame with the same index as query AnnData
    mapping_results = pd.DataFrame({
        'assigned_subclass': pred_subclass,
        'assigned_cluster': pred_cluster,
        'conf_subclass': conf_subclass,
        'conf_cluster': conf_cluster
    }, index=adata_query.obs.index)  # Use query AnnData index
    
    # Check consistency: cluster must belong to assigned subclass
    consistent = np.zeros(n_query, dtype=bool)
    for i in range(n_query):
        if pred_cluster[i] is not None:
            expected_subclass = cluster_to_subclass.get(pred_cluster[i])
            consistent[i] = (expected_subclass == pred_subclass[i])
        else:
            consistent[i] = False
    
    # Determine mapping status
    mapping_status = []
    for i in range(n_query):
        if conf_subclass[i] < conf_threshold_subclass:
            status = 'rejected'
        elif conf_cluster[i] < conf_threshold_cluster or not consistent[i]:
            status = 'ok_subclass_only'
        else:
            status = 'ok_cluster'
        mapping_status.append(status)
    
    mapping_results['consistent'] = consistent
    mapping_results['mapping_status'] = mapping_status
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("MAPPING RESULTS SUMMARY")
    print("=" * 60)
    status_counts = pd.Series(mapping_status).value_counts()
    for status, count in status_counts.items():
        pct = 100 * count / n_query
        print(f"  {status}: {count} cells ({pct:.1f}%)")
    print(f"\n  Mean subclass confidence: {conf_subclass.mean():.3f}")
    print(f"  Mean cluster confidence: {conf_cluster[conf_cluster > 0].mean():.3f}")
    print(f"  Consistency rate: {consistent.sum() / n_query:.1%}")
    
    return mapping_results
