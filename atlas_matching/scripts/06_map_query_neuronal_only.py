#!/usr/bin/env python
"""
Map query Patch-seq cells to reference atlas - NEURONAL CELLS ONLY.

This script implements the critical fix for the glia assignment problem:
- Pre-filters reference to neuronal clusters only (Glut + GABA)
- Prevents mapping of Patch-seq neurons to non-neuronal clusters
- Uses electrophysiology-based class priors (optional)

Usage:
    python atlas_matching/scripts/06_map_query_neuronal_only.py --dataset M1 V1
"""

import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths (06_map_query_neuronal_only.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))
DATA_DIR = ATLAS_MATCHING_ROOT / "results" / "preprocessed"
RESULTS_DIR = ATLAS_MATCHING_ROOT / "results"

from patch_seq_transcriptome_mapping.mapping.hierarchical import hierarchical_assignment
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global model name (set by main)
CURRENT_MODEL_NAME = None


def identify_neuronal_clusters(adata_ref, subclass_key='subclass'):
    """
    Identify neuronal (glutamatergic + GABAergic) clusters.

    Returns masks for glutamatergic and GABAergic cells.
    """
    print("\n" + "=" * 70)
    print("IDENTIFYING NEURONAL CLUSTERS")
    print("=" * 70)

    subclasses = adata_ref.obs[subclass_key].astype(str)

    # Keywords for cell types
    glut_keywords = ['glut', 'l2/3 it', 'l4', 'l5 it', 'l5 pt', 'l6 it', 'l6 ct', 'l6b', 'et']
    gaba_keywords = ['gaba', 'vip', 'pvalb', 'sst', 'lamp5', 'sncg']

    glut_mask = np.zeros(len(subclasses), dtype=bool)
    gaba_mask = np.zeros(len(subclasses), dtype=bool)

    for i, sc_label in enumerate(subclasses):
        sc_lower = sc_label.lower()

        if any(kw in sc_lower for kw in glut_keywords):
            glut_mask[i] = True
        elif any(kw in sc_lower for kw in gaba_keywords):
            gaba_mask[i] = True

    neuronal_mask = glut_mask | gaba_mask

    print(f"\nNeuronal cells identified:")
    print(f"  Glutamatergic: {glut_mask.sum():,} cells ({100*glut_mask.sum()/len(adata_ref):.1f}%)")
    print(f"  GABAergic:     {gaba_mask.sum():,} cells ({100*gaba_mask.sum()/len(adata_ref):.1f}%)")
    print(f"  Total neuronal: {neuronal_mask.sum():,} cells ({100*neuronal_mask.sum()/len(adata_ref):.1f}%)")
    print(f"  Non-neuronal:  {(~neuronal_mask).sum():,} cells ({100*(~neuronal_mask).sum()/len(adata_ref):.1f}%)")

    # Show which subclasses are excluded
    excluded_subclasses = adata_ref.obs.loc[~neuronal_mask, subclass_key].unique()
    if len(excluded_subclasses) > 0:
        print(f"\n  Excluded non-neuronal subclasses ({len(excluded_subclasses)}):")
        for sc in sorted(excluded_subclasses)[:20]:  # Show first 20
            count = (subclasses == sc).sum()
            print(f"    - {sc}: {count} cells")
        if len(excluded_subclasses) > 20:
            print(f"    ... and {len(excluded_subclasses) - 20} more")

    return neuronal_mask, glut_mask, gaba_mask


def predict_ephys_class(adata_query):
    """
    Predict major cell class (Excitatory vs Inhibitory) from electrophysiology.

    This is a placeholder - implement with actual ephys features if available.
    Returns None if ephys data not available.
    """
    # Check if ephys features are available in obs
    ephys_cols = [col for col in adata_query.obs.columns
                  if any(kw in col.lower() for kw in ['spike', 'width', 'ap', 'firing', 'tau'])]

    if len(ephys_cols) == 0:
        print("\n  ⚠️  No electrophysiology features found in query data")
        print("      Cannot apply ephys-based priors")
        return None

    print(f"\n  Found {len(ephys_cols)} ephys features: {ephys_cols[:5]}...")
    print("  TODO: Implement ephys-based classification")
    print("        (e.g., spike width, adaptation index for Exc vs Inh)")

    return None


def map_dataset(dataset_name, use_ephys_prior=False,
                k_candidates=10, k_nn=30,
                conf_threshold_subclass=0.7, conf_threshold_cluster=0.7,
                model_name=None):
    """
    Map a single dataset to neuronal-only reference.
    """
    print("\n" + "=" * 70)
    print(f"MAPPING {dataset_name} DATASET (NEURONAL ONLY)")
    if model_name:
        print(f"Using model: {model_name}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    ref_file = DATA_DIR / "reference_preprocessed.h5ad"
    query_file = DATA_DIR / f"{dataset_name}_query_preprocessed.h5ad"

    adata_ref = sc.read_h5ad(ref_file)
    adata_query = sc.read_h5ad(query_file)

    print(f"  Reference: {adata_ref.shape[0]:,} cells × {adata_ref.shape[1]:,} genes")
    print(f"  Query ({dataset_name}): {adata_query.shape[0]:,} cells × {adata_query.shape[1]:,} genes")

    # Identify neuronal clusters
    neuronal_mask, glut_mask, gaba_mask = identify_neuronal_clusters(adata_ref)

    # Filter reference to neuronal cells only
    print("\n" + "=" * 70)
    print("FILTERING REFERENCE TO NEURONAL CELLS ONLY")
    print("=" * 70)
    adata_ref_neuronal = adata_ref[neuronal_mask].copy()
    print(f"\nFiltered reference: {adata_ref_neuronal.shape[0]:,} cells")

    # Fix batch categories - ensure all are strings (no int/str mixing)
    if 'tech_batch' in adata_ref_neuronal.obs.columns:
        adata_ref_neuronal.obs['tech_batch'] = adata_ref_neuronal.obs['tech_batch'].astype(str).astype('category')
        print(f"  ✓ Fixed batch categories to strings")

    # Optional: Predict ephys-based class
    ephys_predictions = None
    if use_ephys_prior:
        ephys_predictions = predict_ephys_class(adata_query)

    # Load pre-computed latent embeddings (from scArches mapping)
    print("\n" + "=" * 70)
    print("LOADING PRE-COMPUTED LATENT EMBEDDINGS")
    print("=" * 70)

    # Determine mapped directory (model-specific if model_name provided)
    if model_name:
        mapped_dir = RESULTS_DIR / "scvi_mapped" / model_name
    else:
        mapped_dir = RESULTS_DIR / "scvi_mapped"

    # Load reference with latent embeddings
    ref_mapped_file = mapped_dir / "reference_mapped.h5ad"
    if not ref_mapped_file.exists():
        raise FileNotFoundError(
            f"Mapped reference not found at {ref_mapped_file}\n"
            "Please run 03_map_query.py first to generate latent embeddings"
        )

    # Load query with latent embeddings
    query_mapped_file = mapped_dir / f"{dataset_name}_query_mapped.h5ad"
    if not query_mapped_file.exists():
        raise FileNotFoundError(
            f"Mapped query not found at {query_mapped_file}\n"
            "Please run 03_map_query.py first to generate latent embeddings"
        )

    print(f"  Loading reference latent embeddings...")
    adata_ref_full_latent = sc.read_h5ad(ref_mapped_file)

    print(f"  Loading query latent embeddings...")
    adata_query_latent = sc.read_h5ad(query_mapped_file)

    # Extract latent embeddings for neuronal cells only
    print(f"\n  Extracting latent embeddings for neuronal cells...")
    # Match indices between full reference and neuronal subset
    neuronal_indices = adata_ref.obs.index[neuronal_mask]
    adata_ref_neuronal.obsm['X_latent'] = adata_ref_full_latent[neuronal_indices].obsm['X_latent']
    adata_query.obsm['X_latent'] = adata_query_latent.obsm['X_latent']

    print(f"  ✓ Latent embeddings loaded")
    print(f"    Reference neuronal: {adata_ref_neuronal.obsm['X_latent'].shape}")
    print(f"    Query: {adata_query.obsm['X_latent'].shape}")

    # Perform hierarchical mapping
    print("\n" + "=" * 70)
    print(f"PERFORMING HIERARCHICAL MAPPING")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  k_candidates: {k_candidates}")
    print(f"  k_nn: {k_nn}")
    print(f"  conf_threshold_subclass: {conf_threshold_subclass}")
    print(f"  conf_threshold_cluster: {conf_threshold_cluster}")

    mapping_results = hierarchical_assignment(
        adata_ref_neuronal,
        adata_query,
        latent_key='X_latent',
        subclass_key='subclass',
        cluster_key='cluster',
        k_candidates=k_candidates,
        k_nn=k_nn,
        conf_threshold_subclass=conf_threshold_subclass,
        conf_threshold_cluster=conf_threshold_cluster
    )

    # Save results (to model-specific directory if model_name provided)
    if model_name:
        output_dir = RESULTS_DIR / "neuronal_only" / model_name
    else:
        output_dir = RESULTS_DIR / "neuronal_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{dataset_name}_mapping_neuronal_only.csv"
    mapping_results.to_csv(output_file)
    print(f"\n  ✓ Results saved to: {output_file}")

    # Verify no non-neuronal assignments
    print("\n" + "=" * 70)
    print("VERIFICATION: CHECK FOR NON-NEURONAL ASSIGNMENTS")
    print("=" * 70)

    assigned_subclasses = mapping_results['assigned_subclass'].values
    non_neuronal_keywords = ['astro', 'oligo', 'endo', 'peri', 'smc', 'vlmc',
                            'micro', 'immune', 'glia', 'opc', 'macrophage']

    non_neuronal_count = 0
    for subclass in assigned_subclasses:
        if pd.notna(subclass):
            subclass_lower = str(subclass).lower()
            if any(kw in subclass_lower for kw in non_neuronal_keywords):
                non_neuronal_count += 1

    if non_neuronal_count > 0:
        print(f"  ⚠️  WARNING: {non_neuronal_count} cells still assigned to non-neuronal!")
        print("      This should not happen - check filtering logic")
    else:
        print(f"  ✓ SUCCESS: 0 cells assigned to non-neuronal clusters")
        print("      All assignments are to neuronal clusters")

    return mapping_results


def compare_with_original(dataset_name):
    """Compare neuronal-only mapping with original mapping."""
    print("\n" + "=" * 70)
    print(f"COMPARING WITH ORIGINAL MAPPING ({dataset_name})")
    print("=" * 70)

    original_file = RESULTS_DIR / "annotated_metadata" / f"{dataset_name}_annotated_metadata.csv"
    new_file = RESULTS_DIR / f"{dataset_name}_mapping_neuronal_only.csv"

    if not original_file.exists():
        print("  Original mapping file not found, skipping comparison")
        return

    df_orig = pd.read_csv(original_file, index_col=0)
    df_new = pd.read_csv(new_file, index_col=0)

    # Count cells by status
    print("\nMapping status comparison:")
    print(f"  {'Status':<20s} {'Original':>10s} {'Neuronal-only':>15s} {'Difference':>12s}")
    print("  " + "-" * 60)

    for status in ['ok_cluster', 'ok_subclass_only', 'rejected']:
        count_orig = (df_orig['mapping_status'] == status).sum()
        count_new = (df_new['mapping_status'] == status).sum()
        diff = count_new - count_orig
        print(f"  {status:<20s} {count_orig:>10d} {count_new:>15d} {diff:>+12d}")

    # Check glia assignments
    print("\nNon-neuronal assignments:")

    def count_non_neuronal(df):
        non_neuronal_kw = ['astro', 'oligo', 'endo', 'peri', 'smc', 'vlmc',
                          'micro', 'immune', 'glia', 'opc', 'macrophage']
        count = 0
        for subclass in df['assigned_subclass'].values:
            if pd.notna(subclass):
                if any(kw in str(subclass).lower() for kw in non_neuronal_kw):
                    count += 1
        return count

    glia_orig = count_non_neuronal(df_orig)
    glia_new = count_non_neuronal(df_new)

    print(f"  Original:       {glia_orig:>5d} cells")
    print(f"  Neuronal-only:  {glia_new:>5d} cells")
    print(f"  Reduction:      {glia_orig - glia_new:>5d} cells ({100*(glia_orig-glia_new)/glia_orig:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Map Patch-seq cells to neuronal-only reference atlas"
    )
    parser.add_argument(
        '--dataset',
        nargs='+',
        required=True,
        choices=['M1', 'V1'],
        help='Dataset(s) to map (space-separated if multiple)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (determines input/output subdirectory). '
             'If specified, reads from scvi_mapped/{model_name}/ and writes to neuronal_only/{model_name}/'
    )
    parser.add_argument(
        '--use-ephys-prior',
        action='store_true',
        help='Use electrophysiology features to constrain mapping'
    )
    parser.add_argument(
        '--k-candidates',
        type=int,
        default=10,
        help='Number of candidate clusters in Stage 1 (default: 10, reduced from 15)'
    )
    parser.add_argument(
        '--k-nn',
        type=int,
        default=30,
        help='Number of nearest neighbors in Stage 2 (default: 30, increased from 20)'
    )
    parser.add_argument(
        '--conf-threshold-subclass',
        type=float,
        default=0.7,
        help='Confidence threshold for subclass (default: 0.7)'
    )
    parser.add_argument(
        '--conf-threshold-cluster',
        type=float,
        default=0.7,
        help='Confidence threshold for cluster (default: 0.7)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with original mapping results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NEURONAL-ONLY ATLAS MAPPING")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Datasets: {', '.join(args.dataset)}")
    if args.model_name:
        print(f"  Model: {args.model_name}")
    print(f"  Use ephys prior: {args.use_ephys_prior}")
    print(f"  k_candidates: {args.k_candidates}")
    print(f"  k_nn: {args.k_nn}")
    print(f"  Confidence thresholds: {args.conf_threshold_subclass} (subclass), {args.conf_threshold_cluster} (cluster)")

    # Map each dataset
    for dataset in args.dataset:
        mapping_results = map_dataset(
            dataset,
            use_ephys_prior=args.use_ephys_prior,
            k_candidates=args.k_candidates,
            k_nn=args.k_nn,
            conf_threshold_subclass=args.conf_threshold_subclass,
            conf_threshold_cluster=args.conf_threshold_cluster,
            model_name=args.model_name
        )

        # Compare with original if requested
        if args.compare:
            compare_with_original(dataset)

    print("\n" + "=" * 70)
    print("MAPPING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
