#!/usr/bin/env python3
"""
Step 4: Compute canonical expression profiles

This script:
1. Loads mapped data from step 3 (raw counts + latent embeddings)
2. Normalizes to log(CPM+1) for expression analysis
3. Computes canonical cluster/subclass profiles from reference
4. Computes effective expression for query cells using k-NN (k=100 recommended)
5. Optional: Restrict k-NN neighbors to predicted subclass (improves quality)

IMPORTANT: Uses k=100 neighbors (increased from 20) for more robust canonical
expression estimates. Can optionally restrict neighbors by subclass to prevent
cross-type contamination.

Output: Canonical expression profiles, effective expression
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

warnings.filterwarnings('ignore')

# Add module to path (04_compute_canonical_expression.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping import compute_canonical_pipeline


def main():
    parser = argparse.ArgumentParser(
        description='Step 4: Compute canonical expression profiles',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=PROJECT_ROOT,
        help='Project root directory'
    )
    parser.add_argument(
        '--preprocessed-dir',
        type=Path,
        default=None,
        help='Directory with preprocessed data'
    )
    parser.add_argument(
        '--mapped-dir',
        type=Path,
        default=None,
        help='Directory with mapping results from step 3'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--dataset',
        choices=['V1', 'M1'],
        required=True,
        help='Dataset to process'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (determines input/output subdirectory). '
             'If specified, reads from scvi_mapped/{model_name}/ and writes to canonical/{model_name}/'
    )
    parser.add_argument(
        '--no-model-subdir',
        action='store_true',
        help='Do not use model-specific subdirectories (legacy behavior)'
    )
    parser.add_argument(
        '--use-soft-assignment',
        action='store_true',
        default=True,
        help='Use soft assignment for effective expression'
    )
    parser.add_argument(
        '--k-nn-expression',
        type=int,
        default=100,
        help='Number of neighbors for soft assignment (default: 100 for robust estimates)'
    )
    parser.add_argument(
        '--restrict-by-subclass',
        action='store_true',
        default=True,
        help='Restrict k-NN neighbors to predicted subclass (recommended)'
    )

    args = parser.parse_args()

    # Setup paths
    project_root = args.project_root
    preprocessed_dir = args.preprocessed_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'preprocessed'
    )

    # Determine model-specific paths
    model_name = args.model_name
    use_subdir = model_name and not args.no_model_subdir

    if args.mapped_dir:
        mapped_dir = args.mapped_dir
    elif use_subdir:
        mapped_dir = ATLAS_MATCHING_ROOT / 'results' / 'scvi_mapped' / model_name
    else:
        mapped_dir = ATLAS_MATCHING_ROOT / 'results' / 'scvi_mapped'

    if args.output_dir:
        output_dir = args.output_dir
    elif use_subdir:
        output_dir = ATLAS_MATCHING_ROOT / 'results' / 'canonical' / model_name
    else:
        output_dir = ATLAS_MATCHING_ROOT / 'results' / 'canonical'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 4: COMPUTE CANONICAL EXPRESSION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    if model_name:
        print(f"  Model: {model_name}")
    print(f"  Use soft assignment: {args.use_soft_assignment}")
    print(f"  K NN: {args.k_nn_expression}")
    print(f"  Restrict by subclass: {args.restrict_by_subclass}")
    print(f"\nPaths:")
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Mapped dir: {mapped_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)

    # Load mapped data (has raw counts + latent embeddings)
    print(f"\n[LOAD] Loading mapped data...")
    ref_path = mapped_dir / 'reference_mapped.h5ad'
    query_path = mapped_dir / f'{args.dataset}_query_mapped.h5ad'

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference mapped data not found: {ref_path}\n"
            f"Run 03_map_query.py first"
        )
    if not query_path.exists():
        raise FileNotFoundError(
            f"Query mapped data not found: {query_path}\n"
            f"Run 03_map_query.py first"
        )

    adata_ref = sc.read_h5ad(ref_path)
    adata_query = sc.read_h5ad(query_path)
    print(f"  Reference: {adata_ref.shape}")
    print(f"  Query: {adata_query.shape}")
    print(f"  Reference has X_latent: {'X_latent' in adata_ref.obsm}")
    print(f"  Query has X_latent: {'X_latent' in adata_query.obsm}")

    # Load mapping results
    print(f"\n[LOAD] Loading mapping results...")
    mapping_path = mapped_dir / f'{args.dataset}_mapping_results.csv'
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Mapping results not found: {mapping_path}\n"
            f"Run 03_map_query.py first"
        )

    mapping_results = pd.read_csv(mapping_path, index_col=0)
    print(f"  Mapping results: {len(mapping_results)} cells")

    # Normalize to log(CPM+1)
    print(f"\n[NORMALIZE] Normalizing to log(CPM+1)...")
    adata_ref_norm = adata_ref.copy()
    adata_query_norm = adata_query.copy()

    # Use raw counts from .X (already moved there in step 1)
    sc.pp.normalize_total(adata_ref_norm, target_sum=1e6)
    sc.pp.log1p(adata_ref_norm)

    sc.pp.normalize_total(adata_query_norm, target_sum=1e6)
    sc.pp.log1p(adata_query_norm)

    print(f"  Reference normalized: {adata_ref_norm.shape}")
    print(f"  Query normalized: {adata_query_norm.shape}")

    # Compute canonical expression
    print(f"\n[COMPUTE] Computing canonical expression profiles...")
    canonical_cluster, canonical_subclass, effective_expr = compute_canonical_pipeline(
        adata_ref_norm,
        adata_query_norm,
        mapping_results,
        use_soft_assignment=args.use_soft_assignment,
        k_nn=args.k_nn_expression
    )

    print(f"\nCanonical expression computed:")
    print(f"  Cluster profiles: {canonical_cluster.shape}")
    print(f"  Subclass profiles: {canonical_subclass.shape}")
    print(f"  Effective expression: {effective_expr.shape}")

    # Save results
    print(f"\n[SAVE] Saving results...")
    prefix = f"{args.dataset}_"

    cluster_path = output_dir / f'{prefix}canonical_cluster.csv'
    subclass_path = output_dir / f'{prefix}canonical_subclass.csv'
    effective_path = output_dir / f'{prefix}effective_expression.csv'

    canonical_cluster.to_csv(cluster_path)
    canonical_subclass.to_csv(subclass_path)
    effective_expr.to_csv(effective_path)

    print(f"  Canonical cluster: {cluster_path}")
    print(f"  Canonical subclass: {subclass_path}")
    print(f"  Effective expression: {effective_path}")

    print(f"\n{'=' * 80}")
    print("CANONICAL EXPRESSION COMPUTATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nAll pipeline steps complete!")
    print(f"\nOutput files:")
    print(f"  1. Mapping results: {mapped_dir}/{prefix}mapping_results.csv")
    print(f"  2. Canonical profiles: {output_dir}/{prefix}canonical_*.csv")
    print(f"  3. Effective expression: {output_dir}/{prefix}effective_expression.csv")


if __name__ == '__main__':
    main()
