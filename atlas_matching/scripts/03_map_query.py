#!/usr/bin/env python3
"""
Step 3: Map Patch-seq query to reference atlas

This script:
1. Loads trained scVI model from step 02
2. Maps Patch-seq query cells to reference using scArches surgery
3. Performs hierarchical cell type assignment (subclass → cluster)
4. Saves latent embeddings and mapping results

CRITICAL: Query receives its own batch category (e.g., "PatchSeq_Allen")
to ensure Patch-seq technical effects are modeled separately from reference.

Output: Mapped h5ad files, latent embeddings, cell type assignments
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

# Add module to path (03_map_query.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping import (
    map_query_scarches,
    hierarchical_assignment,
)


def main():
    parser = argparse.ArgumentParser(
        description='Step 3: Map Patch-seq query to reference atlas',
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
        help='Directory with preprocessed data (default: results/preprocessed)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: results/scvi_mapped/{model_name})'
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        default=None,
        help='Directory with trained models (default: results/models)'
    )
    parser.add_argument(
        '--dataset',
        nargs='+',
        required=True,
        help='Dataset(s) to process (V1 and/or M1). Can specify multiple: --dataset V1 M1 or --dataset M1,V1'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name to load (default: scvi_reference_latent50_hidden256). '
             'Results will be saved to results/scvi_mapped/{model_name}/'
    )
    parser.add_argument(
        '--no-model-subdir',
        action='store_true',
        help='Do not create model-specific subdirectory for outputs (legacy behavior)'
    )

    # scArches parameters
    parser.add_argument(
        '--max-epochs-surgery',
        type=int,
        default=200,
        help='Max epochs for scArches surgery (default: 200)'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU'
    )

    # Hierarchical assignment parameters
    parser.add_argument(
        '--k-candidates',
        type=int,
        default=15,
        help='Top-K candidate clusters (default: 15)'
    )
    parser.add_argument(
        '--k-nn',
        type=int,
        default=20,
        help='KNN neighbors for label transfer (default: 20)'
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

    args = parser.parse_args()

    # Setup paths
    project_root = args.project_root
    preprocessed_dir = args.preprocessed_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'preprocessed'
    )
    model_dir = args.model_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'models'
    )

    # Parse datasets (handle comma-separated or space-separated)
    datasets = []
    for ds in args.dataset:
        # Handle comma-separated values like "M1,V1"
        if ',' in ds:
            datasets.extend([d.strip() for d in ds.split(',')])
        else:
            datasets.append(ds.strip())
    
    # Remove duplicates while preserving order
    datasets = list(dict.fromkeys(datasets))
    
    # Validate datasets
    valid_datasets = {'V1', 'M1'}
    invalid_datasets = [d for d in datasets if d not in valid_datasets]
    if invalid_datasets:
        raise ValueError(f"Invalid dataset(s): {invalid_datasets}. Must be one of: {valid_datasets}")

    # Determine model path
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = "scvi_reference_latent50_hidden256"
    model_path = model_dir / model_name

    # Setup output directory (model-specific subdirectory by default)
    if args.output_dir:
        output_dir = args.output_dir
    elif args.no_model_subdir:
        # Legacy behavior: flat directory
        output_dir = ATLAS_MATCHING_ROOT / 'results' / 'scvi_mapped'
    else:
        # New behavior: model-specific subdirectory
        output_dir = ATLAS_MATCHING_ROOT / 'results' / 'scvi_mapped' / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 3: MAP PATCH-SEQ QUERY TO REFERENCE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Model: {model_name}")
    print(f"  scArches max epochs: {args.max_epochs_surgery}")
    print(f"  Use GPU: {not args.no_gpu}")
    print(f"\nHierarchical assignment:")
    print(f"  K candidates: {args.k_candidates}")
    print(f"  K NN: {args.k_nn}")
    print(f"  Subclass threshold: {args.conf_threshold_subclass}")
    print(f"  Cluster threshold: {args.conf_threshold_cluster}")
    print(f"\nPaths:")
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Model path: {model_path}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)

    # Check if model exists
    if not model_path.exists() or not (model_path / "model.pt").exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            f"Run 02_train_scvi.py first to train the model"
        )

    # Load reference once (shared across all datasets)
    print(f"\n[LOAD] Loading reference atlas...")
    ref_path = preprocessed_dir / 'reference_preprocessed.h5ad'
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference not found: {ref_path}\n"
            f"Run 01_preprocess_data.py first"
        )
    adata_ref = sc.read_h5ad(ref_path)
    print(f"  Reference: {adata_ref.shape}")

    # Check for counts layer in reference (scVI needs it)
    if "counts" not in adata_ref.layers:
        print("  Warning: No counts layer in reference, using .X as counts")
        adata_ref.layers["counts"] = adata_ref.X.copy()

    # Verify reference has the required columns for hierarchical assignment
    required_cols = ['subclass', 'cluster']
    missing_cols = [col for col in required_cols if col not in adata_ref.obs.columns]
    if missing_cols:
        print(f"  ERROR: Reference missing required columns: {missing_cols}")
        print(f"  Available columns: {list(adata_ref.obs.columns)}")
        raise ValueError(
            f"Reference missing required taxonomy columns: {missing_cols}\n"
            f"Rerun preprocessing (step 01) to preserve cell type annotations."
        )

    # Process each dataset
    processed_datasets = []
    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING DATASET: {dataset_name}")
        print(f"{'=' * 80}")

        # Load query data for this dataset
        print(f"\n[LOAD] Loading {dataset_name} preprocessed data...")
        query_path = preprocessed_dir / f'{dataset_name}_query_preprocessed.h5ad'
        if not query_path.exists():
            print(f"  ERROR: Query not found: {query_path}")
            print(f"  Skipping {dataset_name} - Run 01_preprocess_data.py first")
            continue

        adata_query = sc.read_h5ad(query_path)
        print(f"  Query: {adata_query.shape}")

        # VALIDATION: Verify query and reference have the same number of genes
        if adata_query.n_vars != adata_ref.n_vars:
            raise ValueError(
                f"\n{'=' * 80}\n"
                f"ERROR: Gene count mismatch between reference and {dataset_name}!\n"
                f"  Reference: {adata_ref.n_vars} genes\n"
                f"  {dataset_name} Query: {adata_query.n_vars} genes\n"
                f"\n"
                f"This happens when datasets were preprocessed separately or with different parameters.\n"
                f"SOLUTION: Rerun preprocessing with ALL datasets together:\n"
                f"  python atlas_matching/scripts/01_preprocess_data_patched.py --datasets V1 M1\n"
                f"{'=' * 80}\n"
            )

        # VALIDATION: Verify gene names match (not just counts)
        if not all(adata_query.var_names == adata_ref.var_names):
            raise ValueError(
                f"\n{'=' * 80}\n"
                f"ERROR: Gene names don't match between reference and {dataset_name}!\n"
                f"Even though counts match, gene names/order differ.\n"
                f"SOLUTION: Rerun preprocessing to ensure consistent gene sets.\n"
                f"{'=' * 80}\n"
            )

        print(f"  ✓ Verified: Query and Reference have matching {adata_query.n_vars} genes")

        # Check for counts layer in query (scVI needs it)
        if "counts" not in adata_query.layers:
            print("  Warning: No counts layer in query, using .X as counts")
            adata_query.layers["counts"] = adata_query.X.copy()

        # Load trained model and map query
        print(f"\n[MAP] Mapping {dataset_name} query to reference using scArches surgery...")
        print(f"  Loading model from: {model_path}")
        print(f"  Query will receive NEW batch category (e.g., 'PatchSeq_Allen')")
        print(f"  This ensures Patch-seq technical effects are modeled separately")

        # CRITICAL: scArches requires model path, not model object
        # Note: We need to reload reference for each dataset to avoid state issues
        # But we can reuse the same reference data
        adata_ref_mapped, adata_query_mapped = map_query_scarches(
            model_path,  # Pass path string
            adata_ref.copy(),  # Copy to avoid modifying original
            adata_query,
            use_gpu=not args.no_gpu,
            max_epochs=args.max_epochs_surgery
        )

        print(f"  ✓ Mapping complete")
        print(f"  Reference latent: {adata_ref_mapped.obsm['X_latent'].shape}")
        print(f"  Query latent: {adata_query_mapped.obsm['X_latent'].shape}")

        # Hierarchical assignment
        print(f"\n[ASSIGN] Hierarchical cell type assignment for {dataset_name}...")

        mapping_results = hierarchical_assignment(
            adata_ref_mapped,
            adata_query_mapped,
            latent_key='X_latent',
            subclass_key='subclass',
            cluster_key='cluster',
            k_candidates=args.k_candidates,
            k_nn=args.k_nn,
            conf_threshold_subclass=args.conf_threshold_subclass,
            conf_threshold_cluster=args.conf_threshold_cluster
        )

        # Add mapping results to query
        mapping_results.index = adata_query_mapped.obs.index
        adata_query_mapped.obs = pd.concat([adata_query_mapped.obs, mapping_results], axis=1)

        print(f"\nMapping summary for {dataset_name}:")
        print(f"  Total cells: {len(mapping_results)}")
        print(f"  Mapping status:")
        for status, count in mapping_results['mapping_status'].value_counts().items():
            pct = 100 * count / len(mapping_results)
            print(f"    {status}: {count} ({pct:.1f}%)")

        # Save results
        print(f"\n[SAVE] Saving results for {dataset_name}...")

        # Save mapped h5ad files
        # Only save reference on first dataset (it's the same for all)
        if dataset_idx == 0:
            ref_out_path = output_dir / 'reference_mapped.h5ad'
            adata_ref_mapped.write(ref_out_path)
            print(f"  Reference: {ref_out_path}")
        
        query_out_path = output_dir / f'{dataset_name}_query_mapped.h5ad'
        adata_query_mapped.write(query_out_path)
        print(f"  Query: {query_out_path}")

        # Save mapping results CSV
        mapping_out_path = output_dir / f'{dataset_name}_mapping_results.csv'
        mapping_results.to_csv(mapping_out_path)
        print(f"  Mapping results: {mapping_out_path}")

        # Save latent embeddings
        # Only save reference latent on first dataset
        if dataset_idx == 0:
            latent_ref_path = output_dir / 'reference_latent.npy'
            np.save(latent_ref_path, adata_ref_mapped.obsm['X_latent'])
            print(f"  Reference latent: {latent_ref_path}")
        
        latent_query_path = output_dir / f'{dataset_name}_query_latent.npy'
        np.save(latent_query_path, adata_query_mapped.obsm['X_latent'])
        print(f"  Query latent: {latent_query_path}")

        processed_datasets.append(dataset_name)

        # Clean up for next iteration
        del adata_query, adata_query_mapped, adata_ref_mapped, mapping_results

    print(f"\n{'=' * 80}")
    print("✅ QUERY MAPPING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nProcessed datasets: {', '.join(processed_datasets)}")
    if processed_datasets:
        print(f"\nNext step: Run 04_compute_canonical_expression.py --dataset {' '.join(processed_datasets)}")


if __name__ == '__main__':
    main()
