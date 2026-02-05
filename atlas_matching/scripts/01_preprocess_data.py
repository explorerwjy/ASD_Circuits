#!/usr/bin/env python3
"""
Step 1: Preprocessing - Load and prepare data for scVI training

This script:
1. Loads reference atlas and Patch-seq datasets
2. Finds gene intersection
3. Selects HVGs from the reference only (counts-based, subsampled)
4. Saves preprocessed data with RAW COUNTS ONLY (no full normalization)

Output: Preprocessed h5ad files ready for scVI training
"""

import sys
import argparse
import warnings
import gc
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

warnings.filterwarnings('ignore')

# Add module to path
# __file__ = .../atlas_matching/scripts/01_preprocess_data.py
# .parent = .../atlas_matching/scripts
# .parent.parent = .../atlas_matching
# .parent.parent.parent = project root (ASD_Circuits)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping import (
    load_reference_atlas,
    load_patchseq_v1,
    load_patchseq_m1,
    load_ion_channel_genes,
)


def preprocess_for_scvi(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    ion_channel_genes: list,
    n_hvgs_ref: int = 3000,
    hvg_flavor: str = 'seurat_v3',
    hvg_ncells: int = 200_000,
    random_seed: int = 0,
    extra_force_genes: list | None = None,
) -> tuple[ad.AnnData, ad.AnnData, list]:
    """
    Preprocess reference atlas + Patch-seq query for scVI/scArches.

    Design choices (important):
    - HVGs are selected from the REFERENCE ONLY (Patch-seq HVGs are excluded).
    - HVGs are computed on a SUBSAMPLE of reference cells to control memory.
    - Uses RAW COUNTS ONLY. No global normalization/log1p on the full atlas.
    - Forces inclusion of ion-channel + AIS/modulator genes (and optional extras).

    Memory notes:
    - Avoids copying the full reference matrix for normalization.
    - Subsampling is the main memory lever; tune hvg_ncells (50k–300k typically).
    """
    print("=" * 80)
    print("PREPROCESSING FOR SCVI (REFERENCE-HVG ONLY; RAW COUNTS)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 0: Ensure counts layer exists; drop .X if it's huge and redundant
    # ------------------------------------------------------------------
    if "counts" not in adata_ref.layers:
        print("  [WARN] Reference has no layers['counts']; using .X as counts.")
        adata_ref.layers["counts"] = adata_ref.X
    if "counts" not in adata_query.layers:
        print("  [WARN] Query has no layers['counts']; using .X as counts.")
        adata_query.layers["counts"] = adata_query.X

    # Optional: if .X exists and is not needed, drop it to reduce peak memory.
    # (We will reconstruct final adata with X=counts anyway.)
    try:
        if adata_ref.X is not None and adata_ref.X is not adata_ref.layers["counts"]:
            adata_ref.X = None
        if adata_query.X is not None and adata_query.X is not adata_query.layers["counts"]:
            adata_query.X = None
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 1: Gene intersection (skip if already matched)
    # ------------------------------------------------------------------
    print("[1/5] Checking gene intersection...")
    if set(adata_ref.var_names) == set(adata_query.var_names):
        print(f"  Genes already intersected: {len(adata_ref.var_names)} common genes")
    else:
        print("  Performing gene intersection...")
        from patch_seq_transcriptome_mapping import intersect_genes
        adata_ref, adata_query = intersect_genes(adata_ref, adata_query)
        gc.collect()

    # Refresh counts references after potential intersection
    ref_counts = adata_ref.layers["counts"]
    qry_counts = adata_query.layers["counts"]

    # ------------------------------------------------------------------
    # Step 2: HVG selection on REFERENCE ONLY (subsample; counts-based)
    # ------------------------------------------------------------------
    print("[2/5] Selecting HVGs from reference only (subsample; counts-based)...")
    n_ref = adata_ref.n_obs
    n_sub = int(min(hvg_ncells, n_ref))
    print(f"  Reference cells: {n_ref:,}; HVG subsample: {n_sub:,}; n_hvgs_ref={n_hvgs_ref}")

    rng = np.random.default_rng(random_seed)
    if n_sub < n_ref:
        sub_idx = rng.choice(n_ref, size=n_sub, replace=False)
        # Copy only the subsample AnnData (much smaller) to allow scanpy to write .var fields.
        ref_hvg = adata_ref[sub_idx, :].copy()
    else:
        # Still copy to avoid mutating adata_ref.var during HVG selection
        ref_hvg = adata_ref.copy()

    # Ensure HVG computation uses counts
    ref_hvg.X = ref_hvg.layers["counts"]

    # seurat_v3 expects counts; no normalize/log1p required here
    sc.pp.highly_variable_genes(
        ref_hvg,
        n_top_genes=n_hvgs_ref,
        flavor=hvg_flavor,
        subset=False
    )

    hvg_ref = ref_hvg.var_names[ref_hvg.var["highly_variable"]].tolist()
    del ref_hvg
    gc.collect()
    print(f"  Selected {len(hvg_ref)} HVGs from reference")

    # ------------------------------------------------------------------
    # Step 3: Force include genes (ion channels + AIS/modulators + extras)
    # ------------------------------------------------------------------
    print("[3/5] Building final gene set (HVG_ref + forced genes)...")

    # Default forced set beyond ion channels (AIS anchoring + Nav modulators)
    default_forced = [
        # AIS anchoring / clustering
        "Ank3", "Sptan1", "Sptbn4", "Nfasc", "Cntnap2",
        # Nav modulators
        "Fgf14", "Fgf13",
        # Helpful sodium channel family (in case not in ion_channel_genes)
        "Scn1a", "Scn2a", "Scn8a", "Scn1b", "Scn2b", "Scn3b", "Scn4b"
    ]

    forced = set(ion_channel_genes) | set(default_forced)
    if extra_force_genes is not None:
        forced |= set(extra_force_genes)

    # Keep only genes present after intersection
    forced_present = [g for g in forced if g in adata_ref.var_names]
    final_genes = sorted(set(hvg_ref) | set(forced_present))

    print(f"  Forced genes requested: {len(forced)}; present after intersection: {len(forced_present)}")
    print(f"  Final gene set size: {len(final_genes)}")

    # ------------------------------------------------------------------
    # Step 4: Subset counts matrices (vectorized; avoid get_loc loops)
    # ------------------------------------------------------------------
    print("[4/5] Subsetting to final genes (vectorized masks)...")
    ref_mask = adata_ref.var_names.isin(final_genes)
    qry_mask = adata_query.var_names.isin(final_genes)

    # Subset from counts layer directly
    ref_counts_subset = ref_counts[:, ref_mask].copy()
    qry_counts_subset = qry_counts[:, qry_mask].copy()

    ref_var = adata_ref.var.loc[ref_mask].copy()
    qry_var = adata_query.var.loc[qry_mask].copy()

    # ------------------------------------------------------------------
    # Step 5: Create final AnnData objects (counts in X + layers['counts'])
    # ------------------------------------------------------------------
    print("[5/5] Creating final AnnData objects (raw counts only)...")

    adata_ref_final = ad.AnnData(
        X=ref_counts_subset,
        obs=adata_ref.obs.copy(),
        var=ref_var
    )
    adata_ref_final.layers["counts"] = adata_ref_final.X  # no extra copy
    adata_ref_final.uns["preprocess_notes"] = {
        "hvg_source": "reference_only",
        "hvg_flavor": hvg_flavor,
        "n_hvgs_ref": int(n_hvgs_ref),
        "hvg_ncells": int(n_sub),
        "forced_genes_present": forced_present,
        "final_gene_set_size": int(len(final_genes)),
        "random_seed": int(random_seed),
    }

    adata_query_final = ad.AnnData(
        X=qry_counts_subset,
        obs=adata_query.obs.copy(),
        var=qry_var
    )
    adata_query_final.layers["counts"] = adata_query_final.X  # no extra copy
    adata_query_final.uns["preprocess_notes"] = {
        "hvg_source": "reference_only",
        "hvg_flavor": hvg_flavor,
        "n_hvgs_ref": int(n_hvgs_ref),
        "hvg_ncells": int(n_sub),
        "forced_genes_present": forced_present,
        "final_gene_set_size": int(len(final_genes)),
        "random_seed": int(random_seed),
    }

    print("✅ Preprocessing complete.")
    print(f"  Reference final shape: {adata_ref_final.shape}")
    print(f"  Query final shape:     {adata_query_final.shape}")

    return adata_ref_final, adata_query_final, final_genes

def main():
    parser = argparse.ArgumentParser(
        description='Step 1: Preprocess data for scVI training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=PROJECT_ROOT,
        help='Project root directory'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Data directory (default: project_root/dat/expression)'
    )
    parser.add_argument(
        '--atlas-dir',
        type=Path,
        default=None,
        help='Atlas directory (default: data_dir/reference_atlas)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: project_root/atlas_matching/results/preprocessed)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['V1'],
        help='Dataset(s) to process (V1 and/or M1). Can specify multiple: --datasets V1 M1 or --datasets V1,M1'
    )
    parser.add_argument(
        '--n-hvgs-ref',
        type=int,
        default=3000,
        help='Number of HVGs from reference'
    )

    parser.add_argument(
        '--hvg-ncells',
        type=int,
        default=200000,
        help='Number of reference cells to subsample for HVG selection (memory control)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='Random seed for HVG subsampling'
    )

    parser.add_argument(
        '--hvg-flavor',
        choices=['seurat_v3', 'seurat', 'cell_ranger'],
        default='seurat_v3',
        help='HVG selection method'
    )
    parser.add_argument(
        '--subsample-atlas',
        type=int,
        default=None,
        help='Subsample atlas to N cells (for testing)'
    )

    args = parser.parse_args()

    # Parse datasets (handle comma-separated or space-separated)
    datasets = []
    for ds in args.datasets:
        # Handle comma-separated values like "V1,M1"
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

    # Setup paths
    project_root = args.project_root
    data_dir = args.data_dir or (project_root / 'dat' / 'expression')
    atlas_dir = args.atlas_dir or (data_dir / 'reference_atlas')
    output_dir = args.output_dir or (project_root / 'atlas_matching' / 'results' / 'preprocessed')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: PREPROCESSING DATA FOR SCVI")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  HVGs (ref): {args.n_hvgs_ref}")
    print(f"  HVG method: {args.hvg_flavor}")
    print(f"  Subsample atlas: {args.subsample_atlas}")
    if args.subsample_atlas is None:
        print(f"\n  NOTE: If you encounter OOM errors, use --subsample-atlas N")
        print(f"        to test with fewer cells (e.g., --subsample-atlas 50000)")
    print(f"\nPaths:")
    print(f"  Data dir: {data_dir}")
    print(f"  Atlas dir: {atlas_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 80)

    # Load reference atlas
    print("\n[LOAD] Loading reference atlas...")
    adata_ref = load_reference_atlas(
        atlas_dir,
        use_log2=False,  # Reference is log2(CPM+1), we'll use counts layer
        subsample=args.subsample_atlas
    )
    print(f"  Reference: {adata_ref.shape}")
    print(f"  Obs columns: {list(adata_ref.obs.columns)}")

    # Ensure reference has counts layer
    if "counts" not in adata_ref.layers:
        raise ValueError("Reference atlas must have layers['counts'] with raw counts")

    # MEMORY OPTIMIZATION: Strip unnecessary metadata from reference
    print("\n[MEMORY] Cleaning reference atlas...")
    # Keep only essential obs columns (INCLUDING cell type taxonomy for downstream mapping)
    essential_obs_cols = [
        'tech_batch',           # Batch info
        'class',                # Cell type taxonomy (coarse)
        'subclass',             # Cell type taxonomy (intermediate)
        'cluster',              # Cell type taxonomy (fine)
        'cluster_alias',        # Cluster names
        'neurotransmitter',     # Functional annotation
    ]
    obs_to_keep = [col for col in essential_obs_cols if col in adata_ref.obs.columns]
    if len(obs_to_keep) < len(adata_ref.obs.columns):
        # Create new DataFrame with only essential columns (more memory efficient)
        adata_ref.obs = adata_ref.obs[obs_to_keep].copy()
        print(f"  Kept essential columns: {obs_to_keep}")

    # Remove any other layers besides counts
    layers_to_remove = [k for k in adata_ref.layers.keys() if k != 'counts']
    for layer in layers_to_remove:
        del adata_ref.layers[layer]

    # Clear unnecessary attributes (these are usually small, but good practice)
    adata_ref.uns.clear()
    adata_ref.obsm.clear()
    adata_ref.varm.clear()
    adata_ref.obsp.clear()
    adata_ref.varp.clear()
    print(f"  Cleared unnecessary attributes")
    gc.collect()

    # Load ion channel genes
    print("\n[LOAD] Loading ion channel genes...")
    ion_channel_genes_dict = load_ion_channel_genes(data_dir, source='both')
    ion_channel_genes = ion_channel_genes_dict['all']
    print(f"  Ion channel genes: {len(ion_channel_genes)}")

    # CRITICAL FIX: Find common genes across ALL datasets FIRST
    # This ensures all datasets (and reference) share the SAME gene set
    print("\n[GENE INTERSECTION] Finding common genes across ALL datasets...")
    print("=" * 80)

    ref_genes = set(adata_ref.var_names)
    print(f"  Reference genes: {len(ref_genes)}")

    # Load all query datasets to find gene intersection
    all_query_genes = []
    for dataset_name in datasets:
        print(f"  Loading {dataset_name} to check genes...")
        if dataset_name == 'V1':
            adata_query_temp, _, _ = load_patchseq_v1(data_dir, load_ephys=False)
        elif dataset_name == 'M1':
            adata_query_temp, _, _ = load_patchseq_m1(data_dir, load_ephys=False)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        query_genes = set(adata_query_temp.var_names)
        all_query_genes.append(query_genes)
        print(f"    {dataset_name} genes: {len(query_genes)}")
        del adata_query_temp
        gc.collect()

    # Find intersection: genes present in reference AND all queries
    common_genes_all = ref_genes
    for query_genes in all_query_genes:
        common_genes_all = common_genes_all & query_genes

    common_genes_all = sorted(list(common_genes_all))
    print(f"\n  ✓ Common genes across reference + {len(datasets)} datasets: {len(common_genes_all)}")
    print("=" * 80)

    if len(common_genes_all) == 0:
        raise ValueError("No common genes found across all datasets!")

    # Filter reference to common genes ONCE (will be used for all datasets)
    print(f"\n[FILTER] Filtering reference to common genes...")
    gene_indices_ref = [adata_ref.var_names.get_loc(g) for g in common_genes_all]
    ref_counts = adata_ref.layers.get("counts", adata_ref.X)
    if hasattr(ref_counts, 'toarray'):
        ref_counts_filtered = ref_counts[:, gene_indices_ref].copy()
    else:
        ref_counts_filtered = ref_counts[:, gene_indices_ref].copy()

    adata_ref_common = ad.AnnData(
        X=ref_counts_filtered,
        obs=adata_ref.obs.copy(),
        var=adata_ref.var.loc[common_genes_all].copy()
    )
    adata_ref_common.layers["counts"] = ref_counts_filtered
    print(f"  Reference filtered to: {adata_ref_common.shape}")

    # Clean up original reference to free memory
    del adata_ref, ref_counts, ref_counts_filtered, gene_indices_ref
    gc.collect()

    # Process each dataset (now all using the same common_genes_all)
    adata_ref_proc = None

    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"\n{'=' * 80}")
        print(f"PROCESSING {dataset_name} DATASET")
        print(f"{'=' * 80}")

        # Load query data
        print(f"\n[LOAD] Loading {dataset_name} Patch-seq data...")
        if dataset_name == 'V1':
            adata_query, metadata, ephys = load_patchseq_v1(
                data_dir,
                load_ephys=False
            )
        elif dataset_name == 'M1':
            adata_query, metadata, ephys = load_patchseq_m1(
                data_dir,
                load_ephys=False
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"  Query: {adata_query.shape}")

        # Ensure query has counts layer
        if "counts" not in adata_query.layers:
            raise ValueError(f"{dataset_name} query must have layers['counts'] with raw counts")

        # Filter query to the pre-computed common genes
        # (Reference was already filtered above to common_genes_all)
        print(f"\n[FILTER] Filtering {dataset_name} query to common genes...")
        print(f"  Query genes before: {adata_query.shape[1]}")
        gene_indices_query = [adata_query.var_names.get_loc(g) for g in common_genes_all]
        
        query_counts = adata_query.layers.get("counts", adata_query.X)
        if hasattr(query_counts, 'toarray'):
            query_counts_filtered = query_counts[:, gene_indices_query].copy()
        else:
            query_counts_filtered = query_counts[:, gene_indices_query].copy()
        
        adata_query_filtered = ad.AnnData(
            X=query_counts_filtered,
            obs=adata_query.obs.copy(),
            var=adata_query.var.loc[common_genes_all].copy()
        )
        adata_query_filtered.layers["counts"] = query_counts_filtered

        print(f"  Query filtered to: {adata_query_filtered.shape}")
        print(f"  ✓ Reference and {dataset_name} now have SAME {len(common_genes_all)} genes")

        # Clean up query metadata and intermediate variables
        adata_query_filtered.uns = {}
        adata_query_filtered.obsm = {}
        adata_query_filtered.varm = {}
        adata_query_filtered.obsp = {}
        adata_query_filtered.varp = {}
        del query_counts, query_counts_filtered, gene_indices_query
        gc.collect()

        # Preprocess using the common reference and this query
        # CRITICAL: Use adata_ref_common.copy() to avoid modifying the shared reference
        adata_ref_proc_new, adata_query_proc, final_genes = preprocess_for_scvi(
            adata_ref_common.copy(),  # Use shared reference (copy to avoid modification)
            adata_query_filtered,
            ion_channel_genes,
            n_hvgs_ref=args.n_hvgs_ref,
            hvg_flavor=args.hvg_flavor,
            hvg_ncells=args.hvg_ncells,
            random_seed=args.random_seed
        )

        # Clean up filtered query
        del adata_query_filtered
        gc.collect()
        
        # Force garbage collection to free memory before next dataset
        gc.collect()

        # VALIDATION: Ensure gene sets match
        if dataset_idx == 0:
            adata_ref_proc = adata_ref_proc_new
            print(f"  ✓ Reference genes: {adata_ref_proc.n_vars}")
        else:
            # Verify new reference has same genes as first
            if adata_ref_proc.n_vars != adata_ref_proc_new.n_vars:
                raise ValueError(
                    f"Gene set mismatch! First dataset: {adata_ref_proc.n_vars} genes, "
                    f"{dataset_name}: {adata_ref_proc_new.n_vars} genes. This should not happen!"
                )
            if not all(adata_ref_proc.var_names == adata_ref_proc_new.var_names):
                raise ValueError(
                    f"Gene names mismatch between datasets! This should not happen!"
                )
            # Release memory from redundant reference processing
            del adata_ref_proc_new
            gc.collect()
            print(f"  ✓ Using reference from first dataset (genes verified to match)")

        # Verify query has same genes as reference
        if adata_query_proc.n_vars != adata_ref_proc.n_vars:
            raise ValueError(
                f"Query-Reference gene mismatch! Query: {adata_query_proc.n_vars}, "
                f"Reference: {adata_ref_proc.n_vars}"
            )
        print(f"  ✓ Verified: Query and Reference have same {adata_query_proc.n_vars} genes")

        # Save preprocessed data
        print(f"\n[SAVE] Saving preprocessed data...")
        ref_path = output_dir / 'reference_preprocessed.h5ad'
        query_path = output_dir / f'{dataset_name}_query_preprocessed.h5ad'
        genes_path = output_dir / f'{dataset_name}_genes.txt'

        # Only save reference once (on first dataset)
        if dataset_idx == 0:
            adata_ref_proc.write(ref_path)
            print(f"  Reference: {ref_path}")

        adata_query_proc.write(query_path)

        # Save gene list
        with open(genes_path, 'w') as f:
            f.write('\n'.join(final_genes))

        print(f"  Query: {query_path}")
        print(f"  Genes: {genes_path}")

    print(f"\n{'=' * 80}")
    print("PREPROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nNext step: Run 02_train_scvi_and_map.py")


if __name__ == '__main__':
    main()
