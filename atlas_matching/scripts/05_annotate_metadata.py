#!/usr/bin/env python3
"""
Step 5: Annotate Patch-seq metadata with atlas mapping results

This script merges Patch-seq metadata with atlas mapping results to produce
annotated DataFrames for both M1 and V1 datasets, using configuration from YAML files.

Output: Annotated metadata CSVs with original metadata + atlas assignments
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add module to path (05_annotate_metadata.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping.utils import (
    get_dataset_paths,
    get_directory_paths,
    get_metadata_columns,
    load_dataset_config
)


def validate_mapping(trans_map, meta, key_column):
    """
    Validate that the mapping between transcriptomics map and metadata is 1-1.
    """
    is_unique_in_map = trans_map.index.is_unique
    is_subset = set(trans_map.index).issubset(set(meta[key_column]))
    is_unique_in_meta = meta[key_column].is_unique
    mapping_counts = trans_map.index.value_counts()
    only_single_mapping = (mapping_counts <= 1).all()

    print(f"  Mapping validation:")
    print(f"    - Unique indices in mapping: {is_unique_in_map}")
    print(f"    - All mapping indices in metadata: {is_subset}")
    print(f"    - Unique keys in metadata: {is_unique_in_meta}")
    print(f"    - One-to-one mapping: {only_single_mapping}")

    is_valid = is_unique_in_map and is_subset and only_single_mapping
    if is_valid:
        print("  ✓ Mapping is 1-1")
    else:
        print("  ✗ Mapping is NOT 1-1")

    return is_valid


def annotate_dataset_metadata(
    dataset: str,
    metadata_path: Path,
    mapping_results_path: Path,
    output_path: Path,
    project_root: Path
):
    """
    Annotate dataset metadata with atlas mapping results.

    Parameters
    ----------
    dataset : str
        Dataset identifier ('V1' or 'M1')
    metadata_path : Path
        Path to metadata CSV file
    mapping_results_path : Path
        Path to mapping results CSV file
    output_path : Path
        Path to save annotated CSV
    project_root : Path
        Project root directory

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with metadata and atlas annotations
    """
    print("=" * 60)
    print(f"Processing {dataset} dataset")
    print("=" * 60)

    # Get dataset config
    dataset_key = f"{dataset}_mouse_gouwens" if dataset == "V1" else f"{dataset}_mouse_tasic"
    meta_cols_config = get_metadata_columns(dataset_key)

    # Load metadata
    print(f"\nLoading {dataset} metadata from: {metadata_path}")
    if dataset == "M1":
        # Try tab-separated first, fall back to comma
        try:
            meta = pd.read_csv(metadata_path, sep='\t')
        except (pd.errors.ParserError, pd.errors.EmptyDataError):
            meta = pd.read_csv(metadata_path, sep=',')
    else:
        meta = pd.read_csv(metadata_path)

    print(f"  Loaded {len(meta)} rows")

    # Load mapping results
    print(f"\nLoading {dataset} mapping results from: {mapping_results_path}")
    trans_map = pd.read_csv(mapping_results_path, index_col=0)
    print(f"  Loaded {len(trans_map)} rows")

    # Validate mapping
    print("\nValidating mapping...")
    cell_id_col = meta_cols_config['cell_id_col']
    validate_mapping(trans_map, meta, cell_id_col)

    # Get required columns
    if dataset == "V1":
        required_cols = [
            meta_cols_config.get('ephys_id_col', 'ephys_session_id'),
            meta_cols_config.get('brain_region_col', 'structure'),
            meta_cols_config['cell_id_col'],
            meta_cols_config.get('cell_type_col', 'Patch-seq CT')
        ]
        # Rename if needed
        rename_mapping = meta_cols_config.get('rename_mapping', {})
    else:  # M1
        required_cols = [
            meta_cols_config['cell_id_col'],
            meta_cols_config.get('family_col', 'RNA family'),
            meta_cols_config['cell_type_col']
        ]
        # Add optional columns if they exist
        optional = meta_cols_config.get('optional_cols', [])
        for col in optional:
            if col in meta.columns:
                required_cols.append(col)
        rename_mapping = {}

    # Check required columns exist
    missing_cols = [col for col in required_cols if col not in meta.columns]
    if missing_cols:
        print(f"  Warning: Missing columns: {missing_cols}")
        required_cols = [c for c in required_cols if c in meta.columns]

    # Create merged DataFrame
    print("\nMerging metadata with mapping results...")
    merged_df = meta[required_cols].copy()

    # Apply renaming if needed
    if rename_mapping:
        merged_df = merged_df.rename(columns=rename_mapping)

    # Merge with mapping results
    merged_df = merged_df.merge(
        trans_map,
        left_on=cell_id_col,
        right_index=True,
        how="inner"
    )

    print(f"  Merged DataFrame: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    print(f"  Columns: {', '.join(merged_df.columns)}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving merged DataFrame to: {output_path}")
    merged_df.to_csv(output_path, index=False)
    print(f"  ✓ {dataset} annotation complete")

    return merged_df


def main():
    """Main function to annotate M1 and/or V1 datasets."""
    parser = argparse.ArgumentParser(
        description='Step 5: Annotate Patch-seq metadata',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=PROJECT_ROOT,
        help='Project root directory'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['V1', 'M1'],
        default=['V1', 'M1'],
        help='Datasets to annotate'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (determines input/output subdirectory). '
             'If specified, reads from scvi_mapped/{model_name}/ and writes to annotated_metadata/{model_name}/'
    )
    parser.add_argument(
        '--no-model-subdir',
        action='store_true',
        help='Do not use model-specific subdirectories (legacy behavior)'
    )
    parser.add_argument(
        '--v1-metadata',
        type=Path,
        default=None,
        help='Path to V1 metadata CSV (overrides config)'
    )
    parser.add_argument(
        '--m1-metadata',
        type=Path,
        default=None,
        help='Path to M1 metadata CSV (overrides config)'
    )
    parser.add_argument(
        '--mapping-dir',
        type=Path,
        default=None,
        help='Directory with mapping results (default: results/scvi_mapped/{model_name})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: results/annotated_metadata/{model_name})'
    )

    args = parser.parse_args()

    project_root = args.project_root

    # Get directory paths from config
    dir_paths = get_directory_paths(project_root)

    # Determine model-specific paths
    model_name = args.model_name
    use_subdir = model_name and not args.no_model_subdir

    if args.mapping_dir:
        mapping_dir = args.mapping_dir
    elif use_subdir:
        mapping_dir = ATLAS_MATCHING_ROOT / 'results' / 'scvi_mapped' / model_name
    else:
        mapping_dir = dir_paths['scvi_mapped_dir']

    if args.output_dir:
        output_dir = args.output_dir
    elif use_subdir:
        output_dir = ATLAS_MATCHING_ROOT / 'results' / 'annotated_metadata' / model_name
    else:
        output_dir = dir_paths['annotated_dir']

    print("=" * 60)
    print("STEP 5: ANNOTATE METADATA")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Datasets: {', '.join(args.datasets)}")
    if model_name:
        print(f"  Model: {model_name}")
    print(f"  Mapping dir: {mapping_dir}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)

    results = {}

    for dataset in args.datasets:
        try:
            # Get paths
            if dataset == "V1":
                if args.v1_metadata:
                    metadata_path = args.v1_metadata
                else:
                    # Try to load from config
                    try:
                        dataset_paths = get_dataset_paths('V1_mouse_gouwens', project_root)
                        metadata_path = dataset_paths['metadata']
                    except:
                        # Fallback to default location
                        metadata_path = project_root / "dat/expression" / "V1_patchseq_metadata.csv"
            else:  # M1
                if args.m1_metadata:
                    metadata_path = args.m1_metadata
                else:
                    try:
                        dataset_paths = get_dataset_paths('M1_mouse_tasic', project_root)
                        metadata_path = dataset_paths['metadata']
                    except:
                        metadata_path = project_root / "dat/expression" / "M1_patchseq_metadata.csv"

            mapping_path = mapping_dir / f"{dataset}_mapping_results.csv"
            output_path = output_dir / f"{dataset}_annotated_metadata.csv"

            # Annotate
            merged_df = annotate_dataset_metadata(
                dataset=dataset,
                metadata_path=metadata_path,
                mapping_results_path=mapping_path,
                output_path=output_path,
                project_root=project_root
            )
            results[dataset] = merged_df

        except Exception as e:
            print(f"\n✗ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            results[dataset] = None

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for dataset, df in results.items():
        if df is not None:
            output_path = output_dir / f"{dataset}_annotated_metadata.csv"
            print(f"{dataset}: {len(df)} annotated cells saved to {output_path}")
        else:
            print(f"{dataset}: Failed to process")

    print("\n✅ Annotation complete!")

    return results


if __name__ == "__main__":
    results = main()
