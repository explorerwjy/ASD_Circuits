#!/usr/bin/env python3
"""
Build MERFISH ISH Annotation: Map MERFISH cells to ISH brain structures.

This script creates MERFISH.ISH_Annot.csv by:
1. Loading raw MERFISH cell metadata from Allen Brain Cell Atlas (ABCA-1 through ABCA-4)
2. Adding CCF coordinates and parcellation annotations
3. Mapping each cell's CCF parcellation to ISH brain structure names (213 structures)

Input data (download from Allen Brain Cell Atlas):
  - MERFISH cell metadata: cell_metadata_with_cluster_annotation.csv (per ABCA section)
  - CCF coordinates: ccf_coordinates.csv (per ABCA section)
  - CCF parcellation: parcellation_to_parcellation_term_membership_acronym.csv
  - CCF parcellation colors: parcellation_to_parcellation_term_membership_color.csv

Download instructions:
  Visit https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html
  Or use the ABC Atlas access tools: pip install abc_atlas_access

Output:
  dat/MERFISH/MERFISH.ISH_Annot.csv — All MERFISH cells with ISH_STR and ISH_STR2 columns

Usage:
  python scripts/build_merfish_annotation.py \\
      --merfish-dir /path/to/MERFISH_ZhuangLab \\
      --ccf-dir /path/to/Allen-CCF-2020/views \\
      --output dat/MERFISH/MERFISH.ISH_Annot.csv
"""

import argparse
import os
import sys
import pandas as pd
import yaml


def add_ish_str(cell_meta, ccf_ontology, structures_ish):
    """Map MERFISH cells to ISH brain structures via CCF v3 ontology.

    Parameters
    ----------
    cell_meta : DataFrame
        MERFISH cell metadata with 'parcellation_structure' and
        'parcellation_substructure' columns.
    ccf_ontology : DataFrame
        CCF v3 ontology indexed by abbreviation with 'CleanName' column.
    structures_ish : list
        ISH structure names (space-separated) from the connectome (213 structures).

    Returns
    -------
    DataFrame with added 'ISH_STR' and 'ISH_STR2' columns.
    """
    ccf_index_set = set(ccf_ontology.index.values)
    ish_set = set(structures_ish)

    ish_str_vals = []
    for _, row in cell_meta.iterrows():
        _str = row["parcellation_structure"]
        _substr = row["parcellation_substructure"]

        name_str = ccf_ontology.loc[_str, "CleanName"] if _str in ccf_index_set else "None"
        name_substr = ccf_ontology.loc[_substr, "CleanName"] if _substr in ccf_index_set else "None"

        if name_str in ish_set:
            ish_str = name_str
        elif name_substr in ish_set:
            ish_str = name_substr
        elif _str in ("VISa", "VISrl"):
            ish_str = "Posterior parietal association areas"
        elif name_str == "Subiculum":
            ish_str = "Subiculum"
        else:
            ish_str = "Not in Connectome"

        ish_str_vals.append(ish_str)

    cell_meta["ISH_STR"] = ish_str_vals
    cell_meta["ISH_STR2"] = ["_".join(s.split()) for s in ish_str_vals]
    return cell_meta


def process_section(section_dir, parcellation_annotation, parcellation_color,
                    ccf_ontology, structures_ish):
    """Process a single ABCA section directory.

    Parameters
    ----------
    section_dir : str
        Path to ABCA section directory (e.g., .../ABCA-1/)
    parcellation_annotation : DataFrame
        CCF parcellation term membership (indexed by parcellation_index)
    parcellation_color : DataFrame
        CCF parcellation color info (indexed by parcellation_index)
    ccf_ontology : DataFrame
        CCF v3 → ISH structure mapping
    structures_ish : list
        ISH structure names

    Returns
    -------
    DataFrame with annotated cell metadata
    """
    section_name = os.path.basename(section_dir.rstrip("/"))
    print(f"  Processing {section_name}...")

    cell_meta = pd.read_csv(
        os.path.join(section_dir, "cell_metadata_with_cluster_annotation.csv"),
        index_col=0
    )
    cell_meta.rename(columns={"x": "x_section", "y": "y_section", "z": "z_section"},
                     inplace=True)

    cell_coords = pd.read_csv(
        os.path.join(section_dir, "ccf_coordinates.csv"),
        index_col=0
    )
    cell_coords.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"},
                       inplace=True)

    cell_meta = cell_meta.join(cell_coords, how="inner")
    cell_meta = cell_meta.join(parcellation_annotation, on="parcellation_index")
    cell_meta = cell_meta.join(parcellation_color, on="parcellation_index")
    cell_meta = add_ish_str(cell_meta, ccf_ontology, structures_ish)

    n_mapped = (cell_meta["ISH_STR"] != "Not in Connectome").sum()
    print(f"    {len(cell_meta):,} cells, {n_mapped:,} mapped to ISH structures")

    return cell_meta


def main():
    parser = argparse.ArgumentParser(
        description="Build MERFISH ISH Annotation from raw Allen Brain Cell Atlas data"
    )
    parser.add_argument(
        "--merfish-dir", required=True,
        help="Path to MERFISH_ZhuangLab/ directory containing ABCA-1 through ABCA-4"
    )
    parser.add_argument(
        "--ccf-dir", required=True,
        help="Path to Allen-CCF-2020 views/ directory with parcellation CSV files"
    )
    parser.add_argument(
        "--output", default="dat/MERFISH/MERFISH.ISH_Annot.csv",
        help="Output CSV path (default: dat/MERFISH/MERFISH.ISH_Annot.csv)"
    )
    parser.add_argument(
        "--project-dir", default=None,
        help="Project root directory (default: auto-detect from script location)"
    )
    args = parser.parse_args()

    # Resolve project directory
    if args.project_dir:
        proj_dir = args.project_dir
    else:
        proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load config for ISH structure list
    config_path = os.path.join(proj_dir, "config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load ISH structure list from bias file
    str_bias_path = os.path.join(proj_dir, config["data_files"]["str_bias_fdr"])
    asd_str_bias = pd.read_csv(str_bias_path, index_col=0)
    structures_ish = [s.replace("_", " ") for s in asd_str_bias.index.values]
    print(f"ISH structures: {len(structures_ish)}")

    # Load CCF v3 ontology
    ccf_ontology_path = os.path.join(proj_dir, config["data_files"]["merfish_ccf_ontology"])
    ccf_ontology = pd.read_csv(ccf_ontology_path, index_col=0)
    print(f"CCF v3 ontology: {ccf_ontology.shape[0]} entries")

    # Load CCF parcellation annotations
    print("Loading CCF parcellation data...")
    parcellation_annotation = pd.read_csv(
        os.path.join(args.ccf_dir, "parcellation_to_parcellation_term_membership_acronym.csv")
    )
    parcellation_annotation.set_index("parcellation_index", inplace=True)
    parcellation_annotation.columns = [
        f"parcellation_{x}" for x in parcellation_annotation.columns
    ]

    parcellation_color = pd.read_csv(
        os.path.join(args.ccf_dir, "parcellation_to_parcellation_term_membership_color.csv")
    )
    parcellation_color.set_index("parcellation_index", inplace=True)
    parcellation_color.columns = [
        f"parcellation_{x}" for x in parcellation_color.columns
    ]

    # Process all ABCA sections
    sections = ["ABCA-1", "ABCA-2", "ABCA-3", "ABCA-4"]
    all_cells = []

    print(f"\nProcessing {len(sections)} MERFISH sections...")
    for section in sections:
        section_dir = os.path.join(args.merfish_dir, section)
        if not os.path.isdir(section_dir):
            print(f"  WARNING: {section_dir} not found, skipping")
            continue
        cell_meta = process_section(
            section_dir, parcellation_annotation, parcellation_color,
            ccf_ontology, structures_ish
        )
        all_cells.append(cell_meta)

    if not all_cells:
        print("ERROR: No MERFISH sections found!")
        sys.exit(1)

    # Combine all sections
    combined = pd.concat(all_cells, ignore_index=True)
    n_mapped = (combined["ISH_STR"] != "Not in Connectome").sum()
    print(f"\nCombined: {len(combined):,} cells, {n_mapped:,} mapped to ISH structures "
          f"({100*n_mapped/len(combined):.1f}%)")

    # Save
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(proj_dir, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({os.path.getsize(output_path)/1e6:.0f} MB)")

    # Also save parquet version
    parquet_path = output_path.replace(".csv", ".parquet")
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].astype(str)
    combined.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path} ({os.path.getsize(parquet_path)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
