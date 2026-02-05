"""
Data loading utilities for atlas-grounded biophysical modeling framework.

This module provides functions to load:
- Reference atlas (scRNA-seq) data
- Query Patch-seq datasets (V1 and M1)
- Ion channel gene lists
"""

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle
import gzip


def load_reference_atlas(
    data_dir: Path,
    use_log2: bool = False,
    subsample: Optional[int] = None,
    random_seed: int = 42
) -> ad.AnnData:
    """
    Load reference atlas expression data.

    MEMORY OPTIMIZED: Only loads raw counts (not both raw and log2).
    For scVI preprocessing, use_log2 parameter is ignored - always uses raw counts.

    Parameters
    ----------
    data_dir : Path
        Path to reference_atlas directory containing h5ad files
    use_log2 : bool
        DEPRECATED: Ignored for memory efficiency. Always loads raw counts.
    subsample : int, optional
        If provided, randomly subsample this many cells from the atlas
    random_seed : int
        Random seed for subsampling

    Returns
    -------
    ad.AnnData
        Combined reference atlas with expression and metadata
        - .X: raw UMI counts (sparse)
        - .layers["counts"]: Same as .X (raw UMI counts)
    """
    data_dir = Path(data_dir)

    # MEMORY FIX: Only load raw counts files
    raw_file1 = data_dir / "Isocortex-1-raw.h5ad"
    raw_file2 = data_dir / "Isocortex-2-raw.h5ad"

    if use_log2:
        print(f"  Note: use_log2={use_log2} ignored for memory efficiency. Loading raw counts only.")

    # Load metadata
    metadata_path = data_dir / "cell_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    print(f"Loading reference atlas metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path, index_col=0)
    print(f"  Loaded metadata for {len(metadata)} cells")

    # Load raw expression data only
    print(f"Loading expression data from {raw_file1.name} and {raw_file2.name}...")

    # MEMORY-EFFICIENT APPROACH WITH EARLY SUBSAMPLING:
    # The key is to subsample IMMEDIATELY after loading each file, not at the end!
    # This avoids loading all 456k cells into memory

    if subsample is not None:
        # EARLY SUBSAMPLING: Load each file and immediately subsample
        print(f"  Loading with EARLY subsampling to {subsample} cells total...")

        # Calculate how many cells to take from each file (split 50/50)
        n_per_file = subsample // 2

        # Load file 1 in backed mode and subsample immediately
        print(f"  Loading {raw_file1.name} in backed mode...")
        adata1_backed = sc.read_h5ad(raw_file1, backed='r')

        # Random subsample indices
        np.random.seed(random_seed)
        n_cells1 = adata1_backed.n_obs
        indices1 = np.random.choice(n_cells1, min(n_per_file, n_cells1), replace=False)
        indices1.sort()  # Sort for efficient access

        # Load only subsampled cells
        print(f"  Subsampling {len(indices1)} cells from file 1...")
        adata1 = ad.AnnData(
            X=adata1_backed.X[indices1, :],
            obs=adata1_backed.obs.iloc[indices1].copy(),
            var=adata1_backed.var.copy()
        )
        adata1_backed.file.close()
        del adata1_backed

        # Load file 2 in backed mode and subsample immediately
        print(f"  Loading {raw_file2.name} in backed mode...")
        adata2_backed = sc.read_h5ad(raw_file2, backed='r')

        n_cells2 = adata2_backed.n_obs
        indices2 = np.random.choice(n_cells2, min(n_per_file, n_cells2), replace=False)
        indices2.sort()

        print(f"  Subsampling {len(indices2)} cells from file 2...")
        adata2 = ad.AnnData(
            X=adata2_backed.X[indices2, :],
            obs=adata2_backed.obs.iloc[indices2].copy(),
            var=adata2_backed.var.copy()
        )
        adata2_backed.file.close()
        del adata2_backed

        import gc
        gc.collect()
        print(f"  Memory saved by early subsampling!")

    else:
        # Full dataset loading - use backed mode for large datasets
        print(f"  Loading FULL dataset in backed mode...")
        adata1_backed = sc.read_h5ad(raw_file1, backed='r')
        adata2_backed = sc.read_h5ad(raw_file2, backed='r')

        # Convert backed to in-memory
        print(f"  Converting to memory (may take several minutes and require ~100GB RAM)...")
        adata1 = ad.AnnData(
            X=adata1_backed.X[:],
            obs=adata1_backed.obs.copy(),
            var=adata1_backed.var.copy()
        )
        adata2 = ad.AnnData(
            X=adata2_backed.X[:],
            obs=adata2_backed.obs.copy(),
            var=adata2_backed.var.copy()
        )
        adata1_backed.file.close()
        adata2_backed.file.close()
        del adata1_backed, adata2_backed
        import gc
        gc.collect()

    print(f"  Isocortex-1: {adata1.n_obs} cells × {adata1.n_vars} genes")
    print(f"  Isocortex-2: {adata2.n_obs} cells × {adata2.n_vars} genes")

    # Convert gene identifiers from Ensembl IDs to gene symbols
    for adata in [adata1, adata2]:
        if 'gene_symbol' in adata.var.columns:
            # Store original Ensembl IDs
            adata.var['ensembl_id'] = adata.var_names.copy()
            # Use gene symbols as var_names
            gene_symbols = adata.var['gene_symbol'].values
            # Handle any missing/duplicate symbols
            adata.var_names = pd.Index(gene_symbols).fillna('Unknown')
            # Make unique if there are duplicates
            adata.var_names_make_unique()

    # Store raw counts in layers for scVI
    # Use view for memory efficiency (will be copied during concat anyway)
    adata1.layers["counts"] = adata1.X
    adata2.layers["counts"] = adata2.X

    print(f"  Converted gene IDs to symbols (sample: {list(adata1.var_names[:3])})")

    # Combine datasets (no index_unique needed - cell names are unique across files)
    print(f"  Concatenating datasets...")
    adata_combined = ad.concat([adata1, adata2], join="outer")

    # Clean up to free memory
    del adata1, adata2
    import gc
    gc.collect()
    print(f"  Combined: {adata_combined.n_obs} cells × {adata_combined.n_vars} genes")

    # MEMORY FIX: Keep data sparse, no conversion to dense
    # Ensure counts layer exists (should already from concat)
    if "counts" in adata_combined.layers:
        print(f"  Counts layer preserved (sparse format)")
    
    # Merge metadata
    # Match cells by cell_label (assuming it's in the index or var_names)
    if 'cell_label' in metadata.columns:
        # Try to match by cell_label
        metadata_indexed = metadata.set_index('cell_label', drop=False)
    else:
        metadata_indexed = metadata
    
    # Align metadata with expression data
    common_cells = set(adata_combined.obs_names) & set(metadata_indexed.index)
    print(f"  Found {len(common_cells)} cells with matching metadata")
    
    if len(common_cells) < len(adata_combined):
        print(f"  Warning: {len(adata_combined) - len(common_cells)} cells missing metadata")
        adata_combined = adata_combined[list(common_cells)].copy()
    
    # Add metadata to obs
    for col in metadata_indexed.columns:
        if col not in adata_combined.obs.columns:
            adata_combined.obs[col] = metadata_indexed.loc[adata_combined.obs_names, col].values
    
    # Add tech_batch column to identify reference atlas
    adata_combined.obs['tech_batch'] = 'Allen_Atlas'
    adata_combined.obs['tech_batch'] = adata_combined.obs['tech_batch'].astype('category')
    print(f"  Added tech_batch column: Allen_Atlas")

    # Note: Subsampling is now done EARLY during loading (before concatenation)
    # to dramatically reduce memory usage. No need to subsample again here.
    print(f"  Final dataset: {len(adata_combined)} cells")

    return adata_combined


def load_patchseq_v1(
    data_dir: Path,
    load_ephys: bool = False
) -> Tuple[ad.AnnData, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load V1 Patch-seq dataset.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing V1 Patch-seq files
    load_ephys : bool
        If True, also load pre-extracted ephys features
        
    Returns
    -------
    ad.AnnData
        Expression data (cells × genes)
    pd.DataFrame
        Cell metadata
    pd.DataFrame, optional
        Ephys features if load_ephys=True
    """
    data_dir = Path(data_dir)
    
    # Load expression counts
    counts_path = data_dir / "V1_patchseq_counts.csv"
    print(f"Loading V1 expression from {counts_path}...")
    counts_df = pd.read_csv(counts_path, index_col=0)
    print(f"  Expression: {counts_df.shape[1]} cells × {counts_df.shape[0]} genes")
    
    # Transpose to cells × genes
    counts_df = counts_df.T
    
    # Create AnnData object with raw counts
    # Store raw counts in both .X and layers["counts"] initially
    counts_array = counts_df.values.astype(np.int32)
    adata = ad.AnnData(X=counts_array, var=pd.DataFrame(index=counts_df.columns))
    adata.obs_names = counts_df.index
    adata.var_names = counts_df.columns
    
    # Store raw counts in layers["counts"] for scVI
    adata.layers["counts"] = counts_array.copy()
    
    # Load metadata
    metadata_path = data_dir / "V1_patchseq_metadata.csv"
    print(f"Loading V1 metadata from {metadata_path}...")
    # Try comma-separated first, fall back to tab-separated
    try:
        metadata = pd.read_csv(metadata_path, sep=',')
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        # Fall back to tab-separated
        metadata = pd.read_csv(metadata_path, sep='\t')
    
    # Match metadata to expression data
    if 'transcriptomics_sample_id' in metadata.columns:
        # Use transcriptomics_sample_id to match
        metadata_indexed = metadata.set_index('transcriptomics_sample_id', drop=False)
        common_cells = set(adata.obs_names) & set(metadata_indexed.index)
        if len(common_cells) > 0:
            adata = adata[list(common_cells)].copy()
            for col in metadata.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = metadata_indexed.loc[adata.obs_names, col].values
    else:
        # Try to match by index
        common_cells = set(adata.obs_names) & set(metadata.index)
        if len(common_cells) > 0:
            adata = adata[list(common_cells)].copy()
            for col in metadata.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = metadata.loc[adata.obs_names, col].values
    
    print(f"  Matched {len(adata)} cells with metadata")
    
    # Add tech_batch column
    adata.obs['tech_batch'] = 'PatchSeq_V1'
    adata.obs['tech_batch'] = adata.obs['tech_batch'].astype('category')
    
    # Load ephys features if requested
    ephys_features = None
    if load_ephys:
        ephys_path = data_dir / "V1_patchseq_ephys_features.pickle"
        print(f"Loading V1 ephys features from {ephys_path}...")
        with open(ephys_path, 'rb') as f:
            ephys_data = pickle.load(f)
        
        # Extract features DataFrame
        if isinstance(ephys_data, dict):
            if 'X_o' in ephys_data:
                ephys_features = ephys_data['X_o']
            else:
                # Try to find DataFrame in dict
                for key, val in ephys_data.items():
                    if isinstance(val, pd.DataFrame):
                        ephys_features = val
                        break
        elif isinstance(ephys_data, pd.DataFrame):
            ephys_features = ephys_data
        
        if ephys_features is not None:
            print(f"  Ephys features: {len(ephys_features)} cells × {ephys_features.shape[1]} features")
    
    return adata, metadata, ephys_features


def load_patchseq_m1(
    data_dir: Path,
    load_ephys: bool = False
) -> Tuple[ad.AnnData, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load M1 Patch-seq dataset.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing M1 Patch-seq files
    load_ephys : bool
        If True, also load pre-extracted ephys features
        
    Returns
    -------
    ad.AnnData
        Expression data (cells × genes)
    pd.DataFrame
        Cell metadata
    pd.DataFrame, optional
        Ephys features if load_ephys=True
    """
    data_dir = Path(data_dir)
    
    # Load expression counts (gzipped)
    counts_path = data_dir / "M1_patchseq_counts.csv.gz"
    print(f"Loading M1 expression from {counts_path}...")
    counts_df = pd.read_csv(counts_path, index_col=0, compression='gzip')
    print(f"  Expression: {counts_df.shape[1]} cells × {counts_df.shape[0]} genes")
    
    # Transpose to cells × genes
    counts_df = counts_df.T
    
    # Create AnnData object with raw counts
    # Store raw counts in both .X and layers["counts"] initially
    counts_array = counts_df.values.astype(np.int32)
    adata = ad.AnnData(X=counts_array, var=pd.DataFrame(index=counts_df.columns))
    adata.obs_names = counts_df.index
    adata.var_names = counts_df.columns
    
    # Store raw counts in layers["counts"] for scVI
    adata.layers["counts"] = counts_array.copy()
    
    # Load metadata
    metadata_path = data_dir / "M1_patchseq_metadata.csv"
    print(f"Loading M1 metadata from {metadata_path}...")
    # Try tab-separated first, fall back to comma-separated
    try:
        metadata = pd.read_csv(metadata_path, sep='\t')
    except (pd.errors.ParserError, pd.errors.EmptyDataError):
        # Fall back to comma-separated
        metadata = pd.read_csv(metadata_path, sep=',')
    
    # Match metadata to expression data
    if 'Cell' in metadata.columns:
        # Use Cell column to match
        metadata_indexed = metadata.set_index('Cell', drop=False)
        common_cells = set(adata.obs_names) & set(metadata_indexed.index)
        if len(common_cells) > 0:
            adata = adata[list(common_cells)].copy()
            for col in metadata.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = metadata_indexed.loc[adata.obs_names, col].values
    else:
        # Try to match by index
        common_cells = set(adata.obs_names) & set(metadata.index)
        if len(common_cells) > 0:
            adata = adata[list(common_cells)].copy()
            for col in metadata.columns:
                if col not in adata.obs.columns:
                    adata.obs[col] = metadata.loc[adata.obs_names, col].values
    
    print(f"  Matched {len(adata)} cells with metadata")
    
    # Add tech_batch column
    adata.obs['tech_batch'] = 'PatchSeq_M1'
    adata.obs['tech_batch'] = adata.obs['tech_batch'].astype('category')
    
    # Load ephys features if requested
    ephys_features = None
    if load_ephys:
        ephys_path = data_dir / "M1_patchseq_ephys_features.csv"
        print(f"Loading M1 ephys features from {ephys_path}...")
        ephys_features = pd.read_csv(ephys_path, index_col=0)
        print(f"  Ephys features: {len(ephys_features)} cells × {ephys_features.shape[1]} features")
    
    return adata, metadata, ephys_features


def load_ion_channel_genes(
    data_dir: Path,
    source: str = "both"
) -> Dict[str, List[str]]:
    """
    Load ion channel gene lists from GO terms and/or curated list.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing ion channel gene files
    source : str
        One of "GO", "177", or "both" (default)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'GO': list of mouse gene symbols from GO terms
        - '177': list of mouse gene symbols from curated 177 list (converted from human)
        - 'all': combined unique list
    """
    data_dir = Path(data_dir)
    result = {}
    
    if source in ["GO", "both"]:
        go_path = data_dir / "ion_channel_genes_GO.txt"
        if go_path.exists():
            print(f"Loading GO term ion channel genes from {go_path}...")
            go_df = pd.read_csv(go_path, sep='\t', low_memory=False)
            
            # Column 2 (index 1) contains gene symbols
            # Column 6 (index 5) contains GO terms
            if go_df.shape[1] > 5:
                # Filter for relevant GO terms
                relevant_terms = [
                    'potassium channel activity',
                    'sodium channel activity',
                    'calcium channel activity',
                    'chloride channel activity',
                    'ion channel activity',
                    'voltage-gated',
                    'ligand-gated'
                ]
                
                go_genes = []
                for term in relevant_terms:
                    mask = go_df.iloc[:, 5].str.contains(term, case=False, na=False)
                    genes = go_df.loc[mask, go_df.columns[1]].dropna().unique()
                    go_genes.extend(genes.tolist())
                
                result['GO'] = list(set(go_genes))
                print(f"  Found {len(result['GO'])} unique genes from GO terms")
            else:
                print(f"  Warning: GO file has unexpected format")
                result['GO'] = []
        else:
            print(f"  Warning: GO file not found at {go_path}")
            result['GO'] = []
    
    if source in ["177", "both"]:
        curated_path = data_dir / "ion_channel_genes_177.csv"
        if curated_path.exists():
            print(f"Loading curated 177 ion channel genes from {curated_path}...")
            curated_df = pd.read_csv(curated_path)
            
            # Column 2 (index 1) contains human HGNC symbols
            if curated_df.shape[1] > 1:
                human_symbols = curated_df.iloc[:, 1].dropna().unique()
                
                # Convert human to mouse: lowercase first letter
                # e.g., SCN1A -> Scn1a
                mouse_symbols = []
                for symbol in human_symbols:
                    if len(symbol) > 0:
                        mouse_symbol = symbol[0].lower() + symbol[1:] if len(symbol) > 1 else symbol.lower()
                        mouse_symbols.append(mouse_symbol)
                
                result['177'] = list(set(mouse_symbols))
                print(f"  Converted {len(result['177'])} human symbols to mouse")
            else:
                print(f"  Warning: Curated file has unexpected format")
                result['177'] = []
        else:
            print(f"  Warning: Curated file not found at {curated_path}")
            result['177'] = []
    
    # Combine all unique genes
    all_genes = set()
    for gene_list in result.values():
        all_genes.update(gene_list)
    result['all'] = list(all_genes)
    print(f"  Total unique ion channel genes: {len(result['all'])}")
    
    return result
