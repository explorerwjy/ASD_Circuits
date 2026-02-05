#!/usr/bin/env python3
"""
Atlas-Grounded Biophysical Modeling Pipeline

This script implements the complete pipeline for mapping Patch-seq neurons onto
a high-resolution scRNA-seq reference atlas and computing canonical expression profiles.

Pipeline Overview:
1. Data Loading: Load reference atlas and Patch-seq datasets
2. Preprocessing: Gene intersection, normalization, HVG selection, scaling
3. Reference Integration: scVI/scANVI + scArches (or PCA + Harmony baseline)
4. Atlas Mapping: Two-stage hybrid mapping with hierarchical assignment
5. Canonical Expression: Compute cluster/subclass profiles and query effective expression
"""

import sys
import argparse
import warnings
import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

# Suppress warnings
warnings.filterwarnings('ignore')

# Add module to path (run_full_pipeline.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping import (
    load_reference_atlas,
    load_patchseq_v1,
    load_patchseq_m1,
    load_ion_channel_genes,
    preprocess_pipeline,
    train_scvi,
    train_scanvi,
    map_query_scarches,
    integrate_pca_harmony,
    hierarchical_assignment,
    compute_canonical_pipeline
)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    return {
        # Preprocessing
        'n_hvgs': 3000,
        'hvg_flavor': 'seurat_v3',
        'ref_is_log2': True,  # Reference files are already log2(CPM+1)
        
        # Reference integration
        'integration_method': 'scvi',  # 'scvi', 'scanvi', or 'pca_harmony'
        'n_latent': 30,
        'n_layers': 2,
        'n_hidden': 128,
        'max_epochs': 400,
        'use_gpu': True,
        
        # Atlas mapping
        'k_candidates': 15,  # Top-K clusters in Stage 1
        'k_nn': 20,  # KNN neighbors in Stage 2
        'conf_threshold_subclass': 0.7,
        'conf_threshold_cluster': 0.7,
        
        # Canonical expression
        'use_soft_assignment': True,
        'k_nn_expression': 20,
        
        # Data loading
        'subsample_atlas': None,  # Set to int to subsample (e.g., 100000)
        'load_ephys': False,  # Set to True to load ephys features
    }


def get_config_hash(config: Dict[str, Any]) -> str:
    """Generate a hash of the configuration for checkpoint naming."""
    # Create a stable string representation of config
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_checkpoint_path(checkpoint_dir: Path, dataset_name: str, step: str, config_hash: str) -> Path:
    """Get checkpoint file path."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{dataset_name}_{step}_{config_hash}.pkl"


def get_model_path(model_dir: Path, dataset_name: str, method: str, config_hash: str) -> Path:
    """Get model directory path for saving scVI/scANVI models."""
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{dataset_name}_{method}_{config_hash}"


def model_exists(model_path: Path) -> bool:
    """Check if a trained model exists (check for model.pt file)."""
    if not model_path.exists():
        return False
    # Check if model.pt exists (scvi-tools saves models as directories with model.pt inside)
    model_file = model_path / "model.pt"
    return model_file.exists()


def save_checkpoint(
    checkpoint_path: Path,
    adata_ref: Optional[ad.AnnData] = None,
    adata_query: Optional[ad.AnnData] = None,
    adata_ref_log: Optional[ad.AnnData] = None,
    adata_query_log: Optional[ad.AnnData] = None,
    mapping_results: Optional[pd.DataFrame] = None,
    canonical_cluster: Optional[pd.DataFrame] = None,
    canonical_subclass: Optional[pd.DataFrame] = None,
    effective_expr: Optional[pd.DataFrame] = None,
    genes: Optional[list] = None,
    means: Optional[np.ndarray] = None,
    stds: Optional[np.ndarray] = None,
    **kwargs
) -> None:
    """Save checkpoint data."""
    checkpoint_data = {
        'adata_ref': adata_ref,
        'adata_query': adata_query,
        'adata_ref_log': adata_ref_log,
        'adata_query_log': adata_query_log,
        'mapping_results': mapping_results,
        'canonical_cluster': canonical_cluster,
        'canonical_subclass': canonical_subclass,
        'effective_expr': effective_expr,
        'genes': genes,
        'means': means,
        'stds': stds,
        **kwargs
    }
    
    # Remove None values
    checkpoint_data = {k: v for k, v in checkpoint_data.items() if v is not None}
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load checkpoint data."""
    if not checkpoint_path.exists():
        return None
    
    print(f"  Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    return checkpoint_data


def process_dataset(
    dataset_name: str,
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    ion_channel_genes: list,
    config: Dict[str, Any],
    output_dir: Path,
    checkpoint_dir: Optional[Path] = None,
    use_checkpoints: bool = True,
    force_rerun: bool = False,
    model_dir: Optional[Path] = None
) -> None:
    """
    Process a single dataset (V1 or M1) through the complete pipeline.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name ('V1' or 'M1')
    adata_ref : ad.AnnData
        Reference atlas
    adata_query : ad.AnnData
        Query Patch-seq data
    ion_channel_genes : list
        List of ion channel genes
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory for results
    checkpoint_dir : Path, optional
        Directory for checkpoint files
    use_checkpoints : bool
        If True, save and load checkpoints
    force_rerun : bool
        If True, ignore existing checkpoints and rerun
    """
    print("=" * 80)
    print(f"PROCESSING {dataset_name} DATASET")
    print("=" * 80)
    
    # Generate config hash for checkpoint naming
    config_hash = get_config_hash(config)
    
    # Step 1: Preprocessing
    print(f"\n[{dataset_name}] Step 1: Preprocessing...")
    checkpoint_path_preproc = None
    if checkpoint_dir and use_checkpoints:
        checkpoint_path_preproc = get_checkpoint_path(checkpoint_dir, dataset_name, 'preprocessed', config_hash)

    if checkpoint_path_preproc and checkpoint_path_preproc.exists() and not force_rerun:
        print(f"  Loading preprocessed data from checkpoint...")
        checkpoint_data = load_checkpoint(checkpoint_path_preproc)
        if checkpoint_data:
            adata_ref_proc = checkpoint_data['adata_ref']
            adata_query_proc = checkpoint_data['adata_query']
            # Get log-normalized data for scVI (if available)
            adata_ref_log = checkpoint_data.get('adata_ref_log')
            adata_query_log = checkpoint_data.get('adata_query_log')
            genes = checkpoint_data.get('genes')
            means = checkpoint_data.get('means')
            stds = checkpoint_data.get('stds')
            print(f"  Loaded preprocessed data: ref={adata_ref_proc.shape}, query={adata_query_proc.shape}")
            # Check if log-normalized data is missing (old checkpoint format)
            if adata_ref_log is None:
                print(f"  Warning: Old checkpoint format detected, missing log-normalized data")
                print(f"  Rerunning preprocessing to get log-normalized data for scVI...")
                result = preprocess_pipeline(
                    adata_ref.copy(),
                    adata_query.copy(),
                    ion_channel_genes,
                    n_hvgs=config['n_hvgs'],
                    n_hvgs_query=config.get('n_hvgs_query', 500),
                    hvg_flavor=config['hvg_flavor'],
                    ref_is_log2=config['ref_is_log2'],
                    return_log_normalized=True
                )
                adata_ref_proc, adata_query_proc, genes, means, stds, adata_ref_log, adata_query_log = result
                # Update checkpoint with new format
                save_checkpoint(
                    checkpoint_path_preproc,
                    adata_ref=adata_ref_proc,
                    adata_query=adata_query_proc,
                    adata_ref_log=adata_ref_log,
                    adata_query_log=adata_query_log,
                    genes=genes,
                    means=means,
                    stds=stds
                )
        else:
            # Fallback to recompute
            result = preprocess_pipeline(
                adata_ref.copy(),
                adata_query.copy(),
                ion_channel_genes,
                n_hvgs=config['n_hvgs'],
                n_hvgs_query=config.get('n_hvgs_query', 500),
                hvg_flavor=config['hvg_flavor'],
                ref_is_log2=config['ref_is_log2'],
                return_log_normalized=True
            )
            adata_ref_proc, adata_query_proc, genes, means, stds, adata_ref_log, adata_query_log = result
            if checkpoint_path_preproc:
                save_checkpoint(
                    checkpoint_path_preproc,
                    adata_ref=adata_ref_proc,
                    adata_query=adata_query_proc,
                    adata_ref_log=adata_ref_log,
                    adata_query_log=adata_query_log,
                    genes=genes,
                    means=means,
                    stds=stds
                )
    else:
        result = preprocess_pipeline(
            adata_ref.copy(),
            adata_query.copy(),
            ion_channel_genes,
            n_hvgs=config['n_hvgs'],
            hvg_flavor=config['hvg_flavor'],
            ref_is_log2=config['ref_is_log2'],
            return_log_normalized=True
        )
        adata_ref_proc, adata_query_proc, genes, means, stds, adata_ref_log, adata_query_log = result
        if checkpoint_path_preproc:
            save_checkpoint(
                checkpoint_path_preproc,
                adata_ref=adata_ref_proc,
                adata_query=adata_query_proc,
                adata_ref_log=adata_ref_log,
                adata_query_log=adata_query_log,
                genes=genes,
                means=means,
                stds=stds
            )
    
    # Step 2: Reference Integration
    # NOTE: scVI requires log-normalized data, NOT Z-scored data
    # We train scVI on log-normalized data, then copy latent embeddings to Z-scored data
    print(f"\n[{dataset_name}] Step 2: Reference Integration...")
    checkpoint_path_integration = None
    # Use model_dir passed to function, or create default if None
    if model_dir is None and checkpoint_dir:
        model_dir = checkpoint_dir.parent / 'models'
    
    if checkpoint_dir and use_checkpoints:
        checkpoint_path_integration = get_checkpoint_path(checkpoint_dir, dataset_name, 'integrated', config_hash)

    if checkpoint_path_integration and checkpoint_path_integration.exists() and not force_rerun:
        print(f"  Loading integrated data from checkpoint...")
        checkpoint_data = load_checkpoint(checkpoint_path_integration)
        if checkpoint_data:
            adata_ref_proc = checkpoint_data['adata_ref']
            adata_query_proc = checkpoint_data['adata_query']
            print(f"  Loaded integrated data: ref={adata_ref_proc.shape}, query={adata_query_proc.shape}")
        else:
            # Fallback to recompute - use log-normalized data for scVI
            adata_ref_log, adata_query_log = _run_integration(
                dataset_name, adata_ref_log, adata_query_log, config,
                checkpoint_path=None, model_dir=model_dir, force_rerun=force_rerun
            )
            # Copy latent embeddings to Z-scored data
            adata_ref_proc.obsm['X_latent'] = adata_ref_log.obsm['X_latent']
            adata_query_proc.obsm['X_latent'] = adata_query_log.obsm['X_latent']
            if checkpoint_path_integration:
                save_checkpoint(
                    checkpoint_path_integration,
                    adata_ref=adata_ref_proc,
                    adata_query=adata_query_proc
                )
            # Copy latent embeddings to Z-scored data
            adata_ref_proc.obsm['X_latent'] = adata_ref_log.obsm['X_latent']
            adata_query_proc.obsm['X_latent'] = adata_query_log.obsm['X_latent']
            if checkpoint_path_integration:
                save_checkpoint(
                    checkpoint_path_integration,
                    adata_ref=adata_ref_proc,
                    adata_query=adata_query_proc
                )
    else:
        # Train integration on log-normalized data
        adata_ref_log, adata_query_log = _run_integration(
            dataset_name, adata_ref_log, adata_query_log, config,
            checkpoint_path=checkpoint_path_integration, model_dir=model_dir, force_rerun=force_rerun
        )
        # Copy latent embeddings to Z-scored data
        adata_ref_proc.obsm['X_latent'] = adata_ref_log.obsm['X_latent']
        adata_query_proc.obsm['X_latent'] = adata_query_log.obsm['X_latent']
        if checkpoint_path_integration:
            save_checkpoint(
                checkpoint_path_integration,
                adata_ref=adata_ref_proc,
                adata_query=adata_query_proc
            )
    
    # Step 3: Atlas Mapping
    print(f"\n[{dataset_name}] Step 3: Atlas Mapping...")
    checkpoint_path_mapping = None
    if checkpoint_dir and use_checkpoints:
        checkpoint_path_mapping = get_checkpoint_path(checkpoint_dir, dataset_name, 'mapped', config_hash)
    
    if checkpoint_path_mapping and checkpoint_path_mapping.exists() and not force_rerun:
        print(f"  Loading mapping results from checkpoint...")
        checkpoint_data = load_checkpoint(checkpoint_path_mapping)
        if checkpoint_data:
            mapping_results = checkpoint_data['mapping_results']
            adata_query_proc = checkpoint_data['adata_query']
            print(f"  Loaded mapping results: {len(mapping_results)} cells")
        else:
            # Fallback to recompute
            mapping_results = hierarchical_assignment(
                adata_ref_proc,
                adata_query_proc,
                latent_key='X_latent',
                subclass_key='subclass',
                cluster_key='cluster',
                k_candidates=config['k_candidates'],
                k_nn=config['k_nn'],
                conf_threshold_subclass=config['conf_threshold_subclass'],
                conf_threshold_cluster=config['conf_threshold_cluster']
            )
            
            # Ensure mapping_results has the same index as adata_query_proc
            if not mapping_results.index.equals(adata_query_proc.obs.index):
                print(f"  Warning: Index mismatch detected. Aligning indices...")
                mapping_results.index = adata_query_proc.obs.index
            
            # Check if mapping columns already exist
            mapping_cols = ['assigned_subclass', 'assigned_cluster', 'conf_subclass', 'conf_cluster', 'mapping_status']
            existing_cols = [col for col in mapping_cols if col in adata_query_proc.obs.columns]
            if existing_cols:
                print(f"  Warning: Mapping columns already exist: {existing_cols}. Dropping duplicates...")
                adata_query_proc.obs = adata_query_proc.obs.drop(columns=existing_cols)
            
            # Concatenate mapping results
            adata_query_proc.obs = pd.concat([
                adata_query_proc.obs,
                mapping_results
            ], axis=1)
            if checkpoint_path_mapping:
                save_checkpoint(
                    checkpoint_path_mapping,
                    mapping_results=mapping_results,
                    adata_query=adata_query_proc
                )
    else:
        mapping_results = hierarchical_assignment(
            adata_ref_proc,
            adata_query_proc,
            latent_key='X_latent',
            subclass_key='subclass',
            cluster_key='cluster',
            k_candidates=config['k_candidates'],
            k_nn=config['k_nn'],
            conf_threshold_subclass=config['conf_threshold_subclass'],
            conf_threshold_cluster=config['conf_threshold_cluster']
        )
        
        # Ensure mapping_results has the same index as adata_query_proc
        if not mapping_results.index.equals(adata_query_proc.obs.index):
            print(f"  Warning: Index mismatch detected. Aligning indices...")
            print(f"    adata_query_proc.obs index: {len(adata_query_proc.obs)} rows")
            print(f"    mapping_results index: {len(mapping_results)} rows")
            mapping_results.index = adata_query_proc.obs.index
        
        # Check if mapping columns already exist (shouldn't happen, but be safe)
        mapping_cols = ['assigned_subclass', 'assigned_cluster', 'conf_subclass', 'conf_cluster', 'mapping_status']
        existing_cols = [col for col in mapping_cols if col in adata_query_proc.obs.columns]
        if existing_cols:
            print(f"  Warning: Mapping columns already exist: {existing_cols}. Dropping duplicates...")
            adata_query_proc.obs = adata_query_proc.obs.drop(columns=existing_cols)
        
        # Concatenate mapping results
        adata_query_proc.obs = pd.concat([
            adata_query_proc.obs,
            mapping_results
        ], axis=1)
        if checkpoint_path_mapping:
            save_checkpoint(
                checkpoint_path_mapping,
                mapping_results=mapping_results,
                adata_query=adata_query_proc
            )
    
    # Step 4: Canonical Expression
    print(f"\n[{dataset_name}] Step 4: Canonical Expression...")
    checkpoint_path_canonical = None
    if checkpoint_dir and use_checkpoints:
        checkpoint_path_canonical = get_checkpoint_path(checkpoint_dir, dataset_name, 'canonical', config_hash)
    
    if checkpoint_path_canonical and checkpoint_path_canonical.exists() and not force_rerun:
        print(f"  Loading canonical expression from checkpoint...")
        checkpoint_data = load_checkpoint(checkpoint_path_canonical)
        if checkpoint_data:
            canonical_cluster = checkpoint_data['canonical_cluster']
            canonical_subclass = checkpoint_data['canonical_subclass']
            effective_expr = checkpoint_data['effective_expr']
            print(f"  Loaded canonical expression")
        else:
            # Fallback to recompute
            # CRITICAL: Use log-normalized data (adata_ref_log) for effective expression
            # Biophysical modeling requires gene abundance values (log-CPM), not Z-scores
            canonical_cluster, canonical_subclass, effective_expr = compute_canonical_pipeline(
                adata_ref_log,  # Use log-normalized reference (not Z-scored)
                adata_query_log,  # Use log-normalized query (not Z-scored)
                mapping_results,
                use_soft_assignment=config['use_soft_assignment'],
                k_nn=config['k_nn_expression']
            )
            if checkpoint_path_canonical:
                save_checkpoint(
                    checkpoint_path_canonical,
                    canonical_cluster=canonical_cluster,
                    canonical_subclass=canonical_subclass,
                    effective_expr=effective_expr
                )
    else:
        # CRITICAL: Use log-normalized data (adata_ref_log) for effective expression
        # Biophysical modeling requires gene abundance values (log-CPM), not Z-scores
        canonical_cluster, canonical_subclass, effective_expr = compute_canonical_pipeline(
            adata_ref_log,  # Use log-normalized reference (not Z-scored)
            adata_query_log,  # Use log-normalized query (not Z-scored)
            mapping_results,
            use_soft_assignment=config['use_soft_assignment'],
            k_nn=config['k_nn_expression']
        )
        if checkpoint_path_canonical:
            save_checkpoint(
                checkpoint_path_canonical,
                canonical_cluster=canonical_cluster,
                canonical_subclass=canonical_subclass,
                effective_expr=effective_expr
            )
    
    print(f"\n[{dataset_name}] Canonical expression:")
    print(f"  Cluster profiles: {canonical_cluster.shape}")
    print(f"  Subclass profiles: {canonical_subclass.shape}")
    print(f"  Query effective expression: {effective_expr.shape}")
    
    # Step 5: Save Results
    print(f"\n[{dataset_name}] Step 5: Saving Results...")
    prefix = f"{dataset_name}_"
    mapping_results.to_csv(output_dir / f'{prefix}mapping_results.csv')
    canonical_cluster.to_csv(output_dir / f'{prefix}canonical_cluster.csv')
    canonical_subclass.to_csv(output_dir / f'{prefix}canonical_subclass.csv')
    effective_expr.to_csv(output_dir / f'{prefix}effective_expression.csv')
    adata_query_proc.write(output_dir / f'{prefix}processed.h5ad')
    
    print(f"[{dataset_name}] Results saved to {output_dir}")
    print("=" * 80)


def _run_integration(
    dataset_name: str,
    adata_ref_log: ad.AnnData,
    adata_query_log: ad.AnnData,
    config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    force_rerun: bool = False
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Run reference integration step on log-normalized data."""
    if config['integration_method'] == 'scvi':
        # Check if model exists
        model_path = None
        model_loaded = False
        if model_dir:
            config_hash = get_config_hash(config)
            model_path = get_model_path(model_dir, dataset_name, 'scvi', config_hash)
        
        if model_path and model_exists(model_path) and not force_rerun:
            print(f"  Loading saved scVI model from {model_path}...")
            try:
                from scvi.model import SCVI
                
                # Try loading without AnnData first (if model was saved with AnnData)
                try:
                    model = SCVI.load(model_path)
                    print(f"  Loaded model without AnnData")
                    # Get batch categories from saved model
                    if hasattr(model, 'adata') and model.adata is not None:
                        saved_batch = model.adata.obs.get('_scvi_batch')
                        if saved_batch is not None:
                            if hasattr(saved_batch, 'cat'):
                                # Get categories and convert all to strings to avoid type mismatch
                                # The saved model might have mixed types (int/str), so we normalize to strings
                                raw_categories = saved_batch.cat.categories.tolist()
                                batch_categories = [str(cat) for cat in raw_categories]
                                print(f"  Found batch categories: {raw_categories} (converted to strings: {batch_categories})")
                            else:
                                # Get unique values and convert to strings
                                unique_vals = saved_batch.unique() if hasattr(saved_batch, 'unique') else [saved_batch.iloc[0]] if len(saved_batch) > 0 else ['batch']
                                batch_categories = [str(v) for v in unique_vals]
                        else:
                            batch_categories = ['batch']
                    else:
                        batch_categories = ['batch']
                except Exception as load_error:
                    # If loading without AnnData fails, try to load with minimal setup
                    print(f"  Loading without AnnData failed ({load_error}), trying with AnnData setup...")
                    # The issue might be that batch categories don't match
                    # Try loading with the exact same setup as training
                    adata_ref_log_setup = adata_ref_log.copy()
                    # Ensure batch is a string category
                    adata_ref_log_setup.obs['_scvi_batch'] = 'batch'
                    adata_ref_log_setup.obs['_scvi_batch'] = adata_ref_log_setup.obs['_scvi_batch'].astype('category')
                    SCVI.setup_anndata(adata_ref_log_setup, batch_key='_scvi_batch')
                    try:
                        model = SCVI.load(model_path, adata=adata_ref_log_setup)
                        batch_categories = ['batch']
                    except Exception as e2:
                        # If this also fails, the model might be corrupted or incompatible
                        raise Exception(f"Failed to load model even with AnnData setup: {e2}")
                
                print(f"  Model loaded successfully")
                print(f"  Batch categories from saved model: {batch_categories}")
                
                # The issue is that the saved model has batch categories with mixed types
                # We need to use the model's adata_manager to transfer setup correctly
                # Instead of setting up AnnData ourselves, let scvi-tools handle it via get_latent_representation
                
                # Set up reference AnnData - use the first batch category from saved model
                # All batch categories are now strings to avoid type mismatch
                adata_ref_log_setup = adata_ref_log.copy()
                batch_value = str(batch_categories[0]) if batch_categories else 'batch'
                
                # Check what batch field registry expects from saved model
                try:
                    # Get the batch field from the model's registry to see what it expects
                    batch_registry = model.adata_manager.get_state_registry().get('_scvi_batch', {})
                    # Try to get the expected categories from registry
                    if 'categorical_mapping' in batch_registry:
                        # The registry has the original batch categories
                        # Use those to set up our AnnData correctly
                        expected_categories = list(batch_registry['categorical_mapping'].keys())
                        print(f"  Expected batch categories from registry: {expected_categories}")
                        # Use the first expected category
                        if expected_categories:
                            batch_value = expected_categories[0]
                            # Ensure our batch value matches the expected type
                            if isinstance(batch_value, (int, np.integer)):
                                batch_value = int(batch_value)
                            else:
                                batch_value = str(batch_value)
                    else:
                        # Fallback to using extracted categories
                        batch_value = str(batch_categories[0]) if batch_categories else 'batch'
                except Exception as reg_error:
                    # If we can't get registry info, use extracted categories
                    print(f"  Could not get batch registry ({reg_error}), using extracted categories")
                    batch_value = str(batch_categories[0]) if batch_categories else 'batch'
                
                # Set batch field with the correct value
                adata_ref_log_setup.obs['_scvi_batch'] = batch_value
                adata_ref_log_setup.obs['_scvi_batch'] = adata_ref_log_setup.obs['_scvi_batch'].astype('category')
                
                # Set up query AnnData - use same batch category as reference
                adata_query_log_setup = adata_query_log.copy()
                adata_query_log_setup.obs['_scvi_batch'] = batch_value
                adata_query_log_setup.obs['_scvi_batch'] = adata_query_log_setup.obs['_scvi_batch'].astype('category')
                
                # Get latent representations
                # scvi-tools will automatically transfer setup from saved model
                try:
                    adata_ref_log.obsm["X_latent"] = model.get_latent_representation(adata_ref_log_setup)
                    adata_query_log.obsm["X_latent"] = model.get_latent_representation(adata_query_log_setup)
                    model_loaded = True
                except TypeError as type_error:
                    if "'<' not supported between instances" in str(type_error):
                        # Type mismatch in batch categories - model needs to be retrained
                        print(f"  ERROR: Batch category type mismatch detected in saved model")
                        print(f"  This usually happens when batch values have mixed types (int/str)")
                        print(f"  Deleting corrupted model and will retrain...")
                        import shutil
                        if model_path.exists():
                            shutil.rmtree(model_path)
                            print(f"  Deleted corrupted model at {model_path}")
                        model_loaded = False
                    else:
                        raise
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to load saved model ({e})")
                print(f"  Traceback: {traceback.format_exc()}")
                print(f"  Will retrain...")
                model_loaded = False
        
        if not model_loaded:
            print(f"Training scVI model for {dataset_name}...")
            model, adata_ref_log = train_scvi(
                adata_ref_log,
                n_latent=config['n_latent'],
                n_layers=config['n_layers'],
                n_hidden=config['n_hidden'],
                max_epochs=config['max_epochs'],
                use_gpu=config['use_gpu']
            )
            
            # Save model
            if model_path:
                print(f"  Saving scVI model to {model_path}...")
                # Save with AnnData so it can be loaded without providing AnnData
                model.save(model_path, overwrite=True, save_anndata=True)
                print(f"  Model saved successfully")
            
            # Map query to reference
            adata_ref_log, adata_query_log = map_query_scarches(
                model, adata_ref_log, adata_query_log, use_gpu=config['use_gpu']
            )

    elif config['integration_method'] == 'scanvi':
        # Check if model exists
        model_path = None
        model_loaded = False
        if model_dir:
            config_hash = get_config_hash(config)
            model_path = get_model_path(model_dir, dataset_name, 'scanvi', config_hash)
        
        if model_path and model_exists(model_path) and not force_rerun:
            print(f"  Loading saved scANVI model from {model_path}...")
            try:
                from scvi.model import SCANVI
                # Set up AnnData for scANVI (required before loading)
                adata_ref_log_setup = adata_ref_log.copy()
                if '_scvi_batch' not in adata_ref_log_setup.obs.columns:
                    adata_ref_log_setup.obs['_scvi_batch'] = 'batch'
                if '_scvi_labels' not in adata_ref_log_setup.obs.columns:
                    adata_ref_log_setup.obs['_scvi_labels'] = adata_ref_log_setup.obs.get('subclass', 'Unknown')
                SCANVI.setup_anndata(
                    adata_ref_log_setup,
                    batch_key='_scvi_batch',
                    labels_key='_scvi_labels'
                )
                
                # Load model with AnnData
                model = SCANVI.load(model_path, adata=adata_ref_log_setup)
                print(f"  Model loaded successfully")
                
                # Set up query AnnData - use same batch category as reference
                adata_query_log_setup = adata_query_log.copy()
                # Get batch category from model to match reference
                try:
                    batch_field = model.adata_manager.get_from_registry('_scvi_batch')
                    if hasattr(batch_field, 'cat') and len(batch_field.cat.categories) > 0:
                        query_batch_value = batch_field.cat.categories[0]
                    else:
                        query_batch_value = 'batch'
                except:
                    query_batch_value = 'batch'
                
                adata_query_log_setup.obs['_scvi_batch'] = query_batch_value
                adata_query_log_setup.obs['_scvi_batch'] = adata_query_log_setup.obs['_scvi_batch'].astype('category')
                if '_scvi_labels' not in adata_query_log_setup.obs.columns:
                    adata_query_log_setup.obs['_scvi_labels'] = 'Unknown'
                SCANVI.setup_anndata(
                    adata_query_log_setup,
                    batch_key='_scvi_batch',
                    labels_key='_scvi_labels'
                )
                
                # Get latent representations
                adata_ref_log.obsm["X_latent"] = model.get_latent_representation(adata_ref_log_setup)
                adata_query_log.obsm["X_latent"] = model.get_latent_representation(adata_query_log_setup)
                model_loaded = True
            except Exception as e:
                import traceback
                print(f"  Warning: Failed to load saved model ({e})")
                print(f"  Traceback: {traceback.format_exc()}")
                print(f"  Will retrain...")
                model_loaded = False
        
        if not model_loaded:
            print(f"Training scANVI model for {dataset_name}...")
            model, adata_ref_log = train_scanvi(
                adata_ref_log,
                labels_key='subclass',
                n_latent=config['n_latent'],
                n_layers=config['n_layers'],
                n_hidden=config['n_hidden'],
                max_epochs=config['max_epochs'],
                use_gpu=config['use_gpu']
            )
            
            # Save model
            if model_path:
                print(f"  Saving scANVI model to {model_path}...")
                # Save with AnnData so it can be loaded without providing AnnData
                model.save(model_path, overwrite=True, save_anndata=True)
                print(f"  Model saved successfully")
            
            # Map query to reference
            adata_ref_log, adata_query_log = map_query_scarches(
                model, adata_ref_log, adata_query_log, use_gpu=config['use_gpu']
            )

    elif config['integration_method'] == 'pca_harmony':
        print(f"Using PCA + Harmony for {dataset_name}...")
        adata_ref_log, adata_query_log = integrate_pca_harmony(
            adata_ref_log, adata_query_log,
            n_comps=config['n_latent']
        )
    else:
        raise ValueError(f"Unknown integration method: {config['integration_method']}")

    if checkpoint_path:
        save_checkpoint(
            checkpoint_path,
            adata_ref=adata_ref_log,
            adata_query=adata_query_log
        )

    return adata_ref_log, adata_query_log


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Atlas-Grounded Biophysical Modeling Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Path arguments
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
        help='Output directory (default: project_root/atlas_matching/results)'
    )
    
    # Dataset selection
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['V1', 'M1'],
        default=['V1', 'M1'],
        help='Datasets to process'
    )
    
    # Preprocessing options
    parser.add_argument(
        '--n-hvgs',
        type=int,
        default=3000,
        help='Number of highly variable genes'
    )
    parser.add_argument(
        '--hvg-flavor',
        choices=['seurat_v3', 'seurat', 'cell_ranger'],
        default='seurat_v3',
        help='HVG selection method'
    )
    parser.add_argument(
        '--ref-is-log2',
        action='store_true',
        default=True,
        help='Reference is already log2(CPM+1) transformed'
    )
    
    # Integration options
    parser.add_argument(
        '--integration-method',
        choices=['scvi', 'scanvi', 'pca_harmony'],
        default='scvi',
        help='Reference integration method'
    )
    parser.add_argument(
        '--n-latent',
        type=int,
        default=50,
        help='Latent dimension (default: 50 for high-complexity atlas)'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=2,
        help='Number of hidden layers'
    )
    parser.add_argument(
        '--n-hidden',
        type=int,
        default=256,
        help='Number of hidden units per layer (default: 256 for increased capacity)'
    )
    parser.add_argument(
        '--n-hvgs-query',
        type=int,
        default=500,
        help='Number of HVGs to select from query for union strategy (default: 500)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=400,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    # Mapping options
    parser.add_argument(
        '--k-candidates',
        type=int,
        default=15,
        help='Top-K candidate clusters in Stage 1'
    )
    parser.add_argument(
        '--k-nn',
        type=int,
        default=20,
        help='KNN neighbors in Stage 2'
    )
    parser.add_argument(
        '--conf-threshold-subclass',
        type=float,
        default=0.7,
        help='Confidence threshold for subclass assignment'
    )
    parser.add_argument(
        '--conf-threshold-cluster',
        type=float,
        default=0.7,
        help='Confidence threshold for cluster assignment'
    )
    
    # Canonical expression options
    parser.add_argument(
        '--use-soft-assignment',
        action='store_true',
        default=True,
        help='Use soft assignment for query effective expression'
    )
    parser.add_argument(
        '--k-nn-expression',
        type=int,
        default=20,
        help='Number of neighbors for soft assignment'
    )
    
    # Data loading options
    parser.add_argument(
        '--subsample-atlas',
        type=int,
        default=None,
        help='Subsample atlas to N cells'
    )
    parser.add_argument(
        '--load-ephys',
        action='store_true',
        help='Load ephys features'
    )
    
    # Checkpointing options
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=None,
        help='Checkpoint directory (default: output_dir/checkpoints)'
    )
    parser.add_argument(
        '--no-checkpoints',
        action='store_true',
        help='Disable checkpointing'
    )
    parser.add_argument(
        '--force-rerun',
        action='store_true',
        help='Force rerun even if checkpoints exist'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = args.project_root
    data_dir = args.data_dir or (project_root / 'dat' / 'expression')
    atlas_dir = args.atlas_dir or (data_dir / 'reference_atlas')
    output_dir = args.output_dir or (ATLAS_MATCHING_ROOT / 'results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up checkpoint directory
    checkpoint_dir = args.checkpoint_dir or (output_dir / 'checkpoints')
    use_checkpoints = not args.no_checkpoints
    
    # Set up model directory (for saving trained scVI/scANVI models)
    model_dir = checkpoint_dir.parent / 'models' if checkpoint_dir else (output_dir / 'models')
    
    # Build configuration
    config = {
        'n_hvgs': args.n_hvgs,
        'n_hvgs_query': getattr(args, 'n_hvgs_query', 500),  # Union HVG strategy
        'hvg_flavor': args.hvg_flavor,
        'ref_is_log2': args.ref_is_log2,
        'integration_method': args.integration_method,
        'n_latent': args.n_latent,
        'n_layers': args.n_layers,
        'n_hidden': args.n_hidden,
        'max_epochs': args.max_epochs,
        'use_gpu': not args.no_gpu,
        'k_candidates': args.k_candidates,
        'k_nn': args.k_nn,
        'conf_threshold_subclass': args.conf_threshold_subclass,
        'conf_threshold_cluster': args.conf_threshold_cluster,
        'use_soft_assignment': args.use_soft_assignment,
        'k_nn_expression': args.k_nn_expression,
        'subsample_atlas': args.subsample_atlas,
        'load_ephys': args.load_ephys,
    }
    
    # Print configuration
    print("=" * 80)
    print("ATLAS-GROUNDED BIOPHYSICAL MODELING PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nPaths:")
    print(f"  Project root: {project_root}")
    print(f"  Data directory: {data_dir}")
    print(f"  Atlas directory: {atlas_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print(f"  Model directory: {model_dir}")
    print(f"  Use checkpoints: {use_checkpoints}")
    print(f"  Force rerun: {args.force_rerun}")
    print(f"  Datasets: {args.datasets}")
    print("=" * 80)
    
    # Scanpy settings
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    
    # Step 1: Load reference atlas
    print("\n" + "=" * 80)
    print("STEP 1: LOAD DATA")
    print("=" * 80)
    
    print("\nLoading reference atlas...")
    adata_ref = load_reference_atlas(
        atlas_dir,
        use_log2=config['ref_is_log2'],
        subsample=config['subsample_atlas']
    )
    print(f"\nReference atlas loaded:")
    print(f"  Shape: {adata_ref.shape}")
    print(f"  Obs columns: {list(adata_ref.obs.columns)}")
    
    # Load ion channel genes
    print("\nLoading ion channel gene lists...")
    ion_channel_genes_dict = load_ion_channel_genes(data_dir, source='both')
    ion_channel_genes = ion_channel_genes_dict['all']
    print(f"Ion channel genes loaded: {len(ion_channel_genes)} total")
    
    # Load and concatenate Patch-seq datasets for joint processing
    print("\nLoading Patch-seq datasets...")
    query_datasets = []
    query_metadata_list = []
    
    for dataset_name in args.datasets:
        print(f"\nLoading {dataset_name} Patch-seq data...")
        
        if dataset_name == 'V1':
            adata_query_single, metadata, ephys = load_patchseq_v1(
                data_dir,
                load_ephys=config['load_ephys']
            )
        elif dataset_name == 'M1':
            adata_query_single, metadata, ephys = load_patchseq_m1(
                data_dir,
                load_ephys=config['load_ephys']
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        print(f"{dataset_name} shape: {adata_query_single.shape}")
        query_datasets.append(adata_query_single)
        query_metadata_list.append(metadata)
    
    # Concatenate all query datasets into a single AnnData object
    if len(query_datasets) > 1:
        print(f"\nConcatenating {len(query_datasets)} Patch-seq datasets...")
        # Ensure tech_batch is preserved during concatenation
        adata_query = ad.concat(query_datasets, join="outer", index_unique=None)
        print(f"  Combined query shape: {adata_query.shape}")
        print(f"  tech_batch categories: {adata_query.obs['tech_batch'].cat.categories.tolist()}")
    else:
        adata_query = query_datasets[0]
    
    # Process joint query dataset
    # Use "Joint" as dataset name for joint processing
    dataset_name = "Joint" if len(args.datasets) > 1 else args.datasets[0]
    
    process_dataset(
        dataset_name,
        adata_ref,
        adata_query,
        ion_channel_genes,
        config,
        output_dir,
        checkpoint_dir=checkpoint_dir,
        use_checkpoints=use_checkpoints,
        force_rerun=args.force_rerun,
        model_dir=model_dir
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutputs available:")
    print("  1. Mapping results: Cell-level assignments to subclass/cluster with confidence scores")
    print("  2. Canonical profiles: Cluster and subclass-level expression profiles")
    print("  3. Effective expression: Query cell effective expression (soft assignment)")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
