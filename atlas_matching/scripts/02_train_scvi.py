#!/usr/bin/env python3
"""
Step 2: Train scVI model on reference atlas

This script:
1. Loads preprocessed reference atlas (raw counts)
2. Trains scVI model on reference ONLY using raw counts and Negative Binomial likelihood
3. Saves trained model for later query mapping

CRITICAL: scVI is trained ONLY on the reference atlas.
Query data will be mapped using scArches (surgery) in step 03.

Output: Trained scVI model saved to results/models/
"""

import sys
import argparse
import warnings
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

warnings.filterwarnings('ignore')


def load_config(config_path: Path, preset: str = None) -> dict:
    """
    Load model configuration from YAML file.

    Args:
        config_path: Path to YAML config file
        preset: Optional preset name to use (e.g., 'quick', 'large')

    Returns:
        Dictionary with flattened config parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # If preset is specified, merge preset values into model/training sections
    if preset and preset in config.get('presets', {}):
        preset_config = config['presets'][preset]
        print(f"  Using preset: {preset}")
        for key, value in preset_config.items():
            # Map preset keys to their sections
            if key in ['n_latent', 'n_layers', 'n_hidden', 'dropout_rate',
                       'gene_likelihood', 'use_batch_norm', 'use_layer_norm']:
                config.setdefault('model', {})[key] = value
            elif key in ['max_epochs', 'batch_size', 'learning_rate',
                         'early_stopping', 'early_stopping_patience']:
                config.setdefault('training', {})[key] = value

    # Flatten config for easy access
    flat_config = {}

    # Model parameters
    model_cfg = config.get('model', {})
    flat_config['n_latent'] = model_cfg.get('n_latent', 50)
    flat_config['n_layers'] = model_cfg.get('n_layers', 2)
    flat_config['n_hidden'] = model_cfg.get('n_hidden', 256)
    flat_config['dropout_rate'] = model_cfg.get('dropout_rate', 0.1)
    flat_config['gene_likelihood'] = model_cfg.get('gene_likelihood', 'nb')
    flat_config['use_batch_norm'] = model_cfg.get('use_batch_norm', 'both')
    flat_config['use_layer_norm'] = model_cfg.get('use_layer_norm', 'none')

    # Training parameters
    train_cfg = config.get('training', {})
    flat_config['max_epochs'] = train_cfg.get('max_epochs', 400)
    flat_config['batch_size'] = train_cfg.get('batch_size', 512)
    flat_config['learning_rate'] = train_cfg.get('learning_rate', 0.001)
    flat_config['early_stopping'] = train_cfg.get('early_stopping', True)
    flat_config['early_stopping_patience'] = train_cfg.get('early_stopping_patience', 50)
    flat_config['seed'] = train_cfg.get('seed', 42)

    # Hardware parameters
    hw_cfg = config.get('hardware', {})
    flat_config['use_gpu'] = hw_cfg.get('use_gpu', True)

    return flat_config

# Add module to path (02_train_scvi.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))

from patch_seq_transcriptome_mapping import train_scvi


def main():
    parser = argparse.ArgumentParser(
        description='Step 2: Train scVI model on reference atlas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file options
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to YAML config file for model parameters'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default=None,
        choices=['default', 'quick', 'large', 'memory_efficient', 'patchseq'],
        help='Use a preset configuration from config file'
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
        '--model-dir',
        type=Path,
        default=None,
        help='Directory to save trained models (default: results/models)'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (default: scvi_reference_latent{n_latent}_hidden{n_hidden})'
    )

    # scVI parameters (can be overridden by CLI, defaults from config if provided)
    parser.add_argument(
        '--n-latent',
        type=int,
        default=None,
        help='Latent dimension (default: 50 for high-complexity atlas)'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=None,
        help='Number of hidden layers'
    )
    parser.add_argument(
        '--n-hidden',
        type=int,
        default=None,
        help='Hidden units per layer (default: 256 for increased capacity)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=None,
        help='Max training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU'
    )

    # Model management
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain even if model exists'
    )

    args = parser.parse_args()

    # Load config file if provided
    config = {}
    if args.config:
        if not args.config.exists():
            raise FileNotFoundError(f"Config file not found: {args.config}")
        print(f"[CONFIG] Loading configuration from: {args.config}")
        config = load_config(args.config, args.preset)
    elif args.preset:
        # If preset specified but no config, try default config location
        default_config = ATLAS_MATCHING_ROOT / 'configs' / 'scvi_model_config.yaml'
        if default_config.exists():
            print(f"[CONFIG] Loading default config with preset '{args.preset}': {default_config}")
            config = load_config(default_config, args.preset)
        else:
            print(f"Warning: --preset specified but no config file found at {default_config}")

    # Merge config with CLI args (CLI takes precedence)
    # Use config values as defaults, CLI overrides if provided
    n_latent = args.n_latent if args.n_latent is not None else config.get('n_latent', 50)
    n_layers = args.n_layers if args.n_layers is not None else config.get('n_layers', 2)
    n_hidden = args.n_hidden if args.n_hidden is not None else config.get('n_hidden', 256)
    max_epochs = args.max_epochs if args.max_epochs is not None else config.get('max_epochs', 400)
    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 512)
    use_gpu = not args.no_gpu and config.get('use_gpu', True)

    # Setup paths
    project_root = args.project_root
    preprocessed_dir = args.preprocessed_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'preprocessed'
    )
    model_dir = args.model_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'models'
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine model path
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"scvi_reference_latent{n_latent}_hidden{n_hidden}"
    model_path = model_dir / model_name

    # Check if model exists
    model_exists = model_path.exists() and (model_path / "model.pt").exists()

    print("=" * 80)
    print("STEP 2: TRAIN SCVI ON REFERENCE ATLAS")
    print("=" * 80)
    if args.config:
        print(f"\nConfig source: {args.config}")
        if args.preset:
            print(f"  Preset: {args.preset}")
    print(f"\nConfiguration:")
    print(f"  Latent dim: {n_latent}")
    print(f"  Architecture: {n_layers} layers × {n_hidden} units")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Use GPU: {use_gpu}")
    print(f"\nModel status:")
    if args.force_retrain:
        print(f"  Mode: RETRAIN (--force-retrain flag set)")
        print(f"  Will train new model even if one exists")
    elif model_exists:
        print(f"  Mode: SKIP (model already exists)")
        print(f"  Found trained model at: {model_path}")
        print(f"  Use --force-retrain to retrain from scratch")
        print(f"\n✓ Existing model found. Skipping training.")
        print(f"  To retrain, use: --force-retrain")
        print(f"  To train with different params, specify --model-name")
        return
    else:
        print(f"  Mode: TRAIN NEW MODEL")
        print(f"  No existing model found at: {model_path}")
        print(f"  Will train new model")
    print(f"\nPaths:")
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Model dir: {model_dir}")
    print(f"  Model name: {model_name}")
    print("=" * 80)

    # Load preprocessed reference atlas
    print(f"\n[LOAD] Loading preprocessed reference atlas...")
    ref_path = preprocessed_dir / 'reference_preprocessed.h5ad'

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference not found: {ref_path}\n"
            f"Run 01_preprocess_data.py first"
        )

    adata_ref = sc.read_h5ad(ref_path)

    print(f"  Reference: {adata_ref.shape}")
    print(f"  Reference .X dtype: {adata_ref.X.dtype}")

    # Check for counts layer (scVI needs it)
    if "counts" not in adata_ref.layers:
        print("  Warning: No counts layer in reference, using .X as counts")
        adata_ref.layers["counts"] = adata_ref.X.copy()
    else:
        print(f"  ✓ Reference has layers['counts']")

    # Train scVI model on reference ONLY
    print(f"\n[TRAIN] Training scVI model on reference atlas...")
    print(f"  Parameters: latent={n_latent}, layers={n_layers}, "
          f"hidden={n_hidden}, max_epochs={max_epochs}")
    print(f"  Training on REFERENCE ONLY (query will be mapped later)")

    model, adata_ref = train_scvi(
        adata_ref,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        max_epochs=max_epochs,
        batch_size=batch_size,
        use_gpu=use_gpu
    )

    # Save model
    print(f"\n[SAVE] Saving trained model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, overwrite=True, save_anndata=True)

    # Fix saved adata (scVI saves categorical as int8 codes)
    print(f"\n  Fixing saved adata (scVI encodes categoricals as int8)...")
    saved_adata_path = model_path / "adata.h5ad"
    saved_adata = ad.read_h5ad(saved_adata_path)

    # Fix _scvi_batch: Convert int8 back to categorical strings
    if '_scvi_batch' in saved_adata.obs.columns:
        batch_dtype = saved_adata.obs['_scvi_batch'].dtype
        print(f"    Found _scvi_batch dtype: {batch_dtype}")

        if batch_dtype == 'int8' or not hasattr(saved_adata.obs['_scvi_batch'], 'cat'):
            # scVI encoded as int8 - convert back to categorical
            if 'tech_batch' in saved_adata.obs.columns:
                print(f"    Converting int8 codes back to categorical using tech_batch...")
                saved_adata.obs['_scvi_batch'] = saved_adata.obs['tech_batch'].astype(str).astype('category')
            else:
                # Map int codes to category names
                print(f"    Converting int8 codes to categorical...")
                # Get unique int values and create string categories
                if hasattr(adata_ref.obs['_scvi_batch'], 'cat'):
                    original_cats = adata_ref.obs['_scvi_batch'].cat.categories.tolist()
                    saved_adata.obs['_scvi_batch'] = pd.Categorical.from_codes(
                        saved_adata.obs['_scvi_batch'].values,
                        categories=original_cats
                    )
                else:
                    # Fallback: just convert codes to strings
                    saved_adata.obs['_scvi_batch'] = saved_adata.obs['_scvi_batch'].astype(str).astype('category')

            # Verify the fix
            cats = saved_adata.obs['_scvi_batch'].cat.categories
            cat_types = set(type(x).__name__ for x in cats)
            print(f"    After fix - dtype: {saved_adata.obs['_scvi_batch'].dtype}")
            print(f"    Categories: {cats.tolist()}")
            print(f"    Category types: {cat_types}")

            if cat_types != {'str'}:
                raise TypeError(f"ERROR: Still have mixed types after fix: {cat_types}")

            # Save the fixed adata
            print(f"    Saving fixed adata...")
            saved_adata.write(saved_adata_path)
            print(f"    ✓ Fixed and saved")
        else:
            # Already categorical with string categories
            cats = saved_adata.obs['_scvi_batch'].cat.categories
            cat_types = set(type(x).__name__ for x in cats)
            print(f"    Categories: {cats.tolist()}")
            print(f"    Category types: {cat_types}")
            if cat_types != {'str'}:
                raise TypeError(f"ERROR: Saved model has mixed types in categories: {cat_types}")
            print(f"    ✓ Already has string categories")

    print(f"  ✓ Model saved successfully!")
    print(f"\n✅ Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Next step: Run 03_map_query.py to map Patch-seq data")


if __name__ == "__main__":
    main()
