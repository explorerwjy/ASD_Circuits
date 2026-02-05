#!/usr/bin/env python3
"""
Step 2b: Train scANVI model on reference atlas (SEMI-SUPERVISED)

This script trains scANVI instead of scVI for better cell type separation.

Key difference from scVI:
- scVI: Unsupervised - learns from expression only
- scANVI: Semi-supervised - learns from expression + reference labels
- Result: Better cell type separation, prevents neuron→glia errors

Output: Trained scANVI model saved to results/models/
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

# Add module to path (02_train_scanvi.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))


def train_scanvi(
    adata_ref: ad.AnnData,
    labels_key: str = 'subclass',
    n_latent: int = 50,
    n_layers: int = 2,
    n_hidden: int = 256,
    batch_key: str = 'tech_batch',
    use_gpu: bool = True,
    max_epochs: int = 400,
    batch_size: int = 512
):
    """
    Train scANVI model (semi-supervised) following scvi-tools best practices.

    This creates an untrained scVI model, wraps it with scANVI, and trains
    the combined model. This is the recommended approach for scANVI training.

    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas with raw counts in layers["counts"]
    labels_key : str
        Column in adata_ref.obs with cell type labels (default: 'subclass')
    n_latent : int
        Latent dimension (default: 50)
    n_layers : int
        Number of hidden layers (default: 2)
    n_hidden : int
        Hidden units per layer (default: 256)
    batch_key : str
        Column for batch correction (default: 'tech_batch')
    use_gpu : bool
        Use GPU if available
    max_epochs : int
        Training epochs (default: 400)
    batch_size : int
        Training batch size (default: 512)

    Returns
    -------
    SCANVI
        Trained scANVI model
    ad.AnnData
        Reference with latent embedding
    """
    import scvi
    from scvi.model import SCVI, SCANVI

    print("=" * 80)
    print("SCANVI TRAINING (SEMI-SUPERVISED)")
    print("=" * 80)

    # Validate labels exist
    if labels_key not in adata_ref.obs.columns:
        raise ValueError(
            f"Label column '{labels_key}' not found in adata_ref.obs!\n"
            f"Available columns: {list(adata_ref.obs.columns)}"
        )

    n_labels = adata_ref.obs[labels_key].nunique()
    print(f"\nCell type labels:")
    print(f"  Column: {labels_key}")
    print(f"  Number of types: {n_labels}")
    print(f"  Examples: {adata_ref.obs[labels_key].unique()[:5].tolist()}")

    # Check for counts layer
    if "counts" not in adata_ref.layers:
        raise ValueError(
            "adata_ref must have layers['counts'] with raw UMI counts."
        )

    # Make a copy to avoid modifying original
    adata_ref = adata_ref.copy()

    # Setup batch key - CRITICAL: Ensure batch categories are PURE strings
    if batch_key and batch_key in adata_ref.obs.columns:
        batch_values = [str(x) for x in adata_ref.obs[batch_key].values]
        adata_ref.obs['_scvi_batch'] = pd.Categorical(batch_values)
        print(f"\nBatch correction:")
        print(f"  Using: {batch_key}")
        print(f"  Batch categories (as strings): {adata_ref.obs['_scvi_batch'].cat.categories.tolist()}")
    elif 'tech_batch' in adata_ref.obs.columns:
        batch_values = [str(x) for x in adata_ref.obs['tech_batch'].values]
        adata_ref.obs['_scvi_batch'] = pd.Categorical(batch_values)
        print(f"\nBatch correction using: tech_batch")
        print(f"  Batch categories (as strings): {adata_ref.obs['_scvi_batch'].cat.categories.tolist()}")
    else:
        adata_ref.obs['_scvi_batch'] = 'batch'
        adata_ref.obs['_scvi_batch'] = adata_ref.obs['_scvi_batch'].astype('category')
        print(f"\nBatch correction: Using default 'batch'")

    # Verify batch categories are all strings
    cats = adata_ref.obs['_scvi_batch'].cat.categories
    cat_types = set(type(x).__name__ for x in cats)
    if cat_types != {'str'}:
        raise TypeError(f"Batch categories must be strings, found types: {cat_types}")
    print(f"  ✓ Verified all batch categories are strings")

    # Setup labels - CRITICAL: Convert to string first
    label_values = [str(x) for x in adata_ref.obs[labels_key].values]
    adata_ref.obs['_scvi_labels'] = pd.Categorical(label_values)

    # Verify label categories are all strings
    label_cats = adata_ref.obs['_scvi_labels'].cat.categories
    label_types = set(type(x).__name__ for x in label_cats)
    if label_types != {'str'}:
        raise TypeError(f"Label categories must be strings, found types: {label_types}")
    print(f"  ✓ Verified all label categories are strings")
    print(f"  Number of cell type labels: {len(label_cats)}")

    print(f"\nModel architecture:")
    print(f"  Latent dimension: {n_latent}")
    print(f"  Hidden layers: {n_layers} × {n_hidden} units")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Batch size: {batch_size}")

    # Setup scANVI - this registers the labels field properly
    print("\n  Setting up scANVI...")
    SCANVI.setup_anndata(
        adata_ref,
        batch_key='_scvi_batch',
        labels_key='_scvi_labels',
        unlabeled_category='Unknown',
        layer='counts'
    )

    # Create scANVI model from untrained scVI (this is the proper way)
    # The scVI model is created internally and never trained separately
    print("  Creating scANVI model...")
    scanvi_model = SCANVI(
        adata_ref,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="nb"  # Negative Binomial for count data
    )

    # Configure training parameters
    train_kwargs = {
        'max_epochs': max_epochs,
        'train_size': 0.9,
        'batch_size': batch_size,
        'early_stopping': True,
        'early_stopping_patience': 20,
        'check_val_every_n_epoch': 5,
        'enable_progress_bar': True,
    }
    if use_gpu:
        train_kwargs['accelerator'] = 'gpu'
        train_kwargs['devices'] = 'auto'
    else:
        train_kwargs['accelerator'] = 'cpu'

    # Train scANVI
    print("\n  Training scANVI...")
    print("  (This trains both the VAE and the cell type classifier together)")
    scanvi_model.train(**train_kwargs)

    # Print training summary
    history = scanvi_model.history
    if history is not None:
        try:
            if isinstance(history, dict):
                elbo_train = history.get('elbo_train', [])
                elbo_val = history.get('elbo_validation', elbo_train)
            else:
                elbo_train = history['elbo_train'].values if 'elbo_train' in history else []
                elbo_val = history['elbo_validation'].values if 'elbo_validation' in history else elbo_train

            if len(elbo_train) > 0:
                print(f"\n  ✓ scANVI training complete")
                print(f"    Epochs: {len(elbo_train)}")
                print(f"    Final train ELBO: {elbo_train[-1]:.2f}")
                if len(elbo_val) > 0:
                    print(f"    Final val ELBO: {elbo_val[-1]:.2f}")
        except Exception:
            print(f"\n  ✓ scANVI training complete")

    # Get latent representation
    adata_ref.obsm["X_latent"] = scanvi_model.get_latent_representation(adata_ref)

    print(f"\n  Latent embedding shape: {adata_ref.obsm['X_latent'].shape}")
    print(f"\n  Key benefits of scANVI:")
    print(f"    ✓ Explicit cell type separation in latent space")
    print(f"    ✓ Better label transfer accuracy")
    print(f"    ✓ Prevents neuron→glia mapping errors")

    return scanvi_model, adata_ref


def main():
    parser = argparse.ArgumentParser(
        description='Step 2b: Train scANVI model (semi-supervised)',
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
        '--model-dir',
        type=Path,
        default=None,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='Model name (default: scanvi_reference_latent{n_latent}_hidden{n_hidden})'
    )

    # scANVI parameters
    parser.add_argument(
        '--labels-key',
        type=str,
        default='subclass',
        help='Column name for cell type labels in reference'
    )
    parser.add_argument(
        '--n-latent',
        type=int,
        default=50,
        help='Latent dimension'
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
        help='Hidden units per layer'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=400,
        help='Training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
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

    # Legacy arguments (ignored but accepted for compatibility)
    parser.add_argument('--max-epochs-scvi', type=int, default=200, help='(Ignored - for compatibility)')
    parser.add_argument('--max-epochs-scanvi', type=int, default=200, help='(Ignored - for compatibility)')

    args = parser.parse_args()

    # Setup paths
    project_root = args.project_root
    preprocessed_dir = args.preprocessed_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'preprocessed'
    )
    model_dir = args.model_dir or (
        ATLAS_MATCHING_ROOT / 'results' / 'models'
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"scanvi_reference_latent{args.n_latent}_hidden{args.n_hidden}"
    model_path = model_dir / model_name

    # Check if model exists
    model_exists = model_path.exists() and (model_path / "model.pt").exists()

    print("=" * 80)
    print("STEP 2b: TRAIN scANVI ON REFERENCE ATLAS (SEMI-SUPERVISED)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Labels key: {args.labels_key}")
    print(f"  Latent dim: {args.n_latent}")
    print(f"  Architecture: {args.n_layers} layers × {args.n_hidden} units")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Use GPU: {not args.no_gpu}")
    print(f"\nModel status:")
    if args.force_retrain:
        print(f"  Mode: RETRAIN (--force-retrain flag set)")
    elif model_exists:
        print(f"  Mode: SKIP (model already exists)")
        print(f"  Found trained model at: {model_path}")
        print(f"  Use --force-retrain to retrain from scratch")
        print(f"\n✓ Existing model found. Skipping training.")
        return
    else:
        print(f"  Mode: TRAIN NEW MODEL")
        print(f"  No existing model found at: {model_path}")
    print(f"\nPaths:")
    print(f"  Preprocessed dir: {preprocessed_dir}")
    print(f"  Model dir: {model_dir}")
    print(f"  Model name: {model_name}")
    print("=" * 80)

    # Load preprocessed reference
    print(f"\n[LOAD] Loading preprocessed reference atlas...")
    ref_path = preprocessed_dir / 'reference_preprocessed.h5ad'

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference not found: {ref_path}\n"
            f"Run 01_preprocess_data.py first"
        )

    adata_ref = sc.read_h5ad(ref_path)
    print(f"  Reference: {adata_ref.shape}")

    # Check for counts layer
    if "counts" not in adata_ref.layers:
        print("  Warning: No counts layer, using .X as counts")
        adata_ref.layers["counts"] = adata_ref.X.copy()

    # Train scANVI
    print(f"\n[TRAIN] Training scANVI model...")
    model, adata_ref = train_scanvi(
        adata_ref,
        labels_key=args.labels_key,
        n_latent=args.n_latent,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu
    )

    # Save model
    print(f"\n[SAVE] Saving trained scANVI model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path, overwrite=True, save_anndata=True)

    # Fix saved adata (scANVI also encodes categoricals as int8)
    print(f"\n  Fixing saved adata (scANVI encodes categoricals as int8)...")
    saved_adata_path = model_path / "adata.h5ad"
    saved_adata = ad.read_h5ad(saved_adata_path)

    # Get original categories from our working copy
    original_batch_cats = adata_ref.obs['_scvi_batch'].cat.categories.tolist()
    original_label_cats = adata_ref.obs['_scvi_labels'].cat.categories.tolist()

    # Fix _scvi_batch
    if '_scvi_batch' in saved_adata.obs.columns:
        if saved_adata.obs['_scvi_batch'].dtype == 'int8':
            saved_adata.obs['_scvi_batch'] = pd.Categorical.from_codes(
                saved_adata.obs['_scvi_batch'].values,
                categories=original_batch_cats
            )
            print(f"    ✓ Fixed _scvi_batch: int8 → categorical strings")

    # Fix _scvi_labels
    if '_scvi_labels' in saved_adata.obs.columns:
        if saved_adata.obs['_scvi_labels'].dtype == 'int8':
            saved_adata.obs['_scvi_labels'] = pd.Categorical.from_codes(
                saved_adata.obs['_scvi_labels'].values,
                categories=original_label_cats
            )
            print(f"    ✓ Fixed _scvi_labels: int8 → categorical strings ({len(original_label_cats)} categories)")

    # Save fixed adata
    saved_adata.write(saved_adata_path)
    print(f"  ✓ Fixed and saved")

    print(f"\n✅ Training complete!")
    print(f"  Model saved to: {model_path}")
    print(f"  Model type: scANVI (semi-supervised)")
    print(f"\nNext step: Run 03_map_query.py to map Patch-seq data")
    print(f"  python atlas_matching/scripts/03_map_query.py \\")
    print(f"      --dataset M1 V1 \\")
    print(f"      --model-name {model_name}")


if __name__ == "__main__":
    main()
