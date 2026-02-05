"""
Reference integration and query mapping using scVI/scANVI + scArches.

This module implements:
- Track A: scVI/scANVI + scArches (recommended)
- Track B: PCA + Harmony (baseline)
"""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple, Any, Union
from pathlib import Path
import warnings

# Type aliases for optional imports
SCVI_TYPE = Any
SCANVI_TYPE = Any

try:
    import scvi
    from scvi.model import SCVI, SCANVI
    SCVI_AVAILABLE = True
    SCVI_TYPE = SCVI
    SCANVI_TYPE = SCANVI
except ImportError:
    SCVI_AVAILABLE = False
    warnings.warn("scvi-tools not available. Install with: pip install scvi-tools")

try:
    from harmonypy import run_harmony
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    warnings.warn("harmonypy not available. Install with: pip install harmonypy")


def train_scvi(
    adata_ref: ad.AnnData,
    n_latent: int = 50,
    n_layers: int = 2,
    n_hidden: int = 256,
    batch_key: Optional[str] = None,
    use_gpu: bool = True,
    max_epochs: int = 400,
    train_size: float = 0.9,
    batch_size: int = 512
) -> Tuple[SCVI_TYPE, ad.AnnData]:
    """
    Train scVI model on reference atlas using raw counts and Negative Binomial likelihood.
    
    CRITICAL: Uses layer="counts" for raw UMI counts and gene_likelihood="nb" for proper
    count modeling. Increased model capacity (n_hidden=256, n_latent=50) for high-complexity
    Isocortex atlas.

    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (must have layers["counts"] with raw UMI counts)
    n_latent : int
        Latent dimension (default: 50 for high-complexity atlas)
    n_layers : int
        Number of hidden layers
    n_hidden : int
        Number of hidden units per layer (default: 256 for increased capacity)
    batch_key : str, optional
        Column name in adata_ref.obs for batch correction (default: 'tech_batch')
    use_gpu : bool
        Use GPU if available
    max_epochs : int
        Maximum training epochs
    train_size : float
        Fraction of data for training (rest for validation)
    batch_size : int
        Training batch size (increase for faster GPU training)

    Returns
    -------
    SCVI
        Trained scVI model
    ad.AnnData
        Reference AnnData with latent embedding in obsm['X_latent']
    """
    if not SCVI_AVAILABLE:
        raise ImportError("scvi-tools not available. Install with: pip install scvi-tools")

    print(f"Training scVI model with Negative Binomial likelihood...")
    print(f"  Latent dimension: {n_latent}")
    print(f"  Architecture: {n_layers} layers, {n_hidden} hidden units")
    print(f"  Using layer='counts' for raw UMI counts")
    print(f"  gene_likelihood='nb' for count modeling")
    
    # Check for counts layer
    if "counts" not in adata_ref.layers:
        raise ValueError(
            "adata_ref must have layers['counts'] with raw UMI counts. "
            "Use data_loader.load_reference_atlas() or ensure counts layer exists."
        )

    # Setup AnnData for scVI (make a copy to avoid modifying original)
    adata_ref = adata_ref.copy()

    # Set up batch key - use tech_batch if available, otherwise use provided batch_key
    # CRITICAL: Ensure batch categories are PURE strings (no int/str mixing)
    if batch_key and batch_key in adata_ref.obs.columns:
        # Convert to string to avoid type mismatches (int/str mixing)
        # Use explicit list comprehension for clean conversion
        batch_values = [str(x) for x in adata_ref.obs[batch_key].values]
        adata_ref.obs['_scvi_batch'] = pd.Categorical(batch_values)
        print(f"  Batch correction using: {batch_key}")
        print(f"  Batch categories (as strings): {adata_ref.obs['_scvi_batch'].cat.categories.tolist()}")
    elif 'tech_batch' in adata_ref.obs.columns:
        # Use explicit list comprehension for clean conversion
        batch_values = [str(x) for x in adata_ref.obs['tech_batch'].values]
        adata_ref.obs['_scvi_batch'] = pd.Categorical(batch_values)
        print(f"  Batch correction using: tech_batch")
        print(f"  Batch categories (as strings): {adata_ref.obs['_scvi_batch'].cat.categories.tolist()}")
    else:
        # Always use string 'batch' to ensure consistency
        adata_ref.obs['_scvi_batch'] = 'batch'
        adata_ref.obs['_scvi_batch'] = adata_ref.obs['_scvi_batch'].astype('category')
        print(f"  Using default batch: 'batch'")

    # Verify categories are all strings (safety check)
    cats = adata_ref.obs['_scvi_batch'].cat.categories
    cat_types = set(type(x).__name__ for x in cats)
    if cat_types != {'str'}:
        raise TypeError(f"Batch categories must be strings, found types: {cat_types}")
    print(f"  ✓ Verified all batch categories are strings")

    # Prepare data - CRITICAL: use layer="counts" for raw counts
    SCVI.setup_anndata(
        adata_ref,
        batch_key='_scvi_batch',
        layer='counts'  # Use raw counts layer
    )

    # Create and train model
    # CRITICAL: Use gene_likelihood="nb" for Negative Binomial (proper count modeling)
    # This is required for robust batch correction with discrete count data
    model = SCVI(
        adata_ref,
        n_latent=n_latent,
        n_layers=n_layers,
        n_hidden=n_hidden,
        gene_likelihood="nb"  # Negative Binomial for count data
    )

    # Configure training parameters for newer scvi-tools/PyTorch Lightning
    train_kwargs = {
        'max_epochs': max_epochs,
        'train_size': train_size,
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

    model.train(**train_kwargs)

    # Print training summary
    history = model.history
    if history is not None and hasattr(history, 'columns') and 'elbo_validation' in history.columns:
        final_train = float(history['elbo_train'].iloc[-1])
        final_val = float(history['elbo_validation'].iloc[-1])
        print(f"  Final ELBO - Train: {final_train:.2f}, Validation: {final_val:.2f}")
        print(f"  Epochs completed: {len(history['elbo_train'])}")
    elif history is not None:
        # Try alternative history format
        try:
            if isinstance(history, dict):
                final_train = float(history.get('elbo_train', [0])[-1] if isinstance(history.get('elbo_train'), list) else history.get('elbo_train', 0))
                final_val = float(history.get('elbo_validation', [0])[-1] if isinstance(history.get('elbo_validation'), list) else history.get('elbo_validation', 0))
                print(f"  Final ELBO - Train: {final_train:.2f}, Validation: {final_val:.2f}")
            else:
                print(f"  Training completed")
        except (TypeError, AttributeError, IndexError, ValueError):
            print(f"  Training completed (history format not recognized)")

    # Get latent representation
    adata_ref.obsm["X_latent"] = model.get_latent_representation(adata_ref)

    print(f"  Training complete. Latent embedding shape: {adata_ref.obsm['X_latent'].shape}")

    return model, adata_ref


def train_scanvi(
    adata_ref: ad.AnnData,
    labels_key: str,
    n_latent: int = 30,
    n_layers: int = 2,
    n_hidden: int = 128,
    batch_key: Optional[str] = None,
    use_gpu: bool = True,
    max_epochs: int = 400,
    train_size: float = 0.9,
    batch_size: int = 512
) -> Tuple[SCANVI_TYPE, ad.AnnData]:
    """
    Train scANVI model on reference atlas with known labels.

    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (log-normalized, NOT Z-scored)
    labels_key : str
        Column name in adata_ref.obs containing cell type labels
    n_latent : int
        Latent dimension
    n_layers : int
        Number of hidden layers
    n_hidden : int
        Number of hidden units per layer
    batch_key : str, optional
        Column name for batch correction
    use_gpu : bool
        Use GPU if available
    max_epochs : int
        Maximum training epochs
    train_size : float
        Fraction of data for training
    batch_size : int
        Training batch size (increase for faster GPU training)

    Returns
    -------
    SCANVI
        Trained scANVI model
    ad.AnnData
        Reference AnnData with latent embedding in obsm['X_latent']
    """
    if not SCVI_AVAILABLE:
        raise ImportError("scvi-tools not available. Install with: pip install scvi-tools")

    print(f"Training scANVI model...")
    print(f"  Labels key: {labels_key}")
    print(f"  Latent dimension: {n_latent}")

    # Setup AnnData (make a copy to avoid modifying original)
    adata_ref = adata_ref.copy()

    # Set up batch and labels
    if batch_key and batch_key in adata_ref.obs.columns:
        # Convert to string to avoid type mismatches (int/str mixing)
        adata_ref.obs['_scvi_batch'] = adata_ref.obs[batch_key].astype(str).astype('category')
    else:
        # Always use string 'batch' to ensure consistency
        adata_ref.obs['_scvi_batch'] = 'batch'
        adata_ref.obs['_scvi_batch'] = adata_ref.obs['_scvi_batch'].astype('category')

    adata_ref.obs['_scvi_labels'] = adata_ref.obs[labels_key].astype('category')

    # Prepare data
    SCANVI.setup_anndata(
        adata_ref,
        batch_key='_scvi_batch',
        labels_key='_scvi_labels'
    )

    # Create model (first train scVI, then scANVI)
    # Use gene_likelihood="normal" for log-transformed continuous data
    model = SCANVI.from_scvi_model(
        SCVI(adata_ref, n_latent=n_latent, n_layers=n_layers, n_hidden=n_hidden,
             gene_likelihood="normal"),
        labels_key='_scvi_labels',
        unlabeled_category='Unknown'
    )

    # Configure training parameters for newer scvi-tools/PyTorch Lightning
    train_kwargs = {
        'max_epochs': max_epochs,
        'train_size': train_size,
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

    model.train(**train_kwargs)

    # Print training summary
    history = model.history
    if history is not None and hasattr(history, 'columns') and 'elbo_validation' in history.columns:
        final_train = float(history['elbo_train'].iloc[-1])
        final_val = float(history['elbo_validation'].iloc[-1])
        print(f"  Final ELBO - Train: {final_train:.2f}, Validation: {final_val:.2f}")
        print(f"  Epochs completed: {len(history['elbo_train'])}")
    elif history is not None:
        # Try alternative history format
        try:
            if isinstance(history, dict):
                final_train = float(history.get('elbo_train', [0])[-1] if isinstance(history.get('elbo_train'), list) else history.get('elbo_train', 0))
                final_val = float(history.get('elbo_validation', [0])[-1] if isinstance(history.get('elbo_validation'), list) else history.get('elbo_validation', 0))
                print(f"  Final ELBO - Train: {final_train:.2f}, Validation: {final_val:.2f}")
            else:
                print(f"  Training completed")
        except (TypeError, AttributeError, IndexError, ValueError):
            print(f"  Training completed (history format not recognized)")

    # Get latent representation
    adata_ref.obsm["X_latent"] = model.get_latent_representation(adata_ref)

    print(f"  Training complete. Latent embedding shape: {adata_ref.obsm['X_latent'].shape}")

    return model, adata_ref


def map_query_scarches(
    model: Union[SCVI_TYPE, SCANVI_TYPE, str],
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    use_gpu: bool = True,
    max_epochs: int = 200
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Map query cells into reference latent space using scArches surgery.

    CRITICAL: Uses scArches surgery (SCVI.load_query_data) to freeze reference weights
    and only train batch-specific parameters for query data. This is the recommended
    approach for integrating new batches into a pre-trained reference model.

    Parameters
    ----------
    model : SCVI, SCANVI, or str
        Pre-trained scVI/scANVI model on reference atlas OR path to saved model directory
    adata_ref : ad.AnnData
        Reference atlas (will store latent embedding)
    adata_query : ad.AnnData
        Query Patch-seq data (must have layers["counts"] with raw UMI counts)
    use_gpu : bool
        Use GPU if available
    max_epochs : int
        Maximum epochs for scArches surgery training (default: 200)

    Returns
    -------
    ad.AnnData
        Reference with latent embedding
    ad.AnnData
        Query with latent embedding
    """
    if not SCVI_AVAILABLE:
        raise ImportError("scvi-tools not available")

    print("Mapping query to reference using scArches surgery...")
    print("  Freezing reference weights, training only batch-specific parameters...")

    # Determine if model is a path string or an object
    from scvi.model import SCVI, SCANVI

    model_is_path = isinstance(model, (str, Path))
    if model_is_path:
        model_path = str(model)
        print(f"  Using saved model from: {model_path}")
        # Determine model type from saved model
        # Try to detect if it's SCANVI or SCVI from the path
        if 'scanvi' in model_path.lower():
            model_class = SCANVI
            is_scanvi = True
        else:
            model_class = SCVI
            is_scanvi = False
    else:
        model_path = None
        model_class = type(model)
        is_scanvi = 'SCANVI' in model_class.__name__

    # Check for counts layer in query
    if "counts" not in adata_query.layers:
        raise ValueError(
            "adata_query must have layers['counts'] with raw UMI counts. "
            "Ensure counts layer is preserved during preprocessing."
        )

    # Prepare query data
    adata_query = adata_query.copy()

    # CRITICAL: Ensure query data has NO mixed types in ANY categorical columns
    # This is the root cause of the int/str comparison error
    print(f"  Cleaning query data to ensure type consistency...")
    for col in adata_query.obs.columns:
        if hasattr(adata_query.obs[col], 'cat'):
            # It's categorical - ensure categories are all strings
            adata_query.obs[col] = adata_query.obs[col].astype(str).astype('category')
        elif adata_query.obs[col].dtype == 'object':
            # It's object type - could have mixed types, convert to string
            adata_query.obs[col] = adata_query.obs[col].astype(str)

    # --- Set up query batch: MUST be a new category (Patch-seq specific) ---
    # CRITICAL FIX: Query batch must NEVER reuse reference batch IDs
    # This ensures Patch-seq technical effects are modeled separately from reference
    if 'tech_batch' in adata_query.obs.columns:
        # Prepend "PatchSeq_" to create distinct batch categories
        adata_query.obs['_scvi_batch'] = (
            "PatchSeq_" + adata_query.obs['tech_batch'].astype(str)
        )
        print(f"  Query batch: {adata_query.obs['_scvi_batch'].unique()}")
    else:
        # Default Patch-seq batch name
        adata_query.obs['_scvi_batch'] = "PatchSeq_Allen"
        print(f"  Query batch: PatchSeq_Allen")

    # Convert to categorical
    adata_query.obs['_scvi_batch'] = adata_query.obs['_scvi_batch'].astype('category')
    print(f"  Query _scvi_batch categories: {adata_query.obs['_scvi_batch'].cat.categories.tolist()}")

    # NOTE: Do NOT call setup_anndata ourselves!
    # load_query_data() will handle setup internally and properly reconcile
    # the query's batch categories with the reference model's registry
    print(f"  Skipping manual setup_anndata (load_query_data handles this)...")

    # scArches surgery: Load query data into pre-trained model
    # This freezes reference weights and only trains batch-specific parameters
    print(f"  Performing scArches surgery (max_epochs={max_epochs})...")

    # CRITICAL: For scArches to work, we must use the class method with the model path
    if model_is_path:
        model_query = model_class.load_query_data(
            adata_query,
            model_path,
            freeze_dropout=True
        )
    else:
        # If model is an object, we should still use the class method with a path
        # This is a limitation of scArches - it needs a saved model
        raise ValueError(
            "scArches surgery requires a saved model path. "
            "Please pass the model directory path (string) instead of the model object."
        )
    
    # Train only the batch-specific parameters (reference weights frozen)
    train_kwargs = {
        'max_epochs': max_epochs,
        'batch_size': 128,  # Smaller batch size for query data
        'early_stopping': True,
        'early_stopping_patience': 10,
        'check_val_every_n_epoch': 5,
        'enable_progress_bar': True,
    }
    if use_gpu:
        train_kwargs['accelerator'] = 'gpu'
        train_kwargs['devices'] = 'auto'
    else:
        train_kwargs['accelerator'] = 'cpu'
    
    model_query.train(**train_kwargs)
    print(f"  scArches surgery complete")

    # Get latent representations using the surgery-trained model
    print(f"  Computing latent embeddings for query...")
    adata_query.obsm["X_latent"] = model_query.get_latent_representation(adata_query)

    # Check if reference already has latent embeddings (from earlier computation)
    if "X_latent" not in adata_ref.obsm or adata_ref.obsm["X_latent"].shape[0] != adata_ref.n_obs:
        print(f"  Computing reference latent embeddings...")
        # Need to load original model to compute reference embeddings
        if model_is_path:
            print(f"  Loading reference model from {model_path}...")
            model_ref = model_class.load(model_path)

            # CRITICAL FIX: Use the model's own adata (which has proper scVI setup)
            # not the adata_ref passed in (which lacks _scvi_batch and other setup)
            print(f"  Using model's reference data (has scVI setup)")
            ref_latent = model_ref.get_latent_representation()

            # Transfer latent embeddings to our adata_ref
            # Need to match cells in case order differs
            if all(model_ref.adata.obs_names == adata_ref.obs_names):
                # Same order, direct copy
                adata_ref.obsm["X_latent"] = ref_latent
            else:
                # Different order or subset, need to align
                print(f"  Aligning latent embeddings (model has {len(model_ref.adata)} cells, adata_ref has {len(adata_ref)} cells)")
                ref_latent_df = pd.DataFrame(
                    ref_latent,
                    index=model_ref.adata.obs_names
                )
                adata_ref.obsm["X_latent"] = ref_latent_df.loc[adata_ref.obs_names].values
        else:
            # model is already an object
            adata_ref.obsm["X_latent"] = model.get_latent_representation()
    else:
        print(f"  Reference already has latent embeddings, skipping recomputation...")

    print(f"  Reference latent embedding shape: {adata_ref.obsm['X_latent'].shape}")
    print(f"  Query latent embedding shape: {adata_query.obsm['X_latent'].shape}")

    return adata_ref, adata_query


def integrate_pca_harmony(
    adata_ref: ad.AnnData,
    adata_query: ad.AnnData,
    n_comps: int = 50,
    batch_key: str = 'platform'
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Baseline integration using PCA + Harmony.
    
    Parameters
    ----------
    adata_ref : ad.AnnData
        Reference atlas expression data (preprocessed)
    adata_query : ad.AnnData
        Query Patch-seq expression data (preprocessed)
    n_comps : int
        Number of principal components
    batch_key : str
        Column name for batch/platform (will be created if not exists)
        
    Returns
    -------
    ad.AnnData
        Reference with integrated PCs
    ad.AnnData
        Query with integrated PCs
    """
    if not HARMONY_AVAILABLE:
        raise ImportError("harmonypy not available. Install with: pip install harmonypy")
    
    print("Integrating with PCA + Harmony...")
    
    # Combine datasets for PCA
    adata_ref = adata_ref.copy()
    adata_query = adata_query.copy()
    
    # Mark datasets
    adata_ref.obs['_dataset'] = 'reference'
    adata_query.obs['_dataset'] = 'query'
    
    # Create platform/batch key
    if batch_key not in adata_ref.obs.columns:
        adata_ref.obs[batch_key] = 'reference'
    if batch_key not in adata_query.obs.columns:
        adata_query.obs[batch_key] = 'query'
    
    # Combine for PCA
    adata_combined = ad.concat([adata_ref, adata_query], join='outer', index_unique='-')
    
    # Compute PCA on reference only
    print(f"  Computing PCA on reference ({adata_ref.shape[0]} cells)...")
    sc.tl.pca(adata_ref, n_comps=n_comps, svd_solver='arpack')
    
    # Project query into PCA space
    print(f"  Projecting query into PCA space...")
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)
    pca_model.fit(adata_ref.X)
    adata_query.obsm['X_pca'] = pca_model.transform(adata_query.X)
    
    # Copy PCA to reference
    adata_ref.obsm['X_pca'] = adata_ref.obsm['X_pca']
    
    # Apply Harmony
    print(f"  Applying Harmony integration...")
    adata_combined = ad.concat([adata_ref, adata_query], join='outer', index_unique='-')
    adata_combined.obsm['X_pca'] = np.vstack([
        adata_ref.obsm['X_pca'],
        adata_query.obsm['X_pca']
    ])
    
    # Run Harmony
    ho = run_harmony(
        adata_combined.obsm['X_pca'],
        adata_combined.obs,
        vars_use=[batch_key]
    )
    
    # Store integrated PCs
    adata_combined.obsm['X_harmony'] = ho.Z_corr.T
    
    # Split back
    ref_mask = adata_combined.obs['_dataset'] == 'reference'
    adata_ref.obsm['X_latent'] = adata_combined.obsm['X_harmony'][ref_mask]
    adata_query.obsm['X_latent'] = adata_combined.obsm['X_harmony'][~ref_mask]
    
    print(f"  Integration complete.")
    print(f"  Reference latent shape: {adata_ref.obsm['X_latent'].shape}")
    print(f"  Query latent shape: {adata_query.obsm['X_latent'].shape}")
    
    return adata_ref, adata_query
