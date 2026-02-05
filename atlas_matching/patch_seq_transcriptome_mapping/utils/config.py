"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


def get_config_dir() -> Path:
    """Get the path to the configs directory."""
    project_root = get_project_root()
    return project_root / "atlas_matching" / "configs"


def get_project_root() -> Path:
    """Get the project root directory.

    Checks ASD_CIRCUITS_ROOT env var first, falls back to __file__-based detection.
    In the ASD_Circuits project layout, the atlas_matching package lives at:
        <project_root>/atlas_matching/patch_seq_transcriptome_mapping/utils/config.py
    So project_root = __file__.parent^4
    """
    env_root = os.environ.get('ASD_CIRCUITS_ROOT')
    if env_root:
        return Path(env_root).resolve()
    # __file__ = .../atlas_matching/patch_seq_transcriptome_mapping/utils/config.py
    # .parent = utils/, .parent.parent = patch_seq_transcriptome_mapping/
    # .parent.parent.parent = atlas_matching/, .parent^4 = project_root
    return Path(__file__).resolve().parent.parent.parent.parent


def load_species_config(species: Optional[str] = None) -> Dict[str, Any]:
    """
    Load species-specific configuration.

    Parameters
    ----------
    species : str, optional
        Species name ('mouse' or 'human'). If None, returns all species configs.

    Returns
    -------
    dict
        Species configuration dictionary
    """
    config_path = get_config_dir() / "species_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Species config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if species is not None:
        if species not in config:
            raise ValueError(
                f"Species '{species}' not found in config. "
                f"Available: {list(config.keys())}"
            )
        return config[species]

    return config


def load_dataset_config(dataset: Optional[str] = None) -> Dict[str, Any]:
    """
    Load dataset-specific configuration.

    Parameters
    ----------
    dataset : str, optional
        Dataset name (e.g., 'V1_mouse_gouwens'). If None, returns all configs.

    Returns
    -------
    dict
        Dataset configuration dictionary
    """
    config_path = get_config_dir() / "dataset_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if dataset is not None:
        if dataset not in config['datasets']:
            raise ValueError(
                f"Dataset '{dataset}' not found in config. "
                f"Available: {list(config['datasets'].keys())}"
            )
        return config['datasets'][dataset]

    return config


def get_dataset_paths(
    dataset: str,
    project_root: Optional[Path] = None,
    data_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Get resolved absolute paths for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'V1_mouse_gouwens', 'M1_mouse_tasic', 'reference_atlas')
    project_root : Path, optional
        Project root directory. If None, uses default.
    data_dir : Path, optional
        Data directory override. If None, uses config default.

    Returns
    -------
    dict
        Dictionary of resolved paths
    """
    if project_root is None:
        project_root = get_project_root()

    # Load full config
    full_config = load_dataset_config()
    dataset_config = full_config['datasets'][dataset]

    # Get data directory
    if data_dir is None:
        if dataset == 'reference_atlas':
            data_dir = project_root / full_config['directories']['atlas_dir']
        else:
            data_dir = project_root / full_config['directories']['data_dir']

    # Resolve paths
    paths = {}
    for key, path_str in dataset_config['paths'].items():
        path = Path(path_str)
        if path.is_absolute():
            paths[key] = path
        else:
            paths[key] = data_dir / path_str

    return paths


def get_directory_paths(
    project_root: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Get all directory paths from config.

    Parameters
    ----------
    project_root : Path, optional
        Project root directory. If None, uses default.

    Returns
    -------
    dict
        Dictionary of directory paths
    """
    if project_root is None:
        project_root = get_project_root()

    config = load_dataset_config()
    directories = config.get('directories', {})

    resolved_dirs = {}
    for key, dir_str in directories.items():
        resolved_dirs[key] = project_root / dir_str

    return resolved_dirs


def get_qc_thresholds(
    species: str,
    dataset: Optional[str] = None,
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Get QC thresholds with priority: override > dataset > species > defaults.

    Parameters
    ----------
    species : str
        Species name ('mouse' or 'human')
    dataset : str, optional
        Dataset name for dataset-specific overrides
    override : dict, optional
        Manual override values

    Returns
    -------
    dict
        QC threshold dictionary with keys:
        - min_genes
        - max_genes
        - min_counts
        - max_counts
        - max_pct_mito
    """
    # Start with species defaults
    species_config = load_species_config(species)
    thresholds = species_config.get('qc', {}).copy()

    # Override with dataset-specific if provided
    if dataset is not None:
        dataset_config = load_dataset_config(dataset)
        if 'qc' in dataset_config:
            thresholds.update(dataset_config['qc'])

    # Override with manual values if provided
    if override is not None:
        thresholds.update(override)

    return thresholds


def get_integration_config(
    method: str = "scvi",
    override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get integration method configuration.

    Parameters
    ----------
    method : str
        Integration method ('scvi', 'scanvi', or 'harmony')
    override : dict, optional
        Manual override values

    Returns
    -------
    dict
        Integration configuration
    """
    config = load_dataset_config()
    integration_config = config.get('integration', {})

    if method not in integration_config:
        raise ValueError(
            f"Integration method '{method}' not found. "
            f"Available: {list(integration_config.keys())}"
        )

    params = integration_config[method].copy()

    if override is not None:
        params.update(override)

    return params


def get_mapping_config(override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get atlas mapping configuration.

    Parameters
    ----------
    override : dict, optional
        Manual override values

    Returns
    -------
    dict
        Mapping configuration
    """
    config = load_dataset_config()
    mapping_config = config.get('mapping', {}).copy()

    if override is not None:
        mapping_config.update(override)

    return mapping_config


def get_metadata_columns(dataset: str) -> Dict[str, str]:
    """
    Get metadata column names for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'V1_mouse_gouwens', 'M1_mouse_tasic')

    Returns
    -------
    dict
        Metadata column configuration
    """
    dataset_config = load_dataset_config(dataset)
    return dataset_config.get('metadata', {})


def get_file_formats(dataset: str) -> Dict[str, Any]:
    """
    Get file format specifications for a dataset.

    Parameters
    ----------
    dataset : str
        Dataset name

    Returns
    -------
    dict
        File format configuration
    """
    dataset_config = load_dataset_config(dataset)
    return dataset_config.get('file_formats', {})
