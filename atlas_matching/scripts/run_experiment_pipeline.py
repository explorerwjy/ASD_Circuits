#!/usr/bin/env python3
"""
Run full experiment pipeline for multiple scVI models.

This script automates the process of training multiple models and running
the full analysis pipeline for each, making it easy to compare model performance.

Pipeline stages:
1. Train scVI models (02_train_scvi.py) - parallelized across GPUs
2. Map query data (03_map_query.py)
3. Compute canonical expression (04_compute_canonical_expression.py)
4. Annotate metadata (05_annotate_metadata.py)

Usage:
    # Run full pipeline for all models in config
    python run_experiment_pipeline.py --config ../configs/scvi_model_config.yaml

    # Run only mapping and downstream for existing models
    python run_experiment_pipeline.py --config ../configs/scvi_model_config.yaml --skip-training

    # Run for specific models
    python run_experiment_pipeline.py --models latent50_hidden256_layers2 latent30_hidden256_layers2
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any

import yaml

# Add module to path (run_experiment_pipeline.py)
ATLAS_MATCHING_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = ATLAS_MATCHING_ROOT.parent
sys.path.insert(0, str(ATLAS_MATCHING_ROOT))


def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'experiments' not in config:
        raise ValueError("No 'experiments' section found in config file")

    return config['experiments']


def run_command(cmd: List[str], description: str, cwd: Path = None) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False


def run_pipeline_for_model(
    model_name: str,
    model_config: Dict,
    project_root: Path,
    datasets: List[str],
    skip_training: bool = False,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Run the full pipeline for a single model.

    Returns dict with success status for each stage.
    """
    print(f"\n{'#'*70}")
    print(f"# PIPELINE FOR MODEL: {model_name}")
    print(f"{'#'*70}")

    scripts_dir = ATLAS_MATCHING_ROOT / 'scripts'
    results = {
        'model_name': model_name,
        'training': None,
        'mapping': None,
        'canonical': None,
        'annotation': None
    }

    # Stage 1: Training (optional)
    if not skip_training:
        cmd = [
            sys.executable,
            str(scripts_dir / '02_train_scvi.py'),
            '--model-name', model_name,
            '--n-latent', str(model_config['n_latent']),
            '--n-layers', str(model_config['n_layers']),
            '--n-hidden', str(model_config['n_hidden']),
            '--max-epochs', str(model_config.get('max_epochs', 400)),
            '--batch-size', str(model_config.get('batch_size', 512)),
        ]
        if force_retrain:
            cmd.append('--force-retrain')

        results['training'] = run_command(
            cmd,
            f"Stage 1: Training model {model_name}",
            project_root
        )

        if not results['training']:
            print(f"  ✗ Training failed for {model_name}, skipping downstream stages")
            return results
    else:
        print(f"\n  Skipping training (--skip-training flag)")
        results['training'] = True  # Assume existing model

    # Stage 2: Mapping
    cmd = [
        sys.executable,
        str(scripts_dir / '03_map_query.py'),
        '--dataset', *datasets,
        '--model-name', model_name,
    ]
    results['mapping'] = run_command(
        cmd,
        f"Stage 2: Mapping query data for {model_name}",
        project_root
    )

    if not results['mapping']:
        print(f"  ✗ Mapping failed for {model_name}, skipping downstream stages")
        return results

    # Stage 3: Canonical expression (for each dataset)
    results['canonical'] = True
    for dataset in datasets:
        cmd = [
            sys.executable,
            str(scripts_dir / '04_compute_canonical_expression.py'),
            '--dataset', dataset,
            '--model-name', model_name,
        ]
        success = run_command(
            cmd,
            f"Stage 3: Canonical expression for {model_name}/{dataset}",
            project_root
        )
        if not success:
            results['canonical'] = False

    # Stage 4: Annotation
    cmd = [
        sys.executable,
        str(scripts_dir / '05_annotate_metadata.py'),
        '--datasets', *datasets,
        '--model-name', model_name,
    ]
    results['annotation'] = run_command(
        cmd,
        f"Stage 4: Annotating metadata for {model_name}",
        project_root
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Run full experiment pipeline for multiple models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=ATLAS_MATCHING_ROOT / 'configs' / 'scvi_model_config.yaml',
        help='Path to config file with experiment definitions'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Specific model names to process (default: all models in config)'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['V1', 'M1'],
        help='Datasets to process'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training stage (use existing models)'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain even if models exist'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT PIPELINE")
    print("=" * 70)

    # Load config
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    exp_config = load_experiment_config(args.config)
    all_models = exp_config.get('models', [])

    if not all_models:
        print("No models defined in config file")
        return

    # Filter to specific models if requested
    if args.models:
        models = [m for m in all_models if m['name'] in args.models]
        missing = set(args.models) - {m['name'] for m in models}
        if missing:
            print(f"Warning: Models not found in config: {missing}")
    else:
        models = all_models

    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Models: {len(models)}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Skip training: {args.skip_training}")
    print(f"  Force retrain: {args.force_retrain}")

    print(f"\nModels to process:")
    for model in models:
        print(f"  - {model['name']}: latent={model['n_latent']}, "
              f"layers={model['n_layers']}, hidden={model['n_hidden']}")

    if args.dry_run:
        print("\n[DRY RUN] No commands will be executed")
        return

    # Run pipeline for each model
    start_time = time.time()
    all_results = []

    for model in models:
        result = run_pipeline_for_model(
            model_name=model['name'],
            model_config=model,
            project_root=PROJECT_ROOT,
            datasets=args.datasets,
            skip_training=args.skip_training,
            force_retrain=args.force_retrain
        )
        all_results.append(result)

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\nResults by model:")
    print(f"  {'Model':<35} {'Train':>8} {'Map':>8} {'Canon':>8} {'Annot':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in all_results:
        def status(val):
            if val is None:
                return "skip"
            return "✓" if val else "✗"

        print(f"  {r['model_name']:<35} "
              f"{status(r['training']):>8} "
              f"{status(r['mapping']):>8} "
              f"{status(r['canonical']):>8} "
              f"{status(r['annotation']):>8}")

    # Count successes
    full_success = [r for r in all_results
                    if all(v is True or v is None for v in r.values() if v is not None)]
    print(f"\nFully successful: {len(full_success)}/{len(all_results)} models")

    # Show results directory structure
    print(f"\nResults directory structure:")
    print(f"  atlas_matching/results/")
    print(f"    models/")
    for model in models:
        print(f"      {model['name']}/")
    print(f"    scvi_mapped/")
    for model in models:
        print(f"      {model['name']}/")
    print(f"    canonical/")
    for model in models:
        print(f"      {model['name']}/")
    print(f"    annotated_metadata/")
    for model in models:
        print(f"      {model['name']}/")

    print(f"\n✅ Pipeline complete!")


if __name__ == '__main__':
    main()
