#!/usr/bin/env python3
"""
Train multiple scVI models with different configurations for comparison.

This script:
1. Reads experiment configurations from YAML config
2. Trains multiple models in parallel using available GPUs
3. Creates model-specific subdirectories for results

Usage:
    # Train all models defined in config
    python train_multiple_models.py --config ../configs/scvi_model_config.yaml

    # Train specific models only
    python train_multiple_models.py --config ../configs/scvi_model_config.yaml \
        --models latent50_hidden256_layers2 latent30_hidden256_layers2

    # Dry run to see what would be trained
    python train_multiple_models.py --config ../configs/scvi_model_config.yaml --dry-run
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import yaml

# Add module to path (train_multiple_models.py)
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


def get_gpu_assignment(model_idx: int, model_name: str, exp_config: Dict) -> int:
    """Get GPU ID for a model based on assignment strategy."""
    gpu_config = exp_config.get('gpu_assignment', {})
    strategy = gpu_config.get('strategy', 'round_robin')
    num_gpus = gpu_config.get('num_gpus', 1)

    if strategy == 'manual':
        manual = gpu_config.get('manual', {})
        return manual.get(model_name, model_idx % num_gpus)
    else:  # round_robin
        return model_idx % num_gpus


def train_single_model(
    model_config: Dict,
    gpu_id: int,
    project_root: Path,
    config_path: Path,
    force_retrain: bool = False
) -> Dict[str, Any]:
    """
    Train a single scVI model.

    Returns dict with training result info.
    """
    model_name = model_config['name']
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Training model: {model_name} on GPU {gpu_id}")
    print(f"{'='*60}")

    # Build command
    train_script = ATLAS_MATCHING_ROOT / 'scripts' / '02_train_scvi.py'

    cmd = [
        sys.executable,
        str(train_script),
        '--model-name', model_name,
        '--n-latent', str(model_config['n_latent']),
        '--n-layers', str(model_config['n_layers']),
        '--n-hidden', str(model_config['n_hidden']),
        '--max-epochs', str(model_config.get('max_epochs', 400)),
        '--batch-size', str(model_config.get('batch_size', 512)),
    ]

    if force_retrain:
        cmd.append('--force-retrain')

    # Set GPU environment
    env = {
        'CUDA_VISIBLE_DEVICES': str(gpu_id),
        'PATH': '/usr/bin:/bin:/usr/local/bin',
    }
    # Inherit important environment variables
    import os
    for key in ['HOME', 'USER', 'PYTHONPATH', 'LD_LIBRARY_PATH', 'CONDA_PREFIX']:
        if key in os.environ:
            env[key] = os.environ[key]

    result = {
        'model_name': model_name,
        'gpu_id': gpu_id,
        'success': False,
        'duration': 0,
        'error': None
    }

    try:
        print(f"  Command: {' '.join(cmd)}")
        print(f"  GPU: CUDA_VISIBLE_DEVICES={gpu_id}")

        proc = subprocess.run(
            cmd,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)},
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )

        result['duration'] = time.time() - start_time

        if proc.returncode == 0:
            result['success'] = True
            print(f"  ✓ Model {model_name} trained successfully in {result['duration']:.1f}s")
        else:
            result['error'] = proc.stderr
            print(f"  ✗ Model {model_name} failed:")
            print(f"    {proc.stderr[:500]}")

        # Print stdout for debugging
        if proc.stdout:
            # Print last few lines
            lines = proc.stdout.strip().split('\n')
            print(f"  Output (last 10 lines):")
            for line in lines[-10:]:
                print(f"    {line}")

    except Exception as e:
        result['duration'] = time.time() - start_time
        result['error'] = str(e)
        print(f"  ✗ Model {model_name} exception: {e}")

    return result


def train_models_parallel(
    models: List[Dict],
    exp_config: Dict,
    project_root: Path,
    config_path: Path,
    max_parallel: int = 2,
    force_retrain: bool = False
) -> List[Dict]:
    """Train multiple models in parallel."""
    results = []

    # Group models by GPU
    gpu_assignments = {}
    for idx, model in enumerate(models):
        gpu_id = get_gpu_assignment(idx, model['name'], exp_config)
        if gpu_id not in gpu_assignments:
            gpu_assignments[gpu_id] = []
        gpu_assignments[gpu_id].append(model)

    print(f"\nGPU Assignment:")
    for gpu_id, gpu_models in gpu_assignments.items():
        print(f"  GPU {gpu_id}: {[m['name'] for m in gpu_models]}")

    # Train models - run one per GPU at a time
    num_gpus = exp_config.get('gpu_assignment', {}).get('num_gpus', 2)

    with ProcessPoolExecutor(max_workers=min(max_parallel, num_gpus)) as executor:
        # Submit initial batch (one per GPU)
        futures = {}
        model_queue = list(enumerate(models))

        # Submit first batch
        for gpu_id in range(min(num_gpus, len(model_queue))):
            if model_queue:
                idx, model = model_queue.pop(0)
                gpu = get_gpu_assignment(idx, model['name'], exp_config)
                future = executor.submit(
                    train_single_model,
                    model, gpu, project_root, config_path, force_retrain
                )
                futures[future] = model['name']

        # Process completions and submit new jobs
        while futures:
            done_futures = []
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'model_name': model_name,
                        'success': False,
                        'error': str(e)
                    })
                done_futures.append(future)

                # Submit next model if available
                if model_queue:
                    idx, model = model_queue.pop(0)
                    gpu = get_gpu_assignment(idx, model['name'], exp_config)
                    new_future = executor.submit(
                        train_single_model,
                        model, gpu, project_root, config_path, force_retrain
                    )
                    futures[new_future] = model['name']

            # Remove completed futures
            for f in done_futures:
                del futures[f]

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train multiple scVI models for comparison',
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
        help='Specific model names to train (default: all models in config)'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=2,
        help='Maximum number of models to train in parallel'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain even if models exist'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be trained without actually training'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Train models sequentially instead of in parallel'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-MODEL TRAINING")
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
    print(f"  Models to train: {len(models)}")
    print(f"  Max parallel: {args.max_parallel}")
    print(f"  Force retrain: {args.force_retrain}")

    print(f"\nModels:")
    for i, model in enumerate(models):
        gpu = get_gpu_assignment(i, model['name'], exp_config)
        desc = model.get('description', '')
        print(f"  {i+1}. {model['name']} (GPU {gpu})")
        print(f"     n_latent={model['n_latent']}, n_layers={model['n_layers']}, "
              f"n_hidden={model['n_hidden']}")
        if desc:
            print(f"     {desc}")

    if args.dry_run:
        print("\n[DRY RUN] No models will be trained")
        return

    # Train models
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")

    start_time = time.time()

    if args.sequential:
        results = []
        for idx, model in enumerate(models):
            gpu = get_gpu_assignment(idx, model['name'], exp_config)
            result = train_single_model(
                model, gpu, PROJECT_ROOT, args.config, args.force_retrain
            )
            results.append(result)
    else:
        results = train_models_parallel(
            models, exp_config, PROJECT_ROOT, args.config,
            max_parallel=args.max_parallel,
            force_retrain=args.force_retrain
        )

    total_time = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Successful: {len(successful)}/{len(results)}")

    if successful:
        print(f"\n✓ Successful models:")
        for r in successful:
            print(f"    {r['model_name']} ({r.get('duration', 0):.1f}s)")

    if failed:
        print(f"\n✗ Failed models:")
        for r in failed:
            error = r.get('error', 'Unknown error')[:100]
            print(f"    {r['model_name']}: {error}")

    print(f"\nNext steps:")
    print(f"  1. Run mapping for each model:")
    for model in models:
        print(f"     python 03_map_query.py --dataset V1 M1 --model-name {model['name']}")
    print(f"\n  2. Or use run_experiment_pipeline.py to run full pipeline for all models")


if __name__ == '__main__':
    main()
