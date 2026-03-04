"""
Standalone sibling circuit search — processes each sibling end-to-end.

Unlike the Snakemake pipeline (which scatters SA jobs across siblings),
this script ensures each sibling fully completes (biaslims → SA → extract
→ pareto) before starting the next. Aggregates every batch_size siblings.

Usage:
    conda run -n gencic python scripts/run_sibling_standalone.py \
        --start 0 --end 10000 --workers 10 --batch-size 1000

Output:
    Per-sibling: results/CircuitSearch_Sibling_Mutability/ASD_Sib_Mutability_{id}/
    Batch NPZ:   results/CircuitSearch_Sibling_Summary/Mutability/size_46/batch_{N}/
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'src'))
sys.path.insert(0, os.path.join(_project_root, 'scripts/workflow'))

from ASD_Circuits import BiasLim, ScoreCircuit_SI_Joint
from sibling_utils import load_bias_df

# Try optimized SA (numba)
try:
    from SA_optimized import CircuitSearch_SA_InfoContent_Optimized as CircuitSearch_SA_InfoContent
    SA_ENGINE = "optimized"
except ImportError:
    from ASD_Circuits import CircuitSearch_SA_InfoContent
    SA_ENGINE = "original"

# ── Config ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_ROOT, "results/CircuitSearch_Sibling_Mutability")
AGG_DIR = os.path.join(PROJECT_ROOT, "results/CircuitSearch_Sibling_Summary/Mutability/size_46")
PARQUET = os.path.join(PROJECT_ROOT, "results/STR_ISH/null_bias/ASD_All_null_bias_sibling.parquet")
REFERENCE = os.path.join(PROJECT_ROOT, "results/STR_ISH/ASD_All_bias_addP_sibling.csv")
PREFIX = "ASD_Sib_Mutability_"
WEIGHT_MAT = os.path.join(PROJECT_ROOT, "dat/allen-mouse-conn/ConnectomeScoringMat/WeightMat.Ipsi.csv")
INFO_MAT = os.path.join(PROJECT_ROOT, "dat/allen-mouse-conn/ConnectomeScoringMat/InfoMat.Ipsi.csv")

SIZE = 46
TOP_N = 213
MIN_BIAS_RANK = 50
SA_RUNTIMES = 5
SA_STEPS = 50000
MEASURE = "SI"


# ── Per-sibling functions ────────────────────────────────────────────────

def FindInitState(tmpDF, size, BiasLim_val):
    """Find initial state satisfying bias constraint."""
    STRs = tmpDF.index.values
    Biases = tmpDF["EFFECT"].values
    minBias = np.min(Biases)
    PsudoBiases = Biases - minBias + 1
    PsudoBiases = np.power(PsudoBiases, BiasLim_val * 150 - 17)
    Probs = PsudoBiases / np.sum(PsudoBiases)
    Probs[-1] = 1 - np.sum(Probs[0:-1])

    rand_count = 0
    while True:
        init_STRs = np.random.choice(STRs, size=size, replace=False, p=Probs)
        avg_bias = tmpDF.loc[init_STRs, "EFFECT"].mean()
        if avg_bias >= BiasLim_val:
            return init_STRs, rand_count
        rand_count += 1
        if rand_count > 10000:
            raise ValueError(f"Cannot find initial state with bias >= {BiasLim_val}")


def run_one_sa(BiasDF, adj_mat, InfoMat, minbias, sa_steps):
    """Run one SA optimization, return (score, meanbias, structures)."""
    tmpBiasDF = BiasDF.head(TOP_N)
    CandidateNodes = tmpBiasDF.index.values
    Init_States = np.zeros(TOP_N)
    init_STRs, _ = FindInitState(tmpBiasDF, SIZE, minbias)
    for i, s in enumerate(CandidateNodes):
        if s in init_STRs:
            Init_States[i] = 1

    ins = CircuitSearch_SA_InfoContent(
        tmpBiasDF, Init_States, adj_mat, InfoMat,
        CandidateNodes, minbias=minbias)
    ins.copy_strategy = "method"
    ins.Tmax = 1e-2
    ins.Tmin = 5e-5
    ins.steps = sa_steps
    ins.updates = 0  # suppress output

    _, _, state, e = ins.anneal()

    result_STRs = CandidateNodes[np.where(state == 1)[0]]
    score = -e
    meanbias = BiasDF.loc[result_STRs, "EFFECT"].mean()
    return score, meanbias, list(result_STRs)


def process_one_sibling(sim_id, adj_mat, InfoMat):
    """Run full pipeline for one sibling: biaslims → SA → extract → pareto."""
    dataset_name = f"{PREFIX}{sim_id}"
    out_dir = os.path.join(RESULT_DIR, dataset_name)

    # Check if pareto front already exists
    pf_path = os.path.join(out_dir, "pareto_fronts",
                           f"{dataset_name}_size_{SIZE}_pareto_front.csv")
    if os.path.exists(pf_path):
        return sim_id, "skip"

    # 1. Load bias with transfer
    config_data = {
        'bias_parquet': PARQUET,
        'sim_id': sim_id,
        'reference_bias_df': REFERENCE,
    }
    BiasDF = load_bias_df(config_data)

    # 2. Generate bias limits
    lims = BiasLim(BiasDF, SIZE)
    threshold = BiasDF.iloc[MIN_BIAS_RANK - 1]["EFFECT"]
    filtered = [(s, b) for s, b in lims if b >= threshold]

    if not filtered:
        return sim_id, "no_limits"

    bias_limits = [b for _, b in filtered]

    # 3. Run SA for all bias limits
    sa_results = {}  # bias_limit -> list of (score, meanbias, structures)
    for bl in bias_limits:
        runs = []
        for _ in range(SA_RUNTIMES):
            try:
                score, meanbias, structs = run_one_sa(BiasDF, adj_mat, InfoMat, bl, SA_STEPS)
                runs.append((score, meanbias, structs))
            except (ValueError, Exception):
                pass
        sa_results[bl] = runs

    # 4. Extract best circuit per bias limit + create pareto front
    rows = []
    for bl in bias_limits:
        runs = sa_results.get(bl, [])
        # Filter feasible (meanbias >= limit)
        feasible = [(sc, mb, st) for sc, mb, st in runs if mb >= bl - 0.001]
        if feasible:
            best = max(feasible, key=lambda x: x[0])
            rows.append({
                'bias_limit': bl,
                'circuit_score': best[0],
                'mean_bias': best[1],
                'n_structures': len(best[2]),
                'structures': ','.join(best[2]),
                'circuit_type': 'optimized',
            })

    # Add baseline (top SIZE by bias, no optimization)
    baseline_strs = list(BiasDF.head(SIZE).index)
    baseline_score = ScoreCircuit_SI_Joint(baseline_strs, InfoMat)
    baseline_bias = BiasDF.head(SIZE)['EFFECT'].mean()
    rows.append({
        'bias_limit': 0.0,
        'circuit_score': baseline_score,
        'mean_bias': baseline_bias,
        'n_structures': SIZE,
        'structures': ','.join(baseline_strs),
        'circuit_type': 'baseline',
    })

    df = pd.DataFrame(rows)

    # Save pareto front
    os.makedirs(os.path.join(out_dir, "pareto_fronts"), exist_ok=True)
    df.to_csv(pf_path, index=False)

    # Also save SA results (for compatibility with snakemake pipeline)
    sa_dir = os.path.join(out_dir, "SA_results", f"size_{SIZE}")
    os.makedirs(sa_dir, exist_ok=True)
    for bl, runs in sa_results.items():
        sa_file = os.path.join(sa_dir, f"SA..topN_{TOP_N}-keepN_{SIZE}-minbias_{bl:.6f}.txt")
        with open(sa_file, 'w') as f:
            for score, meanbias, structs in runs:
                f.write(f"{score}\t{meanbias}\t{','.join(structs)}\n")

    n_optimized = len([r for r in rows if r['circuit_type'] == 'optimized'])
    return sim_id, f"ok:{n_optimized}"


def aggregate_batch(start, end, batch_idx):
    """Aggregate a batch of sibling results into NPZ with real bias."""
    # Load full parquet once (all sim_id columns for this batch)
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(PARQUET)
    all_columns = pf.schema.names
    batch_cols = [str(sid) for sid in range(start, end) if str(sid) in all_columns]
    bias_full = pd.read_parquet(PARQUET, columns=batch_cols)
    print(f"  Loaded parquet: {bias_full.shape[1]} columns for batch {batch_idx}")

    sibling_data = []
    for sim_id in range(start, end):
        dataset_name = f"{PREFIX}{sim_id}"
        pf_path = os.path.join(
            RESULT_DIR, dataset_name, "pareto_fronts",
            f"{dataset_name}_size_{SIZE}_pareto_front.csv")
        if not os.path.exists(pf_path):
            continue

        df = pd.read_csv(pf_path)
        df['sim_id'] = sim_id

        # Recalculate real sibling bias from in-memory parquet
        col = str(sim_id)
        real_biases = []
        for _, row in df.iterrows():
            if pd.notna(row.get('structures', np.nan)) and col in bias_full.columns:
                structs = row['structures'].split(',')
                valid = [s for s in structs if s in bias_full.index]
                real_bias = bias_full.loc[valid, col].mean() if valid else np.nan
            else:
                real_bias = np.nan
            real_biases.append(real_bias)

        df['transferred_bias'] = df['mean_bias']
        df['mean_bias'] = real_biases
        sibling_data.append(df)

    if not sibling_data:
        print(f"  WARNING: No data for batch {batch_idx}!")
        return

    df_all = pd.concat(sibling_data, ignore_index=True)
    optimized = df_all[df_all['circuit_type'] == 'optimized']
    bias_limits = sorted(optimized['bias_limit'].unique())
    n_bias_limits = len(bias_limits)
    n_loaded = len(sibling_data)

    all_profiles = np.full((n_loaded, 2, n_bias_limits), np.nan)
    sim_ids_loaded = sorted(set(df_all['sim_id']))
    for idx, sim_id in enumerate(sim_ids_loaded):
        sib_opt = optimized[optimized['sim_id'] == sim_id]
        for j, bl in enumerate(bias_limits):
            row = sib_opt[sib_opt['bias_limit'] == bl]
            if len(row) > 0:
                all_profiles[idx, 0, j] = row['mean_bias'].iloc[0]  # real bias
                all_profiles[idx, 1, j] = row['circuit_score'].iloc[0]

    meanbias = np.nanmean(all_profiles[:, 0, :], axis=0)
    meanSI = np.nanmean(all_profiles[:, 1, :], axis=0)

    batch_dir = os.path.join(AGG_DIR, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)
    npz_path = os.path.join(batch_dir, "sibling_profiles.npz")
    np.savez_compressed(npz_path,
                        meanbias=meanbias, meanSI=meanSI,
                        topbias_sub=all_profiles,
                        bias_limits=np.array(bias_limits))
    print(f"  Batch {batch_idx}: saved {npz_path} "
          f"({n_loaded} siblings, {n_bias_limits} bias limits)")
    if n_bias_limits > 0:
        print(f"    Real bias: [{meanbias.min():.4f}, {meanbias.max():.4f}], "
              f"Score: [{meanSI.min():.4f}, {meanSI.max():.4f}]")


# ── Worker wrapper ────────────────────────────────────────────────────────

def worker_init():
    """Suppress verbose output in workers."""
    import warnings
    warnings.filterwarnings('ignore')


def worker_fn(sim_id):
    """Worker function — loads shared matrices from global, processes one sibling."""
    return process_one_sibling(sim_id, _adj_mat, _InfoMat)


# Module-level globals for shared data (set by main before forking)
_adj_mat = None
_InfoMat = None


def main():
    global _adj_mat, _InfoMat

    parser = argparse.ArgumentParser(description="Standalone sibling circuit search")
    parser.add_argument('--start', type=int, default=0, help="First sibling ID")
    parser.add_argument('--end', type=int, default=10000, help="Last sibling ID (exclusive)")
    parser.add_argument('--workers', type=int, default=10, help="Parallel workers")
    parser.add_argument('--batch-size', type=int, default=1000, help="Aggregate every N siblings")
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help="Skip siblings with existing pareto fronts")
    args = parser.parse_args()

    print(f"=== Standalone Sibling Circuit Search ===")
    print(f"Siblings: {args.start}–{args.end - 1} ({args.end - args.start} total)")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}")
    print(f"SA: {SA_RUNTIMES} runs × {SA_STEPS} steps, size {SIZE}, top {TOP_N}")
    print(f"SA engine: {SA_ENGINE}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load shared data (before forking workers)
    print("Loading connectivity matrices...")
    _adj_mat = pd.read_csv(WEIGHT_MAT, index_col=0)
    _InfoMat = pd.read_csv(INFO_MAT, index_col=0)
    print(f"  WeightMat: {_adj_mat.shape}, InfoMat: {_InfoMat.shape}")

    # Process in batches
    total_ok = 0
    total_skip = 0
    total_fail = 0

    for batch_start in range(args.start, args.end, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end)
        batch_idx = batch_start // args.batch_size
        sim_ids = list(range(batch_start, batch_end))

        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx}: siblings {batch_start}–{batch_end - 1}")
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        t0 = time.time()
        batch_ok = 0
        batch_skip = 0
        batch_fail = 0

        with Pool(args.workers, initializer=worker_init) as pool:
            for i, (sim_id, status) in enumerate(pool.imap_unordered(worker_fn, sim_ids)):
                if status == "skip":
                    batch_skip += 1
                elif status.startswith("ok"):
                    batch_ok += 1
                else:
                    batch_fail += 1

                done = i + 1
                if done % 50 == 0 or done == len(sim_ids):
                    elapsed = time.time() - t0
                    rate = done / elapsed * 3600
                    print(f"  [{done}/{len(sim_ids)}] ok={batch_ok} skip={batch_skip} "
                          f"fail={batch_fail} ({rate:.0f}/hr) "
                          f"elapsed={elapsed/60:.1f}min")

        elapsed = time.time() - t0
        print(f"\nBatch {batch_idx} SA complete: {elapsed/60:.1f} min "
              f"(ok={batch_ok}, skip={batch_skip}, fail={batch_fail})")

        total_ok += batch_ok
        total_skip += batch_skip
        total_fail += batch_fail

        # Aggregate this batch
        print(f"Aggregating batch {batch_idx}...")
        aggregate_batch(batch_start, batch_end, batch_idx)

    print(f"\n{'='*60}")
    print(f"ALL DONE: {total_ok} ok, {total_skip} skip, {total_fail} fail")
    print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
