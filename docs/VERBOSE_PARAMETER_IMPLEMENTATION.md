# Verbose Parameter Implementation

**Date**: 2025-11-01
**Status**: COMPLETED

## Summary

Implemented complete verbose parameter support to suppress SA progress logs during pipeline execution while keeping them available for debugging.

## Problem

During pipeline execution, the SA annealing algorithm prints verbose progress updates:

```
Temperature        Energy    Accept   Improve      Steps        Elapsed   Remaining
     0.00108         -0.84    94.10%     0.00%      42000.00     0:00:01     0:00:02
     0.00252         -0.85    89.90%     0.10%      26000.00     0:00:01     0:00:03
     ...
```

When running many SA iterations in parallel, this creates excessive log output.

## Solution

The `Annealer` class in `src/SA.py` has an `updates` parameter that controls progress output:
- `updates = 0`: No progress output (silent mode)
- `updates = 100`: Show progress updates every 1% (verbose mode)

## Implementation

### 1. Configuration (config/circuit_config.yaml)

```yaml
# Verbose output during SA runs
# Default: False (suppress individual run details during pipeline execution)
# Set to True for debugging
verbose: False
```

### 2. Pipeline Integration (Snakefile.circuit.refactored)

```python
VERBOSE = config.get("verbose", False)
```

The verbose parameter is passed to the workflow script via `params.verbose`.

### 3. Workflow Script (scripts/workflow/run_sa_search.py)

```python
def run_CircuitOpt(BiasDF, adj_mat, InfoMat, topN, keepN, minbias,
                   measure, sa_steps, verbose=False):
    """Run circuit optimization with SA"""
    # ... setup code ...

    # Control verbosity: 0 = no output, 100 = show progress updates
    ins.updates = 100 if verbose else 0

    Tmps, Energys, state, e = ins.anneal()
    # ...
```

### 4. Old Pipeline Script (scripts/script_circuit_search.SI.py)

Updated to suppress verbose output by default:

```python
ins.updates = 0  # Suppress verbose output
```

### 5. Test Scripts

All test scripts already had `ins.updates = 0` to keep test output clean:
- `scripts/test_sa_circuit_size.py`
- `scripts/test_sa_score_comparison.py`
- `scripts/benchmark_sa_optimization.py`

## Usage

### Running Pipeline with Quiet Output (Default)

```bash
snakemake -s Snakefile.circuit.refactored --configfile config/circuit_config.yaml --cores 10
```

Output will only show:
```
[ASD_SPARK_61] Loading bias data...
[ASD_SPARK_61] Loading connectivity matrices...
[ASD_SPARK_61] Running SA search: topN=213, keepN=46, minbias=0.35, measure=SI
[ASD_SPARK_61] SA run 1/20: score=0.802069, bias=0.3500
[ASD_SPARK_61] SA run 2/20: score=0.801234, bias=0.3501
...
```

### Running Pipeline with Verbose Output (Debugging)

Edit `config/circuit_config.yaml`:

```yaml
verbose: True
```

Or override from command line:

```bash
snakemake -s Snakefile.circuit.refactored --configfile config/circuit_config.yaml \
  --config verbose=True --cores 10
```

Output will include SA progress tables:

```
[ASD_SPARK_61] Running SA search: topN=213, keepN=46, minbias=0.35, measure=SI

 Temperature        Energy    Accept   Improve      Steps        Elapsed   Remaining
     0.01000         -0.44                         0:00:00
     0.00948         -0.45    86.00%     2.00%         50.00     0:00:00     0:00:08
     0.00899         -0.47   100.00%     2.00%        100.00     0:00:00     0:00:07
     ...
```

## Testing

Run the verbose parameter test:

```bash
python scripts/test_verbose_parameter.py
```

This test verifies:
1. **Test 1 (verbose=False)**: No SA progress output
2. **Test 2 (verbose=True)**: Full SA progress table displayed

## Files Modified

- `scripts/workflow/run_sa_search.py`: Added `verbose` parameter to `run_CircuitOpt()`
- `scripts/script_circuit_search.SI.py`: Set `ins.updates = 0` for old pipeline
- `scripts/test_verbose_parameter.py`: New test script to verify functionality

## Benefits

1. **Clean Pipeline Output**: No clutter from hundreds of SA progress lines
2. **Easy Debugging**: Set `verbose: True` in config to see detailed progress
3. **Consistent Behavior**: Same quiet output across old and new pipelines
4. **Performance**: No impact on SA performance, only affects logging

## Notes

- The `updates` parameter only controls the progress table output
- The dataset-level log messages (e.g., "SA run 1/20: score=...") are always shown
- This is intentional to track overall pipeline progress
- To suppress ALL output, redirect stderr: `snakemake ... 2>/dev/null`
