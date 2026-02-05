# Pipeline Scripts

This directory contains executable scripts for the Patch-seq transcriptome mapping pipeline.

## Pipeline Overview

The pipeline follows scArches best practices for mapping Patch-seq query data to a reference atlas:

1. **Preprocessing**: Select genes (reference HVGs only), prepare raw counts
2. **Train scVI**: Train scVI on reference atlas only (query not included)
3. **Map Query**: Use scArches surgery to map query with new batch category
4. **Canonical Expression**: Compute robust expression profiles (k=100, subclass-restricted)
5. **Annotate Metadata**: Merge atlas assignments with original metadata

---

## Scripts Overview

### 01_preprocess_data.py
**✅ FIXED**: Now uses reference-only HVGs (query HVGs removed to prevent contamination).
**✅ FIXED**: Adds forced gene inclusion (ion channels + AIS genes).
**✅ FIXED**: Persists raw counts in `layers["counts"]` for scVI.

Preprocesses raw data by selecting genes and preparing counts matrices.

**Usage**:
```bash
# Process V1 dataset
python 01_preprocess_data.py --datasets V1

# Process M1 dataset
python 01_preprocess_data.py --datasets M1

# Process both
python 01_preprocess_data.py --datasets V1 M1

# With custom HVG count (default: 3000)
python 01_preprocess_data.py --datasets V1 --n-hvgs-ref 5000
```

**Key Changes**:
- Removed `--n-hvgs-query` (query HVGs no longer used)
- Added AIS/modulator genes: Ank3, Sptan1, Sptbn4, Nfasc, Cntnap2, Fgf14, Fgf13, Scn1b-4b
- Raw counts preserved in both `.X` and `layers["counts"]`

**Outputs**: `results/preprocessed/`

---

### 02_train_scvi.py
**✅ NEW**: Separate training script (trains on reference ONLY).

Trains scVI model on reference atlas using raw counts and Negative Binomial likelihood.

**Usage**:
```bash
# Train scVI on reference
python 02_train_scvi.py

# Custom architecture
python 02_train_scvi.py \
    --n-latent 100 \
    --n-hidden 512 \
    --max-epochs 500

# Force retrain
python 02_train_scvi.py --force-retrain

# Custom model name
python 02_train_scvi.py --model-name scvi_custom_v1
```

**Key Parameters**:
- `--n-latent 50`: Latent dimension (default: 50 for high-complexity atlas)
- `--n-hidden 256`: Hidden units per layer (increased capacity)
- `--max-epochs 400`: Training epochs
- `--no-gpu`: Disable GPU

**Outputs**: `results/models/scvi_reference_latent50_hidden256/`

---

### 03_map_query.py
**✅ NEW**: Separate mapping script using scArches surgery.
**✅ FIXED**: Query receives NEW batch category ("PatchSeq_Allen") to prevent biological corruption.

Maps Patch-seq query cells to reference using scArches, performs hierarchical cell type assignment.

**Usage**:
```bash
# Map V1 query
python 03_map_query.py --dataset V1

# Map M1 query
python 03_map_query.py --dataset M1

# Use custom model
python 03_map_query.py --dataset V1 \
    --model-name scvi_custom_v1

# Adjust hierarchical assignment thresholds
python 03_map_query.py --dataset V1 \
    --k-candidates 20 \
    --k-nn 30 \
    --conf-threshold-cluster 0.8
```

**Key Parameters**:
- `--max-epochs-surgery 200`: scArches surgery epochs
- `--k-candidates 15`: Top-K candidate clusters
- `--k-nn 20`: KNN neighbors for label transfer
- `--conf-threshold-subclass 0.7`: Subclass confidence threshold
- `--conf-threshold-cluster 0.7`: Cluster confidence threshold

**Outputs**: `results/scvi_mapped/`

---

### 04_compute_canonical_expression.py
**✅ FIXED**: Increased k from 20 to 100 for robust estimates.
**✅ FIXED**: Added subclass restriction to prevent cross-type contamination.

Computes canonical expression profiles from reference and effective expression for query.

**Usage**:
```bash
# Compute for V1 (with defaults: k=100, subclass-restricted)
python 04_compute_canonical_expression.py --dataset V1

# Custom k-NN (not recommended to go below 50)
python 04_compute_canonical_expression.py --dataset V1 \
    --k-nn-expression 50

# Disable subclass restriction (not recommended)
python 04_compute_canonical_expression.py --dataset V1 \
    --no-restrict-by-subclass
```

**Key Changes**:
- Default k increased from 20 → 100 (more robust averaging)
- Neighbors restricted to predicted subclass (prevents rare cell type contamination)

**Outputs**: `results/canonical/`

---

### 05_annotate_metadata.py
Annotates Patch-seq metadata with atlas mapping results (unchanged from before).

**Usage**:
```bash
# Annotate both V1 and M1 (default)
python 05_annotate_metadata.py

# Annotate only V1
python 05_annotate_metadata.py --datasets V1

# Use custom metadata path
python 05_annotate_metadata.py --datasets M1 \
    --m1-metadata /path/to/custom_metadata.csv
```

**Outputs**: `results/annotated_metadata/`

---

## Complete Workflow Example

```bash
# Step 1: Preprocess (reference HVGs only, forced gene inclusion)
python scripts/01_preprocess_data.py --datasets V1 M1

# Step 2: Train scVI on reference ONLY
python scripts/02_train_scvi.py

# Step 3: Map V1 query (new batch category, scArches surgery)
python scripts/03_map_query.py --dataset V1

# Step 4: Compute canonical expression (k=100, subclass-restricted)
python scripts/04_compute_canonical_expression.py --dataset V1

# Step 5: Annotate metadata
python scripts/05_annotate_metadata.py --datasets V1

# Repeat steps 3-5 for M1
python scripts/03_map_query.py --dataset M1
python scripts/04_compute_canonical_expression.py --dataset M1
python scripts/05_annotate_metadata.py --datasets M1
```

---

## Critical Fixes Summary

### ✅ Batch Handling (Step 3)
- **Problem**: Query reused reference batch IDs, causing biological corruption
- **Fix**: Query receives new batch category (`"PatchSeq_Allen"`)
- **Impact**: Patch-seq technical effects modeled separately from reference

### ✅ HVG Selection (Step 1)
- **Problem**: Query HVGs contaminated feature space with Patch-seq noise
- **Fix**: Use reference-only HVGs
- **Impact**: Mapping based on biologically informative genes only

### ✅ Forced Gene Inclusion (Step 1)
- **Problem**: Ion channel genes might not be HVGs but are critical for ephys modeling
- **Fix**: Force-include Scn1a-8a, Scn1b-4b, Ank3, Nfasc, Fgf13/14, etc.
- **Impact**: Ensure genes of interest are preserved

### ✅ Canonical Expression (Step 4)
- **Problem**: k=20 too small, susceptible to outliers; no subclass restriction
- **Fix**: k=100, restrict neighbors by predicted subclass
- **Impact**: More robust expression estimates, prevents rare cell type misassignment

### ✅ Raw Counts Persistence (Step 1)
- **Problem**: counts layer not always preserved
- **Fix**: Explicitly save raw counts to `layers["counts"]`
- **Impact**: scVI can use proper Negative Binomial modeling

---

## Common Options

All scripts support:
- `--project-root`: Set project root directory
- `--help`: Show all options

Step-specific options:
- **01**: `--n-hvgs-ref` (query HVGs removed), `--subsample-atlas`, `--hvg-flavor`
- **02**: `--n-latent`, `--n-hidden`, `--max-epochs`, `--no-gpu`, `--force-retrain`
- **03**: `--model-name`, `--max-epochs-surgery`, `--k-candidates`, `--k-nn`, `--conf-threshold-*`
- **04**: `--k-nn-expression 100` (default), `--restrict-by-subclass` (default: True)
- **05**: `--datasets`, `--v1-metadata`, `--m1-metadata`

---

## Tips

1. **Memory issues**: Use `--subsample-atlas 100000` in step 01
2. **No GPU**: Use `--no-gpu` in steps 02 and 03
3. **Reuse models**: Step 02 automatically skips training if model exists
4. **Force retrain**: Use `--force-retrain` in step 02 if you need to retrain
5. **Quick test**: Subsample atlas and use fewer epochs:
   ```bash
   python 01_preprocess_data.py --datasets V1 --subsample-atlas 10000
   python 02_train_scvi.py --max-epochs 50
   python 03_map_query.py --dataset V1 --max-epochs-surgery 50
   python 04_compute_canonical_expression.py --dataset V1
   ```

---

## Migration from Old Pipeline

If you have results from the old 3-step pipeline (`02_train_scvi_and_map.py`):

1. **Training**: Old step 02 → New steps 02 + 03
   - Models trained with old pipeline need retraining (batch handling fix)
   - Use `02_train_scvi.py` to retrain reference-only model

2. **Preprocessing**: Step 01 changed
   - Rerun with new 01_preprocess_data.py (reference HVGs only)

3. **Canonical Expression**: Step 03 → New step 04
   - Old results used k=20, new uses k=100
   - Recommend rerunning for better quality

---

## See Also

- [scArches Fix Spec](../docs/scarches_patchseq_fix_spec.md): Detailed fix rationale
- [Pipeline Guide](../docs/pipeline.md): Detailed documentation
- [Main README](../README.md): Project overview
