# Team Sample Strategy - UNIFIED 500K APPROACH

**Date**: February 1, 2026  
**Decision**: Use `cf_sample_500k.csv` as the single source of truth for all team experiments

---

## 📊 Current Situation - CLEANED UP

### ✅ What We KEPT

**`data/processed/samples/`**
- ✅ `cf_sample_500k.csv` (34 MB) - **STANDARD TEAM SAMPLE**
  - 500,000 random interactions (seed=42)
  - 18,558 users, 105,803 items
  - Everyone uses THIS file

- ✅ `cf_sample_info.txt` - Sample metadata
- ✅ `create_cf_sample_random.py` - Regeneration script  
- ✅ `README.md` - Documentation

### ❌ What We DELETED

- ❌ `cf_sample_500k_sequential_old.csv` (34 MB saved!)
- ❌ `cf_sample_info_old.txt`
- ❌ `create_cf_sample.py` (old method)
- ❌ Old preprocessing samples (100K, different size)

---

## 🎯 Team Workflow

### For Collaborative Filtering (Pure CF, no features)

```python
import pandas as pd

# Everyone loads the SAME sample
df = pd.read_csv('data/processed/samples/cf_sample_500k.csv')

# Run your CF model
# Expected AUC: ~0.76-0.77
```

**Use for**: 
- `notebooks/03_baselines/baseline_collaborative_filtering.py`
- Matrix factorization
- Pure CF experiments

---

### For Feature-Based Models (with 31 features)

**Step 1**: Generate preprocessed version (run once)

```bash
cd data/processed/preprocessing
python create_preprocessed_500k.py
```

This creates:
- `train_500k_preprocessed.csv` (400K rows, 46 columns)
- `test_500k_preprocessed.csv` (100K rows, 46 columns)  
- `user_stats_500k.csv` (user lookup table)

**Step 2**: Use preprocessed samples

```python
import pandas as pd

# Load preprocessed with all features
train = pd.read_csv('data/processed/preprocessing/train_500k_preprocessed.csv')
test = pd.read_csv('data/processed/preprocessing/test_500k_preprocessed.csv')
user_stats = pd.read_csv('data/processed/preprocessing/user_stats_500k.csv')

# Train your model with 46 features
# Expected AUC: ~0.85-0.90
```

**Use for**:
- Gradient boosting models
- Neural networks
- Any feature-based ML

---

## ⚠️ IMPORTANT: Update Your Scripts

### Before (OLD - Don't use):
```python
# DON'T DO THIS - creates random sequential samples
df = pd.read_csv('../data/raw/train.csv', nrows=500000)
```

### After (NEW - Use this):
```python
# DO THIS - everyone gets same data
df = pd.read_csv('../data/processed/samples/cf_sample_500k.csv')
```

---

## 📁 File Locations Summary

| File | Location | Size | Purpose |
|------|----------|------|---------|
| **cf_sample_500k.csv** | `data/processed/samples/` | 34 MB | Raw 500K sample ⭐ |
| train_500k_preprocessed.csv | `data/processed/preprocessing/` | TBD | Train with 46 features |
| test_500k_preprocessed.csv | `data/processed/preprocessing/` | TBD | Test with 46 features |
| user_stats_500k.csv | `data/processed/preprocessing/` | TBD | User statistics |

---

## 🎉 Benefits

✅ **Everyone uses same data** - Reproducible results  
✅ **No confusion** - One standard sample  
✅ **Consistent splits** - Same train/test for all  
✅ **34 MB saved** - Deleted old files  
✅ **Clear workflow** - Raw CF → Preprocessed ML

---

## 🔧 Scripts to Update

Need to update these to use `cf_sample_500k.csv`:

1. ✅ `notebooks/03_baselines/baseline_collaborative_filtering.py`
   - Change line 38: `df = pd.read_csv('../data/processed/samples/cf_sample_500k.csv')`

2. ✅ `notebooks/02_preprocessing/demo_preprocessing_with_users.py`
   - Use preprocessed 500K samples instead of generating on-the-fly

---

## 💡 To Generate Preprocessed Samples

**Note**: Due to environment issues with `venv311`, use your working Python environment:

```bash
# Make sure you're using Python 3.13 (Anaconda) or working environment
cd data/processed/preprocessing
python create_preprocessed_500k.py
```

This will take ~30 seconds and create all preprocessed files.

---

## 🚀 Next Steps

1. ✅ Pull latest changes: `git pull`
2. ✅ Verify you have: `data/processed/samples/cf_sample_500k.csv`
3. ⏭️ Generate preprocessed: `python create_preprocessed_500k.py`
4. ⏭️ Update your scripts to use new paths
5. ⏭️ Verify results match team expectations

---

**Status**: Cleaned up, simplified, ready for team use!
