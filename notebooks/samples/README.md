# samples/ - Sample Datasets

Reproducible sample datasets for development and experimentation.

## 🎯 Active Sample

### `cf_sample_500k.csv` ⭐ **USE THIS**
**Random sample of 500,000 interactions**
- Method: Random sampling (seed=42)
- Size: 33.5 MB
- Users: 18,558 (93.2% of total)
- Items: 105,803 tracks (23.4% of total)
- Listen rate: 68.4% (matches full dataset)

**Why random?** Better coverage, more representative, best practice

---

## 📋 Sample Information

### `cf_sample_info.txt`
Statistics and metadata for the active sample
- Total interactions, users, items
- Listen/skip rates
- Matrix sparsity
- Usage instructions

---

## 🔧 Creation Scripts

### `create_cf_sample_random.py` ⭐ **CURRENT METHOD**
Creates random sample (reproducible with seed=42)
- Better item coverage (105K vs 20K)
- Representative of full dataset
- No temporal or user bias

**Run**: `python create_cf_sample_random.py`

### `create_cf_sample.py`
Sequential sample (first 500K rows)
- Simpler but less representative
- Kept for reference

---

## 📊 Sample Statistics

```
Total interactions:     500,000
Unique users:           18,558
Unique tracks:          105,803
Unique albums:          15,234
Unique artists:         24,516
Unique genres:          738

Listen rate:            68.4%
Skip rate:              31.6%
Matrix sparsity:        99.95%

Avg interactions/user:  26.9
Avg interactions/item:  4.7
```

---

## 🚀 Usage

### Load Sample
```python
import pandas as pd

df = pd.read_csv('cf_sample_500k.csv')
print(f"Loaded: {len(df):,} interactions")
```

### Verify Correctness
```python
# Check you have the RIGHT sample
assert df['user_id'].nunique() == 18558, "Wrong sample!"
assert df['media_id'].nunique() == 105803, "Wrong sample!"
print("✅ Correct random sample!")
```

---

## 🔄 Reproducibility

- **Random seed**: 42
- **Sampling fraction**: 6.61% of full dataset
- **Method**: `df.sample(n=500000, random_state=42)`

All team members using this file will get:
- ✅ Identical data
- ✅ Identical results
- ✅ Comparable experiments

---

## 📚 Documentation

- `../docs/CF_SAMPLE_TEAM_GUIDE.md` - Team usage guide
- `../docs/README_CF_SAMPLE.md` - Quick start
- `../docs/SAMPLING_COMPARISON.md` - Technical comparison

---

## 🎯 For Team Collaboration

1. **Everyone uses same file**: `cf_sample_500k.csv`
2. **Expected AUC** (User+Item Bias): ~0.75-0.77
3. **Expected accuracy**: ~74-76%
4. **If results match**: ✅ You're all using the same data!

---

## 🗂️ File Sizes

```
cf_sample_500k.csv           33.5 MB  ⭐ Main sample
cf_sample_info.txt          <1 KB    Statistics
create_cf_sample_random.py   5.9 KB   Creation script
create_cf_sample.py          8.2 KB   Sequential method
```

---

*Current sample: Random, seed=42, 500K interactions, 105K items*
