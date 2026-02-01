# Team Collaboration - Reproducible CF Sample

**Date**: February 1, 2026  
**Status**: ✅ Sample Created & Ready to Share

---

## Summary

Your team can now work with the **exact same dataset** for collaborative filtering experiments!

### What Was Created

| File | Size | Purpose |
|------|------|---------|
| `cf_sample_500k.csv` | 34 MB | **The reproducible sample** (500K interactions) |
| `cf_sample_info.txt` | 1.7 KB | Dataset statistics and documentation |
| `CF_SAMPLE_TEAM_GUIDE.md` | 6.8 KB | Complete team usage guide |
| `README_CF_SAMPLE.md` | - | Quick start guide (this summary) |
| `create_cf_sample.py` | 8.1 KB | Script to recreate if needed |

---

## Key Details

- **500,000 interactions** from Deezer DSG17 training set
- **17,429 unique users**
- **20,475 unique tracks (items)**
- **Random seed: 42** (ensures reproducibility)
- **Listen rate: 69.8%** (skip rate: 30.2%)
- **Matrix sparsity: 99.86%**

---

## How to Share

### Recommended Methods

1. **Cloud Storage** (Google Drive, Dropbox, OneDrive)
   - Upload `cf_sample_500k.csv` (34 MB)
   - Share link with team
   - Everyone downloads to `notebooks/cf_sample_500k.csv`

2. **Git Repository** (if your repo allows)
   - File is currently NOT in .gitignore
   - You can commit it: `git add notebooks/cf_sample_500k.csv`
   - Or exclude it by uncommenting line in `.gitignore`

3. **Compressed Transfer** (for email/chat)
   ```bash
   cd notebooks
   gzip -k cf_sample_500k.csv  # Creates ~10MB .gz file
   # Share the .gz file, decompress with: gunzip cf_sample_500k.csv.gz
   ```

---

## Quick Verification

Everyone should run this to verify they have the correct file:

```python
import pandas as pd

df = pd.read_csv('notebooks/cf_sample_500k.csv')

# Verify these match
print(f"Rows: {len(df):,}")                    # Should be: 500,000
print(f"Users: {df['user_id'].nunique():,}")   # Should be: 17,429
print(f"Tracks: {df['media_id'].nunique():,}") # Should be: 20,475
print(f"Listen rate: {df['is_listened'].mean():.3f}")  # Should be: 0.698

# If these match → ✅ Correct sample!
```

---

## Expected Results (for validation)

When everyone runs the same collaborative filtering model:

| Model | Expected AUC |
|-------|--------------|
| User+Item Bias | 0.7699 |
| SVD (50 factors) | 0.6254 |
| NMF (50 factors) | 0.5597 |

**If results match → Everyone is using the same data correctly!** ✅

---

## Usage in Scripts

### Instead of:
```python
# ❌ Each person gets different random sample
df = pd.read_csv('../data/raw/train.csv', nrows=500000)
```

### Use:
```python
# ✅ Everyone gets the same sample
df = pd.read_csv('cf_sample_500k.csv')
```

---

## Documentation for Team

Send your colleagues these files:

1. **`cf_sample_500k.csv`** - The data file
2. **`CF_SAMPLE_TEAM_GUIDE.md`** - Complete usage guide
3. **`README_CF_SAMPLE.md`** - Quick start

Or share this message:

```
Hi team! 👋

We now have a reproducible sample dataset for our collaborative 
filtering experiments. This ensures we all work with identical data.

📁 File: cf_sample_500k.csv (34 MB)
📊 Contains: 500K interactions, 17.4K users, 20.5K tracks

To use:
1. Download cf_sample_500k.csv
2. Place in notebooks/ folder
3. Load with: df = pd.read_csv('cf_sample_500k.csv')
4. Verify: 500,000 rows, 17,429 users, 20,475 tracks

See CF_SAMPLE_TEAM_GUIDE.md for complete instructions!

Random seed: 42 (for reproducibility)
```

---

## Files in Git

Currently in repository (notebooks/):
- ✅ `create_cf_sample.py` - Recreation script
- ✅ `cf_sample_info.txt` - Statistics
- ✅ `CF_SAMPLE_TEAM_GUIDE.md` - Team guide
- ✅ `README_CF_SAMPLE.md` - Quick start
- ⚠️ `cf_sample_500k.csv` - **34 MB data file** (currently NOT ignored)

To exclude the large CSV from git, uncomment this line in `.gitignore`:
```
# notebooks/cf_sample_500k.csv
```

---

## Reproducibility

This sample was created with:
- **Random seed: 42**
- **First 500K rows** from train.csv
- **All columns preserved** (15 columns)

If anyone needs to recreate:
```bash
cd notebooks
python create_cf_sample.py
```

---

## Benefits for Your Team

✅ **No more "works on my machine"** - Same data everywhere  
✅ **Easy to compare approaches** - Apples-to-apples comparison  
✅ **Faster experiments** - 500K vs 7.5M rows  
✅ **Consistent results** - No random variation  
✅ **Clear baseline** - Everyone starts from same point  

---

## Next Steps

1. ✅ **Decide** how to share (cloud storage / git / email)
2. ✅ **Distribute** `cf_sample_500k.csv` to team
3. ✅ **Share** `CF_SAMPLE_TEAM_GUIDE.md` for instructions
4. ✅ **Verify** everyone can load and validate the sample
5. ✅ **Run** experiments - results should match!

---

## Questions?

- **Full documentation**: `CF_SAMPLE_TEAM_GUIDE.md`
- **Dataset stats**: `cf_sample_info.txt`
- **Recreate sample**: `python create_cf_sample.py`

---

**Status: Ready for team collaboration!** 🚀

*Created: February 1, 2026*  
*Location: `notebooks/cf_sample_500k.csv`*  
*Size: 34 MB (500,000 rows)*
