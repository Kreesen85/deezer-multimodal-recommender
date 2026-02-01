# 🎯 Collaborative Filtering Sample - Quick Start

## ✅ You Now Have:

### 📁 Files Created

1. **`cf_sample_500k.csv`** (34 MB) ⭐ **THE SAMPLE FILE**
   - 500,000 interactions
   - 17,429 users × 20,475 tracks
   - Random seed: 42 (reproducible!)
   
2. **`cf_sample_info.txt`** (1.7 KB)
   - Complete dataset statistics
   - Column descriptions
   
3. **`CF_SAMPLE_TEAM_GUIDE.md`** (6.8 KB)
   - How to share with colleagues
   - Usage examples
   - Expected results
   
4. **`create_cf_sample.py`** (8.1 KB)
   - Script to recreate sample if needed

---

## 🚀 Quick Usage

### Load the Sample

```python
import pandas as pd

# Load the reproducible sample
df = pd.read_csv('notebooks/cf_sample_500k.csv')

print(f"Loaded: {len(df):,} interactions")
print(f"Users: {df['user_id'].nunique():,}")    # 17,429
print(f"Tracks: {df['media_id'].nunique():,}")  # 20,475
```

### Verify It's Correct

```python
# These should match for everyone!
assert len(df) == 500000
assert df['user_id'].nunique() == 17429
assert df['media_id'].nunique() == 20475
assert abs(df['is_listened'].mean() - 0.698) < 0.001

print("✅ Correct sample!")
```

---

## 📤 Share with Colleagues

### Option 1: Google Drive / Dropbox

```bash
# Upload cf_sample_500k.csv to shared folder
# Share link with team
# Everyone downloads to: notebooks/cf_sample_500k.csv
```

### Option 2: Git (if repo allows large files)

```bash
# Add to git
git add notebooks/cf_sample_500k.csv
git add notebooks/cf_sample_info.txt
git add notebooks/CF_SAMPLE_TEAM_GUIDE.md

git commit -m "Add reproducible CF sample for team"
git push
```

### Option 3: Email / Slack (compressed)

```bash
# Compress first
cd notebooks
gzip -k cf_sample_500k.csv  # Creates cf_sample_500k.csv.gz (smaller)

# Share the .gz file
# Colleagues decompress with:
gunzip cf_sample_500k.csv.gz
```

---

## 📊 Expected Results

When everyone runs collaborative filtering on this sample:

| Model | AUC | Accuracy |
|-------|-----|----------|
| **User+Item Bias** | **0.7699** | 74.98% |
| SVD (50 factors) | 0.6254 | 53.68% |
| NMF (50 factors) | 0.5597 | 45.21% |
| Global Baseline | 0.5000 | 69.90% |

**If your results match → You're using the correct data!** ✅

---

## 🔧 Update Existing Scripts

### Before (different data for each person):
```python
df = pd.read_csv('../data/raw/train.csv', nrows=500000)
```

### After (same data for everyone):
```python
df = pd.read_csv('cf_sample_500k.csv')
```

---

## ✨ Why This Matters

✅ **Reproducible** - Everyone gets identical results  
✅ **Comparable** - Easy to compare approaches  
✅ **Faster** - Smaller than full 7.5M dataset  
✅ **Consistent** - No random variation  
✅ **Team-friendly** - Clear shared baseline  

---

## 📁 File Locations

```
notebooks/
├── cf_sample_500k.csv          ⭐ The sample data
├── cf_sample_info.txt          📊 Statistics
├── CF_SAMPLE_TEAM_GUIDE.md     📚 Full guide
├── create_cf_sample.py         🔧 Recreation script
└── baseline_collaborative_filtering.py  💡 Example usage
```

---

## 💡 Next Steps

1. ✅ **Share** `cf_sample_500k.csv` with your team
2. ✅ **Update** your scripts to load this file
3. ✅ **Run** experiments - results will match!
4. ✅ **Compare** approaches with teammates
5. ✅ **Iterate** on improvements together

---

## 🆘 Help

- Full guide: See `CF_SAMPLE_TEAM_GUIDE.md`
- Statistics: See `cf_sample_info.txt`
- Recreate sample: Run `python create_cf_sample.py`

---

**Sample verified and ready to share!** 🎉

*Created: February 1, 2026*  
*Random seed: 42*
