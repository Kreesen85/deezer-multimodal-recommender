# Collaborative Filtering Sample Dataset - Team Guide

**Created**: February 1, 2026  
**Purpose**: Reproducible sample for collaborative filtering experiments  
**File**: `cf_sample_500k.csv` (33.8 MB)

---

## 📁 What You Have

### `cf_sample_500k.csv`
- **500,000 interactions** from the Deezer DSG17 training set
- **17,429 unique users**
- **20,475 unique tracks**
- **Random seed 42** (reproducible)
- **Listen rate: 69.8%** (skip rate: 30.2%)

### `cf_sample_info.txt`
- Complete dataset statistics
- Column descriptions
- Usage instructions

---

## 🎯 Why Use This Sample?

### Problem with Random Sampling
```python
# ❌ DON'T DO THIS - Each person gets different data!
df = pd.read_csv('train.csv', nrows=500000)
# Results won't match between team members
```

### Solution: Shared Sample File
```python
# ✅ DO THIS - Everyone gets the same data!
df = pd.read_csv('cf_sample_500k.csv')
# Results will be identical and comparable
```

---

## 📤 Sharing with Your Team

### Option 1: Git LFS (Recommended for Large Files)

If your repo uses Git LFS:
```bash
git lfs track "notebooks/cf_sample_500k.csv"
git add .gitattributes
git add notebooks/cf_sample_500k.csv
git commit -m "Add reproducible CF sample dataset"
git push
```

### Option 2: Cloud Storage (Google Drive, Dropbox, etc.)

1. Upload `cf_sample_500k.csv` to shared folder
2. Share link with team
3. Everyone downloads to `notebooks/cf_sample_500k.csv`

### Option 3: Data Directory (Local Network/Server)

```bash
# Place in shared data directory
cp cf_sample_500k.csv /shared/data/deezer/
```

---

## 💻 How to Use

### Method 1: Use Directly in Scripts

```python
import pandas as pd

# Load the shared sample
df = pd.read_csv('cf_sample_500k.csv')

print(f"Loaded {len(df):,} interactions")
print(f"Users: {df['user_id'].nunique():,}")
print(f"Items: {df['media_id'].nunique():,}")
```

### Method 2: Update Existing Scripts

**Original code (baseline_collaborative_filtering.py):**
```python
# OLD - Don't use this
df = pd.read_csv('../data/raw/train.csv', nrows=500000)
```

**Updated code:**
```python
# NEW - Use shared sample
df = pd.read_csv('cf_sample_500k.csv')
```

---

## 🧪 Example: Run Collaborative Filtering

```python
"""
Example: Collaborative Filtering with Shared Sample
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

# Load shared sample
df = pd.read_csv('cf_sample_500k.csv')

# Create user-item matrix
user_ids = df['user_id'].unique()
item_ids = df['media_id'].unique()

user_map = {uid: idx for idx, uid in enumerate(user_ids)}
item_map = {iid: idx for idx, iid in enumerate(item_ids)}

df['user_idx'] = df['user_id'].map(user_map)
df['item_idx'] = df['media_id'].map(item_map)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create sparse matrix
n_users = len(user_ids)
n_items = len(item_ids)

train_matrix = csr_matrix(
    (train_df['is_listened'].values,
     (train_df['user_idx'].values, train_df['item_idx'].values)),
    shape=(n_users, n_items)
)

# Train SVD model
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(train_matrix)
item_factors = svd.components_.T

# Predict on test set
predictions = []
actuals = []

for _, row in test_df.iterrows():
    pred = np.dot(user_factors[row['user_idx']], 
                  item_factors[row['item_idx']])
    pred = np.clip(pred, 0, 1)
    predictions.append(pred)
    actuals.append(row['is_listened'])

# Evaluate
auc = roc_auc_score(actuals, predictions)
print(f"AUC: {auc:.4f}")

# Expected result: AUC ~0.6254 (should match for everyone!)
```

---

## ✅ Verification Checklist

When you load the sample, verify these stats match:

| Metric | Expected Value |
|--------|----------------|
| Total interactions | 500,000 |
| Unique users | 17,429 |
| Unique items | 20,475 |
| Listen rate | 0.698 (69.8%) |
| First user_id | Check sample |
| First media_id | Check sample |

```python
# Verification script
df = pd.read_csv('cf_sample_500k.csv')

assert len(df) == 500000, "Wrong sample size!"
assert df['user_id'].nunique() == 17429, "Wrong user count!"
assert df['media_id'].nunique() == 20475, "Wrong item count!"
assert abs(df['is_listened'].mean() - 0.698) < 0.001, "Wrong listen rate!"

print("✅ Sample verified - you have the correct data!")
```

---

## 📊 Expected Results

When you run collaborative filtering on this sample:

| Model | Expected AUC | Expected Accuracy |
|-------|-------------|-------------------|
| User+Item Bias | **0.7699** | 74.98% |
| SVD (50 factors) | 0.6254 | 53.68% |
| NMF (50 factors) | 0.5597 | 45.21% |
| Global Baseline | 0.5000 | 69.90% |

If your results match these, you're using the correct sample! ✅

---

## 🔧 Troubleshooting

### File Not Found Error
```python
FileNotFoundError: cf_sample_500k.csv
```

**Solution**: Make sure you're in the `notebooks/` directory
```bash
cd notebooks
python your_script.py
```

### Different Results
If your results don't match the expected values:
1. Check you're using `random_state=42` in train_test_split
2. Verify the file loaded correctly (see verification checklist)
3. Ensure you're using the same model parameters

### File Size Issues
The file is 33.8 MB. If too large for email:
- Use cloud storage (Google Drive, Dropbox)
- Use Git LFS
- Use compression: `gzip cf_sample_500k.csv`

---

## 📝 Quick Reference

### Load Sample
```python
import pandas as pd
df = pd.read_csv('cf_sample_500k.csv')
```

### Verify Sample
```python
print(f"Interactions: {len(df):,}")  # Should be 500,000
print(f"Users: {df['user_id'].nunique():,}")  # Should be 17,429
print(f"Items: {df['media_id'].nunique():,}")  # Should be 20,475
```

### Use in Experiments
```python
# Your collaborative filtering code here
# Results will be reproducible across team members
```

---

## 📚 Related Files

- `create_cf_sample.py` - Script to recreate the sample (if needed)
- `cf_sample_info.txt` - Detailed dataset statistics
- `baseline_collaborative_filtering.py` - Example CF implementation
- `COLLABORATIVE_FILTERING_BASELINE_RESULTS.md` - Expected results

---

## ✨ Benefits

✅ **Reproducibility**: Everyone gets identical results  
✅ **Comparability**: Easy to compare different approaches  
✅ **Efficiency**: Faster experiments (500K vs 7.5M rows)  
✅ **Consistency**: No random variation between runs  
✅ **Collaboration**: Clear shared baseline for the team  

---

## 🚀 Next Steps

1. **Download/access** `cf_sample_500k.csv`
2. **Verify** the sample loads correctly
3. **Run** collaborative filtering experiments
4. **Compare** results with team members
5. **Iterate** on model improvements

---

**Questions?** Check `cf_sample_info.txt` or ask your team lead!

---

*Generated: February 1, 2026*  
*Sample created with random seed 42 for reproducibility*
