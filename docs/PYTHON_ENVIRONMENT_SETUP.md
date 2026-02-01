# Python Environment Setup Guide

**Date**: February 1, 2026  
**Status**: Resolved - Using Anaconda Base Environment

---

## Issue Summary

The project encountered compatibility issues with Python virtual environments:

1. **Python 3.11 venv (`venv311/`)**: Segmentation faults with NumPy and scientific packages
2. **scikit-surprise**: Incompatible with Python 3.13 and NumPy 2.x (compilation errors)
3. **Binary incompatibility**: macOS ARM64 (Apple Silicon) package compatibility issues

---

## ✅ Recommended Solution: Use Anaconda Base Environment

### Current Working Setup

- **Python**: 3.13.9 (Anaconda)
- **NumPy**: 2.4.1
- **Pandas**: 3.0.0  
- **Scikit-learn**: 1.8.0
- **Matplotlib, Seaborn, Scipy**: All working

### How to Use

```bash
# Activate Anaconda base (if not already active)
conda activate base

# Verify Python version
python --version  # Should show Python 3.13.9

# Run your scripts
python notebooks/eda_full_optimized.py
python src/data/preprocessing.py
```

### PATH Configuration

Your system is already configured to use Anaconda Python:
```bash
/opt/anaconda3/bin/python
```

---

## Package Compatibility

### ✅ Working Packages

All packages in `requirements.txt` work **except** `scikit-surprise`:

- pandas 3.0.0
- numpy 2.4.1
- matplotlib 3.10.8
- seaborn 0.13.2
- plotly 6.5.2
- scikit-learn 1.8.0
- scipy 1.17.0
- jupyter, ipykernel, ipywidgets (all working)
- tqdm, python-dateutil (working)

### ❌ Incompatible: scikit-surprise

**Issue**: scikit-surprise 1.1.4 fails to compile with Python 3.13 and NumPy 2.x

**Error**: Cython compilation errors in `co_clustering.pyx`

**Solution**: Use scikit-learn for collaborative filtering instead

---

## Alternative: Collaborative Filtering with Scikit-learn

Instead of scikit-surprise, use scikit-learn's built-in capabilities:

### Option 1: Matrix Factorization with NMF

```python
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

# Non-negative Matrix Factorization
nmf = NMF(n_components=50, init='random', random_state=42)
user_features = nmf.fit_transform(user_item_matrix)
item_features = nmf.components_
```

### Option 2: Nearest Neighbors (User-based CF)

```python
from sklearn.neighbors import NearestNeighbors

# User-based collaborative filtering
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)
distances, indices = knn.kneighbors(user_item_matrix[user_id].reshape(1, -1), n_neighbors=10)
```

### Option 3: Truncated SVD

```python
from sklearn.decomposition import TruncatedSVD

# Dimensionality reduction for collaborative filtering
svd = TruncatedSVD(n_components=100, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_
```

### Option 4: Use implicit Library (Python 3.11 compatible)

If you need advanced CF algorithms, consider using the `implicit` library:

```bash
pip install implicit
```

```python
import implicit

# ALS (Alternating Least Squares)
model = implicit.als.AlternatingLeastSquares(factors=50, iterations=10)
model.fit(sparse_user_item_matrix)
recommendations = model.recommend(user_id, user_item_matrix[user_id], N=10)
```

---

## Folder Structure Status

### Keep

- `venv311_backup_broken/` - Backup of broken Python 3.11 environment (can delete after confirming)
- `venv311/` - Fresh Python 3.11 environment (currently has NumPy issues)

### Recommendation

**Delete both venv folders** and use Anaconda base:

```bash
rm -rf venv311 venv311_backup_broken
```

Add to `.gitignore`:
```
venv311/
venv311_backup_broken/
conda-env/
```

---

## Running Your Project

### 1. Verify Environment

```bash
python --version
# Python 3.13.9

python -c "import numpy, pandas, sklearn; print('All packages work!')"
# All packages work!
```

### 2. Run Scripts

```bash
# Data quality check
cd notebooks
python data_quality_check.py

# EDA
python eda_full_optimized.py

# Preprocessing demo
python demo_preprocessing_with_users.py
```

### 3. Jupyter Notebooks

```bash
jupyter lab
# or
jupyter notebook
```

---

## For Collaborators

If someone else needs to set up this project:

```bash
# Clone repository
git clone <repo-url>
cd deezer-multimodal-recommender

# Option 1: Use Anaconda (Recommended)
conda activate base
# All packages already installed in Anaconda base

# Option 2: Create fresh conda environment
conda create -n deezer python=3.11 -y
conda activate deezer
pip install -r requirements.txt
# Note: Skip scikit-surprise, use scikit-learn instead

# Option 3: Use venv (may have issues on macOS ARM)
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Known Issues & Workarounds

### Issue 1: SSL Certificate Errors

If you see SSL errors when installing packages:

```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org <package>
```

### Issue 2: NumPy Segmentation Faults

**Cause**: Binary incompatibility on macOS ARM64

**Solution**: Use Anaconda's pre-built binaries (as recommended above)

### Issue 3: scikit-surprise Compilation Errors

**Cause**: Incompatible with Python 3.13 and NumPy 2.x

**Solution**: Use scikit-learn alternatives (see above)

---

## Testing Package Imports

Create a test script to verify all packages:

```python
# test_imports.py
import sys
print(f"Python: {sys.version}")

import numpy as np
print(f"✅ NumPy: {np.__version__}")

import pandas as pd
print(f"✅ Pandas: {pd.__version__}")

import sklearn
print(f"✅ Scikit-learn: {sklearn.__version__}")

import matplotlib
print(f"✅ Matplotlib: {matplotlib.__version__}")

import seaborn as sns
print(f"✅ Seaborn: {sns.__version__}")

import scipy
print(f"✅ SciPy: {scipy.__version__}")

print("\n🎉 All packages loaded successfully!")
```

Run it:
```bash
python test_imports.py
```

---

## Summary

✅ **Use Anaconda base environment (Python 3.13.9)**  
✅ **All packages work except scikit-surprise**  
✅ **Use scikit-learn for collaborative filtering**  
✅ **Delete venv311 folders to avoid confusion**  
✅ **Update scripts to use scikit-learn instead of surprise**

**Status**: Ready to proceed with modeling! 🚀

---

*Generated: February 1, 2026*
