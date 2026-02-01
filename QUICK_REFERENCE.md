# Python Environment - Quick Reference

## ✅ RESOLVED: Use Anaconda Base Environment

### Quick Start

```bash
# Verify you're using Anaconda Python
python --version
# Should show: Python 3.13.9 | packaged by Anaconda, Inc.

# Test environment
python test_environment.py

# Run project scripts
python notebooks/eda_full_optimized.py
python notebooks/demo_preprocessing_with_users.py
```

---

## Package Versions (Working)

| Package | Version | Status |
|---------|---------|--------|
| Python | 3.13.9 | ✅ |
| NumPy | 2.4.1 | ✅ |
| Pandas | 3.0.0 | ✅ |
| Scikit-learn | 1.8.0 | ✅ |
| Matplotlib | 3.10.8 | ✅ |
| Seaborn | 0.13.2 | ✅ |
| SciPy | 1.17.0 | ✅ |
| Jupyter | latest | ✅ |
| scikit-surprise | N/A | ❌ (incompatible) |

---

## What Changed

### Files Created
- `PYTHON_ENVIRONMENT_SETUP.md` - Comprehensive setup guide
- `PYTHON_ENVIRONMENT_RESOLUTION.md` - Problem/solution summary
- `test_environment.py` - Environment verification script
- `QUICK_REFERENCE.md` (this file)

### Files Modified
- `requirements.txt` - Removed NumPy<2.0 constraint, commented out scikit-surprise
- `README.md` - Updated Quick Start and Technologies sections
- `.gitignore` - Added venv folders

---

## Collaborative Filtering Alternatives

Since `scikit-surprise` doesn't work with Python 3.13, use these **scikit-learn** alternatives:

### Quick Examples

#### Matrix Factorization (NMF)
```python
from sklearn.decomposition import NMF
model = NMF(n_components=50, random_state=42)
user_factors = model.fit_transform(user_item_matrix)
```

#### Singular Value Decomposition (SVD)
```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
```

#### K-Nearest Neighbors (User-based CF)
```python
from sklearn.neighbors import NearestNeighbors
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)
```

---

## Troubleshooting

### Q: How do I know if I'm using the right Python?

```bash
python --version
# Should show: Python 3.13.9 | packaged by Anaconda

which python
# Should show: /opt/anaconda3/bin/python
```

### Q: Packages not working?

```bash
# Run the test script
python test_environment.py
```

### Q: Need to install a package?

```bash
# Use pip (Anaconda's pip)
pip install package-name

# If SSL errors occur:
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org package-name
```

### Q: Want to use Python 3.11?

Don't! Python 3.13 (Anaconda) works better. But if you insist:

```bash
# Use conda environment (NOT venv)
conda create -n myenv python=3.11 -y
conda activate myenv
pip install -r requirements.txt
```

---

## Status: ✅ READY TO GO

Your environment is set up and tested. You can now:

1. ✅ Run all analysis scripts
2. ✅ Use Jupyter notebooks
3. ✅ Develop machine learning models
4. ✅ Use scikit-learn for collaborative filtering

---

## Documentation

- **`PYTHON_ENVIRONMENT_SETUP.md`** - Full setup guide with alternatives
- **`PYTHON_ENVIRONMENT_RESOLUTION.md`** - What went wrong and how we fixed it
- **`QUICK_REFERENCE.md`** (this file) - Quick commands and tips
- **`README.md`** - Main project documentation

---

**Last Updated**: February 1, 2026  
**Status**: Environment verified and working ✅
