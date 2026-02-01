# Notebooks Directory

Organized structure for all analysis, experiments, and documentation.

## 📁 Directory Structure

```
notebooks/
├── 01_eda/              # Exploratory Data Analysis
├── 02_preprocessing/    # Data preprocessing & feature engineering
├── 03_baselines/        # Baseline models (CF, surprise, etc.)
├── 04_experiments/      # Experiments and model development
├── docs/                # Documentation and reports
├── archive/             # Old/deprecated files
└── README.md            # This file
```

**Data files** are stored in `../data/processed/` (see Data Files section below)

---

## 01_eda/ - Exploratory Data Analysis

Scripts and outputs for understanding the data:

- `eda_full_optimized.py` - Complete EDA on 7.5M dataset
- `data_quality_check.py` - Data quality validation
- `check_temporal_consistency.py` - Temporal analysis
- `user_skip_behavior_analysis.py` - User segmentation analysis
- `eda_full_*.png` - EDA visualizations (kept in this dir)
- `eda_full_summary.txt` - Key findings

**Run order**: data_quality → eda_full → temporal → user_behavior

---

## 02_preprocessing/ - Data Preprocessing

Scripts for feature engineering and data preparation:

- `demo_preprocessing.py` - Basic preprocessing demo
- `demo_preprocessing_with_users.py` ⭐ Complete preprocessing with user features

**Outputs**: See `../data/processed/preprocessing/` for generated CSV files

**See**: `docs/USER_FEATURES_IMPLEMENTATION.md` for details

---

## 03_baselines/ - Baseline Models

Collaborative filtering and baseline model implementations:

- `baseline_collaborative_filtering.py` ⭐ Scikit-learn CF models (AUC 0.77)
- `baseline_surprise_models.py` - Surprise library models (if available)
- `collaborative_filtering_results.png` - Results visualization (kept in this dir)

**Results**: See `../data/processed/collaborative_filtering_results.csv`

**See**: `docs/COLLABORATIVE_FILTERING_BASELINE_RESULTS.md`

---

## 04_experiments/ - Experiments

Advanced models and experimental work:

- `experiments.ipynb` - Main experimentation notebook
- (Add your experiment notebooks here)

---

## docs/ - Documentation

All markdown documentation and reports:

- `COLLABORATIVE_FILTERING_BASELINE_RESULTS.md` - CF baseline results
- `USER_FEATURES_IMPLEMENTATION.md` - Feature engineering guide
- `USER_SKIP_BEHAVIOR_ANALYSIS.md` - User behavior analysis
- `CF_SAMPLE_TEAM_GUIDE.md` - Team collaboration guide
- `README_CF_SAMPLE.md` - Sample dataset quick start
- `SAMPLING_COMPARISON.md` - Sampling methodology comparison

---

## archive/ - Archived Files

Deprecated or old test files kept for reference:

- `test_*.py` - Old test scripts

---

## 📊 Data Files

All data files (CSVs) are stored in `../data/processed/` following proper data management practices:

### Sample Datasets
**Location**: `../data/processed/samples/`
- `cf_sample_500k.csv` (33.5 MB) - Random sample for CF experiments
- `cf_sample_info.txt` - Sample statistics
- `create_cf_sample_random.py` - Sample creation script

### Preprocessing Outputs
**Location**: `../data/processed/preprocessing/`
- `train_preprocessed_sample.csv` - Training data with 31 features
- `test_preprocessed_sample.csv` - Test data with 31 features
- `user_stats_from_train.csv` - User statistics

### EDA Outputs
**Location**: `../data/processed/eda/`
- `user_segments.csv` - User segmentation (19,165 users)
- `temporal_inconsistencies_sample.csv` - Temporal issues

### Results
**Location**: `../data/processed/`
- `collaborative_filtering_results.csv` - Model comparison

**See**: `../data/README.md` for complete data documentation

---

## 🚀 Quick Start

### Run Full EDA
```bash
cd notebooks/01_eda
python eda_full_optimized.py
```

### Preprocess Data
```bash
cd notebooks/02_preprocessing
python demo_preprocessing_with_users.py
```

### Run Baseline Models
```bash
cd notebooks/03_baselines
python baseline_collaborative_filtering.py
```

### Start Experiments
```bash
cd notebooks/04_experiments
jupyter notebook experiments.ipynb
```

### Load Sample Data
```python
import pandas as pd
df = pd.read_csv('../data/processed/samples/cf_sample_500k.csv')
```

---

## 📝 Key Files

| File | Location | Purpose |
|------|----------|---------|
| `eda_full_optimized.py` | `01_eda/` | Complete dataset EDA |
| `demo_preprocessing_with_users.py` | `02_preprocessing/` | Feature engineering pipeline |
| `baseline_collaborative_filtering.py` | `03_baselines/` | CF baseline models |
| `cf_sample_500k.csv` | `../data/processed/samples/` | Sample dataset (500K interactions) |
| `experiments.ipynb` | `04_experiments/` | Main experimentation notebook |

---

## 📚 Documentation

All documentation is in `docs/` directory. Start with:
1. `COLLABORATIVE_FILTERING_BASELINE_RESULTS.md` - Baseline results (AUC 0.77)
2. `USER_FEATURES_IMPLEMENTATION.md` - Feature engineering (31 features)
3. `CF_SAMPLE_TEAM_GUIDE.md` - Team collaboration guide

---

## 🔧 Organization Principles

1. **Code in notebooks/** - All scripts and notebooks
2. **Data in data/processed/** - All CSV files and datasets
3. **Docs in notebooks/docs/** - All markdown documentation
4. **Visualizations with code** - PNG files stay with source scripts

This follows best practices for data science project organization.

---

**Last Updated**: February 1, 2026  
**Total Notebooks**: 43 files organized  
**Status**: ✅ Organized and documented
