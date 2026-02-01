# XGBoost Experiment Setup - Summary

## ✅ What Was Created

### Folder Structure:
```
notebooks/04_experiments/xgboost/
├── xgboost_baseline.py          # Main training script (12 KB)
├── README.md                    # Detailed documentation (5.7 KB)
├── run.sh                       # Quick start script (executable)
└── .gitignore                   # Ignore output files
```

### Updated Files:
- `notebooks/04_experiments/README.md` - Added experiment tracking

---

## 📦 Environment Setup

### ✅ Installed Packages:
- **XGBoost 3.1.3** - Gradient boosting library

### Existing Packages (Already Installed):
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 🎯 The XGBoost Baseline Script

### What It Does:

1. **Loads preprocessed data** (100K or 500K sample)
2. **Selects 46 engineered features**:
   - Temporal (11): hour, day_of_week, is_weekend, etc.
   - Release (10): release_year, days_since_release, etc.
   - Duration (4): duration_minutes, duration_category, etc.
   - User engagement (9): user_listen_rate, user_skip_rate, etc.
   - Context (2): context_type, platform_family
   - Demographics (2): user_age, user_gender
   - Other (8): platform_name, etc.

3. **Trains XGBoost classifier**
   - 300 trees
   - Max depth: 6
   - Learning rate: 0.1
   - Early stopping after 20 rounds

4. **Evaluates with ROC AUC**
   - 80/20 train/validation split
   - Stratified sampling

5. **Generates outputs**:
   - `xgboost_model.json` - Trained model
   - `feature_importance.csv` - Feature rankings
   - `validation_predictions.csv` - Predictions for analysis
   - `metrics_summary.json` - Performance metrics
   - 3 PNG visualizations (feature importance, confusion matrix, probability distribution)

---

## 🚀 How to Run

### Option 1: Quick Start Script
```bash
cd /Users/kreesen/Documents/deezer-multimodal-recommender/notebooks/04_experiments/xgboost
./run.sh
```

### Option 2: Direct Python
```bash
cd /Users/kreesen/Documents/deezer-multimodal-recommender/notebooks/04_experiments/xgboost
/opt/anaconda3/bin/python xgboost_baseline.py
```

### Expected Runtime:
- 100K sample: ~2-5 minutes
- 500K sample: ~10-30 minutes

### Expected Performance:
- **Validation ROC AUC: 0.70 - 0.75**

---

## 📊 Output Files

After running, you'll find these files in the `xgboost/` folder:

1. **xgboost_model.json** - Trained model (can be loaded later)
2. **feature_importance.csv** - All 46 features ranked by importance
3. **validation_predictions.csv** - User/item predictions for error analysis
4. **metrics_summary.json** - ROC AUC, hyperparameters, timestamps
5. **feature_importance_plot.png** - Bar chart of top 20 features
6. **confusion_matrix.png** - Classification performance heatmap
7. **prediction_distribution.png** - Histogram of predicted probabilities

**Note:** These output files are gitignored (won't be committed).

---

## 🔧 Customization Options

### Change Sample Size:
Edit line 22 in `xgboost_baseline.py`:
```python
SAMPLE_SIZE = 100000  # Use first 100K rows
SAMPLE_SIZE = None    # Use ALL rows (slower)
```

### Change Data Source:
Edit line 21:
```python
DATA_PATH = "../../data/processed/preprocessing/train_100k_preprocessed.csv"
DATA_PATH = "../../data/processed/preprocessing/train_preprocessed_sample.csv"
```

### Tune Hyperparameters:
Edit lines 28-41:
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,      # More trees
    'max_depth': 8,           # Deeper trees
    'learning_rate': 0.05,    # Slower learning
    # ... etc
}
```

---

## 📈 Next Steps

### After Getting Baseline Results:

1. **Analyze Feature Importance**
   - Check `feature_importance.csv`
   - Identify top predictive features
   - Remove weak features

2. **Hyperparameter Tuning**
   - Try different `max_depth` (4, 6, 8, 10)
   - Adjust `learning_rate` (0.01, 0.05, 0.1)
   - Tune regularization (`reg_alpha`, `reg_lambda`)

3. **Feature Engineering**
   - Add user-genre affinity
   - Add user-artist affinity
   - Create interaction features

4. **Add Collaborative Filtering**
   - Train SVD/NMF on user-item matrix
   - Add CF predictions as features

5. **Try Other Models**
   - LightGBM (faster, better categorical handling)
   - Neural networks
   - Ensemble multiple models

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: train_preprocessed_sample.csv"

**Solution:** Generate preprocessed data first:
```bash
cd ../../02_preprocessing
/opt/anaconda3/bin/python demo_preprocessing_with_users.py
```

### Issue: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:** Install XGBoost:
```bash
/opt/anaconda3/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org xgboost
```

### Issue: Training takes too long

**Solution:** Reduce sample size or number of trees:
```python
SAMPLE_SIZE = 50000                # Smaller sample
XGBOOST_PARAMS['n_estimators'] = 100  # Fewer trees
```

---

## 📚 Understanding the Task

### ⚠️ IMPORTANT: Task Clarification

**Your colleague's baseline (Item-CF, SVD) was solving a DIFFERENT problem!**

- **Her approach**: Ranking task (recommend top-10 items)
- **Actual task**: Binary classification (predict skip/listen for specific track)

### The Real Problem:

```
Test Dataset Format:
user_id  media_id  [features...]  is_listened
   1       5001         ...            ?
   2       5002         ...            ?
   3       5003         ...            ?
```

**Goal:** For each row, predict probability that user will listen to THAT specific track.

**Evaluation:** ROC AUC (how well you separate listeners from skippers)

**Submission Format:**
```csv
sample_id,is_listened
0,0.73
1,0.42
2,0.89
...
```

### Why XGBoost Fits Better:

✅ Outputs calibrated probabilities (0-1)  
✅ Uses all engineered features  
✅ Optimizes for classification  
✅ ROC AUC is natural evaluation metric

---

## 🎯 Quick Comparison

| Approach | Task Type | Metric | Use Case |
|----------|-----------|--------|----------|
| **Item-CF** | Ranking | Hit@10 | "Recommend top items" |
| **SVD** | Ranking | Hit@10 | "Recommend top items" |
| **XGBoost** ⭐ | Classification | ROC AUC | "Predict this specific track" |

---

**Created:** 2026-02-01  
**Ready to run:** Yes  
**Expected performance:** 0.70-0.75 ROC AUC
