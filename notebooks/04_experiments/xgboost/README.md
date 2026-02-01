# XGBoost Baseline Experiment

## 📋 Overview

This folder contains the XGBoost baseline model for the Deezer Skip Prediction challenge.

**Task:** Predict probability that a user will listen to a recommended track (>30 seconds)  
**Evaluation Metric:** ROC AUC  
**Model:** XGBoost Classifier with 46 engineered features

---

## 📂 Contents

```
xgboost/
├── README.md                          # This file
├── xgboost_baseline.py                # Main training script
├── xgboost_model.json                 # Trained model (generated)
├── feature_importance.csv             # Feature importance scores (generated)
├── validation_predictions.csv         # Predictions on validation set (generated)
├── metrics_summary.json               # Performance metrics (generated)
├── feature_importance_plot.png        # Top 20 features visualization (generated)
├── confusion_matrix.png               # Confusion matrix (generated)
└── prediction_distribution.png        # Distribution of predicted probabilities (generated)
```

---

## 🚀 Quick Start

### Run the Baseline

```bash
cd /Users/kreesen/Documents/deezer-multimodal-recommender/notebooks/04_experiments/xgboost
python xgboost_baseline.py
```

**Note:** The script uses the Anaconda Python environment by default. If you need to specify:

```bash
/opt/anaconda3/bin/python xgboost_baseline.py
```

---

## 📊 Features Used

The model uses **46 engineered features** from preprocessing:

### 🕐 Temporal Features (11)
- `hour`, `day_of_week`, `day_of_month`, `month`
- `is_weekend`, `is_late_night`, `is_evening`, `is_commute_time`
- `time_of_day`

### 🎵 Release Features (10)
- `release_year`, `release_month`, `release_decade`
- `days_since_release`, `is_pre_release_listen`, `is_new_release`
- `track_age_category`

### ⏱️ Duration Features (4)
- `duration_minutes`, `duration_category`, `is_extended_track`
- `media_duration`

### 👤 User Engagement Features (9)
- `user_listen_rate`, `user_skip_rate`, `user_total_listens`
- `user_session_count`, `user_genre_diversity`, `user_artist_diversity`
- `user_context_variety`, `user_engagement_score`, `user_engagement_segment`

### 🎧 Context Features (2)
- `context_type`, `platform_family`, `listen_type`

### 👥 Demographics (2)
- `user_age`, `user_gender`

### 🎸 Other (8)
- `platform_name`, etc.

---

## ⚙️ Hyperparameters

Current configuration (in `xgboost_baseline.py`):

```python
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': 42,
    'eval_metric': 'auc'
}
```

**To tune these**, edit the `XGBOOST_PARAMS` dictionary in the script.

---

## 📈 Expected Performance

| Metric | Expected Range |
|--------|----------------|
| Validation ROC AUC | 0.70 - 0.75 |
| Training Time | 2-5 minutes (100K sample) |
| Training Time | 10-30 minutes (500K sample) |

---

## 🔧 Customization

### Change Sample Size

Edit line 22 in `xgboost_baseline.py`:

```python
SAMPLE_SIZE = 100000  # Use first 100K rows
SAMPLE_SIZE = None    # Use all data
```

### Change Data Source

Edit line 21:

```python
DATA_PATH = "../../data/processed/preprocessing/train_100k_preprocessed.csv"
DATA_PATH = "../../data/processed/preprocessing/train_preprocessed_sample.csv"  # Full data
```

### Adjust Train/Test Split

Edit line 23:

```python
TEST_SIZE = 0.2  # 80/20 split
TEST_SIZE = 0.3  # 70/30 split
```

---

## 📊 Output Files

After running the script, you'll get:

### 1. **xgboost_model.json**
- Trained XGBoost model in JSON format
- Can be loaded later for predictions

### 2. **feature_importance.csv**
- All features ranked by importance
- Use this to identify which features matter most

### 3. **validation_predictions.csv**
- Contains: `user_id`, `media_id`, `y_true`, `y_pred_proba`, `y_pred`
- Useful for error analysis

### 4. **metrics_summary.json**
- ROC AUC scores
- Hyperparameters used
- Dataset statistics
- Timestamp

### 5. **Visualizations (PNG files)**
- `feature_importance_plot.png` - Top 20 features
- `confusion_matrix.png` - Classification performance
- `prediction_distribution.png` - Probability distribution

---

## 🎯 Next Steps

### Immediate Improvements:
1. **Hyperparameter tuning** (GridSearch or Bayesian optimization)
2. **Add categorical encoding** for `genre_id`, `artist_id`, `album_id`
3. **Create interaction features** (e.g., `user_genre_affinity`)
4. **Add collaborative filtering features** (user/item embeddings)

### Advanced Techniques:
1. **Ensemble** with LightGBM, CatBoost
2. **Deep learning** (neural networks)
3. **Sequential models** (LSTM for user history)
4. **Hybrid models** (CF + features)

---

## 📚 References

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **ROC AUC**: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
- **Deezer Dataset**: DSG17 Sequential Skip Prediction Challenge

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'xgboost'`

**Solution:**
```bash
pip install xgboost
```

### Issue: `FileNotFoundError: train_preprocessed_sample.csv not found`

**Solution:** Generate preprocessed data first:
```bash
cd ../../02_preprocessing
python demo_preprocessing_with_users.py
```

### Issue: Training is too slow

**Solution:** Reduce sample size or n_estimators:
```python
SAMPLE_SIZE = 50000  # Use smaller sample
XGBOOST_PARAMS['n_estimators'] = 100  # Fewer trees
```

---

**Last Updated:** 2026-02-01  
**Author:** Deezer Multimodal Recommender Team
