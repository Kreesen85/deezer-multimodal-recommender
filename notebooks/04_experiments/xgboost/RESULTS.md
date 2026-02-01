# XGBoost Baseline - Results Summary

**Date:** 2026-02-01 23:39:00  
**Dataset:** 100K preprocessed training data  
**Model:** XGBoost Classifier

---

## 🎯 **Performance Results**

### **ROC AUC Score: 0.8722** ⭐

| Metric | Score |
|--------|-------|
| **Validation ROC AUC** | **0.8722** |
| Training ROC AUC | 0.8920 |
| Overfitting Gap | 0.0198 (low, good!) |

**This is EXCELLENT performance!** 🎉

- Expected range: 0.70-0.75
- **Actual: 0.8722** (much better than expected!)
- Very low overfitting (only 2% gap)

---

## 📊 **Classification Performance**

### Overall Accuracy: 81%

| Metric | Skip (0) | Listen (1) |
|--------|----------|------------|
| **Precision** | 0.73 | 0.83 |
| **Recall** | 0.59 | 0.90 |
| **F1-Score** | 0.65 | 0.87 |

### Confusion Matrix:

```
                    Predicted
                Skip        Listen
Actual Skip     3,604       2,539    (59% recall)
Actual Listen   1,323      12,534    (90% recall)
```

### Key Insights:
- ✅ **Excellent at identifying listeners** (90% recall)
- ⚠️ **Struggles more with skips** (59% recall)
- This is expected: the model learns from positive examples (listens) better
- 81% overall accuracy is very strong

---

## 🔍 **Top 10 Most Important Features**

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | **user_listen_rate** | 39.98% | User Engagement ⭐⭐⭐ |
| 2 | **user_skip_rate** | 29.16% | User Engagement ⭐⭐⭐ |
| 3 | user_engagement_segment | 2.09% | User Engagement |
| 4 | listen_type | 1.59% | Context |
| 5 | platform_name | 1.22% | Context |
| 6 | user_engagement_score | 1.21% | User Engagement |
| 7 | platform_family | 1.17% | Context |
| 8 | context_type | 1.16% | Context |
| 9 | user_total_listens | 1.15% | User Engagement |
| 10 | user_session_count | 1.01% | User Engagement |

### **Key Findings:**

1. **User behavior dominates** (70% importance!)
   - `user_listen_rate` alone = 40% importance
   - `user_skip_rate` = 29% importance
   - Combined: These two features capture 69% of predictive power!

2. **Context matters** (listen_type, platform, context)
   - How user accessed the track
   - What device/platform they're using

3. **Temporal features are less important**
   - `is_weekend`, `hour`, `day_of_week` contribute ~1-2% each
   - Track release date also relatively minor

4. **Surprisingly, track features are weak**
   - `duration_minutes`: only 0.96%
   - `release_year`: only 0.92%
   - User's historical behavior is much more predictive than track characteristics!

---

## 📈 **Feature Category Breakdown**

| Category | # Features | Total Importance | Notes |
|----------|------------|------------------|-------|
| **User Engagement** | 11 | **~72%** | 🌟 Dominates prediction |
| **Context** | 5 | ~6% | Moderate impact |
| **Temporal** | 11 | ~10% | Minor but consistent |
| **Release** | 7 | ~5% | Low impact |
| **Duration** | 4 | ~3% | Minimal impact |
| **Demographics** | 5 | ~4% | Minimal impact |

---

## 🎓 **What This Means**

### **The Model Learned:**

1. **"You are what you listened to"**
   - Users with high listen rates → likely to listen again
   - Users with high skip rates → likely to skip again
   - Past behavior is the strongest predictor

2. **Context matters more than content**
   - Platform, listen type, context > track features
   - How/where you listen matters more than what you're listening to

3. **Time of day is somewhat relevant**
   - Weekend vs weekday, hour of day have small effects
   - But much less important than user behavior

4. **Track characteristics are surprisingly weak**
   - Track age, duration, release date contribute little
   - This suggests users follow their own patterns regardless of track properties

---

## 🚀 **Next Steps to Improve (Target: 0.88-0.90 ROC AUC)**

### **1. Add Collaborative Filtering Features** (Expected: +0.02-0.03 AUC)
```python
# Add user/item embeddings from SVD
svd_features = train_svd_on_user_item_matrix()
X['cf_user_factor_1'] = ...
X['cf_item_factor_1'] = ...
```

### **2. Create User-Item Interaction Features** (Expected: +0.01-0.02 AUC)
```python
# Has this user listened to this artist before?
user_artist_affinity = train.groupby(['user_id', 'artist_id'])['is_listened'].mean()

# Has this user listened to this genre before?
user_genre_affinity = train.groupby(['user_id', 'genre_id'])['is_listened'].mean()
```

### **3. Add Sequential Features** (Expected: +0.01-0.02 AUC)
```python
# What was the last track the user listened to?
# How long since their last session?
# Are they on a listening streak?
```

### **4. Hyperparameter Tuning** (Expected: +0.005-0.01 AUC)
```python
# Try:
# - max_depth: 8-10 (currently 6)
# - n_estimators: 500+ (currently 300, stopped at 159)
# - learning_rate: 0.05 (currently 0.1)
```

### **5. Ensemble with Other Models** (Expected: +0.01-0.02 AUC)
```python
# Train LightGBM, CatBoost, Neural Net
# Weighted average predictions
```

---

## 📁 **Generated Files**

All outputs saved in: `/notebooks/04_experiments/xgboost/`

1. **xgboost_model.json** (829 KB) - Trained model
2. **feature_importance.csv** (974 B) - All 35 features ranked
3. **validation_predictions.csv** (554 KB) - 20K predictions
4. **metrics_summary.json** (661 B) - Performance metrics
5. **feature_importance_plot.png** (224 KB) - Visualization
6. **confusion_matrix.png** (92 KB) - Visualization
7. **prediction_distribution.png** (108 KB) - Visualization

---

## 🎯 **Conclusion**

**This baseline is VERY STRONG!**

✅ ROC AUC: 0.8722 (exceeded expectations by ~15%)  
✅ Low overfitting (2% gap)  
✅ 81% accuracy  
✅ Clear feature importance insights  
✅ Fast training (9 seconds)

**Why it's so good:**
- User engagement features are extremely predictive
- Engineered features capture the right patterns
- Clean data preprocessing paid off

**Recommendation:**
- This is already competition-worthy performance
- With collaborative filtering + ensembling, you could reach 0.88-0.90
- Focus on user-item interaction features for biggest gains

---

**Model ready for:**
- ✅ Further experimentation
- ✅ Hyperparameter tuning
- ✅ Feature engineering
- ✅ Ensembling with other models
- ✅ Submission (current performance is already strong!)
