# 04_experiments - Model Experiments

This folder contains experimental models and approaches for the Deezer Skip Prediction challenge.

---

## 🎯 Task Overview

**Objective:** Predict whether users will listen to their recommended tracks (>30 seconds) or skip them

**Evaluation Metric:** ROC AUC

**Dataset:**
- Training: User listening history over 1 month (~7.5M interactions)
- Test: First recommended track for each user (one row per user)

---

## 📂 Experiment Structure

```
04_experiments/
├── xgboost/                    # XGBoost baseline (CURRENT)
│   ├── xgboost_baseline.py    # Training script
│   ├── README.md              # Detailed documentation
│   └── run.sh                 # Quick start script
│
├── lightgbm/                   # LightGBM experiments (TODO)
├── neural_net/                 # Deep learning models (TODO)
├── collaborative_filtering/    # CF-based approaches (TODO)
└── ensemble/                   # Model ensembles (TODO)
```

---

## 🚀 Current Experiments

### ✅ 1. XGBoost Baseline

**Status:** Implemented  
**Location:** `xgboost/`

**Features:**
- Uses all 46 engineered features (temporal, release, duration, user engagement)
- ROC AUC evaluation on validation set
- Feature importance analysis
- Comprehensive visualizations

**Quick Start:**
```bash
cd xgboost
./run.sh
```

**Expected Performance:** 0.70-0.75 ROC AUC

**Documentation:** See `xgboost/README.md`

---

## 📋 Planned Experiments

### 🔲 2. LightGBM
- Faster training than XGBoost
- Better handling of categorical features
- Expected: 0.71-0.76 ROC AUC

### 🔲 3. Collaborative Filtering + Features
- Hybrid approach: CF embeddings + engineered features
- User/item latent factors from SVD/NMF
- Expected: 0.72-0.76 ROC AUC

### 🔲 4. Neural Networks
- Deep feedforward network
- Learn non-linear feature interactions
- Expected: 0.69-0.74 ROC AUC

### 🔲 5. Sequential Models (LSTM/GRU)
- Use user's listening sequence
- Capture temporal patterns
- Expected: 0.71-0.75 ROC AUC

### 🔲 6. Model Ensemble
- Weighted combination of best models
- Stacking/blending approaches
- Expected: 0.76-0.80 ROC AUC

---

## 📊 Experiment Tracking

| Experiment | Status | ROC AUC (Val) | Features | Notes |
|------------|--------|---------------|----------|-------|
| XGBoost Baseline | ✅ Done | TBD | 46 | Baseline model |
| LightGBM | 🔲 TODO | - | 46 | - |
| XGB + CF | 🔲 TODO | - | 46 + CF | - |
| Neural Net | 🔲 TODO | - | 46 | - |
| Ensemble | 🔲 TODO | - | All | - |

---

## 🔧 Best Practices

### Experiment Workflow:
1. **Create folder** for each experiment (e.g., `xgboost/`)
2. **Write script** with clear documentation
3. **Add README** with usage instructions
4. **Save outputs**: model, predictions, metrics, plots
5. **Gitignore outputs** (keep only code and docs)
6. **Update tracking table** above with results

### Output Files (per experiment):
- `model.*` - Trained model
- `feature_importance.csv` - Feature analysis
- `validation_predictions.csv` - Predictions for error analysis
- `metrics_summary.json` - Performance metrics
- `*.png` - Visualizations

---

## 🎯 Performance Goals

| Phase | Target ROC AUC | Approaches |
|-------|----------------|------------|
| Phase 1: Baseline | 0.70-0.72 | Single feature-based model |
| Phase 2: Improved | 0.72-0.75 | Better features, tuning, CF hybrid |
| Phase 3: Advanced | 0.75-0.78 | Ensemble, sequential models |
| Phase 4: Optimized | 0.78-0.80 | Full ensemble, feature engineering |

---

## 📚 References

- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Scikit-learn**: https://scikit-learn.org/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **ROC AUC**: https://en.wikipedia.org/wiki/Receiver_operating_characteristic

---

## 🐛 Troubleshooting

### Missing preprocessed data?
```bash
cd ../02_preprocessing
python demo_preprocessing_with_users.py
```

### Missing packages?
```bash
pip install xgboost lightgbm scikit-learn matplotlib seaborn
```

---

**Last Updated:** 2026-02-01  
**Current Focus:** XGBoost Baseline
