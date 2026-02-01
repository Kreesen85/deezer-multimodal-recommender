# 03_baselines/ - Baseline Models

Collaborative filtering and baseline model implementations.

## 🎯 Models

### `baseline_collaborative_filtering.py` ⭐ **MAIN BASELINE**
Scikit-learn collaborative filtering models
- **User+Item Bias**: AUC 0.7699 (BEST)
- **SVD**: Matrix factorization
- **NMF**: Non-negative matrix factorization
- **Global Baseline**: Mean predictor

**Run**: `python baseline_collaborative_filtering.py`

**Expected Runtime**: ~60 seconds for 500K interactions

### `baseline_surprise_models.py`
Surprise library models (requires scikit-surprise)
- SVD, SVD++, NMF
- KNN-based models
- Note: May not work with Python 3.13

---

## 📊 Results

### Best Model: User+Item Bias
```
AUC:      0.7699
Accuracy: 74.98%
RMSE:     0.4168
Speed:    ⚡⚡⚡ Very fast
```

### Performance Comparison
| Model | AUC | Speed |
|-------|-----|-------|
| User+Item Bias | **0.7699** | ⚡⚡⚡ |
| SVD | 0.6254 | ⚡⚡⚡ |
| NMF | 0.5597 | ⚡ |
| Global Baseline | 0.5000 | ⚡⚡⚡ |

---

## 📁 Generated Files

- `collaborative_filtering_results.csv` - Model comparison table
- `collaborative_filtering_results.png` - Visualization (4 plots)

---

## 🔑 Key Insights

1. **Simple bias model outperforms matrix factorization**
   - User+Item Bias: 0.7699 AUC
   - SVD: 0.6254 AUC
   - Why? Binary ratings favor bias over latent factors

2. **Pure CF achieves ~77% AUC without features**
   - No demographics, no temporal, no content
   - Just user-item interactions
   - Strong baseline!

3. **Cold start is a problem**
   - Cannot predict for new users/items
   - Need feature-based fallback

4. **Expected improvement with features: +8-12 AUC points**
   - Target: 0.85-0.90 AUC with full feature set

---

## 📚 Documentation

See `../docs/COLLABORATIVE_FILTERING_BASELINE_RESULTS.md` for:
- Complete results analysis
- Model comparisons
- Next steps for production system
- Feature-based recommendations

---

## 🚀 Quick Start

```bash
# Run baseline CF models
python baseline_collaborative_filtering.py

# Check results
cat collaborative_filtering_results.csv
```

---

## 💡 Next Steps

1. Add features (user demographics, track content, temporal)
2. Hybrid model (CF + features)
3. Handle cold start problem
4. Production deployment (2-stage: retrieval + ranking)

Expected AUC with features: **0.85-0.90**

---

*Baseline established: AUC 0.7699 (pure CF, no features)*
