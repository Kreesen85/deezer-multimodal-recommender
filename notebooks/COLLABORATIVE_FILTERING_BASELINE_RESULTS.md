# Collaborative Filtering Baseline Results

**Date**: February 1, 2026  
**Approach**: Pure Collaborative Filtering (No Features)  
**Dataset**: 500,000 interactions from Deezer DSG17

---

## Executive Summary

✅ **Best Model**: User+Item Bias Model  
✅ **AUC**: 0.7699 (77% accuracy in ranking)  
✅ **Accuracy**: 75.0%  
✅ **Training Time**: < 1 second  

**Key Finding**: Simple collaborative filtering with user and item biases achieves strong performance **without using any features** (demographics, genres, temporal).

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Interactions** | 500,000 |
| **Unique Users** | 17,429 |
| **Unique Items (tracks)** | 20,475 |
| **Listen Rate** | 69.8% (30.2% skip rate) |
| **Matrix Sparsity** | 99.86% |
| **Train/Test Split** | 80/20 |

---

## Models Compared

### 1. **User+Item Bias Model** 🏆 WINNER

**Method**: Predict based on global mean + user tendency + item popularity

```
prediction = global_mean + user_bias + item_bias
where:
  user_bias = user's avg listen rate - global mean
  item_bias = item's avg listen rate - global mean
```

**Results**:
- **AUC**: 0.7699
- **RMSE**: 0.4168
- **MAE**: 0.3134
- **Accuracy**: 74.98%
- **Training Time**: 0.5s

**Pros**:
- ✅ Fast training and prediction
- ✅ Simple, interpretable
- ✅ Production-ready

**Cons**:
- ❌ Cold start problem (new users/items)
- ❌ Doesn't use features

---

### 2. **SVD (Matrix Factorization)**

**Method**: Decompose user-item matrix into latent factors

```
prediction = user_factors · item_factors
```

**Configuration**:
- Latent factors: 50
- Explained variance: 73.12%

**Results**:
- **AUC**: 0.6254
- **RMSE**: 0.6506
- **MAE**: 0.4566
- **Accuracy**: 53.68%
- **Training Time**: 0.3s

**Note**: Underperformed expectations. Possible reasons:
- High sparsity (99.86%)
- Binary ratings (0/1) not ideal for SVD
- May need more tuning

---

### 3. **NMF (Non-Negative Matrix Factorization)**

**Method**: Matrix factorization with non-negativity constraints

**Results**:
- **AUC**: 0.5597
- **RMSE**: 0.7206
- **MAE**: 0.5445
- **Accuracy**: 45.21%
- **Training Time**: 39.1s

**Note**: Slowest and least accurate. Not recommended for this problem.

---

### 4. **Global Baseline (Mean Predictor)**

**Method**: Predict global average for everyone

**Results**:
- **AUC**: 0.5000 (random classifier)
- **RMSE**: 0.4587
- **MAE**: 0.4211
- **Accuracy**: 69.90%
- **Training Time**: 0.1s

**Note**: Reference baseline. AUC = 0.5 confirms no discriminative power.

---

## Performance Ranking

| Rank | Model | AUC | RMSE | Accuracy | Speed |
|------|-------|-----|------|----------|-------|
| **1** | User+Item Bias | **0.7699** | 0.4168 | 74.98% | ⚡⚡⚡ |
| **2** | SVD | 0.6254 | 0.6506 | 53.68% | ⚡⚡⚡ |
| **3** | NMF | 0.5597 | 0.7206 | 45.21% | ⚡ |
| **4** | Global Baseline | 0.5000 | 0.4587 | 69.90% | ⚡⚡⚡ |

---

## Key Insights

### 1. **User+Item Bias is Surprisingly Strong**

The simplest model (User+Item Bias) **outperformed matrix factorization** by **14.5 AUC points**!

**Why?**
- Captures individual user preferences (some users listen more than others)
- Captures item quality (some tracks are more popular)
- Less affected by sparsity
- Binary ratings (0/1) fit bias model better than latent factors

### 2. **Sparsity is a Major Challenge**

- 99.86% sparse matrix
- Average user: 28.7 interactions
- Average item: 24.4 interactions
- Many users/items have very few observations

### 3. **SVD Underperformed**

Expected to be best, but:
- Binary ratings not ideal for SVD (designed for 1-5 star ratings)
- Extreme sparsity hurts latent factor learning
- May improve with more tuning

### 4. **Listen Rate is High (69.8%)**

- Users listen to most tracks they're exposed to
- Predicting "always listen" achieves 70% accuracy
- Real challenge: ranking which tracks to show

---

## Limitations of Pure Collaborative Filtering

### ❌ **Cold Start Problem**

**New Users:**
- Cannot predict for users with no history
- Must fall back to popularity or demographics

**New Items:**
- Cannot recommend newly released tracks
- Must rely on content features (genre, artist)

### ❌ **Feature Blindness**

Ignores valuable information:
- User demographics (age, gender)
- Track features (genre, duration, artist)
- Temporal patterns (time of day, day of week)
- Context (platform, listening situation)

### ❌ **Scalability Concerns**

- Matrix size grows with users × items
- Real Deezer: millions of users, millions of tracks
- Sparse matrix storage helps, but updates are slow

---

## Comparison: With vs Without Features

### Current (CF Only):
- **AUC**: 0.7699
- **Uses**: Only interaction history (user-item pairs)
- **Cold start**: ❌ Cannot handle
- **Features**: ❌ Not used

### Expected (Hybrid Model with Features):
- **AUC**: **0.85-0.90** (expected)
- **Uses**: Interaction history + demographics + content + temporal
- **Cold start**: ✅ Handled with content-based fallback
- **Features**: ✅ All 50+ engineered features

**Expected improvement**: +8-12 AUC points

---

## Next Steps for Competition System (Q1)

### 1. **Hybrid Model with Features** (Priority 1)

Build feature-enhanced recommender:
- Use all 50+ engineered features
- Combine CF signals with content features
- Handle cold start gracefully

**Options:**
- Custom implementation with scikit-learn
- LightFM (if compatibility resolved)
- Gradient boosting (XGBoost/LightGBM) on user-item pairs + features

### 2. **Ensemble Strategy**

Combine multiple approaches:
- User+Item Bias (fast, strong baseline)
- Feature-based model (handles cold start)
- Weighted average or stacking

### 3. **Hyperparameter Tuning**

For SVD:
- Try different latent factor counts (20, 100, 200)
- Experiment with regularization
- Test on larger sample

---

## Next Steps for Production System (Q2)

### 1. **Two-Stage Architecture**

**Stage 1: Candidate Generation (Fast)**
- Use User+Item Bias for quick filtering
- Retrieve top 1000 candidates
- < 10ms latency

**Stage 2: Ranking (Precise)**
- Feature-based model on candidates
- Score and rank top 50
- < 50ms latency

### 2. **Cold Start Strategy**

**New Users:**
- Ask for genre preferences during onboarding
- Use demographic-based defaults (age, gender)
- Quick-learn from first 5-10 listens

**New Items:**
- Content-based similarity (genre, artist)
- Trending tracks (popularity spike)
- Editorial playlists (curated)

### 3. **Continuous Learning**

- Update user/item biases daily
- Retrain feature models weekly
- A/B test new models before full deployment

---

## Files Generated

| File | Description |
|------|-------------|
| `baseline_collaborative_filtering.py` | Complete implementation script |
| `collaborative_filtering_results.csv` | Model comparison table |
| `collaborative_filtering_results.png` | Visualization (4 plots) |

---

## Conclusion

✅ **Collaborative filtering baseline established**: AUC 0.7699  
✅ **User+Item Bias is production-ready**: Fast, simple, effective  
✅ **Clear path forward**: Add features to improve to AUC 0.85-0.90  
✅ **Foundation for report**: Q1 (competition) and Q2 (production) sections  

**Status**: Ready to build hybrid feature-based models next 🚀

---

*For reproduction: `notebooks/baseline_collaborative_filtering.py`*  
*Processing time: ~57 seconds for 500K interactions*
