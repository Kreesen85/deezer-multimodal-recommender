# XGBoost Full Dataset Results (500K)

**Date:** 2026-02-01 23:51:46  
**Dataset:** 500,000 preprocessed samples (FULL DATASET)  
**Training Time:** 13 seconds

---

## 🎯 **Performance: ROC AUC 0.8290**

### Performance Comparison:

| Dataset | Samples | ROC AUC | Accuracy | Training Time |
|---------|---------|---------|----------|---------------|
| **100K** | 100,000 | **0.8722** | 81% | 9 sec |
| **500K (Full)** | 500,000 | **0.8290** | 78% | 13 sec |

### ⚠️ **Interesting Finding: Performance DECREASED with more data!**

**Why?**
- 100K sample: 0.8722 ROC AUC
- 500K sample: 0.8290 ROC AUC
- **Difference: -0.0432 (-4.3%)**

---

## 🔍 **Analysis: Why Did Performance Drop?**

### **Possible Reasons:**

1. **Sample Quality Difference**
   - 100K: Might have been more homogeneous/easier users
   - 500K: More diverse user behaviors, harder patterns

2. **User/Item Coverage**
   - 100K: 12,173 users, 4,977 items
   - 500K: 17,429 users, 20,475 items
   - More items = more cold-start/rare items = harder to predict

3. **Feature Stability**
   - User engagement features computed on different data
   - 500K has more variety → features less predictive

4. **Need More Capacity**
   - Current model (depth 6, 300 trees) might be underfitting on 500K
   - Need deeper trees or more trees for larger dataset

---

## 📊 **Detailed Results (500K)**

### Classification Performance:

| Metric | Skip (0) | Listen (1) |
|--------|----------|------------|
| **Precision** | 0.69 | 0.81 |
| **Recall** | 0.50 | 0.90 |
| **F1-Score** | 0.58 | 0.85 |

### Confusion Matrix:

```
                    Predicted
                Skip        Listen
Actual Skip     15,176      14,981    (50% recall - WORSE)
Actual Listen    6,799      63,044    (90% recall - same)
```

**Key Changes from 100K:**
- ❌ Skip recall dropped: 59% → 50% (-9%)
- ✅ Listen recall stable: 90% → 90%
- ❌ Overall accuracy: 81% → 78% (-3%)

---

## 🔍 **Feature Importance Changes**

### Top Features Comparison:

| Rank | 100K Dataset | Importance | 500K Dataset | Importance |
|------|--------------|------------|--------------|------------|
| 1 | user_listen_rate | 39.98% | **user_skip_rate** | 39.02% |
| 2 | user_skip_rate | 29.16% | **user_listen_rate** | 29.22% |
| 3 | user_engagement_segment | 2.09% | **listen_type** | 3.27% |

**Key Changes:**
- ⚠️ **Rankings swapped!** Skip rate now #1 (was #2)
- ✅ User behavior still dominates (~69% combined)
- ⬆️ `listen_type` jumped from 1.59% → 3.27% (more important in full data)
- ⬆️ `is_new_release` appeared in top features (0.0117)

---

## 🎓 **Insights from Full Dataset**

### **What We Learned:**

1. **More data ≠ better performance** (in this case)
   - Need to tune model for larger dataset
   - Current hyperparameters optimized for smaller data

2. **Skip patterns are harder with more users**
   - Skip recall dropped from 59% → 50%
   - More diverse user behaviors = harder to model

3. **User behavior features remain dominant**
   - Still 68% importance combined
   - But effectiveness reduced on larger, more diverse data

4. **Context (listen_type) matters more at scale**
   - Jumped to 3.27% importance
   - How users access music becomes more relevant

---

## 🚀 **Recommendations to Improve 500K Performance**

### **1. Increase Model Capacity** (Expected: +0.03-0.04)
```python
XGBOOST_PARAMS = {
    'n_estimators': 500,     # More trees (was 300)
    'max_depth': 8,          # Deeper trees (was 6)
    'learning_rate': 0.05,   # Slower learning (was 0.1)
}
```

### **2. Add Collaborative Filtering** (Expected: +0.02-0.03)
- User/item embeddings for cold-start items
- 20,475 items need better representation

### **3. User-Item Interaction Features** (Expected: +0.02)
- Has user listened to this artist before?
- Has user listened to this genre before?
- User-specific track preferences

### **4. Better Feature Engineering** (Expected: +0.01-0.02)
- Normalize user features across full dataset
- Add item popularity features
- Temporal sequence features

### **5. Handle Class Imbalance Better** (Expected: +0.01)
```python
scale_pos_weight = len(y[y==0]) / len(y[y==1])
# Currently: 30.16% skips, 69.84% listens
```

---

## 📈 **Expected Performance After Improvements**

| Improvement | Expected ROC AUC |
|-------------|------------------|
| Current (500K) | 0.8290 |
| + Model capacity | 0.8590 |
| + Collaborative filtering | 0.8790 |
| + User-item features | 0.8990 |
| + Ensembling | 0.9100+ |

---

## 💡 **Key Takeaway**

**The 100K sample performance (0.8722) is actually more impressive than expected!**

It suggests:
- The 100K sample was a "lucky" subset OR
- The model is well-suited for that scale OR
- The 500K dataset has inherently harder patterns

**Next steps:**
1. ✅ Keep the 100K model as a strong baseline
2. 🔧 Tune hyperparameters specifically for 500K
3. 🎯 Add collaborative filtering features
4. 🚀 Try ensemble approaches

---

## 📊 **Summary Statistics**

### Dataset Comparison:

| Metric | 100K | 500K |
|--------|------|------|
| **Samples** | 100,000 | 500,000 |
| **Users** | 12,173 | 17,429 (+43%) |
| **Items** | 4,977 | 20,475 (+311%) |
| **Listen Rate** | 69.29% | 69.84% |
| **ROC AUC** | 0.8722 | 0.8290 |
| **Accuracy** | 81% | 78% |
| **Overfitting** | 2.0% | 1.2% |
| **Training Time** | 9 sec | 13 sec |

**The item count increased by 311%** - this is likely the main challenge!

More items means:
- More cold-start items
- Sparser user-item matrix
- Harder to predict without collaborative filtering

---

**Conclusion:** The 100K model is excellent, but the 500K model needs tuning and additional features (especially collaborative filtering) to handle the increased complexity.
