# Using the Test Dataset for Final Predictions

## 📋 **Understanding the Test Dataset**

### **What We Have:**

**Test File:** `data/raw/test.csv`
- **Rows:** 19,918 (one per user)
- **Size:** 1.4 MB
- **Structure:** Same features as train.csv, **BUT NO `is_listened` COLUMN**

**Key difference from training:**
```
train.csv: Multiple interactions per user (listening history)
test.csv:  ONE row per user (the track we need to predict)
```

---

## 🎯 **The Task**

For each row in test.csv, predict:
> **"Will this user listen to THIS specific recommended track?"**

Output: Probability between 0 and 1

---

## 🔄 **Current Workflow vs Final Workflow**

### **What We've Been Doing (Development):**

```
train.csv (7.5M rows)
    ↓
Split 80/20
    ↓
├── Internal Train (80%) → Train model
└── Internal Test (20%)  → Evaluate (get ROC AUC)
```

**Purpose:** Develop and validate model performance

---

### **What We Need to Do (Final Submission):**

```
train.csv (7.5M rows) + test.csv (19,918 rows)
    ↓
1. Train on ALL of train.csv (no split)
    ↓
2. Preprocess test.csv (add same 31 features)
    ↓
3. Load trained model
    ↓
4. Predict probabilities for test.csv
    ↓
5. Create submission.csv
```

---

## 📝 **Submission Format**

```csv
sample_id,is_listened
0,0.73
1,0.42
2,0.89
3,0.15
...
19917,0.68
```

- `sample_id`: Row number from test.csv (0-19917)
- `is_listened`: Predicted probability (0.0 - 1.0)

---

## 🔧 **Implementation Steps**

### **Step 1: Preprocess Test Data**

Create script: `notebooks/02_preprocessing/preprocess_test.py`

```python
import pandas as pd
import sys
sys.path.append('../..')
from src.data.preprocessing import add_temporal_features, add_release_features, add_duration_features

# Load test data
test_df = pd.read_csv('../../data/raw/test.csv')
print(f"Test data: {len(test_df)} rows")

# Add same features as training
test_df = add_temporal_features(test_df)
test_df = add_release_features(test_df)
test_df = add_duration_features(test_df)

# Load user stats computed from TRAINING data
user_stats = pd.read_csv('../../data/processed/preprocessing/user_stats_from_train.csv')

# Merge user features (use training-based stats!)
test_df = test_df.merge(user_stats, on='user_id', how='left')

# Handle cold-start users (not in training)
# Fill missing user features with defaults
user_feature_cols = [col for col in test_df.columns if col.startswith('user_')]
test_df[user_feature_cols] = test_df[user_feature_cols].fillna({
    'user_listen_rate': test_df['user_listen_rate'].mean(),
    'user_skip_rate': test_df['user_skip_rate'].mean(),
    # ... etc for all user features
})

# Save preprocessed test
test_df.to_csv('../../data/processed/preprocessing/test_preprocessed.csv', index=False)
print("✓ Test data preprocessed")
```

---

### **Step 2: Train Final Model on Full Training Data**

Create script: `notebooks/04_experiments/xgboost/train_final_model.py`

```python
import pandas as pd
import xgboost as xgb

# Load FULL training data (no split!)
train_df = pd.read_csv('../../data/processed/preprocessing/train_preprocessed_sample.csv')

# Define features (same as before)
features = [col for col in train_df.columns 
            if col not in ['user_id', 'media_id', 'is_listened', 
                           'datetime', 'release_date_parsed', 'listen_date', 
                           'ts_listen', 'release_date']]

X_train = train_df[features]
y_train = train_df['is_listened']

print(f"Training final model on {len(X_train):,} samples")

# Train with best hyperparameters found during development
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("✓ Model trained")

# Save final model
model.save_model('xgboost_final_model.json')
print("✓ Model saved")
```

---

### **Step 3: Generate Predictions for Test Data**

Create script: `notebooks/04_experiments/xgboost/predict_test.py`

```python
import pandas as pd
import xgboost as xgb
import numpy as np

# Load preprocessed test data
test_df = pd.read_csv('../../data/processed/preprocessing/test_preprocessed.csv')
print(f"Loaded {len(test_df):,} test samples")

# Load trained model
model = xgb.XGBClassifier()
model.load_model('xgboost_final_model.json')
print("✓ Model loaded")

# Define features (same as training)
features = [col for col in test_df.columns 
            if col not in ['sample_id', 'user_id', 'media_id', 
                           'datetime', 'release_date_parsed', 'listen_date',
                           'ts_listen', 'release_date']]

X_test = test_df[features]

# Handle missing features (fill with median/mean)
X_test = X_test.fillna(X_test.median())

# Generate predictions
print("Generating predictions...")
predictions = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (listened)

# Create submission dataframe
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'is_listened': predictions
})

# Verify format
print(f"\nSubmission preview:")
print(submission.head(10))
print(f"\nPrediction statistics:")
print(f"  Min: {predictions.min():.4f}")
print(f"  Max: {predictions.max():.4f}")
print(f"  Mean: {predictions.mean():.4f}")
print(f"  Median: {np.median(predictions):.4f}")

# Save submission
submission.to_csv('submission.csv', index=False)
print(f"\n✓ Submission saved: submission.csv ({len(submission):,} rows)")
```

---

### **Step 4: Verify Submission**

```python
# Check submission format
import pandas as pd

sub = pd.read_csv('submission.csv')

# Validation checks
assert len(sub) == 19918, "Wrong number of rows!"
assert list(sub.columns) == ['sample_id', 'is_listened'], "Wrong columns!"
assert sub['sample_id'].min() == 0, "sample_id should start at 0"
assert sub['sample_id'].max() == 19917, "sample_id should end at 19917"
assert (sub['is_listened'] >= 0).all(), "Probabilities must be >= 0"
assert (sub['is_listened'] <= 1).all(), "Probabilities must be <= 1"

print("✓ Submission format is valid!")
```

---

## ⚠️ **Important Considerations**

### **1. User Features Must Come from Training Data**

```python
# ✅ CORRECT: Use user stats from training
user_stats = pd.read_csv('user_stats_from_train.csv')
test_df = test_df.merge(user_stats, on='user_id', how='left')

# ❌ WRONG: Computing user stats from test data
# This is data leakage! Test data should not influence features.
```

### **2. Cold-Start Users**

Some test users might not be in training data:

```python
# Handle users not seen in training
missing_mask = test_df['user_listen_rate'].isna()
print(f"Cold-start users: {missing_mask.sum()}")

# Fill with global averages from training
test_df.loc[missing_mask, 'user_listen_rate'] = 0.5  # Or training mean
```

### **3. Feature Consistency**

Test data must have EXACT same features as training:

```python
# Get training feature names
train_features = X_train.columns.tolist()

# Ensure test has same features
missing_features = set(train_features) - set(X_test.columns)
if missing_features:
    print(f"Missing features in test: {missing_features}")
    # Add missing columns with defaults
```

---

## 📊 **Complete Pipeline**

Create master script: `generate_submission.sh`

```bash
#!/bin/bash

echo "=== Generating Final Submission ==="

# Step 1: Preprocess test data
echo "1. Preprocessing test data..."
cd notebooks/02_preprocessing
python preprocess_test.py

# Step 2: Train final model on full training data
echo "2. Training final model..."
cd ../04_experiments/xgboost
python train_final_model.py

# Step 3: Generate predictions
echo "3. Generating predictions..."
python predict_test.py

# Step 4: Verify submission
echo "4. Verifying submission..."
python verify_submission.py

echo "✓ Done! Check submission.csv"
```

---

## 🎯 **Expected Results**

Based on your validation performance:

| Model | Validation ROC AUC | Expected Test Performance |
|-------|-------------------|---------------------------|
| XGBoost (100K) | 0.8722 | ~0.865-0.875 |
| XGBoost (500K) | 0.8290 | ~0.820-0.830 |

**Why lower on test?**
- Different distribution (one track per user vs full history)
- Possible overfitting to training patterns
- Cold-start users

---

## 📝 **Quick Start**

Want me to create these scripts for you? I can:

1. ✅ Create `preprocess_test.py`
2. ✅ Create `train_final_model.py`
3. ✅ Create `predict_test.py`
4. ✅ Create `verify_submission.py`
5. ✅ Generate `submission.csv`

This will give you a complete, ready-to-submit prediction file!

---

**Ready to generate your test predictions?** 🚀
