"""
Full Submission Pipeline - Deezer Skip Prediction
===================================================

This script handles the complete end-to-end pipeline:
1. Load full training data (7.5M rows)
2. Compute user stats from ALL training data
3. Preprocess training data (add 31 features)
4. Train XGBoost on full training data
5. Preprocess test.csv with same features
6. Generate submission.csv

Usage:
    cd notebooks/04_experiments/xgboost
    /opt/anaconda3/bin/python generate_submission.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import json
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = "/Users/kreesen/Documents/deezer-multimodal-recommender"
sys.path.append(PROJECT_ROOT)
from src.data.preprocessing import (
    add_temporal_features,
    add_release_features,
    add_duration_features,
    compute_user_features_from_train,
    apply_user_features,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/raw/test.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Training sample size: None = full 7.5M, or set a number for faster runs
TRAIN_SAMPLE_SIZE = None

# XGBoost hyperparameters (same as baseline that achieved 0.8722 AUC)
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "eval_metric": "auc",
    "n_jobs": -1,
}

# Columns to exclude from features
EXCLUDE_COLS = [
    "is_listened",          # Target
    "sample_id",            # Test set ID
    "user_id",              # ID
    "media_id",             # ID
    "artist_id",            # High cardinality ID
    "album_id",             # High cardinality ID
    "genre_id",             # High cardinality ID
    "datetime",             # Converted to features
    "release_date_parsed",  # Converted to features
    "listen_date",          # Converted to features
    "ts_listen",            # Converted to features
    "release_date",         # Converted to features
]

RANDOM_STATE = 42

# ============================================================================

print("=" * 80)
print("DEEZER SKIP PREDICTION - FULL SUBMISSION PIPELINE")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Project root: {PROJECT_ROOT}\n")


# ============================================================================
# STEP 1: Load Training Data
# ============================================================================
print("STEP 1: Loading training data...")
print(f"Path: {TRAIN_PATH}")

train_df = pd.read_csv(TRAIN_PATH, nrows=TRAIN_SAMPLE_SIZE)
print(f"  Loaded {len(train_df):,} rows")
print(f"  Users: {train_df['user_id'].nunique():,}")
print(f"  Items: {train_df['media_id'].nunique():,}")
print(f"  Listen rate: {train_df['is_listened'].mean():.2%}")
print()


# ============================================================================
# STEP 2: Compute User Stats from FULL Training Data
# ============================================================================
print("STEP 2: Computing user stats from full training data...")

user_stats = compute_user_features_from_train(train_df)
user_stats_path = os.path.join(OUTPUT_DIR, "user_stats_full.csv")
user_stats.to_csv(user_stats_path, index=False)
print(f"  Saved user stats to: {user_stats_path}")
print(f"  Users with stats: {len(user_stats):,}")
print()


# ============================================================================
# STEP 3: Preprocess Training Data (add features)
# ============================================================================
print("STEP 3: Preprocessing training data...")

print("  Adding temporal features...")
train_df = add_temporal_features(train_df)

print("  Adding release features...")
train_df = add_release_features(train_df)

print("  Adding duration features...")
train_df = add_duration_features(train_df)

print("  Applying user features...")
train_df = apply_user_features(train_df, user_stats)

print(f"  Total columns after preprocessing: {len(train_df.columns)}")
print()


# ============================================================================
# STEP 4: Prepare Features and Train XGBoost
# ============================================================================
print("STEP 4: Training XGBoost on full training data...")

# Select feature columns
feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS]
print(f"  Features selected: {len(feature_cols)}")

X_train = train_df[feature_cols].copy()
y_train = train_df["is_listened"].copy()

# Handle any missing values
missing_count = X_train.isnull().sum().sum()
if missing_count > 0:
    print(f"  Filling {missing_count:,} missing values with median...")
    X_train = X_train.fillna(X_train.median())

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Listen rate: {y_train.mean():.2%}")
print()

# Train model (no validation split - using ALL data for final model)
print(f"  Training XGBoost ({XGBOOST_PARAMS['n_estimators']} trees, "
      f"depth {XGBOOST_PARAMS['max_depth']})...")
print(f"  This may take a few minutes on {len(X_train):,} rows...")

model = xgb.XGBClassifier(**XGBOOST_PARAMS)
model.fit(X_train, y_train, verbose=50)

# Save model
model_path = os.path.join(OUTPUT_DIR, "xgboost_final_model.json")
model.save_model(model_path)
print(f"\n  Model saved to: {model_path}")

# Save feature importance
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_path = os.path.join(OUTPUT_DIR, "feature_importance_final.csv")
importance_df.to_csv(importance_path, index=False)
print(f"  Feature importance saved to: {importance_path}")

print("\n  Top 10 features:")
for _, row in importance_df.head(10).iterrows():
    print(f"    {row['feature']:35s} {row['importance']:.4f}")
print()


# ============================================================================
# STEP 5: Load and Preprocess Test Data
# ============================================================================
print("STEP 5: Loading and preprocessing test data...")
print(f"  Path: {TEST_PATH}")

test_df = pd.read_csv(TEST_PATH)
sample_ids = test_df["sample_id"].copy()
print(f"  Loaded {len(test_df):,} test rows")
print(f"  Users: {test_df['user_id'].nunique():,}")
print(f"  sample_id range: {sample_ids.min()} - {sample_ids.max()}")

# Add features (same pipeline as training)
print("  Adding temporal features...")
test_df = add_temporal_features(test_df)

print("  Adding release features...")
test_df = add_release_features(test_df)

print("  Adding duration features...")
test_df = add_duration_features(test_df)

print("  Applying user features from training data...")
test_df = apply_user_features(test_df, user_stats)

# Check user coverage
n_test_users = test_df["user_id"].nunique()
n_with_stats = test_df["user_listen_rate"].notna().sum()
print(f"  Test users with training stats: {n_with_stats:,} / {len(test_df):,}")
print()


# ============================================================================
# STEP 6: Generate Predictions
# ============================================================================
print("STEP 6: Generating predictions...")

# Ensure test has same features as training
test_feature_cols = [col for col in feature_cols if col in test_df.columns]
missing_features = set(feature_cols) - set(test_df.columns)
if missing_features:
    print(f"  WARNING: Missing features in test: {missing_features}")
    for feat in missing_features:
        test_df[feat] = 0  # Fill missing features with 0

X_test = test_df[feature_cols].copy()

# Handle missing values
missing_count = X_test.isnull().sum().sum()
if missing_count > 0:
    print(f"  Filling {missing_count:,} missing values...")
    # Use training medians for consistency
    train_medians = X_train.median()
    X_test = X_test.fillna(train_medians)

print(f"  X_test shape: {X_test.shape}")

# Predict probabilities
predictions = model.predict_proba(X_test)[:, 1]

print(f"\n  Prediction statistics:")
print(f"    Min:    {predictions.min():.4f}")
print(f"    Max:    {predictions.max():.4f}")
print(f"    Mean:   {predictions.mean():.4f}")
print(f"    Median: {np.median(predictions):.4f}")
print(f"    Std:    {predictions.std():.4f}")
print()


# ============================================================================
# STEP 7: Create and Verify Submission
# ============================================================================
print("STEP 7: Creating submission file...")

submission = pd.DataFrame({
    "sample_id": sample_ids,
    "is_listened": predictions,
})

# Sort by sample_id to be safe
submission = submission.sort_values("sample_id").reset_index(drop=True)

# Verification checks
print("\n  Verification checks:")

checks_passed = True

# Check row count
expected_rows = 19918
actual_rows = len(submission)
check_1 = actual_rows == expected_rows
print(f"    Row count: {actual_rows:,} (expected {expected_rows:,}) "
      f"{'PASS' if check_1 else 'FAIL'}")
if not check_1:
    checks_passed = False

# Check columns
expected_cols = ["sample_id", "is_listened"]
check_2 = list(submission.columns) == expected_cols
print(f"    Columns: {list(submission.columns)} "
      f"{'PASS' if check_2 else 'FAIL'}")
if not check_2:
    checks_passed = False

# Check sample_id range
check_3 = submission["sample_id"].min() == 0
check_4 = submission["sample_id"].max() == expected_rows - 1
print(f"    sample_id range: {submission['sample_id'].min()}-"
      f"{submission['sample_id'].max()} "
      f"{'PASS' if (check_3 and check_4) else 'FAIL'}")
if not (check_3 and check_4):
    checks_passed = False

# Check prediction range
check_5 = (submission["is_listened"] >= 0).all()
check_6 = (submission["is_listened"] <= 1).all()
print(f"    Predictions in [0,1]: "
      f"{'PASS' if (check_5 and check_6) else 'FAIL'}")
if not (check_5 and check_6):
    checks_passed = False

# Check no duplicates
check_7 = submission["sample_id"].nunique() == len(submission)
print(f"    No duplicate sample_ids: "
      f"{'PASS' if check_7 else 'FAIL'}")
if not check_7:
    checks_passed = False

# Check no NaN predictions
check_8 = submission["is_listened"].notna().all()
print(f"    No NaN predictions: "
      f"{'PASS' if check_8 else 'FAIL'}")
if not check_8:
    checks_passed = False

if checks_passed:
    print("\n  ALL CHECKS PASSED")
else:
    print("\n  SOME CHECKS FAILED - Review submission before submitting!")

# Save submission
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"\n  Submission saved to: {submission_path}")

# Preview
print(f"\n  Preview (first 10 rows):")
print(submission.head(10).to_string(index=False))
print()


# ============================================================================
# STEP 8: Save Pipeline Summary
# ============================================================================
print("STEP 8: Saving pipeline summary...")

summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_data": TRAIN_PATH,
    "test_data": TEST_PATH,
    "train_rows": int(len(train_df)),
    "train_users": int(train_df["user_id"].nunique()),
    "test_rows": int(len(test_df)),
    "test_users": int(test_df["user_id"].nunique()),
    "n_features": len(feature_cols),
    "feature_list": feature_cols,
    "hyperparameters": XGBOOST_PARAMS,
    "prediction_mean": float(predictions.mean()),
    "prediction_std": float(predictions.std()),
    "all_checks_passed": checks_passed,
}

summary_path = os.path.join(OUTPUT_DIR, "submission_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Summary saved to: {summary_path}")


# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nGenerated files:")
print(f"  - submission.csv              ({len(submission):,} predictions)")
print(f"  - xgboost_final_model.json    (trained model)")
print(f"  - user_stats_full.csv         (user features from full training data)")
print(f"  - feature_importance_final.csv (feature rankings)")
print(f"  - submission_summary.json     (pipeline metadata)")
print(f"\nSubmission format:")
print(f"  sample_id,is_listened")
print(f"  0,{submission.iloc[0]['is_listened']:.4f}")
print(f"  1,{submission.iloc[1]['is_listened']:.4f}")
print(f"  ...")
print(f"  {int(submission.iloc[-1]['sample_id'])},{submission.iloc[-1]['is_listened']:.4f}")
print("=" * 80)
