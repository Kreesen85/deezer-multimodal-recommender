"""
Improved Submission Pipeline v2 - Deezer Skip Prediction
==========================================================

Improvements over v1 (generate_submission.py):
1. Target encoding for genre_id, artist_id, album_id
2. Item-level features (media listen rate, artist popularity, genre popularity)
3. User-artist and user-genre affinity features
4. LightGBM as second model
5. Ensemble of XGBoost + LightGBM
6. Validation split to measure improvement before submitting

Usage:
    cd notebooks/04_experiments/xgboost
    /opt/anaconda3/bin/python generate_submission_v2.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import sys
import os
import json
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
PROJECT_ROOT = "/Users/kreesen/Documents/deezer-multimodal-recommender"
sys.path.append(PROJECT_ROOT)
from src.data.preprocessing import (
    add_temporal_features,
    add_release_features,
    add_duration_features,
    compute_user_features_from_train,
    apply_user_features,
)

# Try importing LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("WARNING: LightGBM not installed. Will use XGBoost only.")
    print("Install with: /opt/anaconda3/bin/pip install --trusted-host pypi.org "
          "--trusted-host files.pythonhosted.org lightgbm\n")

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "data/raw/test.csv")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_SAMPLE_SIZE = None  # None = full 7.5M
RANDOM_STATE = 42
VALIDATION_SIZE = 0.1  # 10% held out for validation

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "random_state": RANDOM_STATE,
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    "n_jobs": -1,
}

# LightGBM hyperparameters
LGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 7,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Ensemble weights (XGBoost, LightGBM)
ENSEMBLE_WEIGHTS = (0.5, 0.5)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def target_encode(train_df, test_df, column, target, smoothing=50):
    """
    Target encoding with smoothing to prevent overfitting.

    For each category value, computes a smoothed mean of the target:
        smoothed_mean = (count * category_mean + smoothing * global_mean) /
                        (count + smoothing)

    This pulls rare categories toward the global mean.
    """
    global_mean = train_df[target].mean()

    # Compute category statistics from training data only
    stats = train_df.groupby(column)[target].agg(["mean", "count"])
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )

    # Map to both train and test
    encoded_col = f"{column}_target_enc"
    train_encoded = train_df[column].map(stats["smoothed"]).fillna(global_mean)
    test_encoded = test_df[column].map(stats["smoothed"]).fillna(global_mean)

    return train_encoded, test_encoded, encoded_col


def add_item_features(train_df, test_df):
    """
    Add item-level aggregate features computed from training data.
    """
    print("  Computing item-level features...")

    # --- Media (track) listen rate ---
    media_stats = train_df.groupby("media_id")["is_listened"].agg(
        ["mean", "count"]
    )
    media_stats.columns = ["media_listen_rate", "media_play_count"]
    # Smooth rare items toward global mean
    global_mean = train_df["is_listened"].mean()
    smoothing = 20
    media_stats["media_listen_rate_smooth"] = (
        (media_stats["media_play_count"] * media_stats["media_listen_rate"]
         + smoothing * global_mean)
        / (media_stats["media_play_count"] + smoothing)
    )

    train_df = train_df.merge(
        media_stats[["media_listen_rate_smooth", "media_play_count"]],
        on="media_id", how="left",
    )
    test_df = test_df.merge(
        media_stats[["media_listen_rate_smooth", "media_play_count"]],
        on="media_id", how="left",
    )

    # Fill missing (unseen items in test) with global mean
    train_df["media_listen_rate_smooth"] = train_df[
        "media_listen_rate_smooth"
    ].fillna(global_mean)
    train_df["media_play_count"] = train_df["media_play_count"].fillna(0)
    test_df["media_listen_rate_smooth"] = test_df[
        "media_listen_rate_smooth"
    ].fillna(global_mean)
    test_df["media_play_count"] = test_df["media_play_count"].fillna(0)

    # Log-transform play count (highly skewed)
    train_df["media_play_count_log"] = np.log1p(train_df["media_play_count"])
    test_df["media_play_count_log"] = np.log1p(test_df["media_play_count"])

    # --- Artist popularity ---
    artist_stats = train_df.groupby("artist_id")["is_listened"].agg(
        ["mean", "count"]
    )
    artist_stats.columns = ["artist_listen_rate", "artist_play_count"]
    artist_stats["artist_listen_rate_smooth"] = (
        (artist_stats["artist_play_count"] * artist_stats["artist_listen_rate"]
         + smoothing * global_mean)
        / (artist_stats["artist_play_count"] + smoothing)
    )

    train_df = train_df.merge(
        artist_stats[["artist_listen_rate_smooth", "artist_play_count"]],
        on="artist_id", how="left",
    )
    test_df = test_df.merge(
        artist_stats[["artist_listen_rate_smooth", "artist_play_count"]],
        on="artist_id", how="left",
    )
    train_df["artist_listen_rate_smooth"] = train_df[
        "artist_listen_rate_smooth"
    ].fillna(global_mean)
    train_df["artist_play_count"] = train_df["artist_play_count"].fillna(0)
    test_df["artist_listen_rate_smooth"] = test_df[
        "artist_listen_rate_smooth"
    ].fillna(global_mean)
    test_df["artist_play_count"] = test_df["artist_play_count"].fillna(0)
    train_df["artist_play_count_log"] = np.log1p(train_df["artist_play_count"])
    test_df["artist_play_count_log"] = np.log1p(test_df["artist_play_count"])

    print(f"    Added: media_listen_rate_smooth, media_play_count_log, "
          f"artist_listen_rate_smooth, artist_play_count_log")

    return train_df, test_df


def add_user_item_affinity(train_df, test_df):
    """
    Add user-artist and user-genre affinity features.
    Measures how much a user likes a specific artist/genre based on history.
    """
    print("  Computing user-item affinity features...")
    global_mean = train_df["is_listened"].mean()
    smoothing = 5

    # --- User-Artist affinity ---
    ua_stats = train_df.groupby(["user_id", "artist_id"])["is_listened"].agg(
        ["mean", "count"]
    )
    ua_stats.columns = ["ua_listen_rate", "ua_count"]
    ua_stats["user_artist_affinity"] = (
        (ua_stats["ua_count"] * ua_stats["ua_listen_rate"]
         + smoothing * global_mean)
        / (ua_stats["ua_count"] + smoothing)
    )
    ua_stats = ua_stats.reset_index()[["user_id", "artist_id", "user_artist_affinity"]]

    train_df = train_df.merge(ua_stats, on=["user_id", "artist_id"], how="left")
    test_df = test_df.merge(ua_stats, on=["user_id", "artist_id"], how="left")
    train_df["user_artist_affinity"] = train_df["user_artist_affinity"].fillna(
        global_mean
    )
    test_df["user_artist_affinity"] = test_df["user_artist_affinity"].fillna(
        global_mean
    )

    # --- User-Genre affinity ---
    ug_stats = train_df.groupby(["user_id", "genre_id"])["is_listened"].agg(
        ["mean", "count"]
    )
    ug_stats.columns = ["ug_listen_rate", "ug_count"]
    ug_stats["user_genre_affinity"] = (
        (ug_stats["ug_count"] * ug_stats["ug_listen_rate"]
         + smoothing * global_mean)
        / (ug_stats["ug_count"] + smoothing)
    )
    ug_stats = ug_stats.reset_index()[["user_id", "genre_id", "user_genre_affinity"]]

    train_df = train_df.merge(ug_stats, on=["user_id", "genre_id"], how="left")
    test_df = test_df.merge(ug_stats, on=["user_id", "genre_id"], how="left")
    train_df["user_genre_affinity"] = train_df["user_genre_affinity"].fillna(
        global_mean
    )
    test_df["user_genre_affinity"] = test_df["user_genre_affinity"].fillna(
        global_mean
    )

    # --- Has user heard this artist before? ---
    train_df["user_knows_artist"] = train_df["user_artist_affinity"].ne(
        global_mean
    ).astype(int)
    test_df["user_knows_artist"] = test_df["user_artist_affinity"].ne(
        global_mean
    ).astype(int)

    print("    Added: user_artist_affinity, user_genre_affinity, user_knows_artist")

    return train_df, test_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

print("=" * 80)
print("DEEZER SKIP PREDICTION - IMPROVED PIPELINE v2")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"LightGBM available: {HAS_LIGHTGBM}\n")


# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("STEP 1: Loading data...")

train_df = pd.read_csv(TRAIN_PATH, nrows=TRAIN_SAMPLE_SIZE)
test_df = pd.read_csv(TEST_PATH)
sample_ids = test_df["sample_id"].copy()

print(f"  Train: {len(train_df):,} rows, {train_df['user_id'].nunique():,} users, "
      f"{train_df['media_id'].nunique():,} items")
print(f"  Test:  {len(test_df):,} rows, {test_df['user_id'].nunique():,} users")
print(f"  Listen rate: {train_df['is_listened'].mean():.2%}\n")


# ============================================================================
# STEP 2: Compute User Stats
# ============================================================================
print("STEP 2: Computing user stats from full training data...")

user_stats = compute_user_features_from_train(train_df)
print()


# ============================================================================
# STEP 3: Add Base Features (temporal, release, duration)
# ============================================================================
print("STEP 3: Adding base features...")

for df, name in [(train_df, "train"), (test_df, "test")]:
    pass  # We process in-place below

print("  Adding temporal features...")
train_df = add_temporal_features(train_df)
test_df = add_temporal_features(test_df)

print("  Adding release features...")
train_df = add_release_features(train_df)
test_df = add_release_features(test_df)

print("  Adding duration features...")
train_df = add_duration_features(train_df)
test_df = add_duration_features(test_df)

print("  Applying user features...")
train_df = apply_user_features(train_df, user_stats)
test_df = apply_user_features(test_df, user_stats)
print()


# ============================================================================
# STEP 4: Add NEW Features (target encoding, item features, affinity)
# ============================================================================
print("STEP 4: Adding new features (v2 improvements)...")

# Target encoding for categorical IDs
print("  Target encoding categorical features...")
for col in ["genre_id", "artist_id", "album_id"]:
    train_enc, test_enc, enc_name = target_encode(
        train_df, test_df, col, "is_listened", smoothing=50
    )
    train_df[enc_name] = train_enc
    test_df[enc_name] = test_enc
    print(f"    {col} -> {enc_name}")

# Item-level features
train_df, test_df = add_item_features(train_df, test_df)

# User-item affinity
train_df, test_df = add_user_item_affinity(train_df, test_df)

print(f"\n  Total columns: {len(train_df.columns)}")
print()


# ============================================================================
# STEP 5: Prepare Features
# ============================================================================
print("STEP 5: Preparing features...")

EXCLUDE_COLS_V2 = [
    "is_listened", "sample_id", "user_id", "media_id",
    "artist_id", "album_id", "genre_id",
    "datetime", "release_date_parsed", "listen_date",
    "ts_listen", "release_date",
]

feature_cols = [col for col in train_df.columns if col not in EXCLUDE_COLS_V2]
print(f"  Features: {len(feature_cols)}")

# Show new features vs v1
v1_features = 35
print(f"  v1 features: {v1_features}")
print(f"  v2 features: {len(feature_cols)} (+{len(feature_cols) - v1_features} new)")

new_features = [
    "genre_id_target_enc", "artist_id_target_enc", "album_id_target_enc",
    "media_listen_rate_smooth", "media_play_count_log",
    "artist_listen_rate_smooth", "artist_play_count_log",
    "user_artist_affinity", "user_genre_affinity", "user_knows_artist",
    "media_play_count", "artist_play_count",
]
print(f"\n  New features added:")
for f in new_features:
    if f in feature_cols:
        print(f"    + {f}")
print()


# ============================================================================
# STEP 6: Validation Split (to measure improvement)
# ============================================================================
print("STEP 6: Validation split to measure improvement...")

X_all = train_df[feature_cols].copy()
y_all = train_df["is_listened"].copy()

# Handle missing values
missing = X_all.isnull().sum().sum()
if missing > 0:
    print(f"  Filling {missing:,} missing values...")
    X_all = X_all.fillna(X_all.median())

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all,
)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
print()


# ============================================================================
# STEP 7: Train Models
# ============================================================================
print("STEP 7: Training models...")

# --- XGBoost ---
print(f"\n  [XGBoost] Training ({XGB_PARAMS['n_estimators']} trees, "
      f"depth {XGB_PARAMS['max_depth']}, lr {XGB_PARAMS['learning_rate']})...")

xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
)

xgb_val_pred = xgb_model.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_val_pred)
print(f"\n  [XGBoost] Validation AUC: {xgb_auc:.4f}")

# --- LightGBM ---
if HAS_LIGHTGBM:
    print(f"\n  [LightGBM] Training ({LGB_PARAMS['n_estimators']} trees, "
          f"depth {LGB_PARAMS['max_depth']}, lr {LGB_PARAMS['learning_rate']})...")

    lgb_model = lgb.LGBMClassifier(**LGB_PARAMS)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    lgb_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    lgb_auc = roc_auc_score(y_val, lgb_val_pred)
    print(f"  [LightGBM] Validation AUC: {lgb_auc:.4f}")

    # --- Ensemble ---
    w_xgb, w_lgb = ENSEMBLE_WEIGHTS
    ensemble_val_pred = w_xgb * xgb_val_pred + w_lgb * lgb_val_pred
    ensemble_auc = roc_auc_score(y_val, ensemble_val_pred)
    print(f"\n  [Ensemble] Validation AUC: {ensemble_auc:.4f} "
          f"(weights: XGB={w_xgb}, LGB={w_lgb})")

    # Find optimal weights
    best_auc = 0
    best_w = 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        ens_pred = w * xgb_val_pred + (1 - w) * lgb_val_pred
        auc = roc_auc_score(y_val, ens_pred)
        if auc > best_auc:
            best_auc = auc
            best_w = w

    print(f"  [Ensemble] Best weights: XGB={best_w:.2f}, LGB={1-best_w:.2f} "
          f"-> AUC: {best_auc:.4f}")
    ENSEMBLE_WEIGHTS = (best_w, 1 - best_w)
else:
    lgb_auc = None
    ensemble_auc = None
    best_auc = xgb_auc

print()

# ============================================================================
# STEP 8: Print Results Comparison
# ============================================================================
print("=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print(f"  v1 Baseline (XGBoost, 35 features):  0.8722 AUC (on 100K)")
print(f"  v2 XGBoost  ({len(feature_cols)} features):  {xgb_auc:.4f} AUC")
if HAS_LIGHTGBM:
    print(f"  v2 LightGBM ({len(feature_cols)} features):  {lgb_auc:.4f} AUC")
    print(f"  v2 Ensemble (XGB+LGB):          {best_auc:.4f} AUC")
print("=" * 80)
print()


# ============================================================================
# STEP 9: Retrain on ALL data and Generate Submission
# ============================================================================
print("STEP 9: Retraining on ALL training data for final submission...")

# XGBoost - retrain on full data (no early stopping)
xgb_params_final = {k: v for k, v in XGB_PARAMS.items()
                    if k != "early_stopping_rounds"}
xgb_params_final["n_estimators"] = xgb_model.best_iteration + 1 \
    if hasattr(xgb_model, "best_iteration") and xgb_model.best_iteration > 0 \
    else XGB_PARAMS["n_estimators"]

print(f"  [XGBoost] Retraining on {len(X_all):,} rows "
      f"({xgb_params_final['n_estimators']} trees)...")
xgb_final = xgb.XGBClassifier(**xgb_params_final)
xgb_final.fit(X_all, y_all, verbose=100)

xgb_final.save_model(os.path.join(OUTPUT_DIR, "xgboost_v2_model.json"))

if HAS_LIGHTGBM:
    lgb_params_final = dict(LGB_PARAMS)
    lgb_params_final["n_estimators"] = lgb_model.best_iteration_ + 1 \
        if hasattr(lgb_model, "best_iteration_") and lgb_model.best_iteration_ > 0 \
        else LGB_PARAMS["n_estimators"]

    print(f"  [LightGBM] Retraining on {len(X_all):,} rows "
          f"({lgb_params_final['n_estimators']} trees)...")
    lgb_final = lgb.LGBMClassifier(**lgb_params_final)
    lgb_final.fit(X_all, y_all)

    lgb_final.booster_.save_model(os.path.join(OUTPUT_DIR, "lightgbm_v2_model.txt"))

print()


# ============================================================================
# STEP 10: Predict on Test Data
# ============================================================================
print("STEP 10: Generating predictions on test data...")

X_test = test_df[feature_cols].copy()
missing = X_test.isnull().sum().sum()
if missing > 0:
    print(f"  Filling {missing:,} missing values...")
    train_medians = X_all.median()
    X_test = X_test.fillna(train_medians)

print(f"  X_test shape: {X_test.shape}")

# XGBoost predictions
xgb_test_pred = xgb_final.predict_proba(X_test)[:, 1]

if HAS_LIGHTGBM:
    # LightGBM predictions
    lgb_test_pred = lgb_final.predict_proba(X_test)[:, 1]

    # Ensemble
    w_xgb, w_lgb = ENSEMBLE_WEIGHTS
    final_pred = w_xgb * xgb_test_pred + w_lgb * lgb_test_pred
    print(f"  Ensemble weights: XGB={w_xgb:.2f}, LGB={w_lgb:.2f}")
else:
    final_pred = xgb_test_pred

print(f"\n  Prediction statistics:")
print(f"    Min:    {final_pred.min():.4f}")
print(f"    Max:    {final_pred.max():.4f}")
print(f"    Mean:   {final_pred.mean():.4f}")
print(f"    Median: {np.median(final_pred):.4f}")
print()


# ============================================================================
# STEP 11: Create and Verify Submission
# ============================================================================
print("STEP 11: Creating submission...")

submission = pd.DataFrame({
    "sample_id": sample_ids,
    "is_listened": final_pred,
})
submission = submission.sort_values("sample_id").reset_index(drop=True)

# Verification
checks = {
    "Row count": len(submission) == 19918,
    "Columns": list(submission.columns) == ["sample_id", "is_listened"],
    "sample_id range": submission["sample_id"].min() == 0
                       and submission["sample_id"].max() == 19917,
    "Predictions in [0,1]": (submission["is_listened"] >= 0).all()
                            and (submission["is_listened"] <= 1).all(),
    "No duplicates": submission["sample_id"].nunique() == len(submission),
    "No NaN": submission["is_listened"].notna().all(),
}

print("  Verification:")
all_passed = True
for name, passed in checks.items():
    print(f"    {name}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        all_passed = False

# Save submission
sub_path = os.path.join(OUTPUT_DIR, "submission_v2.csv")
submission.to_csv(sub_path, index=False)
print(f"\n  Saved: {sub_path}")

# Also save individual model submissions for comparison
xgb_sub = pd.DataFrame({"sample_id": sample_ids, "is_listened": xgb_test_pred})
xgb_sub = xgb_sub.sort_values("sample_id").reset_index(drop=True)
xgb_sub.to_csv(os.path.join(OUTPUT_DIR, "submission_v2_xgb_only.csv"), index=False)

if HAS_LIGHTGBM:
    lgb_sub = pd.DataFrame({"sample_id": sample_ids, "is_listened": lgb_test_pred})
    lgb_sub = lgb_sub.sort_values("sample_id").reset_index(drop=True)
    lgb_sub.to_csv(os.path.join(OUTPUT_DIR, "submission_v2_lgb_only.csv"), index=False)
print()


# ============================================================================
# STEP 12: Feature Importance
# ============================================================================
print("STEP 12: Feature importance...")

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "xgb_importance": xgb_final.feature_importances_,
}).sort_values("xgb_importance", ascending=False)

if HAS_LIGHTGBM:
    lgb_imp = pd.DataFrame({
        "feature": feature_cols,
        "lgb_importance": lgb_final.feature_importances_,
    })
    lgb_imp["lgb_importance"] = lgb_imp["lgb_importance"] / lgb_imp[
        "lgb_importance"
    ].sum()
    importance_df = importance_df.merge(lgb_imp, on="feature")
    importance_df["avg_importance"] = (
        importance_df["xgb_importance"] + importance_df["lgb_importance"]
    ) / 2
    importance_df = importance_df.sort_values("avg_importance", ascending=False)

importance_df.to_csv(
    os.path.join(OUTPUT_DIR, "feature_importance_v2.csv"), index=False
)

print("\n  Top 15 features:")
for _, row in importance_df.head(15).iterrows():
    xgb_imp = row["xgb_importance"]
    extra = ""
    if HAS_LIGHTGBM:
        lgb_imp = row["lgb_importance"]
        extra = f"  LGB: {lgb_imp:.4f}"
    print(f"    {row['feature']:35s} XGB: {xgb_imp:.4f}{extra}")


# ============================================================================
# STEP 13: Save Summary
# ============================================================================
print("\n\nSTEP 13: Saving summary...")

summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "version": "v2",
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "n_features": len(feature_cols),
    "new_features_added": len(feature_cols) - v1_features,
    "xgb_val_auc": float(xgb_auc),
    "lgb_val_auc": float(lgb_auc) if lgb_auc else None,
    "ensemble_val_auc": float(best_auc),
    "ensemble_weights": {
        "xgb": float(ENSEMBLE_WEIGHTS[0]),
        "lgb": float(ENSEMBLE_WEIGHTS[1]),
    },
    "xgb_params": xgb_params_final,
    "prediction_mean": float(final_pred.mean()),
    "all_checks_passed": all_passed,
}

with open(os.path.join(OUTPUT_DIR, "submission_v2_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PIPELINE v2 COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nPerformance:")
print(f"  v1 Baseline:    0.8722 AUC (100K, 35 features)")
print(f"  v2 XGBoost:     {xgb_auc:.4f} AUC ({len(feature_cols)} features)")
if HAS_LIGHTGBM:
    print(f"  v2 LightGBM:    {lgb_auc:.4f} AUC")
    print(f"  v2 Ensemble:    {best_auc:.4f} AUC")
print(f"\nNew features: +{len(feature_cols) - v1_features}")
print(f"  - Target encoded: genre_id, artist_id, album_id")
print(f"  - Item features:  media_listen_rate, artist_listen_rate")
print(f"  - Affinity:       user_artist_affinity, user_genre_affinity")
print(f"\nSubmission files:")
print(f"  - submission_v2.csv          (ensemble)")
print(f"  - submission_v2_xgb_only.csv (XGBoost only)")
if HAS_LIGHTGBM:
    print(f"  - submission_v2_lgb_only.csv (LightGBM only)")
print(f"\nModels saved:")
print(f"  - xgboost_v2_model.json")
if HAS_LIGHTGBM:
    print(f"  - lightgbm_v2_model.txt")
print("=" * 80)
