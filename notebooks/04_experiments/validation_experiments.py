"""
Validation Experiments for Report
==================================
Runs three experiments:
1. Temporal split vs random split comparison
2. Baseline models (popularity, logistic regression)
3. Feature ablation study (remove feature groups, measure AUC drop)

All results printed to stdout for inclusion in the report.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
import json
from datetime import datetime

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/Users/kreesen/Documents/deezer-multimodal-recommender"
sys.path.append(PROJECT_ROOT)

from src.data.preprocessing import (
    add_temporal_features,
    add_release_features,
    add_duration_features,
    compute_user_features_from_train,
    apply_user_features,
)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.csv")
RANDOM_STATE = 42

EXCLUDE_COLS = [
    "is_listened", "sample_id", "user_id", "media_id",
    "artist_id", "album_id", "genre_id",
    "datetime", "release_date_parsed", "listen_date",
    "ts_listen", "release_date",
]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.05,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1,
    "random_state": 42,
    "eval_metric": "auc",
    "n_jobs": -1,
}


# ============================================================================
# HELPER FUNCTIONS (same as v2 pipeline)
# ============================================================================

def target_encode(train_df, test_df, column, target, smoothing=50):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(column)[target].agg(["mean", "count"])
    stats["smoothed"] = (
        (stats["count"] * stats["mean"] + smoothing * global_mean)
        / (stats["count"] + smoothing)
    )
    encoded_col = f"{column}_target_enc"
    train_encoded = train_df[column].map(stats["smoothed"]).fillna(global_mean)
    test_encoded = test_df[column].map(stats["smoothed"]).fillna(global_mean)
    return train_encoded, test_encoded, encoded_col


def add_item_features(train_df, test_df):
    global_mean = train_df["is_listened"].mean()
    smoothing = 20
    media_stats = train_df.groupby("media_id")["is_listened"].agg(["mean", "count"])
    media_stats.columns = ["media_listen_rate", "media_play_count"]
    media_stats["media_listen_rate_smooth"] = (
        (media_stats["media_play_count"] * media_stats["media_listen_rate"]
         + smoothing * global_mean) / (media_stats["media_play_count"] + smoothing)
    )
    for df in [train_df, test_df]:
        merged = df[["media_id"]].merge(
            media_stats[["media_listen_rate_smooth", "media_play_count"]],
            on="media_id", how="left",
        )
        df["media_listen_rate_smooth"] = merged["media_listen_rate_smooth"].fillna(global_mean).values
        df["media_play_count"] = merged["media_play_count"].fillna(0).values
        df["media_play_count_log"] = np.log1p(df["media_play_count"])

    artist_stats = train_df.groupby("artist_id")["is_listened"].agg(["mean", "count"])
    artist_stats.columns = ["artist_listen_rate", "artist_play_count"]
    artist_stats["artist_listen_rate_smooth"] = (
        (artist_stats["artist_play_count"] * artist_stats["artist_listen_rate"]
         + smoothing * global_mean) / (artist_stats["artist_play_count"] + smoothing)
    )
    for df in [train_df, test_df]:
        merged = df[["artist_id"]].merge(
            artist_stats[["artist_listen_rate_smooth", "artist_play_count"]],
            on="artist_id", how="left",
        )
        df["artist_listen_rate_smooth"] = merged["artist_listen_rate_smooth"].fillna(global_mean).values
        df["artist_play_count"] = merged["artist_play_count"].fillna(0).values
        df["artist_play_count_log"] = np.log1p(df["artist_play_count"])
    return train_df, test_df


def add_user_item_affinity(train_df, test_df):
    global_mean = train_df["is_listened"].mean()
    smoothing = 5
    ua = train_df.groupby(["user_id", "artist_id"])["is_listened"].agg(["mean", "count"])
    ua.columns = ["ua_rate", "ua_count"]
    ua["user_artist_affinity"] = (
        (ua["ua_count"] * ua["ua_rate"] + smoothing * global_mean)
        / (ua["ua_count"] + smoothing)
    )
    ua = ua.reset_index()[["user_id", "artist_id", "user_artist_affinity"]]
    for df in [train_df, test_df]:
        merged = df[["user_id", "artist_id"]].merge(ua, on=["user_id", "artist_id"], how="left")
        df["user_artist_affinity"] = merged["user_artist_affinity"].fillna(global_mean).values

    ug = train_df.groupby(["user_id", "genre_id"])["is_listened"].agg(["mean", "count"])
    ug.columns = ["ug_rate", "ug_count"]
    ug["user_genre_affinity"] = (
        (ug["ug_count"] * ug["ug_rate"] + smoothing * global_mean)
        / (ug["ug_count"] + smoothing)
    )
    ug = ug.reset_index()[["user_id", "genre_id", "user_genre_affinity"]]
    for df in [train_df, test_df]:
        merged = df[["user_id", "genre_id"]].merge(ug, on=["user_id", "genre_id"], how="left")
        df["user_genre_affinity"] = merged["user_genre_affinity"].fillna(global_mean).values

    for df in [train_df, test_df]:
        df["user_knows_artist"] = (df["user_artist_affinity"] != global_mean).astype(int)
    return train_df, test_df


def prepare_full_features(train_portion, val_portion):
    """Apply all v2 features, computing stats ONLY from train_portion."""
    # Base features (applied to both)
    for df in [train_portion, val_portion]:
        add_temporal_features(df)
        add_release_features(df)
        add_duration_features(df)

    # User stats from train only
    user_stats = compute_user_features_from_train(train_portion)
    train_portion = apply_user_features(train_portion, user_stats)
    val_portion = apply_user_features(val_portion, user_stats)

    # Target encoding from train only
    for col in ["genre_id", "artist_id", "album_id"]:
        tr_enc, te_enc, enc_name = target_encode(train_portion, val_portion, col, "is_listened")
        train_portion[enc_name] = tr_enc
        val_portion[enc_name] = te_enc

    # Item features from train only
    train_portion, val_portion = add_item_features(train_portion, val_portion)

    # User-item affinity from train only
    train_portion, val_portion = add_user_item_affinity(train_portion, val_portion)

    return train_portion, val_portion


# ============================================================================
# LOAD DATA
# ============================================================================
print("=" * 80)
print("VALIDATION EXPERIMENTS")
print("=" * 80)
print(f"Loading data...")
df = pd.read_csv(TRAIN_PATH)
print(f"  Loaded {len(df):,} rows\n")


# ============================================================================
# EXPERIMENT 1: TEMPORAL SPLIT vs RANDOM SPLIT
# ============================================================================
print("=" * 80)
print("EXPERIMENT 1: TEMPORAL SPLIT vs RANDOM SPLIT")
print("=" * 80)

# Add datetime for sorting
df["datetime"] = pd.to_datetime(df["ts_listen"], unit="s")

# --- Random split ---
print("\n--- Random Split (90/10 stratified) ---")
train_rand, val_rand = train_test_split(
    df.copy(), test_size=0.1, random_state=RANDOM_STATE, stratify=df["is_listened"]
)
print(f"  Train: {len(train_rand):,} | Val: {len(val_rand):,}")

train_rand, val_rand = prepare_full_features(train_rand, val_rand)
feature_cols = [c for c in train_rand.columns if c not in EXCLUDE_COLS]

X_tr = train_rand[feature_cols].values.astype(np.float32)
y_tr = train_rand["is_listened"].values
X_vr = val_rand[feature_cols].values.astype(np.float32)
y_vr = val_rand["is_listened"].values
X_tr = np.nan_to_num(X_tr, nan=0.0)
X_vr = np.nan_to_num(X_vr, nan=0.0)

model_rand = xgb.XGBClassifier(**XGB_PARAMS)
model_rand.fit(X_tr, y_tr, eval_set=[(X_vr, y_vr)], verbose=0)
rand_auc = roc_auc_score(y_vr, model_rand.predict_proba(X_vr)[:, 1])
print(f"  Random Split AUC: {rand_auc:.4f}")

# --- Temporal split ---
print("\n--- Temporal Split (last 10% by time) ---")
df_sorted = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.9)
train_temp = df_sorted.iloc[:split_idx].copy()
val_temp = df_sorted.iloc[split_idx:].copy()

print(f"  Train: {len(train_temp):,} | Val: {len(val_temp):,}")
print(f"  Train period: {train_temp['datetime'].min()} to {train_temp['datetime'].max()}")
print(f"  Val period:   {val_temp['datetime'].min()} to {val_temp['datetime'].max()}")
print(f"  Train listen rate: {train_temp['is_listened'].mean():.3f}")
print(f"  Val listen rate:   {val_temp['is_listened'].mean():.3f}")

# Check user/item overlap
train_users = set(train_temp["user_id"].unique())
val_users = set(val_temp["user_id"].unique())
train_items = set(train_temp["media_id"].unique())
val_items = set(val_temp["media_id"].unique())
user_overlap = len(train_users & val_users) / len(val_users) * 100
item_overlap = len(train_items & val_items) / len(val_items) * 100
new_items_pct = (1 - item_overlap / 100) * 100
print(f"  User overlap: {user_overlap:.1f}%")
print(f"  Item overlap: {item_overlap:.1f}% ({new_items_pct:.1f}% new items in val)")

train_temp, val_temp = prepare_full_features(train_temp, val_temp)
feature_cols_temp = [c for c in train_temp.columns if c not in EXCLUDE_COLS]

X_tt = train_temp[feature_cols_temp].values.astype(np.float32)
y_tt = train_temp["is_listened"].values
X_vt = val_temp[feature_cols_temp].values.astype(np.float32)
y_vt = val_temp["is_listened"].values
X_tt = np.nan_to_num(X_tt, nan=0.0)
X_vt = np.nan_to_num(X_vt, nan=0.0)

model_temp = xgb.XGBClassifier(**XGB_PARAMS)
model_temp.fit(X_tt, y_tt, eval_set=[(X_vt, y_vt)], verbose=0)
temp_auc = roc_auc_score(y_vt, model_temp.predict_proba(X_vt)[:, 1])
print(f"  Temporal Split AUC: {temp_auc:.4f}")

print(f"\n  >>> AUC Drop (random -> temporal): {rand_auc:.4f} -> {temp_auc:.4f} ({temp_auc - rand_auc:+.4f})")

# Feature importance comparison
print("\n--- Feature Importance Shift (Temporal Split) ---")
imp_temp = dict(zip(feature_cols_temp, model_temp.feature_importances_))
imp_rand = dict(zip(feature_cols, model_rand.feature_importances_))
top_temp = sorted(imp_temp.items(), key=lambda x: x[1], reverse=True)[:10]
top_rand = sorted(imp_rand.items(), key=lambda x: x[1], reverse=True)[:10]

print(f"\n  {'Rank':<5} {'Random Split':<35} {'Temporal Split':<35}")
print(f"  {'-'*75}")
for i in range(10):
    r_name, r_imp = top_rand[i]
    t_name, t_imp = top_temp[i]
    print(f"  {i+1:<5} {r_name:<30} {r_imp:.4f}  {t_name:<30} {t_imp:.4f}")


# ============================================================================
# EXPERIMENT 2: BASELINE MODELS
# ============================================================================
print("\n\n" + "=" * 80)
print("EXPERIMENT 2: BASELINE MODELS")
print("=" * 80)

# Use the temporal split for more honest evaluation
# Reuse train_temp, val_temp from above

# --- Popularity baseline ---
print("\n--- Popularity Baseline ---")
global_mean = train_temp["is_listened"].mean()
pop_pred = np.full(len(val_temp), global_mean)
pop_auc = roc_auc_score(y_vt, pop_pred)
print(f"  Global mean baseline AUC: {pop_auc:.4f} (predicts {global_mean:.3f} for all)")

# Per-user popularity
user_rates = train_temp.groupby("user_id")["is_listened"].mean()
user_pred = val_temp["user_id"].map(user_rates).fillna(global_mean).values
user_auc = roc_auc_score(y_vt, user_pred)
print(f"  Per-user listen rate AUC: {user_auc:.4f}")

# Per-item popularity
item_rates = train_temp.groupby("media_id")["is_listened"].mean()
item_pred = val_temp["media_id"].map(item_rates).fillna(global_mean).values
item_auc = roc_auc_score(y_vt, item_pred)
print(f"  Per-item listen rate AUC: {item_auc:.4f}")

# Combined user + item
combined_pred = 0.5 * user_pred + 0.5 * item_pred
combined_auc = roc_auc_score(y_vt, combined_pred)
print(f"  User + Item combined AUC: {combined_auc:.4f}")

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
scaler = StandardScaler()
X_tt_scaled = scaler.fit_transform(X_tt)
X_vt_scaled = scaler.transform(X_vt)
X_tt_scaled = np.nan_to_num(X_tt_scaled, nan=0.0)
X_vt_scaled = np.nan_to_num(X_vt_scaled, nan=0.0)

lr = LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="saga", n_jobs=-1)
lr.fit(X_tt_scaled, y_tt)
lr_pred = lr.predict_proba(X_vt_scaled)[:, 1]
lr_auc = roc_auc_score(y_vt, lr_pred)
print(f"  Logistic Regression AUC: {lr_auc:.4f}")


# ============================================================================
# EXPERIMENT 3: FEATURE ABLATION STUDY
# ============================================================================
print("\n\n" + "=" * 80)
print("EXPERIMENT 3: FEATURE ABLATION STUDY")
print("=" * 80)

# Define feature groups
TEMPORAL_FEATURES = ["hour", "day_of_week", "day_of_month", "month",
                     "is_weekend", "is_late_night", "is_evening",
                     "is_commute_time", "time_of_day"]
RELEASE_FEATURES = ["release_year", "release_month", "release_decade",
                    "days_since_release", "is_pre_release_listen",
                    "is_new_release", "track_age_category"]
DURATION_FEATURES = ["duration_minutes", "duration_category", "is_extended_track"]
USER_FEATURES = ["user_listen_rate", "user_skip_rate", "user_session_count",
                 "user_total_listens", "user_genre_diversity", "user_artist_diversity",
                 "user_context_variety", "user_engagement_segment", "user_engagement_score"]
TARGET_ENC_FEATURES = ["genre_id_target_enc", "artist_id_target_enc", "album_id_target_enc"]
ITEM_FEATURES = ["media_listen_rate_smooth", "media_play_count_log",
                 "artist_listen_rate_smooth", "artist_play_count_log"]
AFFINITY_FEATURES = ["user_artist_affinity", "user_genre_affinity", "user_knows_artist"]

feature_groups = {
    "Temporal (9)": TEMPORAL_FEATURES,
    "Release (7)": RELEASE_FEATURES,
    "Duration (3)": DURATION_FEATURES,
    "User Engagement (9)": USER_FEATURES,
    "Target Encoding (3)": TARGET_ENC_FEATURES,
    "Item-Level (4)": ITEM_FEATURES,
    "User-Item Affinity (3)": AFFINITY_FEATURES,
}

# Full model (temporal split) as reference
full_auc = temp_auc
print(f"\n  Full model AUC (temporal split): {full_auc:.4f}")
print(f"\n  {'Feature Group Removed':<30} {'Features':<8} {'AUC':<10} {'Drop':<10} {'% Drop'}")
print(f"  {'-'*75}")

ablation_results = []

for group_name, group_features in feature_groups.items():
    # Remove this group
    available = [f for f in feature_cols_temp if f in train_temp.columns]
    remaining = [f for f in available if f not in group_features]

    X_tt_abl = train_temp[remaining].values.astype(np.float32)
    X_vt_abl = val_temp[remaining].values.astype(np.float32)
    X_tt_abl = np.nan_to_num(X_tt_abl, nan=0.0)
    X_vt_abl = np.nan_to_num(X_vt_abl, nan=0.0)

    model_abl = xgb.XGBClassifier(**XGB_PARAMS)
    model_abl.fit(X_tt_abl, y_tt, eval_set=[(X_vt_abl, y_vt)], verbose=0)
    abl_auc = roc_auc_score(y_vt, model_abl.predict_proba(X_vt_abl)[:, 1])

    drop = full_auc - abl_auc
    pct_drop = drop / full_auc * 100
    print(f"  {group_name:<30} {len(group_features):<8} {abl_auc:<10.4f} {drop:<10.4f} {pct_drop:.2f}%")
    ablation_results.append((group_name, len(group_features), abl_auc, drop, pct_drop))

# Sort by impact
print(f"\n  Sorted by impact (most important first):")
print(f"  {'-'*75}")
for group_name, n_feat, auc, drop, pct in sorted(ablation_results, key=lambda x: x[3], reverse=True):
    print(f"  {group_name:<30} {n_feat:<8} {auc:<10.4f} {drop:<10.4f} {pct:.2f}%")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("SUMMARY TABLE FOR REPORT")
print("=" * 80)

print(f"\n  {'Model':<40} {'Split':<12} {'AUC'}")
print(f"  {'-'*65}")
print(f"  {'Global mean baseline':<40} {'temporal':<12} {pop_auc:.4f}")
print(f"  {'Per-user listen rate':<40} {'temporal':<12} {user_auc:.4f}")
print(f"  {'Per-item listen rate':<40} {'temporal':<12} {item_auc:.4f}")
print(f"  {'User + Item combined':<40} {'temporal':<12} {combined_auc:.4f}")
print(f"  {'Logistic Regression (47 feat.)':<40} {'temporal':<12} {lr_auc:.4f}")
print(f"  {'XGBoost v2 (47 feat.)':<40} {'temporal':<12} {temp_auc:.4f}")
print(f"  {'XGBoost v2 (47 feat.)':<40} {'random':<12} {rand_auc:.4f}")

print(f"\n  Key finding: Random split inflates AUC by {rand_auc - temp_auc:+.4f}")

# Save results
results = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "random_split_auc": float(rand_auc),
    "temporal_split_auc": float(temp_auc),
    "auc_inflation": float(rand_auc - temp_auc),
    "baselines": {
        "global_mean": float(pop_auc),
        "per_user": float(user_auc),
        "per_item": float(item_auc),
        "user_item_combined": float(combined_auc),
        "logistic_regression": float(lr_auc),
    },
    "ablation": {name: {"auc": float(auc), "drop": float(drop)}
                 for name, _, auc, drop, _ in ablation_results},
}

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {output_path}")
print("=" * 80)
