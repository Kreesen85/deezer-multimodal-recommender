"""
ALS Matrix Factorization + Hybrid Experiment
==============================================
1. Train implicit ALS on user-item interaction matrix
2. Evaluate ALS standalone (temporal split)
3. Extract user/item embeddings -> feed into XGBoost
4. Compare: ALS alone vs XGBoost vs Hybrid (XGBoost + MF embeddings)

All evaluated under temporal split for honest comparison.
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import warnings
from datetime import datetime
from scipy.sparse import csr_matrix

import implicit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

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

TRAIN_PATH = os.path.join(PROJECT_ROOT, "data/raw/train.csv")
RANDOM_STATE = 42
N_FACTORS = 64

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
# HELPER FUNCTIONS
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


def prepare_features(train_portion, val_portion):
    for df in [train_portion, val_portion]:
        add_temporal_features(df)
        add_release_features(df)
        add_duration_features(df)
    user_stats = compute_user_features_from_train(train_portion)
    train_portion = apply_user_features(train_portion, user_stats)
    val_portion = apply_user_features(val_portion, user_stats)
    for col in ["genre_id", "artist_id", "album_id"]:
        tr_enc, te_enc, enc_name = target_encode(train_portion, val_portion, col, "is_listened")
        train_portion[enc_name] = tr_enc
        val_portion[enc_name] = te_enc
    train_portion, val_portion = add_item_features(train_portion, val_portion)
    train_portion, val_portion = add_user_item_affinity(train_portion, val_portion)
    return train_portion, val_portion


# ============================================================================
# LOAD DATA & TEMPORAL SPLIT
# ============================================================================
print("=" * 80)
print("ALS MATRIX FACTORIZATION + HYBRID EXPERIMENT")
print("=" * 80)
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")

df = pd.read_csv(TRAIN_PATH)
df["datetime"] = pd.to_datetime(df["ts_listen"], unit="s")
print(f"Loaded {len(df):,} rows\n")

# Temporal split
df_sorted = df.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.9)
train_df = df_sorted.iloc[:split_idx].copy()
val_df = df_sorted.iloc[split_idx:].copy()
print(f"Temporal split: Train {len(train_df):,} | Val {len(val_df):,}")
print(f"  Train: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
print(f"  Val:   {val_df['datetime'].min()} -> {val_df['datetime'].max()}")

# ============================================================================
# EXPERIMENT 1: ALS STANDALONE
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 1: IMPLICIT ALS (STANDALONE)")
print("=" * 80)

# Build ID mappings from TRAINING data only
train_users = train_df["user_id"].unique()
train_items = train_df["media_id"].unique()
user_to_idx = {u: i for i, u in enumerate(train_users)}
item_to_idx = {m: i for i, m in enumerate(train_items)}
n_users = len(train_users)
n_items = len(train_items)
print(f"  Users: {n_users:,} | Items: {n_items:,}")

# Build sparse interaction matrix (user x item)
# Use listen count as confidence weight
train_df["user_idx"] = train_df["user_id"].map(user_to_idx)
train_df["item_idx"] = train_df["media_id"].map(item_to_idx)

# Aggregate: for each (user, item), count listens as confidence
interactions = train_df.groupby(["user_idx", "item_idx"])["is_listened"].sum().reset_index()
interactions.columns = ["user_idx", "item_idx", "confidence"]

row = interactions["user_idx"].values
col = interactions["item_idx"].values
data = interactions["confidence"].values.astype(np.float32)

user_item_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))
print(f"  Interaction matrix: {user_item_matrix.shape}, nnz={user_item_matrix.nnz:,}")

# Train ALS
print(f"\n  Training ALS (factors={N_FACTORS})...")
als_model = implicit.als.AlternatingLeastSquares(
    factors=N_FACTORS,
    regularization=0.1,
    iterations=15,
    random_state=RANDOM_STATE,
)
als_model.fit(user_item_matrix)

user_factors = als_model.user_factors  # shape: (n_users, N_FACTORS)
item_factors = als_model.item_factors  # shape: (n_items, N_FACTORS)
print(f"  User factors: {user_factors.shape} | Item factors: {item_factors.shape}")

# Evaluate ALS on temporal validation set
# For each (user, item) in val, compute dot product of embeddings
val_df["user_idx"] = val_df["user_id"].map(user_to_idx)
val_df["item_idx"] = val_df["media_id"].map(item_to_idx)

# Handle cold start: users/items not in training
val_has_both = val_df["user_idx"].notna() & val_df["item_idx"].notna()
val_eval = val_df[val_has_both].copy()
val_cold = val_df[~val_has_both].copy()

print(f"\n  Val rows with known user+item: {len(val_eval):,} ({len(val_eval)/len(val_df)*100:.1f}%)")
print(f"  Val rows with cold user/item: {len(val_cold):,} ({len(val_cold)/len(val_df)*100:.1f}%)")

# Compute dot product scores
u_idx = val_eval["user_idx"].astype(int).values
i_idx = val_eval["item_idx"].astype(int).values
als_scores = np.array([
    np.dot(user_factors[u], item_factors[i])
    for u, i in zip(u_idx, i_idx)
])

# For cold-start items, predict global mean
global_mean = train_df["is_listened"].mean()
full_als_scores = np.full(len(val_df), global_mean)
full_als_scores[val_has_both.values] = als_scores

als_auc = roc_auc_score(val_df["is_listened"].values, full_als_scores)
als_auc_warm = roc_auc_score(val_eval["is_listened"].values, als_scores)
print(f"\n  ALS AUC (all val):   {als_auc:.4f}")
print(f"  ALS AUC (warm only): {als_auc_warm:.4f}")


# ============================================================================
# EXPERIMENT 2: EXTRACT MF EMBEDDINGS AS FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("EXPERIMENT 2: MF EMBEDDINGS AS XGBOOST FEATURES")
print("=" * 80)

def add_mf_features(df, user_factors, item_factors, user_to_idx, item_to_idx, n_factors):
    """Add MF embedding dot product, cosine similarity, and top-k factor features."""
    u_idx = df["user_id"].map(user_to_idx)
    i_idx = df["media_id"].map(item_to_idx)

    # Mean embeddings for cold start fallback
    mean_user = user_factors.mean(axis=0)
    mean_item = item_factors.mean(axis=0)

    u_emb = np.array([user_factors[int(u)] if pd.notna(u) else mean_user for u in u_idx])
    i_emb = np.array([item_factors[int(i)] if pd.notna(i) else mean_item for i in i_idx])

    # Dot product (main CF signal)
    df["mf_dot_product"] = np.sum(u_emb * i_emb, axis=1)

    # Cosine similarity
    u_norm = np.linalg.norm(u_emb, axis=1, keepdims=True) + 1e-8
    i_norm = np.linalg.norm(i_emb, axis=1, keepdims=True) + 1e-8
    df["mf_cosine_sim"] = np.sum((u_emb / u_norm) * (i_emb / i_norm), axis=1)

    # User and item embedding norms (proxy for activity level)
    df["mf_user_norm"] = np.linalg.norm(u_emb, axis=1)
    df["mf_item_norm"] = np.linalg.norm(i_emb, axis=1)

    return df


# Prepare features (same pipeline as before)
print("  Preparing base features...")
train_feat = train_df.copy()
val_feat = val_df.copy()
train_feat, val_feat = prepare_features(train_feat, val_feat)

# Add MF features
print("  Adding MF embedding features...")
train_feat = add_mf_features(train_feat, user_factors, item_factors, user_to_idx, item_to_idx, N_FACTORS)
val_feat = add_mf_features(val_feat, user_factors, item_factors, user_to_idx, item_to_idx, N_FACTORS)

# Feature columns
feature_cols = [c for c in train_feat.columns if c not in EXCLUDE_COLS]
mf_feature_names = ["mf_dot_product", "mf_cosine_sim", "mf_user_norm", "mf_item_norm"]
base_feature_cols = [c for c in feature_cols if c not in mf_feature_names]
hybrid_feature_cols = feature_cols  # all including MF

print(f"  Base features: {len(base_feature_cols)}")
print(f"  MF features: {len(mf_feature_names)}")
print(f"  Hybrid features: {len(hybrid_feature_cols)}")

# --- XGBoost with base features only (no MF) ---
print("\n--- XGBoost (base features only, temporal split) ---")
X_tr_base = np.nan_to_num(train_feat[base_feature_cols].values.astype(np.float32), nan=0.0)
X_vl_base = np.nan_to_num(val_feat[base_feature_cols].values.astype(np.float32), nan=0.0)
y_tr = train_feat["is_listened"].values
y_vl = val_feat["is_listened"].values

model_base = xgb.XGBClassifier(**XGB_PARAMS)
model_base.fit(X_tr_base, y_tr, eval_set=[(X_vl_base, y_vl)], verbose=0)
base_auc = roc_auc_score(y_vl, model_base.predict_proba(X_vl_base)[:, 1])
print(f"  XGBoost (base, 47 feat.) AUC: {base_auc:.4f}")

# --- XGBoost WITHOUT affinity features (best from ablation) ---
print("\n--- XGBoost (no affinity features, temporal split) ---")
AFFINITY_FEATURES = ["user_artist_affinity", "user_genre_affinity", "user_knows_artist"]
no_affinity_cols = [c for c in base_feature_cols if c not in AFFINITY_FEATURES]
X_tr_noaff = np.nan_to_num(train_feat[no_affinity_cols].values.astype(np.float32), nan=0.0)
X_vl_noaff = np.nan_to_num(val_feat[no_affinity_cols].values.astype(np.float32), nan=0.0)

model_noaff = xgb.XGBClassifier(**XGB_PARAMS)
model_noaff.fit(X_tr_noaff, y_tr, eval_set=[(X_vl_noaff, y_vl)], verbose=0)
noaff_auc = roc_auc_score(y_vl, model_noaff.predict_proba(X_vl_noaff)[:, 1])
print(f"  XGBoost (no affinity, 44 feat.) AUC: {noaff_auc:.4f}")

# --- Hybrid: XGBoost + MF features ---
print("\n--- Hybrid: XGBoost + MF embeddings (temporal split) ---")
X_tr_hyb = np.nan_to_num(train_feat[hybrid_feature_cols].values.astype(np.float32), nan=0.0)
X_vl_hyb = np.nan_to_num(val_feat[hybrid_feature_cols].values.astype(np.float32), nan=0.0)

model_hyb = xgb.XGBClassifier(**XGB_PARAMS)
model_hyb.fit(X_tr_hyb, y_tr, eval_set=[(X_vl_hyb, y_vl)], verbose=0)
hybrid_auc = roc_auc_score(y_vl, model_hyb.predict_proba(X_vl_hyb)[:, 1])
print(f"  Hybrid (47 + 4 MF feat.) AUC: {hybrid_auc:.4f}")

# --- Hybrid without affinity + MF ---
print("\n--- Hybrid: XGBoost (no affinity) + MF embeddings ---")
hybrid_noaff_cols = [c for c in hybrid_feature_cols if c not in AFFINITY_FEATURES]
X_tr_hyb2 = np.nan_to_num(train_feat[hybrid_noaff_cols].values.astype(np.float32), nan=0.0)
X_vl_hyb2 = np.nan_to_num(val_feat[hybrid_noaff_cols].values.astype(np.float32), nan=0.0)

model_hyb2 = xgb.XGBClassifier(**XGB_PARAMS)
model_hyb2.fit(X_tr_hyb2, y_tr, eval_set=[(X_vl_hyb2, y_vl)], verbose=0)
hybrid_noaff_auc = roc_auc_score(y_vl, model_hyb2.predict_proba(X_vl_hyb2)[:, 1])
print(f"  Hybrid (no affinity + MF, {len(hybrid_noaff_cols)} feat.) AUC: {hybrid_noaff_auc:.4f}")

# --- MF feature importance in hybrid model ---
print("\n--- MF Feature Importance in Hybrid Model ---")
imp = dict(zip(hybrid_feature_cols, model_hyb.feature_importances_))
for mf_feat in mf_feature_names:
    print(f"  {mf_feat}: {imp.get(mf_feat, 0):.4f}")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 80)
print("FINAL COMPARISON TABLE (all temporal split)")
print("=" * 80)

results = [
    ("Global mean baseline", 0.500),
    ("Per-item listen rate", 0.612),
    ("Implicit ALS (64 factors)", als_auc),
    ("Per-user listen rate", 0.728),
    ("Logistic Regression (47 feat.)", 0.743),
    ("XGBoost (47 feat.)", base_auc),
    ("XGBoost (no affinity, 44 feat.)", noaff_auc),
    ("Hybrid: XGBoost + MF (51 feat.)", hybrid_auc),
    (f"Hybrid: no affinity + MF ({len(hybrid_noaff_cols)} feat.)", hybrid_noaff_auc),
]

print(f"\n  {'Model':<45} {'AUC'}")
print(f"  {'-'*55}")
for name, auc in sorted(results, key=lambda x: x[1]):
    marker = " <-- best" if auc == max(r[1] for r in results) else ""
    print(f"  {name:<45} {auc:.4f}{marker}")
print(f"\n  Competition winner:                          0.686")

# Save
output = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "als_factors": N_FACTORS,
    "als_auc_all": float(als_auc),
    "als_auc_warm": float(als_auc_warm),
    "xgb_base_auc": float(base_auc),
    "xgb_no_affinity_auc": float(noaff_auc),
    "hybrid_auc": float(hybrid_auc),
    "hybrid_no_affinity_auc": float(hybrid_noaff_auc),
    "mf_feature_importance": {k: float(imp.get(k, 0)) for k in mf_feature_names},
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "als_hybrid_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved: {out_path}")
print(f"  Done: {datetime.now().strftime('%H:%M:%S')}")
print("=" * 80)
