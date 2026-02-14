"""
Neural Network Baseline - Deezer Skip Prediction
==================================================

A simple feedforward neural network for skip prediction.
Uses the same feature engineering pipeline as XGBoost v2
to allow fair comparison.

Architecture:
    Input (47 features)
    -> BatchNorm -> Linear(256) -> ReLU -> Dropout(0.3)
    -> BatchNorm -> Linear(128) -> ReLU -> Dropout(0.3)
    -> BatchNorm -> Linear(64)  -> ReLU -> Dropout(0.2)
    -> Linear(1) -> Sigmoid

Usage:
    cd notebooks/04_experiments/neural_net
    /opt/anaconda3/bin/python neural_net_baseline.py
"""

import pandas as pd
import numpy as np
import sys
import os
import json
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root
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

RANDOM_STATE = 42
VALIDATION_SIZE = 0.1

# Neural network hyperparameters
BATCH_SIZE = 4096
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Feature columns to exclude
EXCLUDE_COLS = [
    "is_listened", "sample_id", "user_id", "media_id",
    "artist_id", "album_id", "genre_id",
    "datetime", "release_date_parsed", "listen_date",
    "ts_listen", "release_date",
]

# Device
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

    # Media stats
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

    # Artist stats
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

    # User-Artist
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

    # User-Genre
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

    # User knows artist flag
    for df in [train_df, test_df]:
        df["user_knows_artist"] = (df["user_artist_affinity"] != global_mean).astype(int)

    return train_df, test_df


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class SkipPredictor(nn.Module):
    """Simple feedforward network for skip prediction."""

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y_batch.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    auc = roc_auc_score(labels, preds)
    return auc, preds


def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

print("=" * 80)
print("NEURAL NETWORK BASELINE - DEEZER SKIP PREDICTION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {DEVICE}")
print()

# --- Load data ---
print("STEP 1: Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_ids = test_df["sample_id"].copy()
print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows\n")

# --- User stats ---
print("STEP 2: Computing user stats...")
user_stats = compute_user_features_from_train(train_df)
print()

# --- Base features ---
print("STEP 3: Adding features...")
for df in [train_df, test_df]:
    add_temporal_features(df)
    add_release_features(df)
    add_duration_features(df)

train_df = apply_user_features(train_df, user_stats)
test_df = apply_user_features(test_df, user_stats)

# --- New features (same as v2) ---
print("  Adding target encoding...")
for col in ["genre_id", "artist_id", "album_id"]:
    tr_enc, te_enc, enc_name = target_encode(train_df, test_df, col, "is_listened")
    train_df[enc_name] = tr_enc
    test_df[enc_name] = te_enc

print("  Adding item features...")
train_df, test_df = add_item_features(train_df, test_df)

print("  Adding user-item affinity...")
train_df, test_df = add_user_item_affinity(train_df, test_df)
print()

# --- Prepare features ---
print("STEP 4: Preparing features...")
feature_cols = [c for c in train_df.columns if c not in EXCLUDE_COLS]
print(f"  Features: {len(feature_cols)}")

X_all = train_df[feature_cols].values.astype(np.float32)
y_all = train_df["is_listened"].values.astype(np.float32)
X_test_raw = test_df[feature_cols].values.astype(np.float32)

# Handle NaN
X_all = np.nan_to_num(X_all, nan=0.0)
X_test_raw = np.nan_to_num(X_test_raw, nan=0.0)

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE, stratify=y_all
)
print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}\n")

# Scale features (important for neural networks)
print("STEP 5: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_raw)
X_all_scaled = scaler.fit_transform(X_all)  # For final retraining
print("  Done\n")

# --- Create DataLoaders ---
print("STEP 6: Creating data loaders...")
train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32),
)
test_dataset = TensorDataset(
    torch.tensor(X_test_scaled, dtype=torch.float32),
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
print(f"  Batches per epoch: {len(train_loader):,}\n")

# --- Train neural network ---
print("STEP 7: Training neural network...")
print(f"  Architecture: {len(feature_cols)} -> 256 -> 128 -> 64 -> 1")
print(f"  Epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE}")
print()

model = SkipPredictor(len(feature_cols)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)

best_auc = 0
best_state = None

print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val AUC':>10}  {'LR':>10}  {'Status'}")
print("  " + "-" * 55)

for epoch in range(1, EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_auc, _ = evaluate(model, val_loader, DEVICE)
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_auc)

    status = ""
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        status = " *best*"

    print(f"  {epoch:5d}  {train_loss:10.4f}  {val_auc:10.4f}  {current_lr:10.6f}{status}")

print(f"\n  Best Validation AUC: {best_auc:.4f}")

# Load best model
model.load_state_dict(best_state)
print()


# --- Results comparison ---
print("=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)
print(f"  v1 XGBoost (35 features, 100K):     0.8722 AUC")
print(f"  v2 XGBoost (47 features, 7.5M):     0.9341 AUC")
print(f"  v2 LightGBM (47 features, 7.5M):    0.9352 AUC")
print(f"  v2 Ensemble (XGB+LGB):              0.9352 AUC")
print(f"  Neural Net ({len(feature_cols)} features, 7.5M):  {best_auc:.4f} AUC")
print("=" * 80)
print()


# --- Retrain on ALL data ---
print("STEP 8: Retraining on ALL data for final predictions...")

all_dataset = TensorDataset(
    torch.tensor(X_all_scaled, dtype=torch.float32),
    torch.tensor(y_all, dtype=torch.float32),
)
all_loader = DataLoader(all_dataset, batch_size=BATCH_SIZE, shuffle=True)

final_model = SkipPredictor(len(feature_cols)).to(DEVICE)
final_optimizer = torch.optim.Adam(
    final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
final_criterion = nn.BCELoss()

# Train for same number of epochs as best
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(final_model, all_loader, final_optimizer, final_criterion, DEVICE)
    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"  Epoch {epoch}/{EPOCHS}  Loss: {loss:.4f}")

# Rescale test with all data scaler
X_test_final = scaler.transform(X_test_raw)
test_final_dataset = TensorDataset(torch.tensor(X_test_final, dtype=torch.float32))
test_final_loader = DataLoader(test_final_dataset, batch_size=BATCH_SIZE * 2)

print()


# --- Generate predictions ---
print("STEP 9: Generating predictions...")

nn_predictions = predict(final_model, test_final_loader, DEVICE)
nn_predictions = np.clip(nn_predictions, 0.0, 1.0)

print(f"  Min:    {nn_predictions.min():.4f}")
print(f"  Max:    {nn_predictions.max():.4f}")
print(f"  Mean:   {nn_predictions.mean():.4f}")
print(f"  Median: {np.median(nn_predictions):.4f}")
print()


# --- Create submission ---
print("STEP 10: Creating submission...")

submission = pd.DataFrame({
    "sample_id": sample_ids,
    "is_listened": nn_predictions,
})
submission = submission.sort_values("sample_id").reset_index(drop=True)

# Verify
checks_passed = (
    len(submission) == 19918
    and list(submission.columns) == ["sample_id", "is_listened"]
    and submission["sample_id"].min() == 0
    and submission["sample_id"].max() == 19917
    and (submission["is_listened"] >= 0).all()
    and (submission["is_listened"] <= 1).all()
    and submission["is_listened"].notna().all()
)
print(f"  Verification: {'ALL PASSED' if checks_passed else 'FAILED'}")

sub_path = os.path.join(OUTPUT_DIR, "submission_nn.csv")
submission.to_csv(sub_path, index=False)
print(f"  Saved: {sub_path}")

# Save model
model_path = os.path.join(OUTPUT_DIR, "neural_net_model.pt")
torch.save(final_model.state_dict(), model_path)
print(f"  Model saved: {model_path}")

# Save summary
summary = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model": "Feedforward Neural Network",
    "architecture": f"{len(feature_cols)} -> 256 -> 128 -> 64 -> 1",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "best_val_auc": float(best_auc),
    "n_features": len(feature_cols),
    "train_rows": int(len(train_df)),
    "device": str(DEVICE),
}
with open(os.path.join(OUTPUT_DIR, "nn_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)


# --- Done ---
print("\n" + "=" * 80)
print("NEURAL NETWORK EXPERIMENT COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nBest Validation AUC: {best_auc:.4f}")
print(f"\nGenerated files:")
print(f"  - submission_nn.csv")
print(f"  - neural_net_model.pt")
print(f"  - nn_summary.json")
print("=" * 80)
