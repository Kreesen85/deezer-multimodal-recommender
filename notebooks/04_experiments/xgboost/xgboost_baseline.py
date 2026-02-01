"""
XGBoost Baseline for Deezer Skip Prediction
============================================

Task: Predict probability that a user will listen to a recommended track (>30 seconds)
Evaluation: ROC AUC
Model: XGBoost Classifier with engineered features

This script trains an XGBoost model using all 46 engineered features from preprocessing.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Configuration
DATA_PATH = "/Users/kreesen/Documents/deezer-multimodal-recommender/data/processed/preprocessing/train_100k_preprocessed.csv"
SAMPLE_SIZE = None  # None = use all data, or specify number (e.g., 100000)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'auc',
    'early_stopping_rounds': 20,
    'n_jobs': -1
}

print("=" * 80)
print("XGBoost Baseline - Deezer Skip Prediction")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("STEP 1: Loading data...")
print(f"Data path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(df):,} rows")
print(f"  - Users: {df['user_id'].nunique():,}")
print(f"  - Items: {df['media_id'].nunique():,}")
print(f"  - Listen rate: {df['is_listened'].mean():.2%}")
print(f"  - Skip rate: {(1 - df['is_listened'].mean()):.2%}\n")

# ============================================================================
# STEP 2: Feature Selection
# ============================================================================
print("STEP 2: Selecting features...")

# Columns to exclude (not features)
exclude_cols = [
    'is_listened',           # Target variable
    'user_id',               # ID (not a feature, but useful for analysis)
    'media_id',              # ID (not a feature, but useful for analysis)
    'artist_id',             # ID (could be used with encoding, skip for now)
    'album_id',              # ID (could be used with encoding, skip for now)
    'genre_id',              # ID (could be used with encoding, skip for now)
    'datetime',              # String datetime (already converted to features)
    'release_date_parsed',   # Date (already converted to features)
    'listen_date',           # Date (already converted to features)
    'ts_listen',             # Timestamp (already converted to hour, etc.)
    'release_date'           # Raw date (already parsed)
]

# Get feature columns
all_cols = df.columns.tolist()
feature_cols = [col for col in all_cols if col not in exclude_cols]

print(f"✓ Selected {len(feature_cols)} features")
print("\nFeature categories:")

# Group features by type for better understanding
temporal_features = [col for col in feature_cols if any(x in col for x in ['hour', 'day', 'month', 'weekend', 'night', 'evening', 'commute', 'time_of_day'])]
release_features = [col for col in feature_cols if any(x in col for x in ['release', 'track_age', 'decade', 'days_since'])]
duration_features = [col for col in feature_cols if 'duration' in col or 'extended' in col]
user_features = [col for col in feature_cols if col.startswith('user_') and col != 'user_id']
context_features = [col for col in feature_cols if any(x in col for x in ['context', 'platform', 'listen_type'])]
demographic_features = [col for col in feature_cols if any(x in col for x in ['age', 'gender'])]

print(f"  - Temporal: {len(temporal_features)}")
print(f"  - Release: {len(release_features)}")
print(f"  - Duration: {len(duration_features)}")
print(f"  - User engagement: {len(user_features)}")
print(f"  - Context: {len(context_features)}")
print(f"  - Demographics: {len(demographic_features)}\n")

# Prepare X and y
X = df[feature_cols].copy()
y = df['is_listened'].copy()

# Keep user_id and media_id for analysis (not used in training)
user_ids = df['user_id'].copy()
media_ids = df['media_id'].copy()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}\n")

# ============================================================================
# STEP 3: Handle Missing Values
# ============================================================================
print("STEP 3: Checking for missing values...")

missing = X.isnull().sum()
if missing.sum() > 0:
    print("⚠ Missing values found:")
    print(missing[missing > 0])
    print("\n→ Filling missing values with median...\n")
    X = X.fillna(X.median())
else:
    print("✓ No missing values found\n")

# ============================================================================
# STEP 4: Train/Validation Split
# ============================================================================
print("STEP 4: Splitting data...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE,
    stratify=y
)

# Also split user/media IDs for later analysis
user_ids_train, user_ids_val = train_test_split(
    user_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
media_ids_train, media_ids_val = train_test_split(
    media_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {len(X_train):,} samples")
print(f"  - Listen rate: {y_train.mean():.2%}")
print(f"  - Skip rate: {(1 - y_train.mean()):.2%}")
print(f"\nValidation set: {len(X_val):,} samples")
print(f"  - Listen rate: {y_val.mean():.2%}")
print(f"  - Skip rate: {(1 - y_val.mean()):.2%}\n")

# ============================================================================
# STEP 5: Train XGBoost Model
# ============================================================================
print("STEP 5: Training XGBoost model...")
print(f"Hyperparameters: {json.dumps(XGBOOST_PARAMS, indent=2)}\n")

model = xgb.XGBClassifier(**XGBOOST_PARAMS)

print("Training in progress...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50  # Print every 50 iterations
)

print("\n✓ Training complete!\n")

# ============================================================================
# STEP 6: Evaluate Model
# ============================================================================
print("STEP 6: Evaluating model...")

# Predictions (probabilities)
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

# Predictions (binary, using 0.5 threshold)
y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
y_val_pred = (y_val_pred_proba >= 0.5).astype(int)

# ROC AUC (main metric)
train_auc = roc_auc_score(y_train, y_train_pred_proba)
val_auc = roc_auc_score(y_val, y_val_pred_proba)

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Training ROC AUC:   {train_auc:.4f}")
print(f"Validation ROC AUC: {val_auc:.4f}")
print(f"Overfitting gap:    {train_auc - val_auc:.4f}")
print("=" * 80)
print()

# Classification report
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Skip (0)', 'Listen (1)']))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
print("Validation Confusion Matrix:")
print(f"                Predicted")
print(f"                Skip  Listen")
print(f"Actual Skip     {cm[0, 0]:6d}  {cm[0, 1]:6d}")
print(f"Actual Listen   {cm[1, 0]:6d}  {cm[1, 1]:6d}")
print()

# ============================================================================
# STEP 7: Feature Importance Analysis
# ============================================================================
print("STEP 7: Analyzing feature importance...")

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print("=" * 60)
for idx, row in importance_df.head(20).iterrows():
    print(f"{row['feature']:40s} {row['importance']:8.4f}")
print("=" * 60)
print()

# Save full importance
importance_df.to_csv('feature_importance.csv', index=False)
print("✓ Full feature importance saved to: feature_importance.csv\n")

# ============================================================================
# STEP 8: Save Results and Visualizations
# ============================================================================
print("STEP 8: Saving results and visualizations...")

# Save model
model.save_model('xgboost_model.json')
print("✓ Model saved to: xgboost_model.json")

# Save predictions
val_results = pd.DataFrame({
    'user_id': user_ids_val.values,
    'media_id': media_ids_val.values,
    'y_true': y_val.values,
    'y_pred_proba': y_val_pred_proba,
    'y_pred': y_val_pred
})
val_results.to_csv('validation_predictions.csv', index=False)
print("✓ Validation predictions saved to: validation_predictions.csv")

# Save metrics summary
metrics_summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'data_path': DATA_PATH,
    'sample_size': len(df),
    'n_features': len(feature_cols),
    'train_size': len(X_train),
    'val_size': len(X_val),
    'train_auc': float(train_auc),
    'val_auc': float(val_auc),
    'hyperparameters': XGBOOST_PARAMS
}

with open('metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("✓ Metrics summary saved to: metrics_summary.json")

# Visualization 1: Feature Importance (Top 20)
plt.figure(figsize=(10, 8))
top_features = importance_df.head(20)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 20 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved to: feature_importance_plot.png")

# Visualization 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Skip', 'Listen'],
            yticklabels=['Skip', 'Listen'])
plt.title('Confusion Matrix - Validation Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix plot saved to: confusion_matrix.png")

# Visualization 3: Prediction Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_val_pred_proba[y_val == 0], bins=50, alpha=0.6, label='Actual Skip', color='red')
plt.hist(y_val_pred_proba[y_val == 1], bins=50, alpha=0.6, label='Actual Listen', color='green')
plt.xlabel('Predicted Probability (Listen)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Prediction distribution plot saved to: prediction_distribution.png")

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE!")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n🎯 Validation ROC AUC: {val_auc:.4f}")
print("\nGenerated files:")
print("  - xgboost_model.json")
print("  - feature_importance.csv")
print("  - validation_predictions.csv")
print("  - metrics_summary.json")
print("  - feature_importance_plot.png")
print("  - confusion_matrix.png")
print("  - prediction_distribution.png")
print("=" * 80)
