"""
Demo: Preprocessing Pipeline with User Engagement Features
Shows proper train/test workflow to prevent data leakage
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from src.data.preprocessing import (
    add_temporal_features,
    add_release_features, 
    add_duration_features,
    compute_user_features_from_train,
    apply_user_features,
    print_preprocessing_summary,
    get_feature_lists
)

print("=" * 80)
print("PREPROCESSING DEMO: User Engagement Features")
print("=" * 80)

# ============================================================================
# STEP 1: Load Training Data Sample
# ============================================================================
print("\n[Step 1] Loading training data sample...")
SAMPLE_SIZE = 100000
train_df_raw = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(train_df_raw):,} training rows")
print(f"  Columns: {len(train_df_raw.columns)}")
print(f"  Unique users: {train_df_raw['user_id'].nunique():,}")

# ============================================================================
# STEP 2: Add Time-Invariant Features (No Leakage Risk)
# ============================================================================
print("\n[Step 2] Adding temporal, release, and duration features...")
train_df = train_df_raw.copy()
train_df = add_temporal_features(train_df)
train_df = add_release_features(train_df)
train_df = add_duration_features(train_df)
print(f"✓ Features after basic engineering: {len(train_df.columns)}")

# ============================================================================
# STEP 3: Compute User Features from Training Data
# ============================================================================
print("\n[Step 3] Computing user engagement features from training data...")
user_stats = compute_user_features_from_train(train_df)

print("\n--- User Statistics Summary ---")
print(f"Total users: {len(user_stats):,}")
print(f"\nUser Listen Rate:")
print(f"  Mean: {user_stats['user_listen_rate'].mean():.3f}")
print(f"  Median: {user_stats['user_listen_rate'].median():.3f}")
print(f"  Std: {user_stats['user_listen_rate'].std():.3f}")

print(f"\nUser Skip Rate:")
print(f"  Mean: {user_stats['user_skip_rate'].mean():.3f}")
print(f"  Median: {user_stats['user_skip_rate'].median():.3f}")

print(f"\nUser Session Count:")
print(f"  Mean: {user_stats['user_session_count'].mean():.1f}")
print(f"  Median: {user_stats['user_session_count'].median():.1f}")
print(f"  Max: {user_stats['user_session_count'].max():.0f}")

print(f"\nUser Engagement Segments:")
segment_names = ['Never Skips', 'Rarely Skips', 'Occasional', 'Moderate', 'Frequent']
segment_counts = user_stats['user_engagement_segment'].value_counts().sort_index()
for segment_id, count in segment_counts.items():
    pct = (count / len(user_stats)) * 100
    print(f"  {segment_names[segment_id]}: {count:,} users ({pct:.1f}%)")

# ============================================================================
# STEP 4: Apply User Features to Training Data
# ============================================================================
print("\n[Step 4] Applying user features to training data...")
train_df = apply_user_features(train_df, user_stats)
print(f"✓ Final training features: {len(train_df.columns)}")

# ============================================================================
# STEP 5: Load and Process Test Data (Simulating Train/Test Split)
# ============================================================================
print("\n[Step 5] Processing test data with training-derived user features...")
print("(Simulating: loading different portion of data as 'test')")

# Load a different sample to simulate test data
test_df_raw = pd.read_csv('../data/raw/train.csv', skiprows=range(1, SAMPLE_SIZE+1), nrows=20000)
print(f"✓ Loaded {len(test_df_raw):,} 'test' rows")
print(f"  Unique users: {test_df_raw['user_id'].nunique():,}")

# Check for new users
test_users = set(test_df_raw['user_id'])
train_users = set(user_stats['user_id'])
new_users = test_users - train_users
print(f"  New users (cold start): {len(new_users):,} ({len(new_users)/len(test_users)*100:.1f}%)")

# Add basic features
test_df = test_df_raw.copy()
test_df = add_temporal_features(test_df)
test_df = add_release_features(test_df)
test_df = add_duration_features(test_df)

# Apply SAME user stats from training
test_df = apply_user_features(test_df, user_stats)
print(f"✓ Final test features: {len(test_df.columns)}")

# ============================================================================
# STEP 6: Verify Feature Consistency
# ============================================================================
print("\n[Step 6] Verifying feature consistency between train and test...")

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

if train_cols == test_cols:
    print("✓ Feature sets match perfectly!")
else:
    missing_in_test = train_cols - test_cols
    extra_in_test = test_cols - train_cols
    if missing_in_test:
        print(f"⚠ Missing in test: {missing_in_test}")
    if extra_in_test:
        print(f"⚠ Extra in test: {extra_in_test}")

# ============================================================================
# STEP 7: Show Sample of New Features
# ============================================================================
print("\n[Step 7] Sample of user engagement features:")
print("\n--- Training Data Sample ---")
user_feature_cols = ['user_id', 'user_skip_rate', 'user_listen_rate', 
                     'user_session_count', 'user_genre_diversity',
                     'user_engagement_segment', 'user_engagement_score']
print(train_df[user_feature_cols].head(10).to_string(index=False))

print("\n--- Test Data Sample (including new users with defaults) ---")
# Show a mix of known and new users
test_sample = test_df[user_feature_cols].drop_duplicates(subset='user_id').head(10)
print(test_sample.to_string(index=False))

# ============================================================================
# STEP 8: Feature List Summary
# ============================================================================
print("\n[Step 8] Complete feature inventory:")
features = get_feature_lists()

total_features = 0
for category, feat_list in features.items():
    print(f"\n{category}: {len(feat_list)} features")
    print(f"  {', '.join(feat_list)}")
    total_features += len(feat_list)

print(f"\n{'='*80}")
print(f"TOTAL ENGINEERED FEATURES: {total_features}")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"{'='*80}")

# ============================================================================
# STEP 9: Show Preprocessing Summary
# ============================================================================
print_preprocessing_summary(train_df_raw, train_df)

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. ✓ User features computed from TRAINING data only
2. ✓ Same user stats applied to TEST data (prevents leakage)
3. ✓ New users in test handled with sensible defaults
4. ✓ Feature consistency maintained across train/test
5. ✓ 9 new user engagement features added:
   - user_listen_rate
   - user_skip_rate
   - user_session_count
   - user_total_listens
   - user_genre_diversity
   - user_artist_diversity
   - user_context_variety
   - user_engagement_segment (categorical)
   - user_engagement_score (composite metric)

READY FOR MODELING!
""")

# Optional: Save preprocessed data
print("\n💾 Saving preprocessed sample data...")
train_df.to_csv('train_preprocessed_sample.csv', index=False)
test_df.to_csv('test_preprocessed_sample.csv', index=False)
user_stats.to_csv('user_stats_from_train.csv', index=False)
print("✓ Saved:")
print("  - train_preprocessed_sample.csv")
print("  - test_preprocessed_sample.csv")
print("  - user_stats_from_train.csv")

print("\n✅ Demo complete!")
