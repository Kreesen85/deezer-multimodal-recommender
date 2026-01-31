"""
Demo: Preprocessing pipeline with temporal feature engineering
"""

import sys
sys.path.append('..')

from src.data.preprocessing import (
    preprocess_data, 
    print_preprocessing_summary,
    get_feature_lists
)
import pandas as pd

print("=" * 80)
print("PREPROCESSING DEMO - WITH PRE-RELEASE FEATURE")
print("=" * 80)

# Load sample
print("\nLoading sample data...")
df_raw = pd.read_csv('../data/raw/train.csv', nrows=50000)
print(f"✓ Loaded {len(df_raw):,} rows")

print(f"\nOriginal columns ({len(df_raw.columns)}): {list(df_raw.columns)}")

# Preprocess
print("\n" + "-" * 80)
df_processed = preprocess_data(df_raw, add_features=True)

# Print summary
print_preprocessing_summary(df_raw, df_processed)

# Show examples of pre-release listening
print("\n" + "=" * 80)
print("PRE-RELEASE LISTENING EXAMPLES")
print("=" * 80)

pre_release = df_processed[df_processed['is_pre_release_listen'] == 1].head(10)

if len(pre_release) > 0:
    print(f"\nFound {len(pre_release):,} pre-release listening examples in sample")
    print("\nSample records:")
    display_cols = ['listen_date', 'release_date_parsed', 'days_since_release', 
                    'is_pre_release_listen', 'media_id', 'artist_id']
    print(pre_release[display_cols].to_string(index=False))
else:
    print("\nNo pre-release listening in this sample")

# Show feature statistics
print("\n" + "=" * 80)
print("FEATURE STATISTICS")
print("=" * 80)

print("\nTemporal Features:")
print(f"  Weekend listening: {df_processed['is_weekend'].mean()*100:.1f}%")
print(f"  Late night (1-5 AM): {df_processed['is_late_night'].mean()*100:.1f}%")
print(f"  Evening (6-11 PM): {df_processed['is_evening'].mean()*100:.1f}%")
print(f"  Commute time (7-9 AM): {df_processed['is_commute_time'].mean()*100:.1f}%")

print("\nRelease Features:")
print(f"  Pre-release listening: {df_processed['is_pre_release_listen'].mean()*100:.2f}%")
print(f"  New releases (<30 days): {df_processed['is_new_release'].mean()*100:.1f}%")
print(f"  Average days since release: {df_processed['days_since_release'].mean():.0f} days")

print("\nDuration Features:")
print(f"  Extended tracks (>5 min): {df_processed['is_extended_track'].mean()*100:.1f}%")
print(f"  Average duration: {df_processed['duration_minutes'].mean():.1f} minutes")

# Show track age distribution
print("\nTrack Age Distribution:")
age_dist = df_processed['track_age_category'].value_counts().sort_index()
age_labels = ['Pre-release', 'New (0-30d)', 'Recent (1mo-1yr)', 
              'Catalog (1-5yr)', 'Deep Catalog (5+yr)']
for idx, count in age_dist.items():
    pct = (count / len(df_processed)) * 100
    print(f"  {age_labels[idx]}: {count:,} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("READY FOR MODELING")
print("=" * 80)

features = get_feature_lists()
all_features = (features['original_features'] + 
                features['temporal_features'] + 
                features['release_features'] +
                features['duration_features'])

print(f"\nTotal engineered features available: {len(all_features)}")
print("\nFeature categories:")
for category, feat_list in features.items():
    if category != 'high_cardinality_ids' and category != 'low_cardinality_categorical':
        print(f"  - {category}: {len(feat_list)} features")

print(f"\nDataset ready for modeling!")
print(f"  - Rows: {len(df_processed):,}")
print(f"  - Features: {len(all_features)}")
print(f"  - Target: is_listened")
print(f"  - Pre-release feature included: ✓")
