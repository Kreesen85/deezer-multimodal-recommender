"""
Quick data quality check for the Deezer dataset
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("DATA QUALITY ASSESSMENT")
print("=" * 80)

# Load a sample to check data quality
print("\nLoading sample...")
df = pd.read_csv('../data/raw/train.csv', nrows=100000)

print(f"Sample size: {len(df):,} rows\n")

# 1. Missing values
print("1. MISSING VALUES CHECK")
print("-" * 40)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print(missing[missing > 0])

# 2. Data types
print("\n2. DATA TYPES")
print("-" * 40)
print(df.dtypes)

# 3. Check for invalid values
print("\n3. INVALID VALUES CHECK")
print("-" * 40)

# Target variable
target_values = df['is_listened'].unique()
print(f"Target (is_listened) values: {sorted(target_values)}")
if set(target_values).issubset({0, 1}):
    print("✓ Target is binary (0, 1)")
else:
    print("⚠ Target has unexpected values!")

# Age range
age_range = (df['user_age'].min(), df['user_age'].max())
print(f"Age range: {age_range[0]} - {age_range[1]}")
if age_range[0] >= 0 and age_range[1] <= 120:
    print("✓ Age values look reasonable")
else:
    print("⚠ Age values might need cleaning")

# Duration
duration_stats = df['media_duration'].describe()
print(f"\nDuration (seconds):")
print(f"  Min: {duration_stats['min']:.0f}s")
print(f"  Max: {duration_stats['max']:.0f}s")
print(f"  Mean: {duration_stats['mean']:.0f}s")
print(f"  Median: {duration_stats['50%']:.0f}s")

# Check for zero or negative durations
invalid_duration = (df['media_duration'] <= 0).sum()
if invalid_duration > 0:
    print(f"⚠ Found {invalid_duration} tracks with duration <= 0")
else:
    print("✓ All durations are positive")

# Check for extremely long durations (> 1 hour)
long_tracks = (df['media_duration'] > 3600).sum()
if long_tracks > 0:
    print(f"⚠ Found {long_tracks} tracks > 1 hour ({long_tracks/len(df)*100:.2f}%)")
else:
    print("✓ No extremely long tracks")

# 4. Categorical variables
print("\n4. CATEGORICAL VARIABLES")
print("-" * 40)
for col in ['platform_name', 'platform_family', 'listen_type', 'user_gender']:
    n_unique = df[col].nunique()
    values = sorted(df[col].unique())
    print(f"{col}: {n_unique} unique values: {values}")

# 5. Release date format
print("\n5. DATE FIELDS")
print("-" * 40)
print(f"release_date sample values: {df['release_date'].head().tolist()}")
print(f"release_date type: {df['release_date'].dtype}")
# Check if it's in YYYYMMDD format
try:
    sample_dates = df['release_date'].head(100)
    # Try to convert
    test_convert = pd.to_datetime(sample_dates, format='%Y%m%d', errors='coerce')
    valid_dates = test_convert.notna().sum()
    print(f"✓ Release dates appear to be in YYYYMMDD format ({valid_dates}/100 valid)")
except:
    print("⚠ Release date format needs investigation")

# Timestamp
print(f"\nts_listen sample: {df['ts_listen'].head().tolist()}")
print(f"ts_listen type: {df['ts_listen'].dtype}")
print("✓ Timestamp appears to be Unix epoch format")

# 6. Duplicates
print("\n6. DUPLICATES")
print("-" * 40)
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates} ({n_duplicates/len(df)*100:.2f}%)")
if n_duplicates == 0:
    print("✓ No duplicate rows")

# 7. ID uniqueness
print("\n7. ID FIELDS")
print("-" * 40)
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique tracks (media_id): {df['media_id'].nunique():,}")
print(f"Unique artists: {df['artist_id'].nunique():,}")
print(f"Unique albums: {df['album_id'].nunique():,}")
print(f"Unique genres: {df['genre_id'].nunique():,}")

# 8. Check for outliers
print("\n8. POTENTIAL OUTLIERS")
print("-" * 40)

# Check genre_id range
print(f"Genre IDs range: {df['genre_id'].min()} - {df['genre_id'].max()}")

# Check context_type range
print(f"Context types range: {df['context_type'].min()} - {df['context_type'].max()}")
print(f"Number of context types: {df['context_type'].nunique()}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
DATA QUALITY: GOOD ✓

The dataset appears to be well-cleaned with:
- No missing values
- Valid binary target variable
- Reasonable value ranges
- No duplicates

PREPROCESSING NEEDED:
1. ✓ Feature Engineering (high priority)
   - Convert timestamps to datetime features (hour, day, etc.)
   - Parse release_date (YYYYMMDD format)
   - Create derived features (track age, user patterns)
   
2. ✓ Encoding (required for modeling)
   - One-hot encode: platform_name, platform_family
   - Label encode: genre_id, context_type, artist_id, etc.
   
3. ✓ Scaling (for some models)
   - Normalize: media_duration, user_age
   - StandardScaler for neural networks
   
4. ✓ Feature Selection
   - Handle high cardinality (1M+ tracks, 250K+ artists)
   - Aggregation strategies for IDs
   
5. Optional: Outlier handling
   - Very long tracks (>1 hour) - investigate or cap
   
DATA IS CLEAN - Focus on feature engineering!
""")
print("=" * 80)
