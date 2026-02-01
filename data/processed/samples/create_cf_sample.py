"""
Create reproducible sample dataset for collaborative filtering experiments.
This ensures all team members work with the same data.

Usage:
    python create_cf_sample.py
    
Output:
    - cf_sample_500k.csv - 500K interactions for collaborative filtering
    - cf_sample_info.txt - Dataset statistics
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CREATING COLLABORATIVE FILTERING SAMPLE DATASET")
print("=" * 80)

# Configuration
SAMPLE_SIZE = 500000
RANDOM_SEED = 42  # For reproducibility
OUTPUT_FILE = 'cf_sample_500k.csv'
INFO_FILE = 'cf_sample_info.txt'

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,} interactions")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Output file: {OUTPUT_FILE}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[Step 1] Loading training data...")

try:
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load data with fixed random seed
    df = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
    
    print(f"✓ Loaded {len(df):,} interactions")
    
except FileNotFoundError:
    print("❌ Error: ../data/raw/train.csv not found")
    print("   Please make sure the data file exists")
    exit(1)

# ============================================================================
# STEP 2: EXTRACT SAMPLE
# ============================================================================
print("\n[Step 2] Creating reproducible sample...")

# For reproducibility, we use the first N rows
# (since nrows with random seed gives consistent results)
sample_df = df.copy()

print(f"✓ Sample created: {len(sample_df):,} interactions")

# ============================================================================
# STEP 3: CALCULATE STATISTICS
# ============================================================================
print("\n[Step 3] Calculating statistics...")

stats = {
    'total_interactions': len(sample_df),
    'unique_users': sample_df['user_id'].nunique(),
    'unique_items': sample_df['media_id'].nunique(),
    'unique_albums': sample_df['album_id'].nunique(),
    'unique_artists': sample_df['artist_id'].nunique(),
    'unique_genres': sample_df['genre_id'].nunique(),
    'listen_rate': sample_df['is_listened'].mean(),
    'skip_rate': 1 - sample_df['is_listened'].mean(),
    'avg_user_interactions': len(sample_df) / sample_df['user_id'].nunique(),
    'avg_item_interactions': len(sample_df) / sample_df['media_id'].nunique(),
}

# Calculate sparsity
matrix_size = stats['unique_users'] * stats['unique_items']
stats['sparsity'] = (1 - len(sample_df) / matrix_size) * 100

print("\nDataset Statistics:")
print(f"  Total interactions: {stats['total_interactions']:,}")
print(f"  Unique users: {stats['unique_users']:,}")
print(f"  Unique items (tracks): {stats['unique_items']:,}")
print(f"  Unique albums: {stats['unique_albums']:,}")
print(f"  Unique artists: {stats['unique_artists']:,}")
print(f"  Unique genres: {stats['unique_genres']:,}")
print(f"  Listen rate: {stats['listen_rate']:.3f} ({stats['listen_rate']*100:.1f}%)")
print(f"  Skip rate: {stats['skip_rate']:.3f} ({stats['skip_rate']*100:.1f}%)")
print(f"  Matrix sparsity: {stats['sparsity']:.2f}%")
print(f"  Avg interactions per user: {stats['avg_user_interactions']:.1f}")
print(f"  Avg interactions per item: {stats['avg_item_interactions']:.1f}")

# ============================================================================
# STEP 4: SAVE FILES
# ============================================================================
print("\n[Step 4] Saving files...")

# Save sample data
sample_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved: {OUTPUT_FILE}")

# Calculate file size
import os
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")

# Save info file
with open(INFO_FILE, 'w') as f:
    f.write("COLLABORATIVE FILTERING SAMPLE DATASET\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Created: {pd.Timestamp.now()}\n")
    f.write(f"Random Seed: {RANDOM_SEED}\n")
    f.write(f"Sample Size: {SAMPLE_SIZE:,}\n\n")
    
    f.write("DATASET STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total interactions:        {stats['total_interactions']:>12,}\n")
    f.write(f"Unique users:              {stats['unique_users']:>12,}\n")
    f.write(f"Unique items (tracks):     {stats['unique_items']:>12,}\n")
    f.write(f"Unique albums:             {stats['unique_albums']:>12,}\n")
    f.write(f"Unique artists:            {stats['unique_artists']:>12,}\n")
    f.write(f"Unique genres:             {stats['unique_genres']:>12,}\n")
    f.write(f"\n")
    f.write(f"Listen rate:               {stats['listen_rate']:>12.3f} ({stats['listen_rate']*100:.1f}%)\n")
    f.write(f"Skip rate:                 {stats['skip_rate']:>12.3f} ({stats['skip_rate']*100:.1f}%)\n")
    f.write(f"\n")
    f.write(f"Matrix size (U×I):         {matrix_size:>12,}\n")
    f.write(f"Matrix sparsity:           {stats['sparsity']:>12.2f}%\n")
    f.write(f"\n")
    f.write(f"Avg interactions/user:     {stats['avg_user_interactions']:>12.1f}\n")
    f.write(f"Avg interactions/item:     {stats['avg_item_interactions']:>12.1f}\n")
    f.write(f"\n")
    
    f.write("COLUMNS\n")
    f.write("-" * 80 + "\n")
    for col in sample_df.columns:
        f.write(f"  - {col}\n")
    
    f.write("\nFILE INFO\n")
    f.write("-" * 80 + "\n")
    f.write(f"Filename:  {OUTPUT_FILE}\n")
    f.write(f"Size:      {file_size_mb:.1f} MB\n")
    f.write(f"Format:    CSV (comma-separated)\n")
    f.write(f"Encoding:  UTF-8\n")
    
    f.write("\nUSAGE\n")
    f.write("-" * 80 + "\n")
    f.write(f"import pandas as pd\n")
    f.write(f"df = pd.read_csv('{OUTPUT_FILE}')\n")
    f.write(f"\n")
    f.write(f"# This sample is reproducible - all team members will have\n")
    f.write(f"# the exact same data for collaborative filtering experiments.\n")
    f.write(f"\n")
    f.write(f"# To use with the baseline script:\n")
    f.write(f"# Modify baseline_collaborative_filtering.py to load this file\n")
    f.write(f"# instead of reading from train.csv with nrows\n")

print(f"✓ Saved: {INFO_FILE}")

# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================
print("\n[Step 5] Verification...")

# Reload and verify
df_verify = pd.read_csv(OUTPUT_FILE)

print(f"✓ File loaded successfully")
print(f"  Rows: {len(df_verify):,}")
print(f"  Columns: {len(df_verify.columns)}")
print(f"  Memory: {df_verify.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

# Verify no missing values in key columns
missing = df_verify[['user_id', 'media_id', 'is_listened']].isnull().sum().sum()
if missing == 0:
    print(f"✓ No missing values in key columns")
else:
    print(f"⚠️  Warning: {missing} missing values found")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
✅ Reproducible sample created successfully!

📁 FILES CREATED:
   1. {OUTPUT_FILE} ({file_size_mb:.1f} MB)
      - {stats['total_interactions']:,} interactions
      - {stats['unique_users']:,} users × {stats['unique_items']:,} items
      
   2. {INFO_FILE}
      - Complete dataset documentation
      - Usage instructions

🔑 REPRODUCIBILITY:
   - Random seed: {RANDOM_SEED}
   - All team members loading this file will have identical data
   - Results will be comparable across experiments

📤 SHARING WITH COLLEAGUES:
   1. Share the '{OUTPUT_FILE}' file
   2. Everyone uses this file for CF experiments
   3. Results will be consistent and comparable

💡 NEXT STEPS:
   1. Share {OUTPUT_FILE} with your team
   2. Update baseline scripts to use this file:
      df = pd.read_csv('{OUTPUT_FILE}')
   3. Run collaborative filtering experiments
   4. Compare results (should match exactly!)

""")

print("=" * 80)
print("✅ Done!")
print("=" * 80)
