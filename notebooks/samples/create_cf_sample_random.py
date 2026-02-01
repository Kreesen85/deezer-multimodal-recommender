"""
Create RANDOM reproducible sample for collaborative filtering.
This is better than just taking first 500K rows.

Improvements:
- Random sampling (not just first N rows)
- Fixed seed for reproducibility
- Representative of full dataset
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("CREATING RANDOM CF SAMPLE (BETTER METHOD)")
print("=" * 80)

# Configuration
SAMPLE_SIZE = 500000
RANDOM_SEED = 42
OUTPUT_FILE = 'cf_sample_500k_random.csv'
INFO_FILE = 'cf_sample_random_info.txt'

print(f"\nConfiguration:")
print(f"  Sample size: {SAMPLE_SIZE:,}")
print(f"  Random seed: {RANDOM_SEED}")
print(f"  Method: Random sampling (not sequential)")

# ============================================================================
# STEP 1: LOAD FULL DATASET
# ============================================================================
print("\n[Step 1] Loading full training data...")

df = pd.read_csv('../data/raw/train.csv')
print(f"✓ Loaded {len(df):,} total interactions")

# ============================================================================
# STEP 2: RANDOM SAMPLE
# ============================================================================
print("\n[Step 2] Creating random sample...")

np.random.seed(RANDOM_SEED)

# Random sample WITHOUT replacement
sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

print(f"✓ Random sample created: {len(sample_df):,} interactions")
print(f"  Sampling fraction: {SAMPLE_SIZE/len(df)*100:.2f}%")

# ============================================================================
# STEP 3: STATISTICS
# ============================================================================
print("\n[Step 3] Comparing original vs sample...")

stats = {
    'Original': {
        'interactions': len(df),
        'users': df['user_id'].nunique(),
        'items': df['media_id'].nunique(),
        'listen_rate': df['is_listened'].mean(),
    },
    'Sample': {
        'interactions': len(sample_df),
        'users': sample_df['user_id'].nunique(),
        'items': sample_df['media_id'].nunique(),
        'listen_rate': sample_df['is_listened'].mean(),
    }
}

print("\nComparison:")
print(f"                    Original      Sample")
print(f"  Interactions:  {stats['Original']['interactions']:>10,}  {stats['Sample']['interactions']:>10,}")
print(f"  Users:         {stats['Original']['users']:>10,}  {stats['Sample']['users']:>10,}")
print(f"  Items:         {stats['Original']['items']:>10,}  {stats['Sample']['items']:>10,}")
print(f"  Listen rate:   {stats['Original']['listen_rate']:>10.3f}  {stats['Sample']['listen_rate']:>10.3f}")

# Check representativeness
listen_rate_diff = abs(stats['Original']['listen_rate'] - stats['Sample']['listen_rate'])
if listen_rate_diff < 0.01:
    print(f"\n✅ Sample is representative (listen rate diff: {listen_rate_diff:.4f})")
else:
    print(f"\n⚠️  Sample may have some bias (listen rate diff: {listen_rate_diff:.4f})")

# ============================================================================
# STEP 4: SAVE FILES
# ============================================================================
print("\n[Step 4] Saving files...")

sample_df.to_csv(OUTPUT_FILE, index=False)
print(f"✓ Saved: {OUTPUT_FILE}")

import os
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")

# Save info
with open(INFO_FILE, 'w') as f:
    f.write("RANDOM COLLABORATIVE FILTERING SAMPLE\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Created: {pd.Timestamp.now()}\n")
    f.write(f"Method: Random sampling (seed={RANDOM_SEED})\n")
    f.write(f"Sample size: {SAMPLE_SIZE:,}\n\n")
    
    f.write("WHY RANDOM SAMPLING?\n")
    f.write("-" * 80 + "\n")
    f.write("Random sampling is better than taking first N rows because:\n")
    f.write("  ✓ No temporal bias (all time periods represented)\n")
    f.write("  ✓ No user bias (all users have chance to be included)\n")
    f.write("  ✓ More representative of full dataset\n")
    f.write("  ✓ Still reproducible with fixed seed\n\n")
    
    f.write("STATISTICS COMPARISON\n")
    f.write("-" * 80 + "\n")
    f.write(f"                      Original        Sample\n")
    f.write(f"Interactions:      {stats['Original']['interactions']:>10,}   {stats['Sample']['interactions']:>10,}\n")
    f.write(f"Users:             {stats['Original']['users']:>10,}   {stats['Sample']['users']:>10,}\n")
    f.write(f"Items:             {stats['Original']['items']:>10,}   {stats['Sample']['items']:>10,}\n")
    f.write(f"Listen rate:       {stats['Original']['listen_rate']:>10.3f}   {stats['Sample']['listen_rate']:>10.3f}\n")
    f.write(f"\nListen rate difference: {listen_rate_diff:.4f} {'✓ Good' if listen_rate_diff < 0.01 else '⚠ Check'}\n")

print(f"✓ Saved: {INFO_FILE}")

# ============================================================================
# STEP 5: VERIFICATION
# ============================================================================
print("\n[Step 5] Verification...")

df_verify = pd.read_csv(OUTPUT_FILE)
print(f"✓ File verified: {len(df_verify):,} rows")

# Check it's different from sequential first 500K
df_first_500k = df.head(SAMPLE_SIZE)
overlap = len(set(sample_df.index) & set(df_first_500k.index))
print(f"✓ Overlap with first 500K: {overlap:,} ({overlap/SAMPLE_SIZE*100:.1f}%)")
if overlap < SAMPLE_SIZE * 0.9:
    print("  Good! This is a truly random sample, not just first N rows")

print("\n" + "=" * 80)
print("✅ RANDOM SAMPLE CREATED SUCCESSFULLY!")
print("=" * 80)
print(f"""
This sample is BETTER than the previous one because:
  ✓ Random sampling (not sequential)
  ✓ More representative of full dataset
  ✓ No temporal or user bias
  ✓ Still reproducible (seed={RANDOM_SEED})

File: {OUTPUT_FILE} ({file_size_mb:.1f} MB)

Share this file with your team for consistent results!
""")
