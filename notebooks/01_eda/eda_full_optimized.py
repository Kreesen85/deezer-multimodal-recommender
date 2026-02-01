"""
Deezer Sequential Skip Prediction - Optimized Full Dataset EDA

This script efficiently processes the full 7.5M row dataset using:
- Chunked reading for statistics
- Strategic sampling for visualizations
- Memory-efficient processing
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print("=" * 80)
print("DEEZER SKIP PREDICTION - FULL DATASET EDA (OPTIMIZED)")
print("=" * 80)

TRAIN_FILE = '../data/raw/train.csv'
CHUNK_SIZE = 500000
VIZ_SAMPLE_SIZE = 1000000  # 1M rows for visualizations

# ============================================================================
# 1. COMPUTE STATISTICS ON FULL DATASET (CHUNKED READING)
# ============================================================================
print("\n[1/8] Computing statistics on full dataset (chunked)...")

# Initialize accumulators
total_rows = 0
target_sum = 0
target_counts = {0: 0, 1: 0}
numeric_sums = {}
numeric_counts = {}

# Read in chunks
chunk_iter = pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE)

for i, chunk in enumerate(chunk_iter):
    print(f"  Processing chunk {i+1} ({len(chunk):,} rows)...", end='\r')
    
    total_rows += len(chunk)
    target_sum += chunk['is_listened'].sum()
    target_counts[0] += (chunk['is_listened'] == 0).sum()
    target_counts[1] += (chunk['is_listened'] == 1).sum()
    
    # Accumulate numeric stats
    for col in ['user_age', 'media_duration', 'user_gender', 'listen_type']:
        if col not in numeric_sums:
            numeric_sums[col] = 0
            numeric_counts[col] = 0
        numeric_sums[col] += chunk[col].sum()
        numeric_counts[col] += len(chunk)

print(f"\n✓ Processed {total_rows:,} rows total")

# Compute global statistics
listen_rate = target_sum / total_rows
skip_rate = 1 - listen_rate
imbalance_ratio = target_counts[0] / target_counts[1]

print(f"\n=== FULL DATASET STATISTICS ===")
print(f"Total sessions: {total_rows:,}")
print(f"Skipped (0): {target_counts[0]:,} ({(target_counts[0]/total_rows)*100:.2f}%)")
print(f"Listened (1): {target_counts[1]:,} ({(target_counts[1]/total_rows)*100:.2f}%)")
print(f"Imbalance ratio: {imbalance_ratio:.3f}:1")

for col in numeric_sums:
    avg = numeric_sums[col] / numeric_counts[col]
    print(f"Average {col}: {avg:.2f}")

# ============================================================================
# 2. LOAD SAMPLE FOR DETAILED ANALYSIS & VISUALIZATIONS
# ============================================================================
print(f"\n[2/8] Loading {VIZ_SAMPLE_SIZE:,} rows for detailed analysis...")

# Use skiprows to get a representative sample
np.random.seed(42)
n = total_rows
s = VIZ_SAMPLE_SIZE
skip = sorted(np.random.choice(range(1, n+1), n - s, replace=False))
df = pd.read_csv(TRAIN_FILE, skiprows=skip)

print(f"✓ Loaded {len(df):,} rows for visualization")

# Count unique values
print(f"\n=== CONTENT STATISTICS (from sample) ===")
print(f"Unique users: ~{int(df['user_id'].nunique() * (total_rows/len(df))):,} (estimated)")
print(f"Unique tracks: ~{int(df['media_id'].nunique() * (total_rows/len(df))):,} (estimated)")
print(f"Unique artists: ~{int(df['artist_id'].nunique() * (total_rows/len(df))):,} (estimated)")
print(f"Unique genres: {df['genre_id'].nunique():,}")

# ============================================================================
# 3. TARGET DISTRIBUTION
# ============================================================================
print("\n[3/8] Creating target distribution visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Use full dataset counts
axes[0].bar(['Skipped', 'Listened'], [target_counts[0], target_counts[1]], 
            color=['#e74c3c', '#2ecc71'])
axes[0].set_ylabel('Count')
axes[0].set_title(f'Target Distribution (n={total_rows:,})')
axes[0].ticklabel_format(style='plain', axis='y')
for i, (k, v) in enumerate(target_counts.items()):
    axes[0].text(i, v, f'{v:,}\\n({(v/total_rows)*100:.1f}%)', 
                ha='center', va='bottom')

axes[1].pie([target_counts[0], target_counts[1]], 
            labels=['Skipped', 'Listened'], autopct='%1.1f%%', 
            colors=['#e74c3c', '#2ecc71'], startangle=90)
axes[1].set_title('Target Distribution')

plt.tight_layout()
plt.savefig('eda_full_target.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_target.png")
plt.close()

# ============================================================================
# 4. USER DEMOGRAPHICS
# ============================================================================
print("\n[4/8] Analyzing user demographics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Age distribution
axes[0, 0].hist(df['user_age'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Age Distribution (sample n={len(df):,})')

# Age vs Listen Rate
age_stats = df.groupby('user_age')['is_listened'].mean()
axes[0, 1].plot(age_stats.index, age_stats.values, marker='o', linewidth=2)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Listen Rate')
axes[0, 1].set_title('Listen Rate by Age')
axes[0, 1].grid(True, alpha=0.3)

# Gender distribution
gender_counts = df['user_gender'].value_counts()
axes[1, 0].bar(['Female', 'Male'], gender_counts.values, color=['#e91e63', '#2196f3'])
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Gender Distribution')

# Gender vs Listen Rate
gender_listen = df.groupby('user_gender')['is_listened'].mean()
axes[1, 1].bar(['Female', 'Male'], gender_listen.values, color=['#e91e63', '#2196f3'])
axes[1, 1].set_ylabel('Listen Rate')
axes[1, 1].set_title('Listen Rate by Gender')

plt.tight_layout()
plt.savefig('eda_full_demographics.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_demographics.png")
plt.close()

# ============================================================================
# 5. TRACK DURATION ANALYSIS
# ============================================================================
print("\n[5/8] Analyzing track duration...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['media_duration'], bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Duration (seconds)')
axes[0].set_ylabel('Count')
axes[0].set_title('Track Duration Distribution')
axes[0].axvline(df['media_duration'].median(), color='red', linestyle='--', 
                label=f'Median: {df["media_duration"].median():.0f}s')
axes[0].legend()

duration_bins = pd.cut(df['media_duration'], bins=15)
duration_listen = df.groupby(duration_bins)['is_listened'].mean()
duration_centers = [interval.mid for interval in duration_listen.index]
axes[1].plot(duration_centers, duration_listen.values, marker='o', linewidth=2)
axes[1].set_xlabel('Duration (seconds)')
axes[1].set_ylabel('Listen Rate')
axes[1].set_title('Listen Rate by Duration')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_full_duration.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_duration.png")
plt.close()

# ============================================================================
# 6. TEMPORAL PATTERNS
# ============================================================================
print("\n[6/8] Analyzing temporal patterns...")

df['datetime'] = pd.to_datetime(df['ts_listen'], unit='s')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Hourly pattern
hourly_stats = df.groupby('hour')['is_listened'].mean()
axes[0].plot(hourly_stats.index, hourly_stats.values, marker='o', linewidth=2)
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Listen Rate')
axes[0].set_title('Listen Rate by Hour')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(range(0, 24, 3))

# Day of week pattern
dow_stats = df.groupby('day_of_week')['is_listened'].mean()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes[1].plot(range(7), dow_stats.values, marker='o', linewidth=2)
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Listen Rate')
axes[1].set_title('Listen Rate by Day of Week')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(day_names)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('eda_full_temporal.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_temporal.png")
plt.close()

# ============================================================================
# 7. PLATFORM & CONTEXT
# ============================================================================
print("\n[7/8] Analyzing platform and context...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

platform_listen = df.groupby('platform_name')['is_listened'].mean().sort_values()
platform_listen.plot(kind='barh', ax=axes[0, 0])
axes[0, 0].set_xlabel('Listen Rate')
axes[0, 0].set_title('Listen Rate by Platform')

context_listen = df.groupby('context_type')['is_listened'].mean().sort_values()
context_listen.plot(kind='barh', ax=axes[0, 1])
axes[0, 1].set_xlabel('Listen Rate')
axes[0, 1].set_title('Listen Rate by Context')

listen_type_listen = df.groupby('listen_type')['is_listened'].mean()
axes[1, 0].bar(['Type 0', 'Type 1'], listen_type_listen.values)
axes[1, 0].set_ylabel('Listen Rate')
axes[1, 0].set_title('Listen Rate by Listen Type')

platform_family_listen = df.groupby('platform_family')['is_listened'].mean()
axes[1, 1].bar(platform_family_listen.index.astype(str), platform_family_listen.values)
axes[1, 1].set_xlabel('Platform Family')
axes[1, 1].set_ylabel('Listen Rate')
axes[1, 1].set_title('Listen Rate by Platform Family')

plt.tight_layout()
plt.savefig('eda_full_platform.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_platform.png")
plt.close()

# ============================================================================
# 8. CORRELATIONS
# ============================================================================
print("\n[8/8] Computing feature correlations...")

numeric_features = ['user_age', 'user_gender', 'media_duration', 'listen_type', 
                   'platform_name', 'platform_family', 'context_type', 'is_listened']
corr_matrix = df[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('eda_full_correlations.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: eda_full_correlations.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
summary = {
    'Total Sessions': f"{total_rows:,}",
    'Skipped': f"{target_counts[0]:,} ({(target_counts[0]/total_rows)*100:.2f}%)",
    'Listened': f"{target_counts[1]:,} ({(target_counts[1]/total_rows)*100:.2f}%)",
    'Skip Rate': f"{skip_rate*100:.2f}%",
    'Listen Rate': f"{listen_rate*100:.2f}%",
    'Imbalance Ratio': f"{imbalance_ratio:.3f}:1",
    'Avg Track Duration': f"{numeric_sums['media_duration']/numeric_counts['media_duration']:.1f}s",
}

print("\n" + "=" * 80)
print("FULL DATASET SUMMARY")
print("=" * 80)
for key, value in summary.items():
    print(f"  {key:.<30} {value}")
print("=" * 80)

# Save summary
with open('eda_full_summary.txt', 'w') as f:
    f.write("DEEZER SKIP PREDICTION - FULL DATASET EDA SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Complete dataset: {total_rows:,} rows\n")
    f.write(f"Visualization sample: {len(df):,} rows\n\n")
    for key, value in summary.items():
        f.write(f"{key}: {value}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("KEY INSIGHTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"""
1. CLASS DISTRIBUTION: 
   - Listen rate: {listen_rate*100:.2f}%
   - Skip rate: {skip_rate*100:.2f}%
   - Moderate class imbalance (ratio {imbalance_ratio:.2f}:1)

2. DATASET SCALE:
   - {total_rows:,} total sessions
   - Large dataset requires efficient processing strategies

3. KEY PREDICTORS:
   - listen_type shows strong correlation with target
   - Platform and context features are important
   - Track duration affects skip behavior

4. MODELING RECOMMENDATIONS:
   - Use stratified sampling for train/val splits
   - Apply class weights in models
   - Feature engineering: user history, popularity, temporal features
   - Consider ensemble methods for best performance

NEXT STEPS:
- Feature engineering and preprocessing pipeline
- Baseline models (logistic regression, random forest)
- Advanced models (XGBoost, LightGBM, neural networks)
- Evaluation framework with proper metrics (AUC, precision, recall)
""")

print("\n✓ FULL DATASET EDA COMPLETE!")
print(f"\nGenerated 6 visualization files and 1 summary file")
print(f"Files saved with 'eda_full_' prefix in notebooks/")
