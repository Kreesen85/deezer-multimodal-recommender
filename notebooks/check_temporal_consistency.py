"""
Check for temporal inconsistencies: listening before release date
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("TEMPORAL CONSISTENCY CHECK")
print("Listening Timestamp vs. Release Date Validation")
print("=" * 80)

# Load a large sample to get representative results
SAMPLE_SIZE = 1000000
print(f"\nLoading {SAMPLE_SIZE:,} records for analysis...")
df = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(df):,} rows\n")

# Convert timestamps to dates
print("Converting timestamps...")
df['listen_date'] = pd.to_datetime(df['ts_listen'], unit='s')
df['release_date_parsed'] = pd.to_datetime(df['release_date'], format='%Y%m%d')

print("✓ Timestamps converted\n")

# Calculate difference
print("Analyzing temporal consistency...")
df['days_after_release'] = (df['listen_date'] - df['release_date_parsed']).dt.days

# Find inconsistencies
inconsistent = df[df['days_after_release'] < 0]
n_inconsistent = len(inconsistent)
pct_inconsistent = (n_inconsistent / len(df)) * 100

print("=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nTotal records analyzed: {len(df):,}")
print(f"Records with listen_date >= release_date: {len(df) - n_inconsistent:,} ({100 - pct_inconsistent:.2f}%)")
print(f"Records with listen_date < release_date: {n_inconsistent:,} ({pct_inconsistent:.4f}%)")

if n_inconsistent > 0:
    print(f"\n⚠ TEMPORAL INCONSISTENCY DETECTED!")
    print(f"\nFound {n_inconsistent:,} instances where users listened before release")
    
    # Analyze the inconsistencies
    print("\n" + "-" * 80)
    print("INCONSISTENCY ANALYSIS")
    print("-" * 80)
    
    print(f"\nDays before release statistics:")
    print(inconsistent['days_after_release'].describe())
    
    print(f"\nMost extreme cases (listened furthest before release):")
    print(inconsistent.nlargest(10, 'days_after_release', keep='first')[
        ['listen_date', 'release_date_parsed', 'days_after_release', 'media_id', 'artist_id']
    ])
    
    # Check if this is a small data quality issue or systematic
    print(f"\n\nDistribution of temporal gaps:")
    bins = [-float('inf'), -365, -180, -90, -30, -7, -1]
    labels = ['>1 year early', '6mo-1yr early', '3-6mo early', '1-3mo early', '1wk-1mo early', '1-7 days early']
    inconsistent['gap_category'] = pd.cut(inconsistent['days_after_release'], bins=bins, labels=labels)
    print(inconsistent['gap_category'].value_counts().sort_index())
    
else:
    print(f"\n✓ NO TEMPORAL INCONSISTENCIES FOUND")
    print("All listening events occurred on or after the track release date")

# Also check for tracks listened to long after release
print("\n" + "=" * 80)
print("ADDITIONAL TEMPORAL ANALYSIS")
print("=" * 80)

print("\nTracks listened to AFTER release:")
after_release = df[df['days_after_release'] >= 0]
print(f"\nDays after release statistics:")
print(after_release['days_after_release'].describe())

print(f"\nDistribution:")
print(f"  Same day as release: {(after_release['days_after_release'] == 0).sum():,}")
print(f"  Within 1 week: {(after_release['days_after_release'] <= 7).sum():,}")
print(f"  Within 1 month: {(after_release['days_after_release'] <= 30).sum():,}")
print(f"  Within 1 year: {(after_release['days_after_release'] <= 365).sum():,}")
print(f"  Over 1 year: {(after_release['days_after_release'] > 365).sum():,}")
print(f"  Over 5 years: {(after_release['days_after_release'] > 1825).sum():,}")
print(f"  Over 10 years: {(after_release['days_after_release'] > 3650).sum():,}")

# Show oldest tracks being listened to
print(f"\nOldest tracks in listening sessions:")
oldest = after_release.nlargest(5, 'days_after_release')[
    ['release_date_parsed', 'listen_date', 'days_after_release', 'media_id']
]
print(oldest.to_string(index=False))

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if n_inconsistent > 0:
    print(f"""
⚠ TEMPORAL INCONSISTENCIES DETECTED

Found {n_inconsistent:,} records ({pct_inconsistent:.4f}%) where listening occurred 
before the official release date.

POSSIBLE EXPLANATIONS:
1. Pre-release listening (promotional, early access, leaks)
2. Data entry errors in release_date field
3. Timezone mismatches between listening and release timestamps
4. Beta testing or preview releases

RECOMMENDATION:
This is a small percentage but should be addressed:
- Option 1: Flag these records for investigation
- Option 2: Set release_date to listen_date for these cases
- Option 3: Remove these records if considered data errors
- Option 4: Keep as-is if pre-release listening is valid

For modeling, this could be handled as a feature:
- 'is_pre_release_listen' (boolean)
- 'days_before_release' (negative values)
""")
else:
    print(f"""
✓ NO TEMPORAL INCONSISTENCIES

All listening events occurred on or after the track release date.
This confirms proper temporal consistency in the dataset.

The data shows a realistic distribution of catalog listening:
- New releases being listened to immediately
- Catalog tracks (old releases) being discovered years later
- This is expected behavior for a music streaming platform
""")

print("=" * 80)

# Save sample of inconsistent records if found
if n_inconsistent > 0:
    sample_file = 'temporal_inconsistencies_sample.csv'
    inconsistent.head(1000).to_csv(sample_file, index=False)
    print(f"\n✓ Saved sample of inconsistent records to: {sample_file}")
