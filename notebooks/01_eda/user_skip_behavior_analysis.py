"""
User Segmentation Analysis: Skip Behavior
Focus on users who don't skip songs vs. frequent skippers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("USER SEGMENTATION ANALYSIS: SKIP BEHAVIOR")
print("=" * 80)

# Load data
SAMPLE_SIZE = 2000000  # 2M for reliable user statistics
print(f"\nLoading {SAMPLE_SIZE:,} records...")
df = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(df):,} rows")

# ============================================================================
# 1. CALCULATE USER-LEVEL STATISTICS
# ============================================================================
print("\n[1/5] Calculating user-level statistics...")

user_stats = df.groupby('user_id').agg({
    'is_listened': ['mean', 'sum', 'count'],
    'user_age': 'first',
    'user_gender': 'first',
    'media_duration': 'mean',
    'listen_type': 'mean',
    'platform_name': lambda x: x.mode()[0] if len(x) > 0 else x.iloc[0],
    'genre_id': 'nunique',
    'artist_id': 'nunique',
    'context_type': 'nunique'
}).reset_index()

# Flatten column names
user_stats.columns = ['user_id', 'listen_rate', 'total_listens', 'total_sessions',
                      'age', 'gender', 'avg_duration', 'avg_listen_type',
                      'primary_platform', 'unique_genres', 'unique_artists', 
                      'unique_contexts']

user_stats['skip_rate'] = 1 - user_stats['listen_rate']
user_stats['skips'] = user_stats['total_sessions'] - user_stats['total_listens']

print(f"✓ Analyzed {len(user_stats):,} unique users")

# ============================================================================
# 2. SEGMENT USERS BY SKIP BEHAVIOR
# ============================================================================
print("\n[2/5] Segmenting users by skip behavior...")

# Define segments based on skip rate
def segment_user(skip_rate):
    if skip_rate == 0:
        return 'Never Skips'
    elif skip_rate < 0.1:
        return 'Rarely Skips (<10%)'
    elif skip_rate < 0.25:
        return 'Occasional Skipper (10-25%)'
    elif skip_rate < 0.5:
        return 'Moderate Skipper (25-50%)'
    else:
        return 'Frequent Skipper (>50%)'

user_stats['segment'] = user_stats['skip_rate'].apply(segment_user)

# Count users in each segment
segment_counts = user_stats['segment'].value_counts()
print("\n--- User Segments ---")
for segment, count in segment_counts.items():
    pct = (count / len(user_stats)) * 100
    print(f"{segment:30s}: {count:6,} users ({pct:5.2f}%)")

# ============================================================================
# 3. PROFILE NON-SKIPPERS
# ============================================================================
print("\n[3/5] Profiling users who don't skip...")

# Focus on users who never or rarely skip
engaged_users = user_stats[user_stats['skip_rate'] < 0.1]
frequent_skippers = user_stats[user_stats['skip_rate'] > 0.5]

print(f"\nEngaged Users (skip rate < 10%): {len(engaged_users):,}")
print(f"Frequent Skippers (skip rate > 50%): {len(frequent_skippers):,}")

print("\n--- Engaged Users Profile ---")
print(f"Average age: {engaged_users['age'].mean():.1f} years")
print(f"Gender distribution: {dict(engaged_users['gender'].value_counts())}")
print(f"Average sessions: {engaged_users['total_sessions'].mean():.1f}")
print(f"Average track duration: {engaged_users['avg_duration'].mean():.1f} seconds")
print(f"Average unique genres: {engaged_users['unique_genres'].mean():.1f}")
print(f"Average unique artists: {engaged_users['unique_artists'].mean():.1f}")

print("\n--- Frequent Skippers Profile ---")
print(f"Average age: {frequent_skippers['age'].mean():.1f} years")
print(f"Gender distribution: {dict(frequent_skippers['gender'].value_counts())}")
print(f"Average sessions: {frequent_skippers['total_sessions'].mean():.1f}")
print(f"Average track duration: {frequent_skippers['avg_duration'].mean():.1f} seconds")
print(f"Average unique genres: {frequent_skippers['unique_genres'].mean():.1f}")
print(f"Average unique artists: {frequent_skippers['unique_artists'].mean():.1f}")

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("\n[4/5] Creating visualizations...")

# Figure 1: User segment distribution and skip rate histogram
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Segment distribution
segment_order = ['Never Skips', 'Rarely Skips (<10%)', 'Occasional Skipper (10-25%)', 
                 'Moderate Skipper (25-50%)', 'Frequent Skipper (>50%)']
segment_counts_ordered = user_stats['segment'].value_counts().reindex(segment_order)
colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c']
axes[0, 0].bar(range(len(segment_counts_ordered)), segment_counts_ordered.values, color=colors)
axes[0, 0].set_xticks(range(len(segment_order)))
axes[0, 0].set_xticklabels(segment_order, rotation=45, ha='right')
axes[0, 0].set_ylabel('Number of Users')
axes[0, 0].set_title('User Distribution by Skip Behavior')
axes[0, 0].ticklabel_format(style='plain', axis='y')

# Skip rate distribution
axes[0, 1].hist(user_stats['skip_rate'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].axvline(0.1, color='green', linestyle='--', label='10% threshold')
axes[0, 1].axvline(0.5, color='red', linestyle='--', label='50% threshold')
axes[0, 1].set_xlabel('Skip Rate')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_title('Distribution of User Skip Rates')
axes[0, 1].legend()

# Age comparison
engaged_users_age = engaged_users['age'].value_counts().sort_index()
skippers_age = frequent_skippers['age'].value_counts().sort_index()
x = np.arange(len(engaged_users_age))
width = 0.35
axes[1, 0].bar(x - width/2, engaged_users_age.values, width, label='Engaged (<10% skip)', color='#2ecc71', alpha=0.8)
axes[1, 0].bar(x + width/2, skippers_age.values, width, label='Frequent Skippers (>50%)', color='#e74c3c', alpha=0.8)
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Number of Users')
axes[1, 0].set_title('Age Distribution: Engaged vs. Frequent Skippers')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(engaged_users_age.index)
axes[1, 0].legend()

# Sessions per user
axes[1, 1].boxplot([engaged_users['total_sessions'], frequent_skippers['total_sessions']], 
                    labels=['Engaged\n(<10% skip)', 'Frequent Skippers\n(>50%)'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1, 1].set_ylabel('Number of Sessions')
axes[1, 1].set_title('Session Count: Engaged vs. Frequent Skippers')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('user_skip_behavior_analysis.png', dpi=200, bbox_inches='tight')
print("✓ Saved: user_skip_behavior_analysis.png")
plt.close()

# Figure 2: Detailed comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Diversity metrics
metrics = ['unique_genres', 'unique_artists', 'unique_contexts']
titles = ['Unique Genres', 'Unique Artists', 'Unique Contexts']
for idx, (metric, title) in enumerate(zip(metrics, titles)):
    axes[0, idx].boxplot([engaged_users[metric], frequent_skippers[metric]], 
                         labels=['Engaged', 'Skippers'],
                         patch_artist=True,
                         boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[0, idx].set_ylabel('Count')
    axes[0, idx].set_title(f'{title} Distribution')
    axes[0, idx].grid(axis='y', alpha=0.3)

# Gender breakdown
for idx, (segment_df, title, color) in enumerate([
    (engaged_users, 'Engaged Users (<10% skip)', '#2ecc71'),
    (frequent_skippers, 'Frequent Skippers (>50%)', '#e74c3c')
]):
    gender_counts = segment_df['gender'].value_counts()
    axes[1, idx].bar(['Female', 'Male'], gender_counts.values, color=color, alpha=0.7)
    axes[1, idx].set_ylabel('Number of Users')
    axes[1, idx].set_title(f'Gender: {title}')
    axes[1, idx].ticklabel_format(style='plain', axis='y')
    for i, v in enumerate(gender_counts.values):
        axes[1, idx].text(i, v, f'{v:,}\\n({v/gender_counts.sum()*100:.1f}%)', 
                         ha='center', va='bottom')

# Platform preferences
platform_engaged = engaged_users['primary_platform'].value_counts()
platform_skippers = frequent_skippers['primary_platform'].value_counts()
x = np.arange(len(platform_engaged))
width = 0.35
axes[1, 2].bar(x - width/2, platform_engaged.values, width, label='Engaged', color='#2ecc71', alpha=0.8)
axes[1, 2].bar(x + width/2, platform_skippers.values, width, label='Skippers', color='#e74c3c', alpha=0.8)
axes[1, 2].set_xlabel('Platform')
axes[1, 2].set_ylabel('Number of Users')
axes[1, 2].set_title('Platform Preferences')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(platform_engaged.index)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('user_segment_detailed_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Saved: user_segment_detailed_comparison.png")
plt.close()

# ============================================================================
# 5. STATISTICAL COMPARISON
# ============================================================================
print("\n[5/5] Statistical comparison...")

from scipy import stats

print("\n--- Statistical Tests (Engaged vs. Frequent Skippers) ---")

# Age comparison
t_stat, p_value = stats.ttest_ind(engaged_users['age'], frequent_skippers['age'])
print(f"\nAge: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("  → Significant difference in age between groups")
else:
    print("  → No significant age difference")

# Session count comparison
t_stat, p_value = stats.ttest_ind(engaged_users['total_sessions'], frequent_skippers['total_sessions'])
print(f"\nSession count: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("  → Significant difference in session count")
else:
    print("  → No significant difference in session count")

# Genre diversity
t_stat, p_value = stats.ttest_ind(engaged_users['unique_genres'], frequent_skippers['unique_genres'])
print(f"\nGenre diversity: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("  → Significant difference in genre diversity")
else:
    print("  → No significant difference in genre diversity")

# ============================================================================
# 6. KEY INSIGHTS
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

never_skip_pct = (user_stats['skip_rate'] == 0).sum() / len(user_stats) * 100
rarely_skip_pct = (user_stats['skip_rate'] < 0.1).sum() / len(user_stats) * 100
frequent_skip_pct = (user_stats['skip_rate'] > 0.5).sum() / len(user_stats) * 100

print(f"""
1. USER DISTRIBUTION:
   - {never_skip_pct:.1f}% of users NEVER skip songs
   - {rarely_skip_pct:.1f}% of users skip less than 10% of the time
   - {frequent_skip_pct:.1f}% of users skip more than 50% of the time

2. ENGAGED USERS CHARACTERISTICS:
   - Average age: {engaged_users['age'].mean():.1f} years
   - More diverse listening: {engaged_users['unique_genres'].mean():.1f} genres vs {frequent_skippers['unique_genres'].mean():.1f}
   - Track duration: {engaged_users['avg_duration'].mean():.1f}s vs {frequent_skippers['avg_duration'].mean():.1f}s
   
3. BEHAVIORAL DIFFERENCES:
   - Engaged users explore {engaged_users['unique_artists'].mean():.1f} artists on average
   - Frequent skippers explore {frequent_skippers['unique_artists'].mean():.1f} artists
   - Context variety: {engaged_users['unique_contexts'].mean():.1f} vs {frequent_skippers['unique_contexts'].mean():.1f}

4. IMPLICATIONS FOR MODELING:
   - User-level skip rate is a strong feature
   - User listening diversity correlates with engagement
   - Platform and context preferences differ by segment
   - Consider creating user engagement score feature

RECOMMENDATIONS:
- Create user_skip_rate feature (historical)
- Add user_genre_diversity feature
- Include user_artist_diversity feature  
- Build user engagement segments as categorical feature
- Use these for cold-start problem (new users)
""")

print("=" * 80)

# Save user segments for future use
print("\n✓ Saving user segments...")
user_stats[['user_id', 'segment', 'skip_rate', 'listen_rate', 'total_sessions']].to_csv(
    'user_segments.csv', index=False
)
print("✓ Saved: user_segments.csv")

print("\n✓ Analysis complete!")
