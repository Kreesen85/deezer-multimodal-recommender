"""
Data Preprocessing Utilities for Deezer Skip Prediction
Includes feature engineering functions with temporal features
"""

import pandas as pd
import numpy as np
from datetime import datetime

def add_temporal_features(df):
    """
    Extract temporal features from timestamps
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'ts_listen' column (Unix timestamp)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added temporal features
    """
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['ts_listen'], unit='s')
    
    # Extract time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    
    # Create categorical time features
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Fri-Sat
    df['is_late_night'] = ((df['hour'] >= 1) & (df['hour'] <= 5)).astype(int)  # 1-5 AM peak
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)  # 6PM-11PM
    df['is_commute_time'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)  # 7-9 AM
    
    # Time of day categories
    def categorize_time_of_day(hour):
        if 6 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 18:
            return 1  # Afternoon
        elif 18 <= hour < 23:
            return 2  # Evening
        else:
            return 3  # Night
    
    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
    
    return df


def add_release_features(df):
    """
    Extract features from release date and calculate track age
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'release_date' (YYYYMMDD format) and 'ts_listen'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added release-related features
    """
    # Parse release date
    df['release_date_parsed'] = pd.to_datetime(df['release_date'], format='%Y%m%d')
    df['listen_date'] = pd.to_datetime(df['ts_listen'], unit='s')
    
    # Extract release date components
    df['release_year'] = df['release_date_parsed'].dt.year
    df['release_month'] = df['release_date_parsed'].dt.month
    df['release_decade'] = (df['release_year'] // 10) * 10
    
    # Calculate days since release (can be negative for pre-release)
    df['days_since_release'] = (df['listen_date'] - df['release_date_parsed']).dt.days
    
    # Pre-release listening flag
    df['is_pre_release_listen'] = (df['days_since_release'] < 0).astype(int)
    
    # New release flag (within 30 days)
    df['is_new_release'] = ((df['days_since_release'] >= 0) & 
                            (df['days_since_release'] <= 30)).astype(int)
    
    # Track age categories
    def categorize_track_age(days):
        if days < 0:
            return 0  # Pre-release
        elif days <= 30:
            return 1  # New (0-30 days)
        elif days <= 365:
            return 2  # Recent (1 month - 1 year)
        elif days <= 1825:
            return 3  # Catalog (1-5 years)
        else:
            return 4  # Deep catalog (5+ years)
    
    df['track_age_category'] = df['days_since_release'].apply(categorize_track_age)
    
    return df


def add_duration_features(df):
    """
    Create features from media duration
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'media_duration' column (seconds)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added duration features
    """
    # Duration in minutes
    df['duration_minutes'] = df['media_duration'] / 60.0
    
    # Duration categories
    def categorize_duration(seconds):
        if seconds < 120:
            return 0  # Very short (< 2 min)
        elif seconds < 180:
            return 1  # Short (2-3 min)
        elif seconds < 240:
            return 2  # Medium (3-4 min)
        elif seconds < 300:
            return 3  # Long (4-5 min)
        else:
            return 4  # Very long (5+ min)
    
    df['duration_category'] = df['media_duration'].apply(categorize_duration)
    
    # Is extended track (> 5 minutes)
    df['is_extended_track'] = (df['media_duration'] > 300).astype(int)
    
    return df


def add_user_features(df):
    """
    Create user-level engagement features based on historical behavior
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with user behavior data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added user-level features
    
    Note:
    -----
    This function computes user aggregations from the current dataset.
    For production, these should be computed from historical data up to 
    the prediction point to avoid data leakage.
    """
    print("  Calculating user statistics...")
    
    # Calculate user-level aggregations
    user_stats = df.groupby('user_id').agg({
        'is_listened': ['mean', 'sum', 'count'],  # Listen rate, total listens, sessions
        'genre_id': 'nunique',  # Genre diversity
        'artist_id': 'nunique',  # Artist diversity
        'context_type': 'nunique',  # Context variety
    }).reset_index()
    
    # Flatten column names
    user_stats.columns = ['user_id', 'user_listen_rate', 'user_total_listens', 
                          'user_session_count', 'user_genre_diversity', 
                          'user_artist_diversity', 'user_context_variety']
    
    # Calculate skip rate
    user_stats['user_skip_rate'] = 1 - user_stats['user_listen_rate']
    
    # User engagement segment based on skip rate
    def categorize_user_segment(skip_rate):
        if skip_rate == 0:
            return 0  # Never Skips
        elif skip_rate < 0.1:
            return 1  # Rarely Skips
        elif skip_rate < 0.25:
            return 2  # Occasional Skipper
        elif skip_rate < 0.5:
            return 3  # Moderate Skipper
        else:
            return 4  # Frequent Skipper
    
    user_stats['user_engagement_segment'] = user_stats['user_skip_rate'].apply(categorize_user_segment)
    
    # Calculate composite engagement score (0-1 scale)
    max_sessions = user_stats['user_session_count'].max()
    max_genres = user_stats['user_genre_diversity'].max()
    
    user_stats['user_engagement_score'] = (
        user_stats['user_listen_rate'] * 0.5 +  # 50% weight on listen rate
        (user_stats['user_session_count'] / max_sessions) * 0.3 +  # 30% on activity
        (user_stats['user_genre_diversity'] / max_genres) * 0.2  # 20% on diversity
    )
    
    print(f"  Computed features for {len(user_stats):,} unique users")
    
    # Merge back to original dataframe
    df = df.merge(user_stats, on='user_id', how='left')
    
    return df


def preprocess_data(df, add_features=True, add_user_features_flag=True):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame
    add_features : bool
        Whether to add engineered features (default: True)
    add_user_features_flag : bool
        Whether to add user-level engagement features (default: True)
        Note: Set to False for test data if computing from training data separately
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame with engineered features
    
    Warning:
    --------
    User features computed from the same dataset can cause data leakage in training.
    For proper ML workflow:
    1. Compute user stats from training data only
    2. Apply those stats to validation/test data
    3. Handle new users (cold start) with defaults
    """
    df = df.copy()
    
    if add_features:
        print("Adding temporal features...")
        df = add_temporal_features(df)
        
        print("Adding release features...")
        df = add_release_features(df)
        
        print("Adding duration features...")
        df = add_duration_features(df)
        
        if add_user_features_flag:
            print("Adding user engagement features...")
            df = add_user_features(df)
        
        print("✓ Feature engineering complete!")
    
    return df


def get_feature_lists():
    """
    Get lists of features by category
    
    Returns:
    --------
    dict
        Dictionary with feature lists
    """
    return {
        'original_features': [
            'genre_id', 'media_id', 'album_id', 'context_type', 
            'platform_name', 'platform_family', 'media_duration', 
            'listen_type', 'user_gender', 'user_id', 'artist_id', 'user_age'
        ],
        
        'temporal_features': [
            'hour', 'day_of_week', 'day_of_month', 'month', 
            'is_weekend', 'is_late_night', 'is_evening', 'is_commute_time',
            'time_of_day'
        ],
        
        'release_features': [
            'release_year', 'release_month', 'release_decade',
            'days_since_release', 'is_pre_release_listen', 'is_new_release',
            'track_age_category'
        ],
        
        'duration_features': [
            'duration_minutes', 'duration_category', 'is_extended_track'
        ],
        
        'user_features': [
            'user_listen_rate', 'user_skip_rate', 'user_session_count',
            'user_total_listens', 'user_genre_diversity', 'user_artist_diversity',
            'user_context_variety', 'user_engagement_segment', 'user_engagement_score'
        ],
        
        'high_cardinality_ids': [
            'media_id', 'artist_id', 'album_id', 'user_id', 'genre_id'
        ],
        
        'low_cardinality_categorical': [
            'platform_name', 'platform_family', 'listen_type', 
            'user_gender', 'context_type'
        ]
    }


def print_preprocessing_summary(df_before, df_after):
    """
    Print summary of preprocessing changes
    
    Parameters:
    -----------
    df_before : pandas.DataFrame
        DataFrame before preprocessing
    df_after : pandas.DataFrame
        DataFrame after preprocessing
    """
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    
    print(f"\nOriginal features: {len(df_before.columns)}")
    print(f"Features after engineering: {len(df_after.columns)}")
    print(f"New features added: {len(df_after.columns) - len(df_before.columns)}")
    
    new_features = set(df_after.columns) - set(df_before.columns)
    print(f"\nNew features:")
    for feat in sorted(new_features):
        print(f"  - {feat}")
    
    # Check for pre-release listening
    if 'is_pre_release_listen' in df_after.columns:
        n_pre_release = df_after['is_pre_release_listen'].sum()
        pct_pre_release = (n_pre_release / len(df_after)) * 100
        print(f"\nPre-release listening detected: {n_pre_release:,} ({pct_pre_release:.2f}%)")
    
    # Check for user features
    if 'user_engagement_segment' in df_after.columns:
        segment_counts = df_after['user_engagement_segment'].value_counts().sort_index()
        segment_names = ['Never Skips', 'Rarely Skips', 'Occasional', 'Moderate', 'Frequent']
        print(f"\nUser Engagement Distribution:")
        for segment_id, count in segment_counts.items():
            pct = (count / len(df_after)) * 100
            segment_name = segment_names[segment_id] if segment_id < len(segment_names) else f"Unknown ({segment_id})"
            print(f"  {segment_name}: {count:,} sessions ({pct:.1f}%)")
    
    print("=" * 70)


def compute_user_features_from_train(train_df):
    """
    Compute user features from training data to apply to test data
    This prevents data leakage by ensuring user stats come only from training
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training DataFrame with user behavior
    
    Returns:
    --------
    pandas.DataFrame
        User statistics DataFrame with user_id as key
    """
    print("Computing user features from training data...")
    
    user_stats = train_df.groupby('user_id').agg({
        'is_listened': ['mean', 'sum', 'count'],
        'genre_id': 'nunique',
        'artist_id': 'nunique',
        'context_type': 'nunique',
    }).reset_index()
    
    user_stats.columns = ['user_id', 'user_listen_rate', 'user_total_listens', 
                          'user_session_count', 'user_genre_diversity', 
                          'user_artist_diversity', 'user_context_variety']
    
    user_stats['user_skip_rate'] = 1 - user_stats['user_listen_rate']
    
    # Engagement segment
    def categorize_user_segment(skip_rate):
        if skip_rate == 0:
            return 0
        elif skip_rate < 0.1:
            return 1
        elif skip_rate < 0.25:
            return 2
        elif skip_rate < 0.5:
            return 3
        else:
            return 4
    
    user_stats['user_engagement_segment'] = user_stats['user_skip_rate'].apply(categorize_user_segment)
    
    # Engagement score
    max_sessions = user_stats['user_session_count'].max()
    max_genres = user_stats['user_genre_diversity'].max()
    
    user_stats['user_engagement_score'] = (
        user_stats['user_listen_rate'] * 0.5 +
        (user_stats['user_session_count'] / max_sessions) * 0.3 +
        (user_stats['user_genre_diversity'] / max_genres) * 0.2
    )
    
    print(f"✓ Computed features for {len(user_stats):,} unique users")
    
    return user_stats


def apply_user_features(df, user_stats, default_values=None):
    """
    Apply pre-computed user features to a DataFrame
    Handles new users (cold start) with default values
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to add user features to
    user_stats : pandas.DataFrame
        Pre-computed user statistics from training data
    default_values : dict, optional
        Default values for new users. If None, uses global averages from user_stats
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with user features added
    """
    print("Applying user features...")
    
    # Merge user stats
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Handle new users (cold start)
    if default_values is None:
        # Use training data averages as defaults
        default_values = {
            'user_listen_rate': user_stats['user_listen_rate'].mean(),
            'user_skip_rate': user_stats['user_skip_rate'].mean(),
            'user_session_count': user_stats['user_session_count'].mean(),
            'user_total_listens': user_stats['user_total_listens'].mean(),
            'user_genre_diversity': user_stats['user_genre_diversity'].mean(),
            'user_artist_diversity': user_stats['user_artist_diversity'].mean(),
            'user_context_variety': user_stats['user_context_variety'].mean(),
            'user_engagement_segment': 3,  # Default to "Moderate"
            'user_engagement_score': user_stats['user_engagement_score'].mean()
        }
    
    # Fill missing values for new users
    n_new_users = df['user_listen_rate'].isna().sum()
    if n_new_users > 0:
        print(f"  Found {n_new_users:,} sessions from new users - applying defaults")
        for col, default_val in default_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
    
    print("✓ User features applied")
    
    return df


# Example usage
if __name__ == "__main__":
    print("Deezer Data Preprocessing Utilities")
    print("=" * 70)
    
    print("\n=== BASIC USAGE ===")
    print("""
    # Load data
    df = pd.read_csv('data/raw/train.csv')
    
    # Preprocess with feature engineering (including user features)
    df_processed = preprocess_data(df, add_features=True)
    
    # Get feature lists
    features = get_feature_lists()
    
    # Use specific feature sets for modeling
    model_features = (features['temporal_features'] + 
                     features['release_features'] +
                     features['user_features'])
    
    X = df_processed[model_features]
    y = df_processed['is_listened']
    """)
    
    print("\n=== PROPER TRAIN/TEST WORKFLOW (Prevents Data Leakage) ===")
    print("""
    # 1. Load training data
    train_df = pd.read_csv('data/raw/train.csv')
    
    # 2. Add temporal, release, duration features (no leakage risk)
    train_df = add_temporal_features(train_df)
    train_df = add_release_features(train_df)
    train_df = add_duration_features(train_df)
    
    # 3. Compute user features FROM TRAINING DATA ONLY
    user_stats = compute_user_features_from_train(train_df)
    
    # 4. Apply user features to training data
    train_df = apply_user_features(train_df, user_stats)
    
    # 5. Load test data
    test_df = pd.read_csv('data/raw/test.csv')
    
    # 6. Add temporal, release, duration features to test
    test_df = add_temporal_features(test_df)
    test_df = add_release_features(test_df)
    test_df = add_duration_features(test_df)
    
    # 7. Apply SAME user stats from training to test (handles new users)
    test_df = apply_user_features(test_df, user_stats)
    
    # Now train_df and test_df have consistent features without leakage
    """)
    
    print("\n=== FEATURE CATEGORIES ===")
    features = get_feature_lists()
    for category, feat_list in features.items():
        print(f"\n{category}:")
        print(f"  Count: {len(feat_list)}")
        print(f"  Features: {', '.join(feat_list[:5])}" + 
              (f", ... (+{len(feat_list)-5} more)" if len(feat_list) > 5 else ""))
