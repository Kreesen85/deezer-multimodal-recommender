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


def preprocess_data(df, add_features=True):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw DataFrame
    add_features : bool
        Whether to add engineered features (default: True)
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed DataFrame with engineered features
    """
    df = df.copy()
    
    if add_features:
        print("Adding temporal features...")
        df = add_temporal_features(df)
        
        print("Adding release features...")
        df = add_release_features(df)
        
        print("Adding duration features...")
        df = add_duration_features(df)
        
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
    
    print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("Deezer Data Preprocessing Utilities")
    print("=" * 70)
    
    print("\nExample usage:")
    print("""
    # Load data
    df = pd.read_csv('data/raw/train.csv')
    
    # Preprocess with feature engineering
    df_processed = preprocess_data(df, add_features=True)
    
    # Get feature lists
    features = get_feature_lists()
    
    # Use specific feature sets for modeling
    model_features = (features['original_features'] + 
                     features['temporal_features'] + 
                     features['release_features'])
    
    X = df_processed[model_features]
    y = df_processed['is_listened']
    """)
    
    print("\nFeature Categories:")
    features = get_feature_lists()
    for category, feat_list in features.items():
        print(f"\n{category}:")
        print(f"  Count: {len(feat_list)}")
        print(f"  Features: {', '.join(feat_list[:5])}" + 
              (f", ... (+{len(feat_list)-5} more)" if len(feat_list) > 5 else ""))
