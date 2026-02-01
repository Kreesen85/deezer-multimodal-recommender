# 02_preprocessing/ - Data Preprocessing

Feature engineering and data preparation scripts.

## 📋 Scripts

### `demo_preprocessing_with_users.py` ⭐ **RECOMMENDED**
Complete preprocessing pipeline with all features
- Temporal features (9)
- Release features (7)
- Duration features (3)
- **User engagement features (9)**
- Proper train/test split (no data leakage!)
- Cold start handling

**Run**: `python demo_preprocessing_with_users.py`

### `demo_preprocessing.py`
Basic preprocessing (no user features)
- Temporal, release, duration features only
- Quick demo of feature engineering

**Run**: `python demo_preprocessing.py`

---

## 📁 Generated Files

### Sample Outputs
- `train_preprocessed_sample.csv` - Training data with all features
- `test_preprocessed_sample.csv` - Test data with all features  
- `user_stats_from_train.csv` - User statistics (for applying to test)

---

## 🎯 Features Generated (31 total)

### Temporal Features (9)
`hour`, `day_of_week`, `is_weekend`, `is_late_night`, `is_evening`, `is_commute_time`, `time_of_day`, etc.

### Release Features (7)
`release_year`, `days_since_release`, `is_pre_release_listen`, `is_new_release`, `track_age_category`, etc.

### Duration Features (3)
`duration_minutes`, `duration_category`, `is_extended_track`

### User Engagement Features (9) 🆕
`user_listen_rate`, `user_skip_rate`, `user_session_count`, `user_genre_diversity`, `user_engagement_score`, etc.

---

## 🚀 Usage

```python
from src.data.preprocessing import (
    add_temporal_features,
    add_release_features,
    add_duration_features,
    compute_user_features_from_train,
    apply_user_features
)

# Load data
train_df = pd.read_csv('../data/raw/train.csv')

# Add features
train_df = add_temporal_features(train_df)
train_df = add_release_features(train_df)
train_df = add_duration_features(train_df)

# User features (prevent data leakage!)
user_stats = compute_user_features_from_train(train_df)
train_df = apply_user_features(train_df, user_stats)
```

---

## 📚 Documentation

See `../docs/USER_FEATURES_IMPLEMENTATION.md` for:
- Detailed feature descriptions
- Train/test workflow
- Cold start strategy
- Implementation guide

---

## ⚡ Performance

- 100K rows: ~1-2 seconds
- 7.5M rows: ~1-2 minutes
- Memory efficient (chunked processing available)

---

*All preprocessing maintains proper train/test separation to prevent data leakage*
