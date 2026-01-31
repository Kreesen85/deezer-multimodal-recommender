# Data Directory

⚠️ **Data files are not included in this repository** due to licensing restrictions.

## Dataset: Deezer Music Streaming Sessions (DSG17)

### Source
- Dataset: Deezer Music Streaming Sessions Dataset
- Challenge: Deezer Sequential Skip Prediction Challenge
- Access: Available through the RecSys Challenge or Deezer's official channels

### Directory Structure

```
data/
├── README.md (this file)
├── raw/
│   ├── train.csv (7.5M rows, 507 MB)
│   └── test.csv (19.9K rows, 1.4 MB)
└── processed/
    └── (processed files will be saved here)
```

### Dataset Description

#### train.csv (Training Data)
Contains user listening sessions with the following features:
- `user_id`: Unique user identifier
- `user_age`: User age
- `user_gender`: User gender (0/1)
- `media_id`: Track identifier
- `album_id`: Album identifier
- `artist_id`: Artist identifier
- `genre_id`: Genre identifier
- `ts_listen`: Timestamp of listening event
- `context_type`: Listening context
- `release_date`: Release date of track
- `platform_name`: Platform used for listening
- `platform_family`: Platform family
- `media_duration`: Duration of track in seconds
- `listen_type`: Type of listening (0/1)
- `is_listened`: Target variable (0 = skipped, 1 = listened)

**Size**: 7,558,835 rows, ~507 MB

#### test.csv (Test Data)
Same features as training data, excluding the target variable `is_listened`.

**Size**: 19,919 rows, ~1.4 MB

### Data Privacy
This data is kept locally and excluded from version control via `.gitignore`.
