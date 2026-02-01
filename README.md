# Deezer Multimodal Recommender Systems

Course project for Recommender Systems - Sequential Skip Prediction Challenge

## Objective

Design and evaluate recommender system approaches for the **Deezer DSG17 dataset**, focusing on:
- **Competition-oriented optimization**: Predict whether users will skip songs
- **Production-oriented recommendation design**: Build scalable, real-world systems

---

## Dataset: Deezer Music Streaming Sessions

- **Training**: 7.5M listening sessions from ~19K users
- **Test**: 19.9K sessions
- **Target**: Binary classification - `is_listened` (0=skipped, 1=listened)
- **Features**: User demographics, track metadata, temporal context, platform info

### Key Statistics:
- Users aged 15-25 years
- 45% listen rate (55% skip rate)
- 19.3% of users rarely skip (< 10% skip rate)
- Age and user engagement are strong predictors

📊 See `notebooks/USER_SKIP_BEHAVIOR_ANALYSIS.md` for detailed insights

---

## Project Status

### ✅ Completed

- [x] **Data Quality Assessment**: Comprehensive checks, 99.55% data consistency
- [x] **Exploratory Data Analysis**: Full dataset analysis (7.5M rows)
- [x] **User Behavior Analysis**: User segmentation by skip patterns
- [x] **Feature Engineering**: 31 engineered features implemented
  - Temporal features (hour, day, weekend, time-of-day categories)
  - Release features (track age, pre-release detection, release decade)
  - Duration features (minutes, categories, extended tracks)
  - **User engagement features (9 features)**: skip rate, session count, diversity metrics
- [x] **Preprocessing Pipeline**: Complete with train/test workflow and cold start handling
- [x] **Documentation**: Comprehensive reports and implementation guides

### 🔄 In Progress

- [ ] Baseline model development
- [ ] Feature importance analysis
- [ ] Model evaluation and optimization

### 📋 Planned

- [ ] Advanced feature engineering (interaction features)
- [ ] Ensemble methods
- [ ] Production deployment strategy
- [ ] Final report

---

## Repository Structure

```
deezer-multimodal-recommender/
├── src/
│   ├── data/
│   │   └── preprocessing.py        # Feature engineering pipeline
│   ├── evaluation/
│   │   └── metrics.py              # Model evaluation utilities
│   └── utils/
│       └── config.py               # Configuration management
├── notebooks/
│   ├── eda_full_optimized.py       # Full dataset EDA
│   ├── data_quality_check.py       # Data quality validation
│   ├── user_skip_behavior_analysis.py  # User segmentation
│   ├── demo_preprocessing_with_users.py  # Complete preprocessing demo
│   ├── USER_SKIP_BEHAVIOR_ANALYSIS.md  # User analysis report
│   ├── USER_FEATURES_IMPLEMENTATION.md  # Feature engineering guide
│   └── README.md                   # Notebooks documentation
├── data/
│   ├── README.md                   # Dataset documentation
│   ├── DATA_QUALITY_REPORT.txt     # Quality assessment
│   ├── DATA_QUALITY_SUMMARY.md     # Quality summary
│   ├── TEMPORAL_CONSISTENCY_ANALYSIS.md  # Temporal analysis
│   └── raw/                        # Data files (not in repo)
│       ├── train.csv
│       └── test.csv
├── report/
│   ├── main.tex                    # LaTeX report (CEUR-WS format)
│   └── references.bib              # Bibliography
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Quick Start

### 1. Setup Environment

**Recommended: Use Anaconda (Python 3.13)**

```bash
# Clone repository
git clone <repository-url>
cd deezer-multimodal-recommender

# Use Anaconda base environment (all packages already installed)
conda activate base

# OR install dependencies if needed
# pip install -r requirements.txt

# Test environment
python test_environment.py
```

**Note**: See `PYTHON_ENVIRONMENT_SETUP.md` for detailed setup instructions and troubleshooting.

### 2. Get the Data

Download the Deezer DSG17 dataset and place files in `data/raw/`:
- `train.csv` (7.5M rows, 507 MB)
- `test.csv` (19.9K rows, 1.4 MB)

### 3. Run Exploratory Analysis

```bash
cd notebooks

# Data quality check
python data_quality_check.py

# Full dataset EDA
python eda_full_optimized.py

# User behavior analysis
python user_skip_behavior_analysis.py
```

### 4. Feature Engineering Demo

```bash
cd notebooks
python demo_preprocessing_with_users.py
```

This demonstrates the complete preprocessing workflow including:
- Temporal, release, and duration features
- User engagement features (proper train/test split)
- Cold start handling for new users

---

## Feature Engineering

### 31 Engineered Features

#### Temporal Features (9)
- `hour`, `day_of_week`, `day_of_month`, `month`
- `is_weekend`, `is_late_night`, `is_evening`, `is_commute_time`
- `time_of_day` (categories: morning/afternoon/evening/night)

#### Release Features (7)
- `release_year`, `release_month`, `release_decade`
- `days_since_release` (can be negative for pre-release)
- `is_pre_release_listen`, `is_new_release`
- `track_age_category` (pre-release/new/recent/catalog/deep-catalog)

#### Duration Features (3)
- `duration_minutes`, `duration_category`, `is_extended_track`

#### **User Engagement Features (9)** 🆕
- `user_listen_rate`, `user_skip_rate` - Historical behavior
- `user_session_count`, `user_total_listens` - Activity metrics
- `user_genre_diversity`, `user_artist_diversity`, `user_context_variety` - Exploration patterns
- `user_engagement_segment` - Categorical (Never/Rarely/Occasional/Moderate/Frequent skipper)
- `user_engagement_score` - Composite engagement metric (0-1)

### Usage Example

```python
from src.data.preprocessing import (
    compute_user_features_from_train,
    apply_user_features,
    add_temporal_features,
    add_release_features,
    add_duration_features
)

# Load training data
train_df = pd.read_csv('data/raw/train.csv')

# Add features
train_df = add_temporal_features(train_df)
train_df = add_release_features(train_df)
train_df = add_duration_features(train_df)

# Compute user features (from training only - no leakage!)
user_stats = compute_user_features_from_train(train_df)
train_df = apply_user_features(train_df, user_stats)

# Apply to test data with cold start handling
test_df = pd.read_csv('data/raw/test.csv')
test_df = add_temporal_features(test_df)
test_df = add_release_features(test_df)
test_df = add_duration_features(test_df)
test_df = apply_user_features(test_df, user_stats)  # Uses training stats
```

See `notebooks/USER_FEATURES_IMPLEMENTATION.md` for detailed guide.

---

## Key Findings

### Data Quality
✅ **99.55%** data consistency  
✅ **0% missing values** in critical fields  
⚠️ **0.45%** pre-release listening (kept as feature)  
✅ Minimal cleaning required - production-ready data

### User Behavior Insights

| User Segment | % of Users | Avg Age | Skip Rate | Sessions |
|-------------|-----------|---------|-----------|----------|
| **Never Skips** | 6.8% | 25.0 | 0% | 110 |
| **Rarely Skips** | 12.5% | - | <10% | - |
| **Occasional** | 23.1% | - | 10-25% | - |
| **Moderate** | 30.7% | - | 25-50% | - |
| **Frequent Skippers** | 26.9% | 23.3 | >50% | 91 |

**Statistical Significance:**
- Age difference: p < 0.000001 ✅
- Session count: p < 0.000001 ✅
- Genre diversity: Not significant

### Expected Model Performance

With user engagement features:
- **AUC**: 0.75-0.78 (baseline ~0.70)
- **Accuracy**: 0.72-0.75 (baseline ~0.65)
- Expected **+5-8 point improvement** from user features

---

## Documentation

- **`data/README.md`** - Dataset structure and column descriptions
- **`data/DATA_QUALITY_REPORT.txt`** - Comprehensive data quality assessment
- **`notebooks/README.md`** - Analysis scripts and outputs
- **`notebooks/USER_SKIP_BEHAVIOR_ANALYSIS.md`** - User segmentation insights
- **`notebooks/USER_FEATURES_IMPLEMENTATION.md`** - Feature engineering guide

---

## Technologies

- **Python 3.13.9** (Anaconda)
- **pandas 3.0.0** - Data manipulation
- **numpy 2.4.1** - Numerical computing
- **scikit-learn 1.8.0** - Machine learning & collaborative filtering
- **matplotlib 3.10.8, seaborn 0.13.2** - Visualization
- **scipy 1.17.0** - Statistical analysis

**Note**: scikit-surprise is incompatible with Python 3.13. Use scikit-learn for collaborative filtering instead.

---

## Contributing

This is a course project. Contributions follow academic collaboration guidelines.

See `CONTRIBUTIONS.md` for details.

---

## License

Academic project for Recommender Systems course.

---

## Contact

For questions or collaboration inquiries, please open an issue or contact the project maintainer.

---

## Acknowledgments

- **Deezer** for the DSG17 dataset
- **RecSys course** instructors and teaching assistants
- **CEUR-WS** for LaTeX report template

---

**Last Updated**: February 1, 2026  
**Status**: Feature engineering complete, ready for modeling 🚀