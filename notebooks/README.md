# Notebooks

This directory contains exploratory analysis, experiments, and data quality checks for the Deezer Sequential Skip Prediction project.

---

## Exploratory Data Analysis (EDA)

### `eda_full_optimized.py`
**Comprehensive full-dataset EDA** - Analyzes entire 7.5M training dataset efficiently.

**Features:**
- Chunked reading for exact statistics on full dataset
- 1M-row sample for visualizations
- Target distribution, user demographics, temporal patterns
- Platform/context analysis, feature correlations
- Optimized for performance (~30-40 seconds runtime)

**Run it:**
```bash
cd notebooks
python eda_full_optimized.py
```

**Outputs:**
- `eda_full_target.png` - Target variable distribution
- `eda_full_demographics.png` - Age and gender analysis
- `eda_full_duration.png` - Track duration patterns
- `eda_full_temporal.png` - Time-of-day and day-of-week patterns
- `eda_full_platform.png` - Platform and context statistics
- `eda_full_correlations.png` - Feature correlation heatmap
- `eda_full_summary.txt` - Key findings and statistics

---

## Data Quality & Consistency Checks

### `data_quality_check.py`
**Systematic data quality assessment** - Checks for common data issues.

**Validates:**
- Missing values and data types
- Target variable integrity (binary 0/1)
- Value ranges (age 15-25, duration > 0)
- Categorical variables and date formats
- Duplicate records

**Run it:**
```bash
cd notebooks
python data_quality_check.py
```

**Output:** Console summary with quality assessment

### `check_temporal_consistency.py`
**Temporal consistency analysis** - Identifies pre-release listening patterns.

**Checks:**
- Listening events before track release dates
- Distribution of temporal gaps
- Track age categories
- Statistical analysis of inconsistencies

**Run it:**
```bash
cd notebooks
python check_temporal_consistency.py
```

**Outputs:**
- Console report with statistics
- `temporal_inconsistencies_sample.csv` - Sample of problematic records

**Finding:** 0.45% of records show pre-release listening (kept as feature)

---

## User Behavior Analysis

### `user_skip_behavior_analysis.py`
**User segmentation by skip behavior** - Analyzes users who don't skip vs. frequent skippers.

**Analyzes:**
- User-level skip rate distribution
- User segments (Never/Rarely/Occasional/Moderate/Frequent skippers)
- Demographics by segment (age, gender)
- Listening diversity (genres, artists, contexts)
- Statistical significance tests

**Run it:**
```bash
cd notebooks
python user_skip_behavior_analysis.py
```

**Outputs:**
- `user_skip_behavior_analysis.png` - Segment distributions and comparisons
- `user_segment_detailed_comparison.png` - Detailed demographic analysis
- `user_segments.csv` - User-level data with segment labels (19,165 users)
- `USER_SKIP_BEHAVIOR_ANALYSIS.md` - Complete analysis report

**Key Finding:** 19.3% of users rarely skip, age and session count are significant predictors

### `USER_SKIP_BEHAVIOR_ANALYSIS.md`
**Detailed report** on user skip behavior analysis with:
- User segment distribution and profiles
- Statistical comparisons (engaged vs. skippers)
- Business insights and recommendations
- Feature engineering implications

---

## Feature Engineering & Preprocessing

### `demo_preprocessing.py`
**Demonstrates basic preprocessing** pipeline (temporal, release, duration features).

**Run it:**
```bash
cd notebooks
python demo_preprocessing.py
```

**Output:** Console summary showing 22 new features added

### `demo_preprocessing_with_users.py`
**Complete preprocessing workflow** - Shows proper train/test split with user features.

**Demonstrates:**
- Adding temporal, release, and duration features
- Computing user engagement features from training data
- Applying user stats to test data (prevents data leakage)
- Handling new users (cold start) with defaults
- Feature consistency validation

**Run it:**
```bash
cd notebooks
python demo_preprocessing_with_users.py
```

**Outputs:**
- `train_preprocessed_sample.csv` - Training data with all features
- `test_preprocessed_sample.csv` - Test data with all features
- `user_stats_from_train.csv` - User statistics lookup table

**Features Generated:** 31 new features (9 user-level + 22 track/temporal)

### `USER_FEATURES_IMPLEMENTATION.md`
**Implementation guide** for user engagement features:
- 9 user-level features explained
- Proper train/test workflow to prevent data leakage
- Cold start strategy for new users
- Model integration guidelines
- Performance expectations

---

## Documentation

### `USER_SKIP_BEHAVIOR_ANALYSIS.md`
Full report on user skip behavior analysis (see above)

### `USER_FEATURES_IMPLEMENTATION.md`
Implementation guide for user engagement features (see above)

---

## Interactive Notebooks

### `exploration.ipynb`
Interactive Jupyter notebook for exploratory analysis (create as needed)

### `experiments.ipynb`
Experimental model development notebook (create as needed)

---

## Output Files Summary

### Visualizations:
- `eda_full_*.png` - Full dataset EDA visualizations
- `user_skip_behavior_analysis.png` - User segmentation
- `user_segment_detailed_comparison.png` - Demographic comparisons

### Data Files:
- `eda_full_summary.txt` - EDA key findings
- `temporal_inconsistencies_sample.csv` - Pre-release listening samples
- `user_segments.csv` - User segmentation (19K users)
- `train_preprocessed_sample.csv` - Preprocessed training sample
- `test_preprocessed_sample.csv` - Preprocessed test sample
- `user_stats_from_train.csv` - User statistics for modeling

### Reports:
- `USER_SKIP_BEHAVIOR_ANALYSIS.md` - User behavior analysis
- `USER_FEATURES_IMPLEMENTATION.md` - Feature engineering guide

---

## Setup

Make sure you have installed the required packages:
```bash
pip install -r ../requirements.txt
```

---

## Recommended Workflow

1. **Data Quality**: Run `data_quality_check.py` and `check_temporal_consistency.py`
2. **EDA**: Run `eda_full_optimized.py` to understand the data
3. **User Analysis**: Run `user_skip_behavior_analysis.py` for user insights
4. **Feature Engineering**: Run `demo_preprocessing_with_users.py` to see full pipeline
5. **Modeling**: Proceed to model development with preprocessed features

---

## Key Findings Summary

✅ **Data Quality**: High quality, 99.55% consistent, minimal cleaning required  
✅ **Pre-release Listening**: 0.45% of records (kept as feature)  
✅ **User Segments**: 5 clear segments identified (6.8% never skip, 26.9% frequent skippers)  
✅ **Predictive Features**: Age, session count, skip rate are statistically significant  
✅ **Feature Count**: 31 engineered features (9 user + 22 track/temporal)  

---

*All scripts are optimized for the 7.5M-row training dataset and run in < 1 minute*
