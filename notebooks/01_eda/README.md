# 01_eda/ - Exploratory Data Analysis

Scripts and outputs for understanding the Deezer dataset.

## 🔍 Analysis Scripts

### `eda_full_optimized.py` ⭐
Complete EDA on 7.5M training dataset
- Runtime: ~30-40 seconds
- Generates 7 visualizations + summary text
- Analyzes: target, demographics, duration, temporal, platform, correlations

**Run**: `python eda_full_optimized.py`

### `data_quality_check.py`
Validates data quality and integrity
- Checks for missing values, data types
- Validates ranges and formats
- Finding: 99.55% data consistency ✅

**Run**: `python data_quality_check.py`

### `check_temporal_consistency.py`
Analyzes temporal patterns and pre-release listening
- Finding: 0.45% pre-release events (kept as feature)

**Run**: `python check_temporal_consistency.py`

### `user_skip_behavior_analysis.py`
User segmentation by skip behavior
- Identifies 5 user segments
- Statistical significance testing
- Generates user segments dataset

**Run**: `python user_skip_behavior_analysis.py`

---

## 📊 Generated Outputs

### Visualizations
- `eda_full_target.png` - Target distribution
- `eda_full_demographics.png` - Age/gender analysis
- `eda_full_duration.png` - Track duration patterns
- `eda_full_temporal.png` - Time patterns
- `eda_full_platform.png` - Platform statistics  
- `eda_full_platform_cleaned.png` - Cleaned version
- `eda_full_context_detailed.png` - Context analysis
- `eda_full_correlations.png` - Feature correlations
- `user_skip_behavior_analysis.png` - User segments
- `user_segment_detailed_comparison.png` - Demographics by segment

### Data Files
- `eda_full_summary.txt` - Key findings (in this directory)
- `../../data/processed/eda/temporal_inconsistencies_sample.csv` - Temporal issues
- `../../data/processed/eda/user_segments.csv` - User segmentation (19,165 users)

---

## 🚀 Quick Start

```bash
# Run full EDA pipeline
python eda_full_optimized.py
python data_quality_check.py  
python check_temporal_consistency.py
python user_skip_behavior_analysis.py
```

---

## 📈 Key Findings

- ✅ 99.55% data consistency
- ✅ 45% listen rate, 55% skip rate
- ✅ 19.3% users rarely skip (<10%)
- ✅ Age and activity predict skip behavior
- ✅ 0.45% pre-release listening (kept as feature)

---

*See `../docs/` for detailed reports*
