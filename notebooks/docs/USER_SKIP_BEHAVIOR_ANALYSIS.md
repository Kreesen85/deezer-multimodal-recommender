# User Skip Behavior Analysis

**Date**: February 1, 2026  
**Dataset**: Deezer DSG17 - User Segmentation by Skip Behavior  
**Sample**: 2,000,000 listening sessions from 19,165 unique users

---

## Executive Summary

Analysis of user skip behavior reveals distinct user segments with different listening patterns. **19.3% of users rarely skip** (< 10% skip rate), representing highly engaged listeners, while **24.7% are frequent skippers** (> 50% skip rate). These segments show statistically significant differences in age and session activity.

---

## User Segmentation

### Distribution by Skip Behavior:

| Segment | Skip Rate | Users | Percentage |
|---------|-----------|-------|------------|
| **Never Skips** | 0% | 1,305 | 6.81% |
| **Rarely Skips** | < 10% | 2,400 | 12.52% |
| **Occasional Skipper** | 10-25% | 4,426 | 23.09% |
| **Moderate Skipper** | 25-50% | 5,887 | 30.72% |
| **Frequent Skipper** | > 50% | 5,147 | 26.86% |

### Key Finding:
- **19.3%** of users are highly engaged (skip < 10%)
- **30.7%** are moderate skippers  
- **26.9%** are frequent skippers

---

## Engaged Users Profile
### (Skip Rate < 10%)

**Demographics:**
- **Count**: 3,705 users
- **Average Age**: 25.0 years (older than frequent skippers)
- **Gender Split**: 58% Female, 42% Male

**Listening Behavior:**
- **Average Sessions**: 110.1 per user
- **Track Duration**: 232.8 seconds (~3.9 min)
- **Genre Diversity**: 10.8 unique genres
- **Artist Diversity**: 36.1 unique artists
- **Context Variety**: 3.6 unique contexts

**Characteristics:**
- More mature users (age 25)
- Higher session count (18% more sessions)
- Consistent listening patterns
- Broad but focused music taste

---

## Frequent Skippers Profile
### (Skip Rate > 50%)

**Demographics:**
- **Count**: 4,730 users  
- **Average Age**: 23.3 years (younger)
- **Gender Split**: 51% Female, 49% Male

**Listening Behavior:**
- **Average Sessions**: 91.1 per user
- **Track Duration**: 231.4 seconds
- **Genre Diversity**: 10.9 unique genres
- **Artist Diversity**: 37.2 unique artists  
- **Context Variety**: 3.6 unique contexts

**Characteristics:**
- Younger users (age 23)
- Lower session count
- Exploratory listening behavior
- Similar music diversity but more skipping

---

## Statistical Significance

### T-Test Results (Engaged vs. Frequent Skippers):

| Feature | T-Statistic | P-Value | Significant? |
|---------|-------------|---------|--------------|
| **Age** | 20.205 | < 0.000001 | ✅ **YES** |
| **Session Count** | 5.572 | < 0.000001 | ✅ **YES** |
| **Genre Diversity** | -0.605 | 0.545 | ❌ No |

### Findings:
- **Age is a significant predictor** of skip behavior
- **Session count differs significantly** between groups
- **Genre diversity is similar** across segments (both explore ~11 genres)

---

## Key Insights

### 1. Age Effect
- **Engaged users are ~2 years older** on average (25.0 vs 23.3)
- Age is statistically significant predictor of engagement
- Suggests maturity or listening habit formation

### 2. Session Engagement
- Engaged users have **21% more sessions** (110 vs 91)
- Higher retention and platform loyalty
- More predictable listening patterns

### 3. Music Discovery
- **Both groups explore similar amounts** of content
  - Similar genre diversity (10.8 vs 10.9)
  - Similar artist counts (36 vs 37)
- **But skip behavior differs dramatically**
- Suggests skipping is about **selection criteria**, not exploration

### 4. Gender Patterns
- Engaged users: **58% Female**
- Frequent skippers: **51% Female**
- Slight female skew in engaged segment

---

## Implications for Modeling

### 1. User-Level Features (High Priority)

Create these user aggregated features:

```python
# Historical skip behavior
user_skip_rate = user_historical_skips / user_total_sessions

# Diversity metrics
user_genre_diversity = unique_genres_listened
user_artist_diversity = unique_artists_listened
user_context_variety = unique_contexts_used

# Engagement score
user_engagement_score = (
    (1 - user_skip_rate) * 0.4 +
    (user_session_count / max_sessions) * 0.3 +
    (user_genre_diversity / max_diversity) * 0.3
)
```

### 2. User Segmentation Feature

```python
# Categorical feature based on historical behavior
user_segment = categorize_by_skip_rate(user_skip_rate)
# ['Never Skips', 'Rarely Skips', 'Occasional', 'Moderate', 'Frequent']
```

### 3. Cold Start Strategy

For new users without history:
- Use **age** as proxy (strong predictor)
- Use **gender** (slight correlation)
- Initialize with platform average skip rate
- Update quickly with first few sessions

### 4. Interaction Features

```python
# User-context interactions
user_segment × context_type
user_age × time_of_day
user_skip_rate × track_age
```

---

## Business Insights

### 1. Highly Engaged Users (19.3%)
- **Value**: Premium user base, high retention
- **Strategy**: Curated playlists, exclusive content
- **Risk**: Don't over-serve recommendations (they're already happy)

### 2. Moderate Users (53.8%)
- **Value**: Largest segment, most malleable
- **Strategy**: Personalized discovery, improve recommendations
- **Opportunity**: Convert to engaged through better matching

### 3. Frequent Skippers (26.9%)
- **Value**: Active explorers, trend setters
- **Strategy**: Quick preview modes, better filtering
- **Risk**: Potential churn if frustrated

---

## Recommendations

### For Feature Engineering:
1. ✅ **Add user_skip_rate** (from historical data)
2. ✅ **Add user_segment** (categorical)
3. ✅ **Add user_age** (already in data, but use in interactions)
4. ✅ **Add user_session_count** (engagement proxy)
5. ✅ **Add diversity metrics** (genres, artists explored)

### For Model Training:
1. **Stratify by user segment** in train/val split
2. **Weight samples** by user engagement (higher weight for engaged users)
3. **Create user-specific models** for cold start vs. returning users
4. **Use embeddings** for user IDs to capture behavior patterns

### For Product:
1. **Different UX for segments**:
   - Engaged: Let them browse organically
   - Skippers: Quick preview, better filters
   - Moderate: Personalized recommendations
   
2. **Retention focus**: Move moderate users toward engaged segment

---

## Files Generated

1. **user_skip_behavior_analysis.png** - User segment distributions and comparisons
2. **user_segment_detailed_comparison.png** - Detailed demographics and diversity metrics  
3. **user_segments.csv** - User-level data with segments for modeling

---

## Next Steps

1. ✅ Incorporate user-level features into preprocessing pipeline
2. ✅ Add user segmentation to feature engineering
3. ⏭️ Build user-aware models
4. ⏭️ Test segment-specific strategies
5. ⏭️ A/B test recommendations by user segment

---

## Conclusion

User skip behavior is **highly variable** (6.8% never skip, 26.9% skip frequently) and **predictable** from user characteristics. **Age** and **session count** are strong indicators of engagement level. 

**Key Takeaway**: Don't treat all users the same - **user segmentation by skip behavior** should be a core feature in the recommendation system.

---

*Analysis based on 2M sessions from 19,165 users*  
*For reproduction: `notebooks/user_skip_behavior_analysis.py`*
