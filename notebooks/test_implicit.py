"""
Test Implicit library - Fast collaborative filtering for implicit feedback
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.metrics import roc_auc_score
import time

print("=" * 80)
print("IMPLICIT LIBRARY TEST - Collaborative Filtering")
print("=" * 80)

# Load small sample
print("\n[1/4] Loading data...")
df = pd.read_csv('../data/raw/train.csv', nrows=100000)
print(f"✓ Loaded {len(df):,} interactions")

# Create user-item matrix
print("\n[2/4] Creating sparse matrix...")
users = df['user_id'].astype('category')
items = df['media_id'].astype('category')
ratings = df['is_listened'].values

user_item_matrix = csr_matrix(
    (ratings, (users.cat.codes, items.cat.codes))
)
print(f"✓ Matrix: {user_item_matrix.shape}")

# Split train/test (simple: first 80% vs last 20%)
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print(f"  Train: {len(train_df):,}")
print(f"  Test: {len(test_df):,}")

# Model 1: ALS
print("\n[3/4] Testing ALS (Alternating Least Squares)...")
start = time.time()
als_model = AlternatingLeastSquares(
    factors=50,
    iterations=15,
    regularization=0.01,
    random_state=42
)
als_model.fit(user_item_matrix)
print(f"✓ ALS trained in {time.time()-start:.1f}s")

# Model 2: BPR
print("\n[4/4] Testing BPR (Bayesian Personalized Ranking)...")
start = time.time()
bpr_model = BayesianPersonalizedRanking(
    factors=50,
    iterations=100,
    learning_rate=0.01,
    regularization=0.01,
    random_state=42
)
bpr_model.fit(user_item_matrix)
print(f"✓ BPR trained in {time.time()-start:.1f}s")

print("\n" + "=" * 80)
print("✅ IMPLICIT LIBRARY WORKS!")
print("=" * 80)
print("\nImplicit library features:")
print("  ✓ Very fast (optimized C++)")
print("  ✓ Built for implicit feedback (like skip/listen)")
print("  ✓ ALS: Fast collaborative filtering")
print("  ✓ BPR: Ranking-optimized (good for RecSys)")
print("\nThis is a great alternative to Surprise!")
