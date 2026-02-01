"""
Surprise Library Test - Simple Baseline
Testing Surprise with minimal complexity to identify issues
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("SURPRISE LIBRARY - MINIMAL TEST")
print("=" * 80)

# Test 1: Import Surprise
print("\n[Test 1] Importing Surprise library...")
try:
    from surprise import SVD, NMF, KNNBasic, BaselineOnly
    from surprise import Dataset, Reader
    from surprise.model_selection import train_test_split
    from surprise import accuracy
    print("✓ Surprise imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Load small sample
print("\n[Test 2] Loading small data sample...")
SAMPLE_SIZE = 50000  # Start with 50K (smaller than before)
df = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(df):,} interactions")
print(f"  Users: {df['user_id'].nunique():,}")
print(f"  Items: {df['media_id'].nunique():,}")

# Test 3: Format data for Surprise
print("\n[Test 3] Formatting data...")
surprise_data = df[['user_id', 'media_id', 'is_listened']].copy()
reader = Reader(rating_scale=(0, 1))
surprise_dataset = Dataset.load_from_df(surprise_data, reader)
print("✓ Data formatted")

# Test 4: Create train/test split
print("\n[Test 4] Creating train/test split...")
trainset, testset = train_test_split(surprise_dataset, test_size=0.2, random_state=42)
print(f"✓ Split complete")
print(f"  Train: {trainset.n_ratings:,} ratings")
print(f"  Test: {len(testset):,} ratings")

# Test 5: Try simplest model first (BaselineOnly)
print("\n[Test 5] Testing BaselineOnly (simplest model)...")
try:
    model = BaselineOnly()
    print("  Training...")
    model.fit(trainset)
    print("  Predicting...")
    predictions = model.test(testset)
    print("  Calculating metrics...")
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"✓ BaselineOnly WORKS!")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
except Exception as e:
    print(f"✗ BaselineOnly failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Try SVD
print("\n[Test 6] Testing SVD...")
try:
    model = SVD(n_factors=50, n_epochs=10, verbose=False)
    print("  Training...")
    model.fit(trainset)
    print("  Predicting...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"✓ SVD WORKS!")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
except Exception as e:
    print(f"✗ SVD failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Try KNN
print("\n[Test 7] Testing KNN...")
try:
    model = KNNBasic(k=20, sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
    print("  Training...")
    model.fit(trainset)
    print("  Predicting...")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"✓ KNN WORKS!")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
except Exception as e:
    print(f"✗ KNN failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("If all tests passed, Surprise is working correctly!")
print("We can now run the full baseline script.")
print("=" * 80)
