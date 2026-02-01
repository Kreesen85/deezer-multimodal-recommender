"""
Baseline Recommender Models using Surprise Library
Collaborative Filtering approaches for sequential skip prediction
"""

import pandas as pd
import numpy as np
from surprise import SVD, SVDpp, NMF, KNNBasic, KNNWithMeans, BaselineOnly
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("SURPRISE LIBRARY: COLLABORATIVE FILTERING BASELINES")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[Step 1] Loading data...")

# Load sample for faster iteration (full dataset takes longer)
SAMPLE_SIZE = 500000  # 500K for reasonable speed
print(f"Loading {SAMPLE_SIZE:,} samples...")

df = pd.read_csv('../data/raw/train.csv', nrows=SAMPLE_SIZE)
print(f"✓ Loaded {len(df):,} interactions")
print(f"  Unique users: {df['user_id'].nunique():,}")
print(f"  Unique items: {df['media_id'].nunique():,}")
print(f"  Listen rate: {df['is_listened'].mean():.3f}")

# ============================================================================
# STEP 2: FORMAT DATA FOR SURPRISE
# ============================================================================
print("\n[Step 2] Formatting data for Surprise...")

# Surprise expects: user_id, item_id, rating
# We use: user_id, media_id, is_listened (0=skip, 1=listen)
surprise_data = df[['user_id', 'media_id', 'is_listened']].copy()

# Surprise requires specific format
reader = Reader(rating_scale=(0, 1))
surprise_dataset = Dataset.load_from_df(
    surprise_data, 
    reader
)

print("✓ Data formatted for Surprise")
print(f"  Format: (user_id, media_id, rating)")
print(f"  Rating scale: 0 (skip) to 1 (listen)")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================
print("\n[Step 3] Creating train/test split...")

# Build full trainset and testset
trainset, testset = train_test_split(surprise_dataset, test_size=0.2, random_state=42)

print(f"✓ Split complete")
print(f"  Training set: {trainset.n_ratings:,} ratings")
print(f"  Test set: {len(testset):,} ratings")
print(f"  Users in train: {trainset.n_users:,}")
print(f"  Items in train: {trainset.n_items:,}")

# ============================================================================
# STEP 4: BUILD MODELS
# ============================================================================
print("\n[Step 4] Building collaborative filtering models...")

models = {
    'SVD': SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=False
    ),
    
    'SVD++': SVDpp(
        n_factors=50,  # Fewer factors for speed
        n_epochs=10,   # Fewer epochs for speed
        lr_all=0.005,
        reg_all=0.02,
        random_state=42,
        verbose=False
    ),
    
    'NMF': NMF(
        n_factors=50,
        n_epochs=20,
        reg_pu=0.06,
        reg_qi=0.06,
        random_state=42,
        verbose=False
    ),
    
    'KNN Basic (User-based)': KNNBasic(
        k=40,
        min_k=1,
        sim_options={
            'name': 'cosine',
            'user_based': True
        },
        verbose=False
    ),
    
    'KNN Basic (Item-based)': KNNBasic(
        k=40,
        min_k=1,
        sim_options={
            'name': 'cosine',
            'user_based': False
        },
        verbose=False
    ),
    
    'KNN with Means': KNNWithMeans(
        k=40,
        min_k=1,
        sim_options={
            'name': 'pearson',
            'user_based': True
        },
        verbose=False
    ),
    
    'Baseline (Global Mean)': BaselineOnly(
        bsl_options={
            'method': 'als',
            'n_epochs': 10,
            'reg_u': 15,
            'reg_i': 10
        }
    )
}

print(f"✓ Configured {len(models)} models")
for model_name in models.keys():
    print(f"  - {model_name}")

# ============================================================================
# STEP 5: TRAIN AND EVALUATE MODELS
# ============================================================================
print("\n[Step 5] Training and evaluating models...")
print("(This may take a few minutes...)")
print()

results = []

for model_name, model in models.items():
    print(f"Training {model_name}...")
    start_time = time.time()
    
    # Train on trainset
    model.fit(trainset)
    
    # Predict on testset
    predictions = model.test(testset)
    
    # Calculate metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    
    training_time = time.time() - start_time
    
    print(f"  ✓ RMSE: {rmse:.4f} | MAE: {mae:.4f} | Time: {training_time:.1f}s")
    
    results.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'Training Time (s)': training_time
    })

print("\n✓ All models trained and evaluated")

# ============================================================================
# STEP 6: RESULTS SUMMARY
# ============================================================================
print("\n[Step 6] Results Summary")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')

print("\n--- MODEL COMPARISON (sorted by RMSE) ---")
print(results_df.to_string(index=False))

# ============================================================================
# STEP 7: DETAILED ANALYSIS - BEST MODEL
# ============================================================================
print("\n[Step 7] Detailed Analysis - Best Model")
print("=" * 80)

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   RMSE: {results_df.iloc[0]['RMSE']:.4f}")
print(f"   MAE: {results_df.iloc[0]['MAE']:.4f}")

# Analyze predictions
predictions = best_model.test(testset)

# Calculate additional metrics
actual_ratings = [pred.r_ui for pred in predictions]
predicted_ratings = [pred.est for pred in predictions]

# Convert to binary for classification metrics
actual_binary = [1 if r > 0.5 else 0 for r in actual_ratings]
predicted_binary = [1 if r > 0.5 else 0 for r in predicted_ratings]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(actual_binary, predicted_binary)
precision = precision_score(actual_binary, predicted_binary, zero_division=0)
recall = recall_score(actual_binary, predicted_binary, zero_division=0)
f1 = f1_score(actual_binary, predicted_binary, zero_division=0)

# AUC requires continuous predictions
try:
    auc = roc_auc_score(actual_binary, predicted_ratings)
except:
    auc = 0.0

print(f"\n--- Classification Metrics (threshold=0.5) ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# ============================================================================
# STEP 8: CROSS-VALIDATION (ON SUBSET FOR SPEED)
# ============================================================================
print("\n[Step 8] Cross-Validation on Top 3 Models")
print("=" * 80)

# Use smaller sample for CV (faster)
print("\nRunning 3-fold cross-validation...")

top_3_models = results_df.head(3)['Model'].tolist()
cv_results = []

for model_name in top_3_models:
    print(f"\nCV for {model_name}...")
    model = models[model_name]
    
    cv_result = cross_validate(
        model, 
        surprise_dataset, 
        measures=['RMSE', 'MAE'],
        cv=3,
        verbose=False
    )
    
    mean_rmse = np.mean(cv_result['test_rmse'])
    std_rmse = np.std(cv_result['test_rmse'])
    mean_mae = np.mean(cv_result['test_mae'])
    std_mae = np.std(cv_result['test_mae'])
    
    print(f"  RMSE: {mean_rmse:.4f} (±{std_rmse:.4f})")
    print(f"  MAE:  {mean_mae:.4f} (±{std_mae:.4f})")
    
    cv_results.append({
        'Model': model_name,
        'CV RMSE': mean_rmse,
        'CV RMSE Std': std_rmse,
        'CV MAE': mean_mae,
        'CV MAE Std': std_mae
    })

cv_df = pd.DataFrame(cv_results)
print("\n--- Cross-Validation Results ---")
print(cv_df.to_string(index=False))

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n[Step 9] Creating visualizations...")

# Figure 1: Model comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# RMSE comparison
ax1 = axes[0]
results_sorted = results_df.sort_values('RMSE', ascending=False)
colors = ['#e74c3c' if i == len(results_sorted)-1 else '#3498db' 
          for i in range(len(results_sorted))]
ax1.barh(results_sorted['Model'], results_sorted['RMSE'], color=colors, alpha=0.7)
ax1.set_xlabel('RMSE (lower is better)')
ax1.set_title('Model Comparison: RMSE')
ax1.axvline(results_sorted['RMSE'].min(), color='green', linestyle='--', 
            label=f'Best: {results_sorted["RMSE"].min():.4f}')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# MAE comparison
ax2 = axes[1]
results_sorted_mae = results_df.sort_values('MAE', ascending=False)
colors = ['#e74c3c' if i == len(results_sorted_mae)-1 else '#3498db' 
          for i in range(len(results_sorted_mae))]
ax2.barh(results_sorted_mae['Model'], results_sorted_mae['MAE'], color=colors, alpha=0.7)
ax2.set_xlabel('MAE (lower is better)')
ax2.set_title('Model Comparison: MAE')
ax2.axvline(results_sorted_mae['MAE'].min(), color='green', linestyle='--',
            label=f'Best: {results_sorted_mae["MAE"].min():.4f}')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Training time comparison
ax3 = axes[2]
results_sorted_time = results_df.sort_values('Training Time (s)', ascending=False)
colors = ['#e67e22' if t > 60 else '#27ae60' 
          for t in results_sorted_time['Training Time (s)']]
ax3.barh(results_sorted_time['Model'], results_sorted_time['Training Time (s)'], 
         color=colors, alpha=0.7)
ax3.set_xlabel('Training Time (seconds)')
ax3.set_title('Model Comparison: Training Time')
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('surprise_models_comparison.png', dpi=200, bbox_inches='tight')
print("✓ Saved: surprise_models_comparison.png")
plt.close()

# Figure 2: Prediction distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Best model predictions
predictions_best = best_model.test(testset)
actual = [pred.r_ui for pred in predictions_best]
predicted = [pred.est for pred in predictions_best]

ax1 = axes[0]
ax1.scatter(actual, predicted, alpha=0.1, s=1)
ax1.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
ax1.set_xlabel('Actual Rating (0=skip, 1=listen)')
ax1.set_ylabel('Predicted Rating')
ax1.set_title(f'Prediction Scatter: {best_model_name}')
ax1.legend()
ax1.grid(alpha=0.3)

# Prediction histogram
ax2 = axes[1]
ax2.hist(predicted, bins=50, alpha=0.7, label='Predicted', color='blue', edgecolor='black')
ax2.axvline(np.mean(predicted), color='blue', linestyle='--', 
            label=f'Mean predicted: {np.mean(predicted):.3f}')
ax2.axvline(np.mean(actual), color='red', linestyle='--', 
            label=f'Mean actual: {np.mean(actual):.3f}')
ax2.set_xlabel('Rating')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Predictions')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('surprise_predictions_analysis.png', dpi=200, bbox_inches='tight')
print("✓ Saved: surprise_predictions_analysis.png")
plt.close()

# ============================================================================
# STEP 10: SAVE RESULTS
# ============================================================================
print("\n[Step 10] Saving results...")

# Save model comparison
results_df.to_csv('surprise_model_comparison.csv', index=False)
print("✓ Saved: surprise_model_comparison.csv")

# Save detailed predictions sample
predictions_sample = pd.DataFrame([
    {
        'user_id': pred.uid,
        'item_id': pred.iid,
        'actual_rating': pred.r_ui,
        'predicted_rating': pred.est,
        'error': abs(pred.r_ui - pred.est)
    }
    for pred in predictions_best[:1000]  # First 1000 predictions
])
predictions_sample.to_csv('surprise_predictions_sample.csv', index=False)
print("✓ Saved: surprise_predictions_sample.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY & INSIGHTS")
print("=" * 80)

print(f"""
📊 EXPERIMENT DETAILS:
   - Dataset: {SAMPLE_SIZE:,} interactions
   - Users: {df['user_id'].nunique():,}
   - Items: {df['media_id'].nunique():,}
   - Sparsity: {1 - (len(df) / (df['user_id'].nunique() * df['media_id'].nunique())):.4f}
   - Train/Test Split: 80/20

🏆 BEST MODEL: {best_model_name}
   - RMSE: {results_df.iloc[0]['RMSE']:.4f}
   - MAE: {results_df.iloc[0]['MAE']:.4f}
   - Accuracy: {accuracy:.4f}
   - AUC: {auc:.4f}

📈 MODEL RANKINGS (by RMSE):
""")

for idx, row in results_df.iterrows():
    print(f"   {idx+1}. {row['Model']:30s} - RMSE: {row['RMSE']:.4f}")

print(f"""

💡 KEY INSIGHTS:

1. COLLABORATIVE FILTERING PERFORMANCE:
   - Pure CF achieves ~{auc:.2f} AUC without any features
   - This is the baseline to beat with hybrid models
   
2. MODEL CHARACTERISTICS:
   - SVD/SVD++: Fast, good for dense data
   - NMF: Interpretable, non-negative factors
   - KNN: Simple, but slower for large datasets
   
3. LIMITATIONS OF PURE CF:
   ❌ Cannot handle new users (cold start)
   ❌ Cannot use features (age, genre, time)
   ❌ Only uses interaction history
   
4. NEXT STEPS:
   ✅ Build LightFM hybrid model (CF + features)
   ✅ Expected AUC improvement: +0.05-0.10
   ✅ Solves cold start problem
   ✅ Uses all engineered features

⏭️  READY FOR: Hybrid model with LightFM
""")

print("=" * 80)
print("✅ Surprise baseline complete!")
print("=" * 80)
