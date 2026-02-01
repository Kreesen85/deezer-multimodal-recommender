## 03_baselines – Collaborative Filtering

This folder contains baseline recommender system models
used for comparison in the project.

### Implemented Models

- Item-based Collaborative Filtering  
  Uses co-occurrence of users between items (implicit feedback).

- Matrix Factorization (SVD)  
  Simple latent factor model trained on implicit interactions.

### Dataset

- train_100k_preprocessed.csv
- Implicit feedback (is_listened ∈ {0,1})
- Train/test split: 80/20 (random)

### Evaluation

- Offline evaluation
- Metric: Hit@10
- Same evaluation procedure used for all baseline models

### Notes

- Models are intentionally kept simple for clarity and learning purposes.
- No user or item features are used.
- Cold-start users/items are not handled at this stage.