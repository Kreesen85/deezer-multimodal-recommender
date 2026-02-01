# Item-based Collaborative Filtering (CF)

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Step 1: Load data and keep only needed columns

DATA_PATH = "data/processed/samples/train_100k_preprocessed.csv"

df = pd.read_csv(DATA_PATH)
df = df[["user_id", "media_id", "is_listened"]].copy()
df["is_listened"] = df["is_listened"].astype(int)

print("Number of rows:", len(df))
print("Number of users:", df["user_id"].nunique())
print("Number of items:", df["media_id"].nunique())
print("Listen rate:", df["is_listened"].mean())

# Step 2: Train/test split and simple train statistics
# We split inside the sample so we can evaluate methods on the same data.

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["is_listened"]
)

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))
print("Train listen rate:", train_df["is_listened"].mean())
print("Test listen rate:", test_df["is_listened"].mean())

global_mean = train_df["is_listened"].mean()
print("Global mean (fallback):", global_mean)

# Step 3: Build mappings from TRAIN only
# We use only positive interactions (implicit feedback).

train_pos = train_df[train_df["is_listened"] == 1]

user_listened = (
    train_pos.groupby("user_id")["media_id"]
    .apply(set)
    .to_dict()
)

print("Users with at least one listened item in train:", len(user_listened))

item_users = (
    train_pos.groupby("media_id")["user_id"]
    .apply(set)
    .to_dict()
)

print("Items with at least one listen in train:", len(item_users))

# Helpful for faster scoring:
# user_items: user -> items they listened (train, positives only)
user_items = (
    train_pos.groupby("user_id")["media_id"]
    .apply(set)
    .to_dict()
)

example_item = list(item_users.keys())[0]
print("Example item:", example_item)
print("Number of users who listened:", len(item_users[example_item]))

# Step 4: Simple item-item similarity (sanity check)
# Here: similarity = number of common users

def item_similarity(item_a, item_b, item_users):
    users_a = item_users.get(item_a, set())
    users_b = item_users.get(item_b, set())
    return len(users_a & users_b)

items = list(item_users.keys())
item_1 = items[0]
item_2 = items[1]
sim = item_similarity(item_1, item_2, item_users)

print("Item 1:", item_1)
print("Item 2:", item_2)
print("Common users:", sim)



# Step 5: Score candidate items for a user (item-based CF)

def score_items_for_user(user_id, user_listened, item_users):
    """
    Item-based scoring using co-occurrence.

    Logic:
    - Take items the user listened
    - Find other users who listened to these items
    - Recommend other items these users listened
    """
    listened_items = user_listened.get(user_id, set())
    scores = {}

    for item in listened_items:
        users_who_listened = item_users.get(item, set())

        for other_user in users_who_listened:
            other_items = user_listened.get(other_user, set())

            for other_item in other_items:
                if other_item in listened_items:
                    continue
                scores[other_item] = scores.get(other_item, 0) + 1

    return scores


# Example recommendation for one test user
test_user = test_df["user_id"].iloc[0]
scores = score_items_for_user(test_user, user_listened, item_users)

print("User:", test_user)
print("Number of candidate items:", len(scores))

top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top recommendations (item_id, score):")
for item, score in top_items:
    print(item, score)


# Step 6: Evaluation with Hit@K

def hit_at_k(test_df, user_listened, item_users, k=10, max_users=200):
    """
    Hit@K:
    At least one of the top-K recommended items
    is actually listened in test.
    """
    test_pos = test_df[test_df["is_listened"] == 1]
    users = test_pos["user_id"].unique()[:max_users]

    hits = 0
    total = 0

    for user_id in users:
        scores = score_items_for_user(user_id, user_listened, item_users)
        if len(scores) == 0:
            continue

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rec_items = set(item for item, _ in top_k)

        true_items = set(
            test_pos[test_pos["user_id"] == user_id]["media_id"].values
        )

        if len(rec_items & true_items) > 0:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0, total


k = 10
hit_rate, n_eval = hit_at_k(test_df, user_listened, item_users, k=k, max_users=200)

print("Hit@{}:".format(k), round(hit_rate, 4))
print("Evaluated users:", n_eval)


# Step 7: Matrix Factorization baseline (SVD)

train_users = train_df["user_id"].unique()
train_items = train_df["media_id"].unique()

user_to_idx = {u: i for i, u in enumerate(train_users)}
item_to_idx = {m: i for i, m in enumerate(train_items)}
idx_to_item = {i: m for m, i in item_to_idx.items()}

train_pos = train_df[train_df["is_listened"] == 1].copy()
train_pos["u"] = train_pos["user_id"].map(user_to_idx)
train_pos["i"] = train_pos["media_id"].map(item_to_idx)

X = csr_matrix(
    (np.ones(len(train_pos)), (train_pos["u"], train_pos["i"])),
    shape=(len(train_users), len(train_items))
)

print("SVD matrix shape:", X.shape)
print("SVD positives:", X.nnz)

n_factors = 50
svd = TruncatedSVD(n_components=n_factors, random_state=42)
U = svd.fit_transform(X)
V = svd.components_.T

print("SVD fitted. Explained variance:",
      round(svd.explained_variance_ratio_.sum(), 4))


def recommend_svd(user_id, user_listened, k=10):
    if user_id not in user_to_idx:
        return []

    listened = user_listened.get(user_id, set())
    u = user_to_idx[user_id]

    scores = U[u].dot(V.T)
    ranking = np.argsort(-scores)

    recs = []
    for idx in ranking:
        item_id = idx_to_item[idx]
        if item_id in listened:
            continue
        recs.append(item_id)
        if len(recs) == k:
            break
    return recs


def hit_at_k_svd(test_df, user_listened, k=10, max_users=200):
    test_pos = test_df[test_df["is_listened"] == 1]
    users = test_pos["user_id"].unique()[:max_users]

    hits = 0
    total = 0

    for user_id in users:
        recs = recommend_svd(user_id, user_listened, k=k)
        if len(recs) == 0:
            continue

        true_items = set(
            test_pos[test_pos["user_id"] == user_id]["media_id"].values
        )

        if len(set(recs) & true_items) > 0:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0, total


hit_svd, n_eval_svd = hit_at_k_svd(test_df, user_listened, k=10, max_users=200)

print("SVD Hit@10:", round(hit_svd, 4))
print("SVD evaluated users:", n_eval_svd)