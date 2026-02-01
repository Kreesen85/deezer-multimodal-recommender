"""
Surprise baselines for implicit feedback (skip/listen) data.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from surprise import BaselineOnly, Dataset, KNNBasic, KNNWithMeans, NMF, Reader, SVD, SVDpp, accuracy
from surprise.model_selection import train_test_split


def build_surprise_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "SVD": SVD(
            n_factors=100,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=random_state,
            verbose=False,
        ),
        "SVD++": SVDpp(
            n_factors=50,
            n_epochs=10,
            lr_all=0.005,
            reg_all=0.02,
            random_state=random_state,
            verbose=False,
        ),
        "NMF": NMF(
            n_factors=50,
            n_epochs=20,
            reg_pu=0.06,
            reg_qi=0.06,
            random_state=random_state,
            verbose=False,
        ),
        "KNN Basic (User-based)": KNNBasic(
            k=40,
            min_k=1,
            sim_options={"name": "cosine", "user_based": True},
            verbose=False,
        ),
        "KNN Basic (Item-based)": KNNBasic(
            k=40,
            min_k=1,
            sim_options={"name": "cosine", "user_based": False},
            verbose=False,
        ),
        "KNN with Means": KNNWithMeans(
            k=40,
            min_k=1,
            sim_options={"name": "pearson", "user_based": True},
            verbose=False,
        ),
        "Baseline (Global Mean)": BaselineOnly(
            bsl_options={"method": "als", "n_epochs": 10, "reg_u": 15, "reg_i": 10}
        ),
    }


def run_surprise_baselines(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "media_id",
    rating_col: str = "is_listened",
    sample_size: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """
    Train and evaluate Surprise baselines on implicit feedback data.

    Returns
    -------
    results_df : pandas.DataFrame
        Sorted by RMSE ascending.
    best_model_name : str
        Model name with lowest RMSE.
    """
    required_cols = {user_col, item_col, rating_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df[[user_col, item_col, rating_col]].copy()
    if sample_size is not None and len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)

    reader = Reader(rating_scale=(0, 1))
    surprise_dataset = Dataset.load_from_df(data, reader)

    trainset, testset = train_test_split(
        surprise_dataset, test_size=test_size, random_state=random_state
    )

    models = build_surprise_models(random_state=random_state)
    results = []

    for model_name, model in models.items():
        start_time = time.time()
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        training_time = time.time() - start_time

        results.append(
            {
                "Model": model_name,
                "RMSE": rmse,
                "MAE": mae,
                "Training Time (s)": training_time,
            }
        )

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]

    return results_df, best_model_name
