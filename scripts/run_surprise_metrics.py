#!/usr/bin/env python3
"""
Run Surprise recommender baselines on raw Deezer data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.surprise_baselines import run_surprise_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Surprise baseline metrics on raw Deezer data."
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/train.csv",
        help="Path to raw train.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for faster runs",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-csv",
        default="notebooks/surprise_model_comparison.csv",
        help="Where to save the model comparison CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(
            f"Missing data file: {data_path}. "
            "Download the dataset and place train.csv under data/raw/."
        )
        return 1

    required_cols = ["user_id", "media_id", "is_listened"]
    df = pd.read_csv(data_path, usecols=required_cols)

    results_df, best_model = run_surprise_baselines(
        df,
        user_col="user_id",
        item_col="media_id",
        rating_col="is_listened",
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\nSurprise model comparison (sorted by RMSE):")
    print(results_df.to_string(index=False))
    print(f"\nBest model: {best_model}")
    print(f"Saved results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
