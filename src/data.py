from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


TARGET_COL = "SalePrice"
ID_COL = "Id"


def load_train_test(data_dir: str | Path = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kaggle House Prices train/test CSVs."""
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df


def split_features_target(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split training data into features and target."""
    X = train_df.drop(columns=[TARGET_COL])
    y = train_df[TARGET_COL]
    return X, y


def save_submission(test_ids: pd.Series, preds: pd.Series, output_path: str | Path) -> None:
    """Save Kaggle submission file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: preds})
    submission.to_csv(output_path, index=False)
