from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
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


def save_submission(
    preds: np.ndarray | pd.Series,
    test_df: pd.DataFrame,
    output_path: str | Path,
    pred_is_log_scale: bool = True,
) -> None:
    """Save Kaggle submission file in a consistent format.

    Parameters
    ----------
    preds:
        Model predictions aligned with `test_df` row order.
    test_df:
        DataFrame containing the test rows and `Id` column.
    output_path:
        Destination CSV path.
    pred_is_log_scale:
        If True, convert predictions with expm1 before saving.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds_array = np.asarray(preds)
    if pred_is_log_scale:
        sale_price = np.expm1(preds_array)
    else:
        sale_price = preds_array

    submission = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].to_numpy(),
            TARGET_COL: sale_price,
        }
    )
    submission.to_csv(output_path, index=False)
