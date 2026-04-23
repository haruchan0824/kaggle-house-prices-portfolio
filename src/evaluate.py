from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_cv_result(cv_result: dict, output_path: str | Path = "outputs/cv_result.json") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cv_result, f, indent=2)


def save_feature_importance(
    feature_importance: pd.DataFrame,
    output_path: str | Path = "outputs/feature_importance.csv",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(output_path, index=False)


def save_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: str | Path = "outputs/model_comparison.csv",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
