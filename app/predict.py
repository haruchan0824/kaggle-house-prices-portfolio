from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT_DIR / "app" / "artifacts"

model = joblib.load(ARTIFACT_DIR / "model.pkl")
feature_columns = joblib.load(ARTIFACT_DIR / "feature_columns.pkl")


def create_features_for_api(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    # API用の列名をKaggle元データの列名に戻す
    df = df.rename(
        columns={
            "FirstFlrSF": "1stFlrSF",
            "SecondFlrSF": "2ndFlrSF",
        }
    )

    # 学習時と同じドメイン特徴量を作る
    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0).fillna(0)
        + df.get("1stFlrSF", 0).fillna(0)
        + df.get("2ndFlrSF", 0).fillna(0)
    )

    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    df["TotalBath"] = (
        df.get("FullBath", 0).fillna(0)
        + 0.5 * df.get("HalfBath", 0).fillna(0)
        + df.get("BsmtFullBath", 0).fillna(0)
        + 0.5 * df.get("BsmtHalfBath", 0).fillna(0)
    )

    df["HasGarage"] = (df.get("GarageArea", 0).fillna(0) > 0).astype(int)
    df["HasBsmt"] = (df.get("TotalBsmtSF", 0).fillna(0) > 0).astype(int)
    df["HasFireplace"] = (df.get("Fireplaces", 0).fillna(0) > 0).astype(int)

    # 学習時に存在したがAPI入力にない列は0で補完
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # 学習時と同じ列順に揃える
    df = df[feature_columns]

    return df


def predict_price(input_dict: dict) -> float:
    X = create_features_for_api(input_dict)

    pred_log = model.predict(X)[0]

    # log1p(SalePrice)で学習しているため元スケールに戻す
    pred_price = np.expm1(pred_log)

    return float(pred_price)