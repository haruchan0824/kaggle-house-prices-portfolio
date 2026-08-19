import pandas as pd

from src.data import split_features_target
from src.features import add_domain_features, cleanup_feature_columns


def test_domain_features_and_identifier_cleanup() -> None:
    train = pd.DataFrame(
        {
            "Id": [1, 2],
            "SalePrice": [100_000, 120_000],
            "TotalBsmtSF": [500, 600],
            "1stFlrSF": [800, 900],
            "2ndFlrSF": [0, 200],
            "YrSold": [2010, 2010],
            "YearBuilt": [2000, 1990],
            "YearRemodAdd": [2005, 2000],
            "FullBath": [1, 2],
            "HalfBath": [1, 0],
            "BsmtFullBath": [0, 1],
            "BsmtHalfBath": [0, 0],
            "GarageArea": [0, 400],
            "Fireplaces": [0, 1],
            "OverallQual": [5, 7],
            "GrLivArea": [800, 1100],
        }
    )
    engineered = add_domain_features(train)
    expected = {
        "TotalSF", "HouseAge", "RemodAge", "TotalBath", "HasGarage",
        "HasBsmt", "HasFireplace", "OverallQual_x_GrLivArea",
    }
    assert expected <= set(engineered.columns)

    features, target = split_features_target(engineered)
    cleaned, aligned, dropped = cleanup_feature_columns(features, features.copy())
    assert "Id" not in cleaned.columns
    assert "SalePrice" not in cleaned.columns
    assert list(cleaned.columns) == list(aligned.columns)
    assert "Id" not in target.name
    assert "Id" in dropped or "Id" not in features.columns
