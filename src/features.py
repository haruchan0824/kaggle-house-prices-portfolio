from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


MISSING_CATEGORY_VALUE = "Missing"


def _get_numeric_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a numeric Series for a column, or zeros if the column is missing."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce")
    return pd.Series(0.0, index=df.index, dtype="float64")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple domain-driven features for House Prices.

    This function is deterministic and should be applied to both train and test
    before fitting/transforming the preprocessing pipeline.
    """
    df = df.copy()

    total_bsmt_sf = _get_numeric_column(df, "TotalBsmtSF").fillna(0)
    first_flr_sf = _get_numeric_column(df, "1stFlrSF").fillna(0)
    second_flr_sf = _get_numeric_column(df, "2ndFlrSF").fillna(0)
    yr_sold = _get_numeric_column(df, "YrSold").fillna(0)
    year_built = _get_numeric_column(df, "YearBuilt").fillna(0)
    year_remod = _get_numeric_column(df, "YearRemodAdd").fillna(0)
    full_bath = _get_numeric_column(df, "FullBath").fillna(0)
    half_bath = _get_numeric_column(df, "HalfBath").fillna(0)
    bsmt_full_bath = _get_numeric_column(df, "BsmtFullBath").fillna(0)
    bsmt_half_bath = _get_numeric_column(df, "BsmtHalfBath").fillna(0)
    garage_area = _get_numeric_column(df, "GarageArea").fillna(0)
    fireplaces = _get_numeric_column(df, "Fireplaces").fillna(0)
    overall_qual = _get_numeric_column(df, "OverallQual").fillna(0)
    gr_liv_area = _get_numeric_column(df, "GrLivArea").fillna(0)

    # Total living-related floor area (basement + 1st floor + 2nd floor).
    df.loc[:, "TotalSF"] = total_bsmt_sf + first_flr_sf + second_flr_sf

    # Approximate age of the house at sale time.
    df.loc[:, "HouseAge"] = yr_sold - year_built

    # Years since last remodel at sale time.
    df.loc[:, "RemodAge"] = yr_sold - year_remod

    # Weighted count of full/half bathrooms above and below ground.
    df.loc[:, "TotalBath"] = full_bath + 0.5 * half_bath + bsmt_full_bath + 0.5 * bsmt_half_bath

    # Binary indicator: house has a usable garage area.
    df.loc[:, "HasGarage"] = (garage_area > 0).astype("int64")

    # Binary indicator: house has a basement area.
    df.loc[:, "HasBsmt"] = (total_bsmt_sf > 0).astype("int64")

    # Binary indicator: house has at least one fireplace.
    df.loc[:, "HasFireplace"] = (fireplaces > 0).astype("int64")

    # Interaction: quality score scaled by above-ground living area.
    df.loc[:, "OverallQual_x_GrLivArea"] = overall_qual * gr_liv_area

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a simple preprocessing pipeline for tabular data."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY_VALUE)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor
