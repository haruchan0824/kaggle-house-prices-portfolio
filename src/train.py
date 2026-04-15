from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor


BASE_LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

BASE_CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": 0,
}


@dataclass
class EnsembleArtifacts:
    lgbm_model: Pipeline
    catboost_model: Pipeline
    cv_result: dict
    feature_importance: pd.DataFrame


def _build_lgbm_model(random_state: int = 42) -> LGBMRegressor:
    params = BASE_LGBM_PARAMS.copy()
    params["random_state"] = random_state
    return LGBMRegressor(**params)


def _build_catboost_model(random_state: int = 42) -> CatBoostRegressor:
    params = BASE_CATBOOST_PARAMS.copy()
    params["random_seed"] = random_state
    return CatBoostRegressor(**params)


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return importance_df


def train_ensemble_with_cv(
    X: pd.DataFrame,
    y_log: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> EnsembleArtifacts:
    """Train LightGBM + CatBoost and evaluate simple average ensemble via 5-fold CV.

    Purpose:
    - Keep the ensemble logic easy to explain for interviews.
    - Average two different tree models to improve robustness.
    - Avoid complex stacking/blending optimization in this portfolio baseline.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rmse_lgbm = []
    fold_rmse_catboost = []
    fold_rmse_ensemble = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y_log.iloc[train_idx], y_log.iloc[valid_idx]

        lgbm_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("regressor", _build_lgbm_model(random_state=random_state)),
            ]
        )
        catboost_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("regressor", _build_catboost_model(random_state=random_state)),
            ]
        )

        lgbm_pipeline.fit(X_train, y_train)
        catboost_pipeline.fit(X_train, y_train)

        pred_lgbm = lgbm_pipeline.predict(X_valid)
        pred_catboost = catboost_pipeline.predict(X_valid)

        # Simple mean ensemble: transparent and reproducible for portfolio usage.
        pred_ensemble = 0.5 * pred_lgbm + 0.5 * pred_catboost

        rmse_lgbm = float(np.sqrt(mean_squared_error(y_valid, pred_lgbm)))
        rmse_catboost = float(np.sqrt(mean_squared_error(y_valid, pred_catboost)))
        rmse_ensemble = float(np.sqrt(mean_squared_error(y_valid, pred_ensemble)))

        fold_rmse_lgbm.append(rmse_lgbm)
        fold_rmse_catboost.append(rmse_catboost)
        fold_rmse_ensemble.append(rmse_ensemble)

    # Retrain both models on the full training set for final test prediction.
    lgbm_full = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("regressor", _build_lgbm_model(random_state=random_state)),
        ]
    )
    catboost_full = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("regressor", _build_catboost_model(random_state=random_state)),
        ]
    )
    lgbm_full.fit(X, y_log)
    catboost_full.fit(X, y_log)

    cv_result = {
        "metric": "rmse_on_log_target",
        "n_splits": n_splits,
        "fold_rmse_lgbm": fold_rmse_lgbm,
        "fold_rmse_catboost": fold_rmse_catboost,
        "fold_rmse_ensemble": fold_rmse_ensemble,
        "lgbm_mean_rmse": float(np.mean(fold_rmse_lgbm)),
        "lgbm_std_rmse": float(np.std(fold_rmse_lgbm)),
        "catboost_mean_rmse": float(np.mean(fold_rmse_catboost)),
        "catboost_std_rmse": float(np.std(fold_rmse_catboost)),
        "ensemble_mean_rmse": float(np.mean(fold_rmse_ensemble)),
        "ensemble_std_rmse": float(np.std(fold_rmse_ensemble)),
    }
    feature_importance = _extract_feature_importance(lgbm_full)

    return EnsembleArtifacts(
        lgbm_model=lgbm_full,
        catboost_model=catboost_full,
        cv_result=cv_result,
        feature_importance=feature_importance,
    )
