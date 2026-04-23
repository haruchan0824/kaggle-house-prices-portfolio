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
class TrainingArtifacts:
    lgbm_model: Pipeline
    catboost_model: Pipeline
    cv_result: dict
    comparison_df: pd.DataFrame
    feature_importance: pd.DataFrame
    recommended_model_name: str


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

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)


def _build_cv_model_stats(fold_rmse: list[float]) -> dict:
    return {
        "fold_rmse": fold_rmse,
        "mean_rmse": float(np.mean(fold_rmse)),
        "std_rmse": float(np.std(fold_rmse)),
    }


def train_and_compare_models_with_cv(
    X: pd.DataFrame,
    y_log: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    catboost_weight: float = 0.7,
) -> TrainingArtifacts:
    """Train and compare LightGBM/CatBoost plus simple & weighted ensembles.

    Design choices kept intentionally simple for portfolio explainability:
    - Same preprocessing and CV splits for all candidates.
    - Weighted ensemble is only kept as a valid candidate if it beats CatBoost.
    - Final recommendation is data-driven (best mean CV RMSE among valid candidates).
    """
    if not 0.0 < catboost_weight < 1.0:
        raise ValueError("catboost_weight must be between 0 and 1")

    lgbm_weight = 1.0 - catboost_weight
    weighted_model_name = f"weighted_ensemble_cb_{catboost_weight:.2f}_lgbm_{lgbm_weight:.2f}"

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rmse_lgbm: list[float] = []
    fold_rmse_catboost: list[float] = []
    fold_rmse_simple_ensemble: list[float] = []
    fold_rmse_weighted_ensemble: list[float] = []

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

        pred_simple = 0.5 * pred_lgbm + 0.5 * pred_catboost
        pred_weighted = lgbm_weight * pred_lgbm + catboost_weight * pred_catboost

        fold_rmse_lgbm.append(float(np.sqrt(mean_squared_error(y_valid, pred_lgbm))))
        fold_rmse_catboost.append(float(np.sqrt(mean_squared_error(y_valid, pred_catboost))))
        fold_rmse_simple_ensemble.append(float(np.sqrt(mean_squared_error(y_valid, pred_simple))))
        fold_rmse_weighted_ensemble.append(float(np.sqrt(mean_squared_error(y_valid, pred_weighted))))

    models = {
        "lightgbm": _build_cv_model_stats(fold_rmse_lgbm),
        "catboost": _build_cv_model_stats(fold_rmse_catboost),
        "simple_average_ensemble": _build_cv_model_stats(fold_rmse_simple_ensemble),
        weighted_model_name: _build_cv_model_stats(fold_rmse_weighted_ensemble),
    }

    # Keep weighted candidate only when it improves over CatBoost.
    include_weighted_candidate = (
        models[weighted_model_name]["mean_rmse"] < models["catboost"]["mean_rmse"]
    )

    candidate_names = ["lightgbm", "catboost", "simple_average_ensemble"]
    if include_weighted_candidate:
        candidate_names.append(weighted_model_name)

    recommended_model_name = min(candidate_names, key=lambda name: models[name]["mean_rmse"])

    comparison_rows = []
    for model_name, stats in models.items():
        comparison_rows.append(
            {
                "model": model_name,
                "mean_rmse": stats["mean_rmse"],
                "std_rmse": stats["std_rmse"],
                "is_candidate": model_name in candidate_names,
                "is_recommended": model_name == recommended_model_name,
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("mean_rmse", ascending=True)

    # Retrain base models on full data for final inference.
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

    # Feature importance from LightGBM for stable, interview-friendly interpretation.
    feature_importance = _extract_feature_importance(lgbm_full)

    cv_result = {
        "metric": "rmse_on_log_target",
        "n_splits": n_splits,
        "weighted_ensemble": {
            "catboost_weight": catboost_weight,
            "lightgbm_weight": lgbm_weight,
            "included_as_candidate": include_weighted_candidate,
            "inclusion_rule": "include only if weighted mean RMSE < CatBoost mean RMSE",
        },
        "models": models,
        "candidate_models": candidate_names,
        "recommended_final_model": recommended_model_name,
        "recommendation_reason": (
            "lowest mean CV RMSE among valid candidates; "
            "weighted ensemble kept only when it beats CatBoost"
        ),
    }

    return TrainingArtifacts(
        lgbm_model=lgbm_full,
        catboost_model=catboost_full,
        cv_result=cv_result,
        comparison_df=comparison_df,
        feature_importance=feature_importance,
        recommended_model_name=recommended_model_name,
    )
