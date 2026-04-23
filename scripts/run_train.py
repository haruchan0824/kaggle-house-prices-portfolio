from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import ID_COL, load_train_test, save_submission, split_features_target
from src.evaluate import save_cv_result, save_feature_importance, save_model_comparison
from src.features import add_domain_features, cleanup_feature_columns
from src.train import train_and_compare_models_with_cv


def main() -> None:
    train_df, test_df = load_train_test("data/raw")

    # Keep submission IDs before cleanup removes identifier columns.
    test_ids = test_df[ID_COL].copy()

    # Add deterministic domain features to both train and test BEFORE encoding.
    train_df = add_domain_features(train_df)
    test_df = add_domain_features(test_df)

    X_train, y_train = split_features_target(train_df)

    # Lightweight cleanup: remove non-predictive identifier and near-constant columns.
    X_train, X_test, dropped_cols = cleanup_feature_columns(
        X_train,
        test_df,
        remove_id=True,
        near_constant_threshold=0.995,
    )

    y_log = np.log1p(y_train)

    artifacts = train_and_compare_models_with_cv(
        X_train,
        y_log,
        n_splits=5,
        random_state=42,
        catboost_weight=0.7,
    )

    pred_lgbm_log = artifacts.lgbm_model.predict(X_test)
    pred_catboost_log = artifacts.catboost_model.predict(X_test)

    # Build candidate predictions and choose exactly the recommended final model.
    candidate_predictions = {
        "lightgbm": pred_lgbm_log,
        "catboost": pred_catboost_log,
        "simple_average_ensemble": 0.5 * pred_lgbm_log + 0.5 * pred_catboost_log,
        "weighted_ensemble_cb_0.70_lgbm_0.30": 0.3 * pred_lgbm_log + 0.7 * pred_catboost_log,
    }
    final_pred_log = candidate_predictions[artifacts.recommended_model_name]
    test_pred = np.expm1(final_pred_log)

    save_cv_result(artifacts.cv_result, "outputs/cv_result.json")
    save_model_comparison(artifacts.comparison_df, "outputs/model_comparison.csv")
    save_feature_importance(artifacts.feature_importance, "outputs/feature_importance.csv")
    save_submission(test_ids, test_pred, Path("data/submissions/submission.csv"))

    print("Training complete.")
    print(f"Dropped columns: {dropped_cols}")
    print(f"Recommended final model: {artifacts.recommended_model_name}")
    print("Saved: outputs/cv_result.json")
    print("Saved: outputs/model_comparison.csv")
    print("Saved: outputs/feature_importance.csv")
    print("Saved: data/submissions/submission.csv")


if __name__ == "__main__":
    main()
