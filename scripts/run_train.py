from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import ID_COL, load_train_test, save_submission, split_features_target
from src.evaluate import save_cv_result, save_feature_importance
from src.features import add_domain_features
from src.train import train_ensemble_with_cv


def main() -> None:
    train_df, test_df = load_train_test("data/raw")

    # Add deterministic domain features to both train and test BEFORE encoding.
    train_df = add_domain_features(train_df)
    test_df = add_domain_features(test_df)

    X_train, y_train = split_features_target(train_df)
    y_log = np.log1p(y_train)

    artifacts = train_ensemble_with_cv(
        X_train,
        y_log,
        n_splits=5,
        random_state=42,
    )

    pred_lgbm_log = artifacts.lgbm_model.predict(test_df)
    pred_catboost_log = artifacts.catboost_model.predict(test_df)

    # Simple average ensemble prediction.
    pred_ensemble_log = 0.5 * pred_lgbm_log + 0.5 * pred_catboost_log
    test_pred = np.expm1(pred_ensemble_log)

    save_cv_result(artifacts.cv_result, "outputs/cv_result.json")
    save_feature_importance(artifacts.feature_importance, "outputs/feature_importance.csv")
    save_submission(test_df[ID_COL], test_pred, Path("data/submissions/submission.csv"))

    print("Training complete.")
    print("Saved: outputs/cv_result.json")
    print("Saved: outputs/feature_importance.csv")
    print("Saved: data/submissions/submission.csv")


if __name__ == "__main__":
    main()
