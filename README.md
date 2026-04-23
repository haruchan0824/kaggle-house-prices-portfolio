# Kaggle House Prices — Portfolio Pipeline (Entry-Level ML Engineer)

A clean, interview-friendly tabular ML project based on the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

> **Project intent:** demonstrate reproducibility, explainability, and iterative improvement — not leaderboard chasing.

## Overview

This repository shows a practical regression workflow for structured data:

- clear preprocessing
- modular training code
- 5-fold cross-validation
- artifact generation for reproducibility

It is designed to be easy to explain in interviews for entry-level AI/ML/Data Engineer roles.

## Problem

- **Task:** predict `SalePrice` from tabular features
- **Target transform:** `log1p(SalePrice)` during training, `expm1` at prediction output
- **Validation metric:** RMSE on log-transformed target

## Iterative Project Flow

This project is structured as an iterative pipeline:

1. **Baseline:** LightGBM with simple, robust preprocessing
2. **Feature engineering:** add small domain-driven features that are easy to explain
3. **Lightweight tuning:** limited Optuna search for key hyperparameters only
4. **Simple ensemble:** LightGBM + CatBoost 50/50 average (no stacking)

Important: more complexity did **not** automatically produce the best model in this project.

## Models

- LightGBM Regressor (`LGBMRegressor`)
- CatBoost Regressor (`CatBoostRegressor`)
- Simple average ensemble (`0.5 * LightGBM + 0.5 * CatBoost`)

## Preprocessing

- Numeric: median imputation
- Categorical: fill missing with `"Missing"`
- Encoding: one-hot encoding (`handle_unknown="ignore"`)
- Target: `log1p(SalePrice)`

## Added Domain Features

- `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- `HouseAge = YrSold - YearBuilt`
- `RemodAge = YrSold - YearRemodAdd`
- `TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`
- `HasGarage = (GarageArea > 0)`
- `HasBsmt = (TotalBsmtSF > 0)`
- `HasFireplace = (Fireplaces > 0)`
- `OverallQual_x_GrLivArea = OverallQual * GrLivArea`

These features were chosen because they are intuitive (size, age, remodeling, amenities, quality-area interaction) and easy to justify in interviews.

## Results (5-fold CV)

| Model | Mean RMSE | Std RMSE |
|---|---:|---:|
| LightGBM | 0.13323640336408846 | 0.01783933807588219 |
| CatBoost | **0.12380264917734725** | 0.017687264357621062 |
| Ensemble (LGBM + CatBoost average) | 0.12537854076497493 | 0.018206999267681857 |

### Interpretation

- **CatBoost was the best-performing model** in this run.
- The simple ensemble improved over **LightGBM alone**, but **did not beat CatBoost**.
- This is a useful portfolio lesson: a more complex setup is not always the top performer.

## Feature Importance Note

Engineered features such as `OverallQual_x_GrLivArea` and `TotalSF` appear useful in importance analysis.

At the same time, high importance for **`Id`** should be treated as a warning signal, not a success: it likely indicates a feature-selection issue and should be fixed in the next iteration (e.g., explicitly dropping identifier columns).

## Outputs

Running training saves:

- `outputs/cv_result.json`
- `outputs/feature_importance.csv`
- `data/submissions/submission.csv`

## Repository Structure

```text
kaggle-house-prices-portfolio/
├── data/
│   ├── raw/                 # Kaggle CSVs (not committed)
│   └── submissions/         # submission.csv
├── notebooks/
│   └── train.ipynb          # original notebook reference
├── outputs/
├── reports/
│   ├── summary.md
│   └── interview_qa.md
├── scripts/
│   └── run_train.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

2. Put Kaggle files in `data/raw/`

   - `train.csv`
   - `test.csv`

3. Train and generate outputs

   ```bash
   python scripts/run_train.py
   ```

## Lessons Learned

- Clean modular design is valuable for portfolio clarity.
- Small, explainable feature engineering can improve baseline quality.
- CV mean + std gives better evidence than a single split.
- Model complexity should be judged by measured results, not assumptions.

## Next Steps

- Explicitly remove `Id` (and similar identifier leakage risks) from training features.
- Continue limited, controlled tuning rather than broad brute-force search.
- Add experiment tracking + simple tests/CI for stronger engineering quality.
