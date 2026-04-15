# Kaggle House Prices — Portfolio Baseline

A clean, interview-ready machine learning project based on the [Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Overview

This repository demonstrates a **simple, production-style tabular ML workflow** using LightGBM.
The focus is not leaderboard optimization; it is code quality, reproducibility, and clear ML fundamentals.

## Goal

Build a baseline regression pipeline that is:

- Easy to read and explain in interviews
- Modular and maintainable
- Reproducible from raw CSV files to Kaggle submission

## Dataset

Competition: **House Prices: Advanced Regression Techniques**

Expected local files:

- `data/raw/train.csv`
- `data/raw/test.csv`

Target variable:

- `SalePrice`

ID column:

- `Id`

## Method

### Model

- **LightGBM Regressor** (`LGBMRegressor`)
- **CatBoost Regressor** (`CatBoostRegressor`)
- Final prediction: simple average of both model predictions

### Preprocessing

- `log1p` transformation on target (`SalePrice`)
- Numeric columns: median imputation
- Categorical columns: fill missing values with `"Missing"`
- One-hot encoding for categorical features (`handle_unknown="ignore"`)
- Simple domain-driven features (added before encoding):
  - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
  - `HouseAge = YrSold - YearBuilt`
  - `RemodAge = YrSold - YearRemodAdd`
  - `TotalBath = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`
  - `HasGarage`, `HasBsmt`, `HasFireplace` (binary flags)
  - `OverallQual_x_GrLivArea = OverallQual * GrLivArea`

Why these features are useful:

- They capture intuitive housing signals (size, age, bathroom capacity, key amenities, and quality-size interaction).
- They are simple to explain in interviews and easy to reproduce.
- They improve feature expressiveness without over-engineering.

### Validation

- 5-fold cross-validation (`KFold`, shuffled, fixed random state)
- Metric tracked in this project: RMSE on log-transformed target

### Simple ensemble rationale

- We train two different tree-based models (LightGBM and CatBoost) and average predictions.
- Different models can make different errors on the same sample, so averaging can reduce variance and improve robustness.
- We intentionally use a plain 50/50 average to keep the method transparent and easy to explain in interviews.
- We intentionally do **not** use stacking or complex blending optimization in this portfolio project.

### Artifacts

Running training generates:

- `outputs/cv_result.json`
- `outputs/feature_importance.csv`
- `data/submissions/submission.csv`

Pipeline order (important):

1. Load train/test CSVs
2. Add domain-driven features (`add_domain_features`)
3. Split target (`SalePrice`) and apply `log1p`
4. Run 5-fold CV for LightGBM, CatBoost, and their 50/50 ensemble
5. Retrain both models on full training data
6. Average test predictions, reverse target with `expm1`, and save artifacts

## Results

This project records fold-level and aggregate CV metrics in:

- `outputs/cv_result.json`

Example fields include:

- `fold_rmse`
- `mean_rmse`
- `std_rmse`

Feature importance rankings are exported to:

- `outputs/feature_importance.csv`

## Repository Structure

```text
kaggle-house-prices-portfolio/
├── data/
│   ├── raw/                 # Kaggle train/test CSVs (not committed)
│   └── submissions/         # Generated submission.csv
├── notebooks/
│   └── train.ipynb          # Original notebook reference
├── outputs/                 # CV + feature importance outputs
├── scripts/
│   └── run_train.py         # End-to-end training entrypoint
├── src/
│   ├── data.py              # Data loading and submission writing
│   ├── evaluate.py          # Saving CV and feature importance artifacts
│   ├── features.py          # Preprocessing pipeline
│   └── train.py             # CV training + final fit
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download Kaggle data and place files in `data/raw/`:

   - `train.csv`
   - `test.csv`

3. Train and generate outputs:

   ```bash
   python scripts/run_train.py
   ```

4. Check generated files:

   - `outputs/cv_result.json`
   - `outputs/feature_importance.csv`
   - `data/submissions/submission.csv`

## Lessons Learned

- A strong baseline with clean structure is often more valuable for portfolios than a heavily tuned notebook.
- Separating data, features, training, and evaluation logic improves readability and maintainability.
- Reproducible outputs (metrics + artifacts) make experimentation and explanation much easier.

## Limitations / Future Work

- Minimal feature engineering by design; performance can improve with richer domain features.
- No experiment tracking framework yet (e.g., MLflow/W&B).
- Ensemble is intentionally simple (equal averaging); more complex methods are possible but intentionally out of scope.
- Could add unit tests and CI checks for stronger engineering rigor.
- Could add model persistence and inference script for deployment-style workflows.
