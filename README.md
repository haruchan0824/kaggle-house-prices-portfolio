# Kaggle House Prices Portfolio Pipeline

An interview-focused tabular regression project for the
[Kaggle House Prices competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
It demonstrates reproducible training, explainable feature engineering, model comparison,
Kaggle submissions, and a containerized inference API.

## Problem and pipeline

The task is to predict `SalePrice`. Training uses `log1p(SalePrice)` and evaluates
5-fold cross-validation RMSE on that log target; submissions and API responses use
`expm1` to return prices. The pipeline is:

```text
Kaggle CSVs -> domain features -> train-only cleanup -> imputation/one-hot encoding
            -> LightGBM + CatBoost -> CV comparison -> submissions + API artifact
```

`SalePrice` and identifier `Id` are excluded from model features. Near-constant
columns are selected from training statistics only, then the same retained schema is
applied to test and inference data.

## Feature engineering

The shared feature function creates `TotalSF`, `HouseAge`, `RemodAge`, `TotalBath`,
`HasGarage`, `HasBsmt`, `HasFireplace`, and `OverallQual_x_GrLivArea`. These encode
size, age, remodeling, amenities, and a quality/area interaction without opaque methods.

## Models and results

Training compares LightGBM, CatBoost, a 50/50 simple average, and a 70% CatBoost /
30% LightGBM weighted average. The model with the lowest mean local CV RMSE is the
recommendation; an ensemble is not forced to win.

Previously recorded 5-fold results were:

| Model | Mean log-RMSE | Std |
|---|---:|---:|
| LightGBM | 0.13305 | 0.01931 |
| CatBoost | **0.12371** | 0.01778 |
| 50/50 ensemble | 0.12516 | 0.01881 |
| 70% CatBoost / 30% LightGBM | 0.12377 | 0.01848 |

CatBoost was strongest in the verified local CV run. Local CV remains the primary selection
criterion. No verifiable Kaggle Public Score record is committed, so this repository
does not claim a leaderboard result.

## Reproduce training

Use Python 3.11 or 3.12. Kaggle data is intentionally not distributed in this repository.

```bash
python -m venv .venv
pip install -r requirements.txt
# Place train.csv and test.csv under data/raw/
python -m scripts.run_train
```

Training writes CV and importance files under `outputs/`, four directly submittable
files under `data/submissions/`, and `artifacts/model.joblib` for inference:

- `submission_lgbm.csv`
- `submission_catboost.csv`
- `submission_simple_ensemble.csv`
- `submission_weighted_ensemble.csv`

Generated data, outputs, and model artifacts are ignored by Git.

## Inference API

After training:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"OverallQual":7,"GrLivArea":1500,"YearBuilt":2005,"YrSold":2010}'
```

`GET /health` reports model availability. `POST /predict` accepts one JSON object of
House Prices fields and returns a numeric USD prediction. A missing artifact produces
a clear `503` response.

## Docker and AWS

Build only after training so the model is embedded in the image:

```bash
docker build -t house-prices-api .
docker run --rm -p 8000:8000 house-prices-api
```

The image listens on `0.0.0.0` and honors `PORT`. AWS deployment is **prepared, not
completed**. The minimal target is ECR -> ECS/Fargate, with an optional ALB. See
[the AWS deployment runbook](docs/aws_deployment.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The smoke suite checks features and schema cleanup, submission validity, and API
health/prediction behavior.

## Repository structure

```text
app/          FastAPI entrypoint
artifacts/    locally generated inference model
docs/         AWS runbook and interview material
notebooks/    historical notebook reference (not authoritative)
scripts/      training entrypoint
src/          data, features, evaluation, and model training
tests/        focused smoke tests
```

## Limitations and future work

- Re-run training to reproduce the committed historical scores and generate the API artifact.
- Deploy the verified image manually to ECR/ECS and record the endpoint evidence.
- The API intentionally accepts the competition's sparse raw feature shape; production
  use would require a stricter versioned request contract and monitoring.
- The notebook is retained as project history; `python -m scripts.run_train` is authoritative.
