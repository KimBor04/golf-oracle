# Golf Oracle

Golf Oracle is an MLOps project that predicts PGA Tour tournament outcomes by forecasting player round scores, aggregating tournament totals, and generating a predicted leaderboard through a lightweight Streamlit UI.

The project is designed as a modular, reproducible machine learning system with separate feature, training, inference, API backfill, testing, tracking, and UI stages.

---

## Project Goal

The goal of Golf Oracle is to build a production-style ML workflow for golf tournament prediction.

The system predicts tournaments in a round-by-round way:

1. predict **Round 1** from historical pre-tournament data
2. predict **Round 2** using historical data plus Round 1 information
3. apply tournament-aware **cut logic** after Round 2
4. filter the predicted weekend field
5. predict **Round 3** and **Round 4**
6. aggregate predicted scores into a projected final leaderboard

---

## Current Scope

The current working baseline supports:

- **Round 1 prediction**
- **Round 2 live-style prediction** using `predicted_round1`
- **Round 2 backtest evaluation** using `actual_round1`
- **tournament-aware cut logic** after Round 2
- **Round 3 prediction**
- **Round 4 prediction**
- full predicted tournament leaderboard through Round 4
- separate prediction and backtest artifacts
- Streamlit visualization of precomputed outputs
- MLflow tracking for training runs
- Docker-based reproducible execution
- unit tests and GitHub Actions CI
- dedicated end-to-end smoke test
- cached FreeWebAPI Golf ingestion/backfill
- optional API-field source for inference
- CLI-configurable inference without manual constant edits

Still planned / in progress:

- continued quota-safe 2026 API result backfill
- Open-Meteo weather integration
- stats dataset enrichment in the training baseline
- improved score spread, cut realism, and ranking quality
- Streamlit UI polish for final demo/presentation quality
- deployment to Hugging Face Spaces

---

## Architecture

Golf Oracle is structured as a modular pipeline system:

- **Feature pipeline**  
  Cleans raw historical data and builds leakage-safe historical features.

- **FreeWebAPI backfill pipeline**  
  Imports cached 2026 PGA Tour schedules, completed results, and near-future player fields with strict API call limits.

- **Training pipeline**  
  Trains baseline XGBoost models for Round 1, Round 2, Round 3, and Round 4 and logs results to MLflow.

- **Inference pipeline**  
  Builds tournament-level predictions for a selected event, applies cut logic, predicts the final leaderboard, and saves prediction/backtest artifacts.

- **Streamlit UI**  
  Displays precomputed prediction and backtest outputs. The UI does not run inference and does not call external APIs.

- **MLflow tracking**  
  Stores parameters, metrics, and model artifacts for training runs.

- **GitHub Actions automation**  
  Runs the fast test suite on push and pull request.

- **Dockerized execution**  
  Supports reproducible local pipeline and UI runs inside containers.

```text
Feature Pipeline / API Backfill
        ↓
Feature Store (Parquet)
        ↓
Training Pipeline → MLflow + Model Artifacts
        ↓
Inference Pipeline → Prediction + Backtest Artifacts
        ↓
Streamlit UI
```

---

## Data Sources

### Current baseline

- historical PGA tournament results dataset
- historical PGA statistics dataset
- cached FreeWebAPI Golf data

### Current usage

- the **results dataset** is used for the active training baseline
- the **stats dataset** is cleaned and stored separately, but is **not yet merged** into the training feature base
- FreeWebAPI Golf is used for:
  - schedule lookup
  - completed 2026 result backfill
  - near-future/current player field import
  - optional API-field inference source

### Planned later

- Open-Meteo weather data
- safer stats enrichment after leakage/matching validation

---

## Repository Structure

```text
golf-oracle/
│
├── pipelines/
│   ├── feature_pipeline.py
│   ├── freewebapi_backfill.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── src/
│   ├── config.py
│   ├── freewebapi_golf_client.py
│   └── paths.py
│
├── ui/
│   └── app.py
│
├── tests/
│   ├── test_feature_logic.py
│   ├── test_training_pipeline.py
│   ├── test_inference_pipeline.py
│   ├── test_inference_api_fields.py
│   ├── test_freewebapi_backfill.py
│   └── test_smoke_pipeline.py
│
├── data/
│   └── raw/                  # raw Kaggle files, gitignored
│
├── features/
│   ├── .gitkeep
│   ├── api_cache/            # raw cached API JSON responses, gitignored
│   ├── api_schedule_2026.parquet
│   ├── api_target_tournaments_2026.parquet
│   ├── api_results_2026.parquet
│   ├── api_fields_2026.parquet
│   ├── historical_results_clean.parquet
│   ├── historical_stats_clean.parquet
│   ├── historical_features.parquet
│   └── live_features.parquet
│
├── predictions/
│   ├── .gitkeep
│   ├── leaderboard_predictions.parquet
│   └── leaderboard_backtest.parquet
│
├── models/
│   ├── .gitkeep
│   ├── xgb_round1_baseline.joblib
│   ├── xgb_round2_baseline.joblib
│   ├── xgb_round3_baseline.joblib
│   └── xgb_round4_baseline.joblib
│
├── mlruns/                   # local MLflow output, gitignored
│
├── .github/workflows/
│   └── tests.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── pytest.ini
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

Create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS / WSL

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

FreeWebAPI Golf credentials should be stored locally in `.env` and must not be committed.

Use `.env.example` as the template.

```bash
cp .env.example .env
```

Then fill in the required RapidAPI / FreeWebAPI values in `.env`.

---

## Project Workflow

The intended baseline workflow is:

1. run the feature pipeline
2. run the training pipeline
3. run the inference pipeline
4. launch the Streamlit UI
5. inspect MLflow runs if needed

```bash
python -m pipelines.feature_pipeline
python -m pipelines.training_pipeline
python -m pipelines.inference_pipeline
streamlit run ui/app.py
```

---

## Run Pipelines

### Feature pipeline

Builds cleaned historical files and the historical feature store.

```bash
python -m pipelines.feature_pipeline
```

Optional live leaderboard fetch example:

```bash
python pipelines/feature_pipeline.py --fetch-live --live-tournament "Masters Tournament" --live-year 2024 --live-round 1
```

---

### FreeWebAPI backfill pipeline

The backfill pipeline is designed to be quota-safe and cache-first.

It saves raw API responses under `features/api_cache/` and writes normalized Parquet artifacts to `features/`.

#### Schedule import

```bash
python pipelines/freewebapi_backfill.py --year 2026 --mode schedule --max-api-calls 1
```

#### Completed result backfill

```bash
python pipelines/freewebapi_backfill.py --year 2026 --mode results --max-api-calls 5
```

#### Near-future field import

```bash
python pipelines/freewebapi_backfill.py --year 2026 --mode fields --next-events 3 --max-api-calls 3
```

Field import uses a default 7-day lookahead window and skips fields that are already cached as available.

---

### Training pipeline

Trains the Round 1, Round 2, Round 3, and Round 4 baseline models, saves model files, and logs runs to MLflow.

```bash
python -m pipelines.training_pipeline
```

Current model artifacts:

- `models/xgb_round1_baseline.joblib`
- `models/xgb_round2_baseline.joblib`
- `models/xgb_round3_baseline.joblib`
- `models/xgb_round4_baseline.joblib`

---

### Inference pipeline

The inference pipeline loads trained models, selects a tournament field, generates predictions, applies cut logic, predicts the weekend rounds, and saves output artifacts.

#### Historical backtest inference

Use this for reproducible historical evaluation.

```bash
python pipelines/inference_pipeline.py \
  --field-source historical \
  --target-tournament "Masters Tournament" \
  --target-start-date 2025-04-10
```

#### API-field live-style inference

Use this with cached API field data such as `features/api_fields_2026.parquet`.

```bash
python pipelines/inference_pipeline.py \
  --field-source api_fields \
  --target-tournament "Cadillac Championship" \
  --target-start-date 2026-04-30
```

#### Optional explicit inference mode

```bash
python pipelines/inference_pipeline.py \
  --field-source historical \
  --target-tournament "Masters Tournament" \
  --target-start-date 2025-04-10 \
  --inference-mode live
```

Supported field sources:

- `historical`
- `api_fields`

Supported inference modes:

- `live`
- `backtest`

Important behavior:

- `historical` is the safe default for backtests and reproducibility
- `api_fields` uses cached API field artifacts
- API-field inference can produce prediction artifacts even when actual scores are unavailable
- if actual round scores are unavailable, the pipeline safely skips the backtest artifact

---

## Run UI

Launch the Streamlit app:

```bash
streamlit run ui/app.py
```

The UI is read-only and loads precomputed artifacts from the `predictions/` folder.

The UI must not:

- call external APIs
- train models
- run heavy inference

---

## MLflow Tracking

Training runs are logged locally with MLflow.

### Run training

```bash
python -m pipelines.training_pipeline
```

### Start the MLflow UI

```bash
mlflow ui
```

### Open in browser

```text
http://127.0.0.1:5000
```

Current MLflow logging includes:

- round name
- target column
- selected feature list
- train/test row counts
- train/test date ranges
- MAE
- RMSE
- saved model artifact
- MLflow model package

---

## Docker Usage

The project supports Docker-based local execution.

### Build and run with Docker Compose

```bash
docker compose up --build
```

Typical Docker-supported workflows:

- run feature pipeline in container
- run training pipeline in container
- run inference pipeline in container
- run Streamlit UI in container
- run pytest in container

Docker validation has already been completed for the current baseline workflow.

---

## Generated Artifacts

The main generated outputs are:

### Features

- `features/historical_results_clean.parquet`
- `features/historical_stats_clean.parquet`
- `features/historical_features.parquet`
- `features/live_features.parquet`
- `features/api_schedule_2026.parquet`
- `features/api_target_tournaments_2026.parquet`
- `features/api_results_2026.parquet`
- `features/api_fields_2026.parquet`
- `features/api_cache/`

### Models

- `models/xgb_round1_baseline.joblib`
- `models/xgb_round2_baseline.joblib`
- `models/xgb_round3_baseline.joblib`
- `models/xgb_round4_baseline.joblib`

### Predictions

- `predictions/leaderboard_predictions.parquet`
- `predictions/leaderboard_backtest.parquet`

### MLflow

- `mlruns/`

---

## Artifact Policy

Generated artifacts are treated separately from source code.

Tracked in Git:

- source code
- configuration files
- Docker files
- test files
- CI workflow
- documentation

Not tracked in Git:

- raw datasets
- generated Parquet files
- trained model binaries
- local MLflow tracking outputs
- raw API cache files

The UI reads finalized artifact paths defined in `src/paths.py`.

---

## Current Baseline Models

### Round 1 model

- target: `round1`
- feature type: historical pre-tournament baseline features
- artifact: `models/xgb_round1_baseline.joblib`

### Round 2 model

- target: `round2`
- feature type: historical baseline features + `round1`
- artifact: `models/xgb_round2_baseline.joblib`

### Round 3 model

- target: `round3`
- feature type: historical baseline features + `round1` + `round2`
- artifact: `models/xgb_round3_baseline.joblib`

### Round 4 model

- target: `round4`
- feature type: historical baseline features + `round1` + `round2` + `round3`
- artifact: `models/xgb_round4_baseline.joblib`

Live/backtest source behavior:

- **live mode** uses predicted prior-round values where actual live round values are not yet safely available
- **backtest mode** can use actual prior-round values for historical evaluation
- future live-score integration should only use completed round scores when `live_round_complete == True`

---

## Cut Logic

The project uses tournament-aware cut rules instead of a fixed percentile shortcut.

Current rule baseline:

- Masters: top 50 and ties, plus players within 10 strokes of the leader
- U.S. Open: top 60 and ties
- The Open / Open Championship: top 70 and ties
- PGA Championship: top 70 and ties
- Default PGA Tour events: top 65 and ties

Important note:

The cut is a predicted cut simulation based on predicted Round 1 + Round 2 scores. It is not an official live rules engine.

---

## Testing

### Normal test suite

Run the normal fast test suite with:

```bash
pytest -v -m "not smoke"
```

This is the command used by the default GitHub Actions workflow.

### End-to-end smoke test

Run the dedicated smoke test with:

```bash
pytest -v -m smoke
```

The smoke test runs the full feature, training, and inference pipeline and verifies that expected feature, model, and prediction artifacts are created.

It is intentionally excluded from default CI because it is heavier than the normal unit test suite.

### Current test coverage

Current test coverage includes:

- feature leakage checks
- shifted historical feature logic
- time-based split validation
- target/metadata exclusion from model input
- Round 2 / Round 3 / Round 4 feature validation
- inference field selection
- historical field source behavior
- API field source behavior
- ranking/output checks
- live vs backtest behavior
- invalid inference mode handling
- cut rule behavior
- ties at the cut line
- Masters special cut behavior
- weekend-field filtering
- final leaderboard output shape
- FreeWebAPI backfill cache/budget/lookahead helper behavior
- empty backtest behavior when API-field inference has no actual scores
- end-to-end smoke validation

---

## GitHub Actions CI

The default CI workflow runs on push and pull request.

Current quality gate:

```bash
pytest -v -m "not smoke"
```

The smoke test is kept as a manual reproducibility check.

---

## UI Pages

The Streamlit UI is intended to show:

### Leaderboard view

- predicted final ranking through Round 4
- predicted Round 1 / Round 2 / Round 3 / Round 4 scores
- predicted total
- predicted cut status
- source context used for inference

### Backtest view

- predicted vs actual rounds
- predicted vs actual final total where available
- predicted vs actual ranking comparison
- player-level error inspection

### Model info view

- baseline model information
- training metrics
- event backtest metrics
- feature importance or model diagnostics later

---

## Current Limitations

The current system is still a baseline and has several important limitations:

- stats dataset enrichment is not yet merged into training
- weather data is not yet connected
- new players with no usable history are dropped
- predictions are still conservative and clustered around central score ranges
- cut realism depends strongly on prediction score spread
- MLflow is currently local only
- API-generated 2026 artifacts are useful locally but should not be committed directly to Git
- FreeWebAPI backfill must use conservative request limits

---

## Tech Stack

- **Version Control:** GitHub
- **Tracking:** MLflow
- **Modeling:** XGBoost
- **Testing:** pytest
- **Automation:** GitHub Actions
- **Containerization:** Docker
- **Frontend:** Streamlit
- **API ingestion:** FreeWebAPI Golf via RapidAPI
- **Deployment target:** Hugging Face Spaces
