# Golf Oracle

Golf Oracle is an MLOps project that predicts PGA Tour tournament outcomes by forecasting player round scores, aggregating tournament results, and generating a predicted leaderboard through a lightweight Streamlit UI.

The project is designed as a modular, reproducible machine learning system with separate feature, training, inference, and UI stages.

---

## Project Goal

The goal of Golf Oracle is to build a production-style ML workflow for golf tournament prediction.

The system is being developed in a round-by-round way:

- predict **Round 1** from historical pre-tournament data
- predict **Round 2** using historical data plus Round 1 information
- later estimate the **cut line**
- later predict **Round 3** and **Round 4**
- aggregate predicted scores into a projected tournament leaderboard

---

## Current Scope

The current working baseline supports:

- **Round 1 prediction**
- **Round 2 live-style prediction** using `predicted_round1`
- **Round 2 backtest evaluation** using `actual_round1`
- separate prediction and backtest artifacts
- Streamlit visualization of precomputed outputs
- MLflow tracking for training runs
- Docker-based reproducible execution
- unit tests and GitHub Actions CI

Not yet implemented:

- dynamic cut logic
- Round 3 prediction
- Round 4 prediction
- external live golf/weather API integration
- stats dataset enrichment in the training baseline

---

## Architecture

Golf Oracle is structured as a modular pipeline system:

- **Feature pipeline**  
  Cleans raw historical data and builds leakage-safe historical features

- **Training pipeline**  
  Trains baseline XGBoost models for Round 1 and Round 2 and logs results to MLflow

- **Inference pipeline**  
  Builds tournament-level predictions for a selected event and generates leaderboard artifacts

- **Streamlit UI**  
  Displays precomputed prediction and backtest outputs

- **MLflow tracking**  
  Stores parameters, metrics, and model artifacts for training runs

- **GitHub Actions automation**  
  Runs tests on push and pull request

- **Dockerized execution**  
  Supports reproducible local pipeline and UI runs inside containers

---

## Data Sources

### Current baseline
- historical PGA tournament results dataset
- historical PGA statistics dataset

### Current usage
- the **results dataset** is used for the active baseline
- the **stats dataset** is cleaned and stored separately, but is **not yet merged** into the training feature base

### Planned later
- live golf API data
- weather data

---

## Repository Structure

```text
golf-oracle/
│
├── pipelines/
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   └── inference_pipeline.py
│
├── src/
│   ├── config.py
│   └── paths.py
│
├── ui/
│   └── app.py
│
├── tests/
│   ├── test_feature_logic.py
│   ├── test_training_pipeline.py
│   └── test_inference_pipeline.py
│
├── data/
│   └── raw/                  # raw Kaggle files (gitignored)
│
├── features/
│   ├── .gitkeep
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
│   └── xgb_round2_baseline.joblib
│
├── mlruns/                   # local MLflow output (gitignored)
│
├── .github/workflows/
│   └── tests.yml
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
````

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

## Project Workflow

The intended baseline workflow is:

1. run the feature pipeline
2. run the training pipeline
3. run the inference pipeline
4. launch the Streamlit UI
5. inspect MLflow runs if needed

---

## Run Pipelines

### Feature pipeline

Builds cleaned historical files and the historical feature store.

```bash
python -m pipelines.feature_pipeline
```

### Training pipeline

Trains the Round 1 and Round 2 baseline models, saves model files, and logs runs to MLflow.

```bash
python -m pipelines.training_pipeline
```

### Inference pipeline

Loads trained models, selects a tournament field, generates predictions, and saves output artifacts.

```bash
python -m pipelines.inference_pipeline
```

---

## Run UI

Launch the Streamlit app:

```bash
streamlit run ui/app.py
```

The UI is read-only and loads precomputed artifacts from the `predictions/` folder.

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

* round name
* target column
* selected feature list
* train/test row counts
* train/test date ranges
* MAE
* RMSE
* saved model artifact
* MLflow model package

---

## Docker Usage

The project also supports Docker-based local execution.

### Build and run with Docker Compose

```bash
docker compose up --build
```

Depending on your compose setup, you can also run pipelines or the UI separately.

### Typical Docker-supported workflows

* run feature pipeline in container
* run training pipeline in container
* run inference pipeline in container
* run Streamlit UI in container
* run pytest in container

Docker validation has already been completed for the current baseline workflow. 

---

## Generated Artifacts

The main generated outputs are:

### Features

* `features/historical_results_clean.parquet`
* `features/historical_stats_clean.parquet`
* `features/historical_features.parquet`

### Models

* `models/xgb_round1_baseline.joblib`
* `models/xgb_round2_baseline.joblib`

### Predictions

* `predictions/leaderboard_predictions.parquet`
* `predictions/leaderboard_backtest.parquet`

### MLflow

* `mlruns/`

---

## Artifact Policy

Generated artifacts are treated separately from source code.

Tracked in Git:

* source code
* configuration files
* Docker files
* test files
* CI workflow
* documentation

Not tracked in Git:

* raw datasets
* generated parquet files
* trained model binaries
* local MLflow tracking outputs

The UI reads finalized artifact paths defined in `src/paths.py`. 

---

## Current Baseline Models

### Round 1 model

* target: `round1`
* feature type: historical pre-tournament baseline features
* artifact: `models/xgb_round1_baseline.joblib`

### Round 2 model

* target: `round2`
* feature type: historical baseline features + `round1`
* artifact: `models/xgb_round2_baseline.joblib`

Round 2 supports two modes:

* **live mode**: uses `predicted_round1`
* **backtest mode**: uses `actual_round1`

---

## Testing

Run all tests with:

```bash
pytest -v
```

Current test coverage includes:

* feature leakage checks
* shifted historical feature logic
* time-based split validation
* target/metadata exclusion from model input
* inference field selection
* ranking/output checks
* live vs backtest Round 2 behavior
* invalid inference mode handling

GitHub Actions runs the test suite automatically on push and pull request. 

---

## UI Pages

The Streamlit UI is intended to show:

### Leaderboard view

* predicted Round 1 and Round 2 results
* predicted total through Round 2
* predicted ranking

### Backtest view

* predicted vs actual Round 1 and Round 2
* total through Round 2
* player-level error inspection

### Model info view

* baseline model information
* training metrics
* event backtest metrics
* feature importance (planned / expandable)

---

## Important Design Decisions

### No heavy inference inside Streamlit

The UI only reads precomputed outputs.
Inference is handled in the pipeline layer.

### Leakage prevention is critical

Only information available before the predicted round may be used in live prediction logic.

### Round-by-round modeling is required

A single tournament-total target was rejected because historical totals mix missed cuts and full four-round results.

### API calls must be cached later

External API usage will be added only after the baseline remains stable.

---

## Current Limitations

The current system is still a baseline and has several important limitations:

* no dynamic cut logic yet
* no Round 3 / Round 4 models yet
* no live external API integration yet
* stats dataset enrichment is not yet merged into training
* new players with no usable history are dropped
* current predictions are still conservative and clustered around central score ranges
* MLflow is currently local only

These items are planned next according to the project roadmap. 

---

## Next Steps

The current next development priorities are:

1. update and finalize the README
2. add a simple end-to-end smoke test
3. design dynamic cut logic
4. extend the system to Round 3 and Round 4
5. integrate external live data only after the baseline is stable

These priorities follow the current project plan. 

---

## Tech Stack

* **Version Control:** GitHub
* **Tracking:** MLflow
* **Modeling:** XGBoost
* **Testing:** pytest
* **Automation:** GitHub Actions
* **Containerization:** Docker
* **Frontend:** Streamlit
* **Deployment target:** Hugging Face Spaces

---

## Success Criteria

The target end state of the project includes:

* modular pipeline design
* reproducible workflow
* Docker-based execution
* leakage-safe training flow
* tracked experiments in MLflow
* tested pipeline logic
* precomputed UI outputs
* clean repository structure
* public GitHub repository
* deployment-ready lightweight interface
