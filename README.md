# Golf Oracle

Golf Oracle is an MLOps project that predicts PGA Tour tournament outcomes by forecasting player round scores, aggregating tournament totals, and generating a predicted leaderboard.

## Project Goal

The system predicts:
- Round-by-round player strokes
- Dynamic tournament cut outcomes
- Final projected leaderboard

## Architecture

- Feature pipeline
- Training pipeline
- Inference pipeline
- Streamlit UI
- MLflow tracking
- GitHub Actions automation
- Dockerized execution

## Repository Structure

- `pipelines/` orchestration entrypoints
- `src/` reusable implementation logic
- `ui/` Streamlit frontend
- `tests/` unit tests
- `features/` feature store outputs
- `predictions/` generated leaderboard outputs

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## Run pipelines

```bash
python pipelines/feature_pipeline.py
python pipelines/training_pipeline.py
python pipelines/inference_pipeline.py
```

## Run UI

```bash
streamlit run ui/app.py
```

