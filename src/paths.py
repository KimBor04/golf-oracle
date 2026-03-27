from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

FEATURES_DIR = PROJECT_ROOT / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
TESTS_DIR = PROJECT_ROOT / "tests"
UI_DIR = PROJECT_ROOT / "ui"

HISTORICAL_FEATURES_PATH = FEATURES_DIR / "historical_features.parquet"
LIVE_FEATURES_PATH = FEATURES_DIR / "live_features.parquet"
LEADERBOARD_PATH = PREDICTIONS_DIR / "leaderboard.parquet"