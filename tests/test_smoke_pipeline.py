from __future__ import annotations

import subprocess
import sys

import pytest

from src.paths import (
    FEATURES_DIR,
    HISTORICAL_FEATURES_PATH,
    LEADERBOARD_BACKTEST_PATH,
    LEADERBOARD_PREDICTIONS_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)


EXPECTED_FEATURE_FILES = [
    FEATURES_DIR / "historical_results_clean.parquet",
    FEATURES_DIR / "historical_stats_clean.parquet",
    HISTORICAL_FEATURES_PATH,
]

EXPECTED_MODEL_FILES = [
    MODELS_DIR / "xgb_round1_baseline.joblib",
    MODELS_DIR / "xgb_round2_baseline.joblib",
]

EXPECTED_PREDICTION_FILES = [
    LEADERBOARD_PREDICTIONS_PATH,
    LEADERBOARD_BACKTEST_PATH,
]


def run_module(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"{module_name} failed.\n\n"
        f"STDOUT:\n{result.stdout}\n\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.smoke
def test_end_to_end_pipeline_smoke() -> None:
    assert RAW_DATA_DIR.exists(), f"Missing raw data directory: {RAW_DATA_DIR}"

    raw_files = [path for path in RAW_DATA_DIR.iterdir() if path.is_file()]
    assert raw_files, f"No raw data files found in: {RAW_DATA_DIR}"

    run_module("pipelines.feature_pipeline")
    for path in EXPECTED_FEATURE_FILES:
        assert path.exists(), f"Missing feature artifact: {path}"

    run_module("pipelines.training_pipeline")
    for path in EXPECTED_MODEL_FILES:
        assert path.exists(), f"Missing model artifact: {path}"

    run_module("pipelines.inference_pipeline")
    for path in EXPECTED_PREDICTION_FILES:
        assert path.exists(), f"Missing prediction artifact: {path}"