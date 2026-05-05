from pathlib import Path
import joblib

import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import (
    HISTORICAL_FEATURES_PATH,
    MODELS_DIR,
)

FEATURES_PATH = HISTORICAL_FEATURES_PATH

ROUND1_MODEL_PATH = MODELS_DIR / "xgb_round1_baseline.joblib"
ROUND2_MODEL_PATH = MODELS_DIR / "xgb_round2_baseline.joblib"
ROUND3_MODEL_PATH = MODELS_DIR / "xgb_round3_baseline.joblib"
ROUND4_MODEL_PATH = MODELS_DIR / "xgb_round4_baseline.joblib"

MLFLOW_EXPERIMENT_NAME = "golf-oracle-baselines"

BASE_FEATURE_COLUMNS = [
    "season",
    "prev_tournament_avg_score",
    "prev_tournament_total",
    "prev_tournament_made_cut",
    "prev_tournament_earnings",
    "rolling_avg_last_3",
    "rolling_avg_last_5",
    "rolling_total_last_3",
    "made_cut_rate_last_5",
    "form_index_last_3",
    "career_tournament_count",
    "days_since_last_tournament",
    "tournaments_last_30",
    "tournaments_last_60",
    "tournaments_last_90",
    "made_cut_streak",
    "missed_cut_streak",
    "round_std_last_5",
    "round_std_last_10",
    "score_range_last_5",
    "best_round_last_10",
    "worst_round_last_10",
    "best_total_last_10",
    "worst_total_last_10",
    "missed_cut_rate_last_10",
]

ROUND_FEATURE_CONFIG = {
    "round1": {
        "target_col": "round1",
        "feature_cols": BASE_FEATURE_COLUMNS,
        "model_path": ROUND1_MODEL_PATH,
        "calibration_alpha": 1.8,
    },
    "round2": {
        "target_col": "round2",
        "feature_cols": BASE_FEATURE_COLUMNS + ["round1"],
        "model_path": ROUND2_MODEL_PATH,
        "calibration_alpha": 1.6,
    },
    "round3": {
        "target_col": "round3",
        "feature_cols": BASE_FEATURE_COLUMNS + ["round1", "round2"],
        "model_path": ROUND3_MODEL_PATH,
        "calibration_alpha": 1.5,
    },
    "round4": {
        "target_col": "round4",
        "feature_cols": BASE_FEATURE_COLUMNS + ["round1", "round2", "round3"],
        "model_path": ROUND4_MODEL_PATH,
        "calibration_alpha": 1.4,
    },
}

ROUND_SCORE_MIN = 55.0
ROUND_SCORE_MAX = 95.0


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")

    if df["start"].isna().any():
        raise ValueError("Some 'start' values could not be parsed as datetimes.")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    round_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if round_name not in ROUND_FEATURE_CONFIG:
        raise ValueError(f"Unsupported round_name: {round_name}")

    config = ROUND_FEATURE_CONFIG[round_name]
    target_col = config["target_col"]
    feature_cols = config["feature_cols"]

    required_cols = {
        "start",
        "tournament",
        "name",
        "player_name_clean",
        target_col,
        *feature_cols,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {round_name}: {sorted(missing)}")

    keep_cols = [
        "start",
        "tournament",
        "name",
        "player_name_clean",
        target_col,
        *feature_cols,
    ]

    df = df[keep_cols].copy()

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].notna()].copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=feature_cols).copy()

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    meta = df[["start", "tournament", "name", "player_name_clean", target_col]].copy()

    return X, y, meta


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.2,
):
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    if len(X) == 0:
        raise ValueError("Cannot split empty training data.")

    order = np.argsort(meta["start"].values)

    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    split_idx = int(len(X) * (1 - test_size))

    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError(
            f"Invalid split_idx={split_idx} for {len(X)} rows. "
            "Adjust test_size or provide more data."
        )

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    meta_train = meta.iloc[:split_idx].copy()
    meta_test = meta.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def calibrate_predictions(
    preds: np.ndarray,
    alpha: float,
    min_score: float = ROUND_SCORE_MIN,
    max_score: float = ROUND_SCORE_MAX,
) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    pred_mean = preds.mean()

    calibrated = pred_mean + alpha * (preds - pred_mean)
    calibrated = np.clip(calibrated, min_score, max_score)

    return calibrated


def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    round_name: str,
) -> dict:
    raw_preds = model.predict(X_test)

    alpha = ROUND_FEATURE_CONFIG[round_name]["calibration_alpha"]
    calibrated_preds = calibrate_predictions(raw_preds, alpha=alpha)

    mae = mean_absolute_error(y_test, calibrated_preds)
    rmse = mean_squared_error(y_test, calibrated_preds) ** 0.5

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "predictions": calibrated_preds,
        "raw_predictions": raw_preds,
        "calibration_alpha": float(alpha),
    }


def log_run_to_mlflow(
    round_name: str,
    model: XGBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    results: dict,
    model_path: Path,
) -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"{round_name}_baseline"):
        mlflow.log_param("round_name", round_name)
        mlflow.log_param("model_type", "XGBRegressor")
        mlflow.log_param("target_col", ROUND_FEATURE_CONFIG[round_name]["target_col"])
        mlflow.log_param("n_features", len(X_train.columns))
        mlflow.log_param("feature_columns", ",".join(X_train.columns.tolist()))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("train_start_min", str(meta_train["start"].min()))
        mlflow.log_param("train_start_max", str(meta_train["start"].max()))
        mlflow.log_param("test_start_min", str(meta_test["start"].min()))
        mlflow.log_param("test_start_max", str(meta_test["start"].max()))
        mlflow.log_param("calibration_alpha", results["calibration_alpha"])
        mlflow.log_param("round_score_min_clip", ROUND_SCORE_MIN)
        mlflow.log_param("round_score_max_clip", ROUND_SCORE_MAX)

        mlflow.log_metric("mae", results["mae"])
        mlflow.log_metric("rmse", results["rmse"])

        mlflow.log_artifact(str(model_path))

        input_example = X_test.head(5).astype(float)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=input_example,
        )


def print_distribution_diagnostics(
    y_test: pd.Series,
    preds: np.ndarray,
    raw_preds: np.ndarray,
) -> None:
    preds_series = pd.Series(preds)
    raw_preds_series = pd.Series(raw_preds)

    print("\nDistribution diagnostics:")
    print(f"Actual mean:          {y_test.mean():.4f}")
    print(f"Raw predicted mean:   {raw_preds_series.mean():.4f}")
    print(f"Calibrated mean:      {preds_series.mean():.4f}")
    print(f"Actual std:           {y_test.std():.4f}")
    print(f"Raw predicted std:    {raw_preds_series.std():.4f}")
    print(f"Calibrated std:       {preds_series.std():.4f}")
    print(f"Actual min/max:       {y_test.min():.4f} / {y_test.max():.4f}")
    print(f"Raw predicted min/max:{raw_preds_series.min():.4f} / {raw_preds_series.max():.4f}")
    print(f"Calibrated min/max:   {preds_series.min():.4f} / {preds_series.max():.4f}")


def print_summary(
    round_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    meta_test: pd.DataFrame,
    results: dict,
) -> None:
    print(f"\n=== TRAINING SUMMARY ({round_name.upper()}) ===")
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Features:", X_train.columns.tolist())

    print("\nTest date range:")
    print(meta_test["start"].min(), "to", meta_test["start"].max())

    print("\nMetrics:")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"Calibration alpha: {results['calibration_alpha']:.2f}")

    print_distribution_diagnostics(
        y_test=y_test,
        preds=results["predictions"],
        raw_preds=results["raw_predictions"],
    )

    preview = meta_test.copy()
    preview[f"raw_prediction_{round_name}"] = results["raw_predictions"]
    preview[f"prediction_{round_name}"] = results["predictions"]
    print("\nSample predictions:")
    print(preview.head(10))


def run_training_for_round(round_name: str) -> None:
    if round_name not in ROUND_FEATURE_CONFIG:
        raise ValueError(f"Unsupported round_name: {round_name}")

    model_path = ROUND_FEATURE_CONFIG[round_name]["model_path"]

    df = load_data()
    X, y, meta = prepare_training_data(df, round_name)
    X_train, X_test, y_train, y_test, meta_train, meta_test = time_split(X, y, meta)

    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, round_name=round_name)

    print_summary(round_name, X_train, X_test, y_test, meta_test, results)

    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")

    log_run_to_mlflow(
        round_name=round_name,
        model=model,
        X_train=X_train,
        X_test=X_test,
        meta_train=meta_train,
        meta_test=meta_test,
        results=results,
        model_path=model_path,
    )
    print(f"Logged MLflow run for: {round_name}")


def main() -> None:
    ensure_directories()

    run_training_for_round("round1")
    run_training_for_round("round2")
    run_training_for_round("round3")
    run_training_for_round("round4")


if __name__ == "__main__":
    main()