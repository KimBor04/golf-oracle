from pathlib import Path
import joblib

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = PROJECT_ROOT / "features" / "historical_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "xgb_round1_baseline.joblib"


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    keep_cols = [
        "season",
        "round1",
        "start",
        "tournament",
        "name",
        "player_name_clean",
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
    ]

    df = df[keep_cols].copy()

    df["round1"] = pd.to_numeric(df["round1"], errors="coerce")
    df = df[df["round1"].notna()].copy()
    df = df[df["prev_tournament_avg_score"].notna()].copy()

    feature_cols = [
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
    ]

    X = df[feature_cols].copy()
    y = df["round1"].copy()
    meta = df[["start", "tournament", "name", "player_name_clean", "round1"]].copy()

    return X, y, meta


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.2,
):
    order = np.argsort(meta["start"].values)

    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    split_idx = int(len(X) * (1 - test_size))

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


def evaluate_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    return {
        "mae": mae,
        "rmse": rmse,
        "predictions": preds,
    }


def print_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    meta_test: pd.DataFrame,
    results: dict,
) -> None:
    print("\n=== TRAINING SUMMARY ===")
    print("Train rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Features:", X_train.columns.tolist())

    print("\nTest date range:")
    print(meta_test["start"].min(), "to", meta_test["start"].max())

    print("\nMetrics:")
    print(f"MAE:  {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")

    preview = meta_test.copy()
    preview["prediction_round1"] = results["predictions"]
    print("\nSample predictions:")
    print(preview.head(10))


def main() -> None:
    ensure_directories()

    df = load_data()
    X, y, meta = prepare_training_data(df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = time_split(X, y, meta)

    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print_summary(X_train, X_test, meta_test, results)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()