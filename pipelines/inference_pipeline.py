from __future__ import annotations

import joblib
import pandas as pd

from src.paths import FEATURES_DIR, MODELS_DIR, PREDICTIONS_DIR


MODEL_PATH = MODELS_DIR / "xgb_round1_baseline.joblib"
FEATURES_PATH = FEATURES_DIR / "historical_features.parquet"

PREDICTION_OUTPUT_PATH = PREDICTIONS_DIR / "leaderboard_predictions.parquet"
BACKTEST_OUTPUT_PATH = PREDICTIONS_DIR / "leaderboard_backtest.parquet"

TARGET_TOURNAMENT = "Masters Tournament"
TARGET_START_DATE = "2025-04-10"

FEATURE_COLUMNS = [
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


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_PATH}")

    df = pd.read_parquet(FEATURES_PATH)

    required_columns = {
        "player_name_clean",
        "start",
        "tournament",
        "season",
        "round1",
        *FEATURE_COLUMNS,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in feature file: {sorted(missing)}")

    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")

    if df["start"].isna().any():
        raise ValueError("Some 'start' values could not be parsed as datetimes.")

    return df


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_target_field(df: pd.DataFrame, tournament: str, start_date: str) -> pd.DataFrame:
    target_start = pd.to_datetime(start_date)

    field_df = df[
        (df["tournament"] == tournament) &
        (df["start"] == target_start)
    ].copy()

    if field_df.empty:
        available = (
            df.loc[df["tournament"] == tournament, ["tournament", "start"]]
            .drop_duplicates()
            .sort_values("start")
        )
        raise ValueError(
            f"No rows found for tournament='{tournament}' and start='{start_date}'.\n"
            f"Available dates for this tournament:\n{available.to_string(index=False)}"
        )

    field_df = field_df.sort_values(["player_name_clean"]).drop_duplicates(
        subset=["player_name_clean"],
        keep="first",
    )

    return field_df


def build_pre_tournament_feature_rows(
    df: pd.DataFrame,
    field_df: pd.DataFrame,
    target_start: pd.Timestamp,
) -> pd.DataFrame:
    players = field_df["player_name_clean"].unique().tolist()
    history_df = df[df["start"] < target_start].copy()

    pre_tournament_rows = []

    for player in players:
        player_history = history_df[history_df["player_name_clean"] == player].copy()

        if player_history.empty:
            continue

        latest_row = player_history.sort_values("start").tail(1).copy()

        latest_row["feature_source_start"] = latest_row["start"]
        latest_row["feature_source_tournament"] = latest_row["tournament"]
        latest_row["feature_source_season"] = latest_row["season"]

        pre_tournament_rows.append(latest_row)

    if not pre_tournament_rows:
        raise ValueError("No pre-tournament history found for any player in the selected field.")

    inference_df = pd.concat(pre_tournament_rows, ignore_index=True)

    field_meta = field_df[
        ["player_name_clean", "tournament", "start", "season", "round1"]
    ].copy()

    field_meta = field_meta.rename(
        columns={
            "tournament": "target_tournament",
            "start": "target_start",
            "season": "target_season",
            "round1": "actual_round1",
        }
    )

    inference_df = field_meta.merge(inference_df, on="player_name_clean", how="left")

    missing_history = inference_df[inference_df["feature_source_start"].isna()].copy()
    if not missing_history.empty:
        print("\nWarning: these players have no pre-tournament history and will be dropped:")
        print(missing_history["player_name_clean"].to_string(index=False))

    inference_df = inference_df.dropna(subset=["feature_source_start"]).copy()

    missing_feature_cols = [col for col in FEATURE_COLUMNS if col not in inference_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing feature columns for inference: {missing_feature_cols}")

    inference_df[FEATURE_COLUMNS] = inference_df[FEATURE_COLUMNS].fillna(0)

    if inference_df.empty:
        raise ValueError("All players were dropped because none had usable pre-tournament history.")

    return inference_df


def predict_round1(model, inference_df: pd.DataFrame) -> pd.DataFrame:
    X = inference_df[FEATURE_COLUMNS].copy()

    predictions = inference_df.copy()
    predictions["predicted_round1"] = model.predict(X)
    predictions["abs_error"] = (predictions["predicted_round1"] - predictions["actual_round1"]).abs()

    predictions = predictions.sort_values(
        ["predicted_round1", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    predictions["predicted_rank"] = predictions.index + 1

    actual_rank_df = (
        predictions[["player_name_clean", "actual_round1"]]
        .sort_values(["actual_round1", "player_name_clean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    actual_rank_df["actual_rank"] = actual_rank_df.index + 1

    predictions = predictions.merge(
        actual_rank_df[["player_name_clean", "actual_rank"]],
        on="player_name_clean",
        how="left",
    )

    return predictions


def build_prediction_output(predictions: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "predicted_rank",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
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
        "predicted_round1",
    ]
    return predictions[output_columns].copy()


def build_backtest_output(predictions: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "predicted_rank",
        "actual_rank",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
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
        "predicted_round1",
        "actual_round1",
        "abs_error",
    ]
    return predictions[output_columns].copy()


def save_outputs(prediction_df: pd.DataFrame, backtest_df: pd.DataFrame) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(PREDICTION_OUTPUT_PATH, index=False)
    backtest_df.to_parquet(BACKTEST_OUTPUT_PATH, index=False)


def print_event_summary(backtest_df: pd.DataFrame) -> None:
    event_mae = backtest_df["abs_error"].mean()
    event_rmse = ((backtest_df["predicted_round1"] - backtest_df["actual_round1"]) ** 2).mean() ** 0.5

    print("\n=== EVENT BACKTEST SUMMARY ===")
    print(f"Tournament: {backtest_df['target_tournament'].iloc[0]}")
    print(f"Start date: {backtest_df['target_start'].iloc[0].date()}")
    print(f"Players evaluated: {len(backtest_df)}")
    print(f"Event MAE:  {event_mae:.4f}")
    print(f"Event RMSE: {event_rmse:.4f}")

    print("\n=== TOP 10 PREDICTED ROUND 1 LEADERBOARD ===")
    print(
        backtest_df[
            ["predicted_rank", "player_name_clean", "predicted_round1", "actual_round1", "actual_rank", "abs_error"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n=== TOP 10 ACTUAL ROUND 1 LEADERBOARD ===")
    print(
        backtest_df.sort_values(["actual_rank", "player_name_clean"])[
            ["actual_rank", "player_name_clean", "actual_round1", "predicted_round1", "predicted_rank", "abs_error"]
        ]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    print("Loading historical features...")
    df = load_features()

    print(f"Selecting field for: {TARGET_TOURNAMENT} ({TARGET_START_DATE})")
    field_df = get_target_field(df, TARGET_TOURNAMENT, TARGET_START_DATE)

    print(f"Players in field: {len(field_df)}")

    print("Building pre-tournament feature rows...")
    target_start = pd.to_datetime(TARGET_START_DATE)
    inference_df = build_pre_tournament_feature_rows(df, field_df, target_start)

    print(f"Players with usable history: {len(inference_df)}")

    print("Loading model...")
    model = load_model()

    print("Running predictions...")
    full_predictions_df = predict_round1(model, inference_df)

    prediction_output_df = build_prediction_output(full_predictions_df)
    backtest_output_df = build_backtest_output(full_predictions_df)

    print(f"Saving prediction artifact to: {PREDICTION_OUTPUT_PATH}")
    print(f"Saving backtest artifact to:   {BACKTEST_OUTPUT_PATH}")
    save_outputs(prediction_output_df, backtest_output_df)

    print_event_summary(backtest_output_df)

    print("\nDone.")


if __name__ == "__main__":
    main()