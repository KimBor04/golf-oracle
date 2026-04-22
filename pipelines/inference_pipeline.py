from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.config import get_cut_rule
from src.paths import (
    FEATURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    LEADERBOARD_PREDICTIONS_PATH,
    LEADERBOARD_BACKTEST_PATH,
)

ROUND1_MODEL_PATH = MODELS_DIR / "xgb_round1_baseline.joblib"
ROUND2_MODEL_PATH = MODELS_DIR / "xgb_round2_baseline.joblib"
ROUND3_MODEL_PATH = MODELS_DIR / "xgb_round3_baseline.joblib"
ROUND4_MODEL_PATH = MODELS_DIR / "xgb_round4_baseline.joblib"

FEATURES_PATH = FEATURES_DIR / "historical_features.parquet"

PREDICTION_OUTPUT_PATH = LEADERBOARD_PREDICTIONS_PATH
BACKTEST_OUTPUT_PATH = LEADERBOARD_BACKTEST_PATH

TARGET_TOURNAMENT = "Masters Tournament"
TARGET_START_DATE = "2025-04-10"
INFERENCE_MODE = "live"  # allowed: "live", "backtest"

ROUND1_FEATURE_COLUMNS = [
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
    "round_std_last_5",
    "round_std_last_10",
    "score_range_last_5",
    "best_round_last_10",
    "worst_round_last_10",
    "best_total_last_10",
    "worst_total_last_10",
    "missed_cut_rate_last_10",
]

ROUND2_FEATURE_COLUMNS = ROUND1_FEATURE_COLUMNS + ["round1"]
ROUND3_FEATURE_COLUMNS = ROUND1_FEATURE_COLUMNS + ["round1", "round2"]
ROUND4_FEATURE_COLUMNS = ROUND1_FEATURE_COLUMNS + ["round1", "round2", "round3"]

ROUND_CALIBRATION_ALPHA = {
    "round1": 1.8,
    "round2": 1.6,
    "round3": 1.5,
    "round4": 1.4,
}

ROUND_SCORE_MIN = 55.0
ROUND_SCORE_MAX = 95.0


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
        "round2",
        "round3",
        "round4",
        *ROUND1_FEATURE_COLUMNS,
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in feature file: {sorted(missing)}")

    df = df.copy()
    df["start"] = pd.to_datetime(df["start"], errors="coerce")

    if df["start"].isna().any():
        raise ValueError("Some 'start' values could not be parsed as datetimes.")

    return df


def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


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


def predict_with_calibration(
    model,
    X: pd.DataFrame,
    round_name: str,
    apply_calibration: bool = True,
) -> np.ndarray:
    raw_preds = model.predict(X)

    if not apply_calibration:
        return raw_preds

    alpha = ROUND_CALIBRATION_ALPHA[round_name]
    return calibrate_predictions(raw_preds, alpha=alpha)


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
        ["player_name_clean", "tournament", "start", "season", "round1", "round2", "round3", "round4"]
    ].copy()

    field_meta = field_meta.rename(
        columns={
            "tournament": "target_tournament",
            "start": "target_start",
            "season": "target_season",
            "round1": "actual_round1",
            "round2": "actual_round2",
            "round3": "actual_round3",
            "round4": "actual_round4",
        }
    )

    inference_df = field_meta.merge(inference_df, on="player_name_clean", how="left")

    missing_history = inference_df[inference_df["feature_source_start"].isna()].copy()
    if not missing_history.empty:
        print("\nWarning: these players have no pre-tournament history and will be dropped:")
        print(missing_history["player_name_clean"].to_string(index=False))

    inference_df = inference_df.dropna(subset=["feature_source_start"]).copy()

    missing_feature_cols = [col for col in ROUND1_FEATURE_COLUMNS if col not in inference_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing feature columns for inference: {missing_feature_cols}")

    inference_df[ROUND1_FEATURE_COLUMNS] = inference_df[ROUND1_FEATURE_COLUMNS].fillna(0)

    if inference_df.empty:
        raise ValueError("All players were dropped because none had usable pre-tournament history.")

    return inference_df


def validate_inference_mode(mode: str) -> None:
    valid_modes = {"live", "backtest"}
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid inference mode: {mode}. Expected one of {sorted(valid_modes)}."
        )


def prepare_round2_features(predictions_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    validate_inference_mode(mode)
    predictions = predictions_df.copy()

    if mode == "backtest":
        predictions["round1"] = predictions["actual_round1"]
        predictions["round1_input_source"] = "actual_round1"
    else:
        predictions["round1"] = predictions["predicted_round1"]
        predictions["round1_input_source"] = "predicted_round1"

    predictions["inference_mode"] = mode
    return predictions


def prepare_round3_features(predictions_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    validate_inference_mode(mode)
    predictions = predictions_df.copy()

    if mode == "backtest":
        predictions["round1"] = predictions["actual_round1"]
        predictions["round2"] = predictions["actual_round2"]
        predictions["round3_input_source"] = "actual_round1_actual_round2"
    else:
        predictions["round1"] = predictions["predicted_round1"]
        predictions["round2"] = predictions["predicted_round2"]
        predictions["round3_input_source"] = "predicted_round1_predicted_round2"

    predictions["inference_mode"] = mode
    return predictions


def prepare_round4_features(predictions_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    validate_inference_mode(mode)
    predictions = predictions_df.copy()

    if mode == "backtest":
        predictions["round1"] = predictions["actual_round1"]
        predictions["round2"] = predictions["actual_round2"]
        predictions["round3"] = predictions["actual_round3"]
        predictions["round4_input_source"] = "actual_round1_actual_round2_actual_round3"
    else:
        predictions["round1"] = predictions["predicted_round1"]
        predictions["round2"] = predictions["predicted_round2"]
        predictions["round3"] = predictions["predicted_round3"]
        predictions["round4_input_source"] = "predicted_round1_predicted_round2_predicted_round3"

    predictions["inference_mode"] = mode
    return predictions


def predict_round1(
    round1_model,
    inference_df: pd.DataFrame,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    X = inference_df[ROUND1_FEATURE_COLUMNS].copy()

    predictions = inference_df.copy()
    predictions["predicted_round1"] = predict_with_calibration(
        round1_model,
        X,
        round_name="round1",
        apply_calibration=apply_calibration,
    )
    predictions["abs_error_round1"] = (
        predictions["predicted_round1"] - predictions["actual_round1"]
    ).abs()

    predictions = predictions.sort_values(
        ["predicted_round1", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    predictions["predicted_rank_round1"] = predictions.index + 1

    actual_rank_df = (
        predictions[["player_name_clean", "actual_round1"]]
        .sort_values(["actual_round1", "player_name_clean"], ascending=[True, True])
        .reset_index(drop=True)
    )
    actual_rank_df["actual_rank_round1"] = actual_rank_df.index + 1

    predictions = predictions.merge(
        actual_rank_df[["player_name_clean", "actual_rank_round1"]],
        on="player_name_clean",
        how="left",
    )

    return predictions


def predict_round2(
    round2_model,
    predictions_df: pd.DataFrame,
    mode: str,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    predictions = prepare_round2_features(predictions_df, mode=mode)

    X = predictions[ROUND2_FEATURE_COLUMNS].copy()
    predictions["predicted_round2"] = predict_with_calibration(
        round2_model,
        X,
        round_name="round2",
        apply_calibration=apply_calibration,
    )

    predictions["predicted_total_through_round2"] = (
        predictions["predicted_round1"] + predictions["predicted_round2"]
    )

    predictions = predictions.sort_values(
        ["predicted_total_through_round2", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    predictions["predicted_rank_through_round2"] = predictions.index + 1

    if mode == "backtest":
        predictions["abs_error_round2"] = (
            predictions["predicted_round2"] - predictions["actual_round2"]
        ).abs()

        predictions["actual_total_through_round2"] = (
            predictions["actual_round1"] + predictions["actual_round2"]
        )
        predictions["abs_error_total_through_round2"] = (
            predictions["predicted_total_through_round2"]
            - predictions["actual_total_through_round2"]
        ).abs()

        actual_rank_df = (
            predictions[["player_name_clean", "actual_total_through_round2"]]
            .sort_values(
                ["actual_total_through_round2", "player_name_clean"],
                ascending=[True, True],
            )
            .reset_index(drop=True)
        )
        actual_rank_df["actual_rank_through_round2"] = actual_rank_df.index + 1

        predictions = predictions.merge(
            actual_rank_df[["player_name_clean", "actual_rank_through_round2"]],
            on="player_name_clean",
            how="left",
        )

    return predictions


def apply_cut(predictions_df: pd.DataFrame, tournament_name: str) -> pd.DataFrame:
    predictions = predictions_df.copy()

    required_cols = {"predicted_round1", "predicted_round2", "predicted_total_through_round2"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns for cut logic: {sorted(missing)}")

    if predictions.empty:
        predictions["cut_rule_top_n"] = pd.NA
        predictions["cut_rule_ties"] = pd.NA
        predictions["cut_rule_within_leader_strokes"] = pd.NA
        predictions["leader_score_r2"] = pd.NA
        predictions["cut_line"] = pd.NA
        predictions["made_cut_predicted"] = False
        return predictions

    predictions = predictions.sort_values(
        ["predicted_total_through_round2", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)

    rule = get_cut_rule(tournament_name)
    top_n = rule["top_n"]
    ties = rule["ties"]
    within_leader_strokes = rule["within_leader_strokes"]

    bubble_index = min(top_n, len(predictions)) - 1
    cut_score = predictions.iloc[bubble_index]["predicted_total_through_round2"]
    leader_score = predictions.iloc[0]["predicted_total_through_round2"]

    if ties:
        made_cut = predictions["predicted_total_through_round2"] <= cut_score
    else:
        made_cut = pd.Series(False, index=predictions.index)
        made_cut.iloc[:top_n] = True

    if within_leader_strokes is not None:
        made_cut = made_cut | (
            predictions["predicted_total_through_round2"] <= leader_score + within_leader_strokes
        )

    predictions["cut_rule_top_n"] = top_n
    predictions["cut_rule_ties"] = ties
    predictions["cut_rule_within_leader_strokes"] = within_leader_strokes
    predictions["leader_score_r2"] = leader_score
    predictions["cut_line"] = cut_score
    predictions["made_cut_predicted"] = made_cut

    return predictions


def filter_players_making_cut(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if "made_cut_predicted" not in predictions_df.columns:
        raise ValueError("DataFrame must include 'made_cut_predicted' before filtering cut players.")
    return predictions_df[predictions_df["made_cut_predicted"]].copy()


def predict_round3(
    round3_model,
    predictions_df: pd.DataFrame,
    mode: str,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    predictions = filter_players_making_cut(predictions_df)
    predictions = prepare_round3_features(predictions, mode=mode)

    X = predictions[ROUND3_FEATURE_COLUMNS].copy()
    predictions["predicted_round3"] = predict_with_calibration(
        round3_model,
        X,
        round_name="round3",
        apply_calibration=apply_calibration,
    )

    predictions["predicted_total_through_round3"] = (
        predictions["predicted_round1"]
        + predictions["predicted_round2"]
        + predictions["predicted_round3"]
    )

    predictions = predictions.sort_values(
        ["predicted_total_through_round3", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    predictions["predicted_rank_through_round3"] = predictions.index + 1

    if mode == "backtest":
        predictions["abs_error_round3"] = (
            predictions["predicted_round3"] - predictions["actual_round3"]
        ).abs()

        predictions["actual_total_through_round3"] = (
            predictions["actual_round1"] + predictions["actual_round2"] + predictions["actual_round3"]
        )
        predictions["abs_error_total_through_round3"] = (
            predictions["predicted_total_through_round3"]
            - predictions["actual_total_through_round3"]
        ).abs()

    return predictions


def predict_round4(
    round4_model,
    predictions_df: pd.DataFrame,
    mode: str,
    apply_calibration: bool = True,
) -> pd.DataFrame:
    predictions = prepare_round4_features(predictions_df, mode=mode)

    X = predictions[ROUND4_FEATURE_COLUMNS].copy()
    predictions["predicted_round4"] = predict_with_calibration(
        round4_model,
        X,
        round_name="round4",
        apply_calibration=apply_calibration,
    )

    predictions["predicted_total"] = (
        predictions["predicted_round1"]
        + predictions["predicted_round2"]
        + predictions["predicted_round3"]
        + predictions["predicted_round4"]
    )

    predictions = predictions.sort_values(
        ["predicted_total", "player_name_clean"],
        ascending=[True, True],
    ).reset_index(drop=True)
    predictions["predicted_rank_final"] = predictions.index + 1

    if mode == "backtest":
        predictions["abs_error_round4"] = (
            predictions["predicted_round4"] - predictions["actual_round4"]
        ).abs()

        predictions["actual_total"] = (
            predictions["actual_round1"]
            + predictions["actual_round2"]
            + predictions["actual_round3"]
            + predictions["actual_round4"]
        )
        predictions["abs_error_total"] = (
            predictions["predicted_total"] - predictions["actual_total"]
        ).abs()

        actual_rank_df = (
            predictions[["player_name_clean", "actual_total"]]
            .dropna(subset=["actual_total"])
            .sort_values(["actual_total", "player_name_clean"], ascending=[True, True])
            .reset_index(drop=True)
        )
        actual_rank_df["actual_rank_final"] = actual_rank_df.index + 1

        predictions = predictions.merge(
            actual_rank_df[["player_name_clean", "actual_rank_final"]],
            on="player_name_clean",
            how="left",
        )

    return predictions


def build_prediction_output(predictions: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "predicted_rank_round1",
        "predicted_rank_through_round2",
        "predicted_rank_through_round3",
        "predicted_rank_final",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
        "inference_mode",
        "round1_input_source",
        "round3_input_source",
        "round4_input_source",
        "predicted_round1",
        "predicted_round2",
        "predicted_round3",
        "predicted_round4",
        "predicted_total_through_round2",
        "predicted_total_through_round3",
        "predicted_total",
        "cut_rule_top_n",
        "cut_rule_ties",
        "cut_rule_within_leader_strokes",
        "leader_score_r2",
        "cut_line",
        "made_cut_predicted",
    ]
    return predictions[output_columns].copy()


def build_backtest_output(predictions: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "predicted_rank_round1",
        "actual_rank_round1",
        "predicted_rank_through_round2",
        "actual_rank_through_round2",
        "predicted_rank_through_round3",
        "predicted_rank_final",
        "actual_rank_final",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
        "inference_mode",
        "round1_input_source",
        "round3_input_source",
        "round4_input_source",
        "predicted_round1",
        "actual_round1",
        "abs_error_round1",
        "predicted_round2",
        "actual_round2",
        "abs_error_round2",
        "predicted_round3",
        "actual_round3",
        "abs_error_round3",
        "predicted_round4",
        "actual_round4",
        "abs_error_round4",
        "predicted_total_through_round2",
        "actual_total_through_round2",
        "abs_error_total_through_round2",
        "predicted_total_through_round3",
        "actual_total_through_round3",
        "abs_error_total_through_round3",
        "predicted_total",
        "actual_total",
        "abs_error_total",
        "cut_rule_top_n",
        "cut_rule_ties",
        "cut_rule_within_leader_strokes",
        "leader_score_r2",
        "cut_line",
        "made_cut_predicted",
    ]
    return predictions[output_columns].copy()


def save_outputs(prediction_df: pd.DataFrame, backtest_df: pd.DataFrame) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(PREDICTION_OUTPUT_PATH, index=False)
    backtest_df.to_parquet(BACKTEST_OUTPUT_PATH, index=False)


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

    print("Loading models...")
    round1_model = load_model(ROUND1_MODEL_PATH)
    round2_model = load_model(ROUND2_MODEL_PATH)
    round3_model = load_model(ROUND3_MODEL_PATH)
    round4_model = load_model(ROUND4_MODEL_PATH)

    print("Running Round 1 predictions...")
    predictions_df = predict_round1(round1_model, inference_df)

    print(f"Running Round 2 {INFERENCE_MODE} predictions for prediction artifact...")
    prediction_mode_df = predict_round2(round2_model, predictions_df, mode=INFERENCE_MODE)

    print("Applying cut logic to prediction artifact...")
    prediction_mode_df = apply_cut(prediction_mode_df, tournament_name=TARGET_TOURNAMENT)

    print(f"Running Round 3 {INFERENCE_MODE} predictions for prediction artifact...")
    prediction_mode_df = predict_round3(round3_model, prediction_mode_df, mode=INFERENCE_MODE)

    print(f"Running Round 4 {INFERENCE_MODE} predictions for prediction artifact...")
    prediction_mode_df = predict_round4(round4_model, prediction_mode_df, mode=INFERENCE_MODE)

    prediction_output_df = build_prediction_output(prediction_mode_df)

    print("Running Round 2 backtest predictions for evaluation artifact...")
    backtest_mode_df = predict_round2(round2_model, predictions_df, mode="backtest")

    print("Applying cut logic to backtest artifact...")
    backtest_mode_df = apply_cut(backtest_mode_df, tournament_name=TARGET_TOURNAMENT)

    print("Running Round 3 backtest predictions for evaluation artifact...")
    backtest_mode_df = predict_round3(round3_model, backtest_mode_df, mode="backtest")

    print("Running Round 4 backtest predictions for evaluation artifact...")
    backtest_mode_df = predict_round4(round4_model, backtest_mode_df, mode="backtest")

    backtest_output_df = build_backtest_output(backtest_mode_df)

    print(f"Saving prediction artifact to: {PREDICTION_OUTPUT_PATH}")
    print(f"Saving backtest artifact to:   {BACKTEST_OUTPUT_PATH}")
    save_outputs(prediction_output_df, backtest_output_df)

    print("\nDone.")


if __name__ == "__main__":
    main()