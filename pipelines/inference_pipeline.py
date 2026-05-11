from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_cut_rule
from src.artifact_validation import (
    validate_backtest_artifact,
    validate_prediction_artifact,
)
from src.paths import (
    DATA_DIR,
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
LIVE_FEATURES_PATH = FEATURES_DIR / "live_features.parquet"
MANUAL_FIELDS_DIR = DATA_DIR / "manual_fields"

PREDICTION_OUTPUT_PATH = LEADERBOARD_PREDICTIONS_PATH
BACKTEST_OUTPUT_PATH = LEADERBOARD_BACKTEST_PATH

TARGET_TOURNAMENT = "Masters Tournament"
TARGET_START_DATE = "2025-04-10"
INFERENCE_MODE = "live"  # allowed: "live", "backtest"

# Field source options:
# - "historical": use field from historical_features.parquet
# - "api_fields": use field from features/api_fields_<year>.parquet
#                 and automatically fall back to data/manual_fields/*.csv
# - "manual_fields": use a local CSV field from data/manual_fields/
# - "live_features": use field from features/live_features.parquet
FIELD_SOURCE = "historical"

VALID_INFERENCE_MODES = {"live", "backtest"}
VALID_FIELD_SOURCES = {"historical", "api_fields", "manual_fields", "live_features"}

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Golf Oracle tournament inference and save prediction/backtest artifacts."
        )
    )

    parser.add_argument(
        "--target-tournament",
        default=TARGET_TOURNAMENT,
        help=(
            "Tournament name to predict. "
            f"Default: '{TARGET_TOURNAMENT}'."
        ),
    )
    parser.add_argument(
        "--target-start-date",
        default=TARGET_START_DATE,
        help=(
            "Tournament start date in YYYY-MM-DD format. "
            f"Default: '{TARGET_START_DATE}'."
        ),
    )
    parser.add_argument(
        "--inference-mode",
        default=INFERENCE_MODE,
        choices=sorted(VALID_INFERENCE_MODES),
        help=(
            "Prediction mode for the prediction artifact. "
            "'live' uses predicted prior rounds. "
            "'backtest' uses actual prior rounds where available. "
            f"Default: '{INFERENCE_MODE}'."
        ),
    )
    parser.add_argument(
        "--field-source",
        default=FIELD_SOURCE,
        choices=sorted(VALID_FIELD_SOURCES),
        help=(
            "Source for the tournament field. "
            "'historical' uses historical_features.parquet. "
            "'api_fields' uses features/api_fields_<year>.parquet and falls back "
            "to data/manual_fields/*.csv if no usable API field exists. "
            "'manual_fields' uses data/manual_fields/*.csv directly. "
            "'live_features' uses features/live_features.parquet. "
            f"Default: '{FIELD_SOURCE}'."
        ),
    )

    return parser.parse_args(argv)


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
        (df["tournament"] == tournament)
        & (df["start"] == target_start)
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


def api_fields_path_for_year(year: int) -> Path:
    return FEATURES_DIR / f"api_fields_{year}.parquet"


def load_api_fields(year: int) -> pd.DataFrame:
    api_fields_path = api_fields_path_for_year(year)

    if not api_fields_path.exists():
        raise FileNotFoundError(
            f"API fields file not found: {api_fields_path}. "
            "Run the FreeWebAPI fields backfill first."
        )

    api_fields_df = pd.read_parquet(api_fields_path)

    required_metadata_columns = {
        "target_tournament",
        "season",
        "field_cache_status",
        "api_record_type",
        "api_start_date",
    }

    missing_metadata = required_metadata_columns - set(api_fields_df.columns)
    if missing_metadata:
        raise ValueError(
            "Missing required columns in API fields file: "
            f"{sorted(missing_metadata)}"
        )

    api_fields_df = api_fields_df.copy()
    api_fields_df["api_start_date"] = pd.to_datetime(
        api_fields_df["api_start_date"],
        errors="coerce",
    )

    if "api_end_date" in api_fields_df.columns:
        api_fields_df["api_end_date"] = pd.to_datetime(
            api_fields_df["api_end_date"],
            errors="coerce",
        )

    if "player_name_clean" not in api_fields_df.columns:
        if "player_name" in api_fields_df.columns:
            api_fields_df["player_name_clean"] = (
                api_fields_df["player_name"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        elif {"firstname", "lastname"}.issubset(api_fields_df.columns):
            api_fields_df["player_name_clean"] = (
                api_fields_df["firstname"].fillna("").astype(str).str.strip()
                + " "
                + api_fields_df["lastname"].fillna("").astype(str).str.strip()
            ).str.lower().str.strip()

        elif {"first_name", "last_name"}.issubset(api_fields_df.columns):
            api_fields_df["player_name_clean"] = (
                api_fields_df["first_name"].fillna("").astype(str).str.strip()
                + " "
                + api_fields_df["last_name"].fillna("").astype(str).str.strip()
            ).str.lower().str.strip()

        elif "name" in api_fields_df.columns:
            api_fields_df["player_name_clean"] = (
                api_fields_df["name"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        else:
            api_fields_df["player_name_clean"] = pd.NA

    return api_fields_df


def get_target_field_from_api_fields(
    api_fields_df: pd.DataFrame,
    tournament: str,
    start_date: str,
) -> pd.DataFrame:
    target_start = pd.to_datetime(start_date)
    target_year = int(target_start.year)

    field_df = api_fields_df.copy()

    field_df = field_df[field_df["season"].astype(int) == target_year].copy()
    field_df = field_df[field_df["target_tournament"] == tournament].copy()

    if "api_start_date" in field_df.columns and field_df["api_start_date"].notna().any():
        field_df = field_df[field_df["api_start_date"] == target_start].copy()

    if "api_record_type" in field_df.columns:
        field_df = field_df[field_df["api_record_type"] == "future_field"].copy()

    if field_df.empty:
        available = (
            api_fields_df[
                [
                    "target_tournament",
                    "season",
                    "api_start_date",
                    "field_cache_status",
                ]
            ]
            .drop_duplicates()
            .sort_values(["season", "api_start_date", "target_tournament"])
        )
        raise ValueError(
            f"No API field rows found for tournament='{tournament}' and start='{start_date}'.\n"
            f"Available API field records:\n{available.to_string(index=False)}"
        )

    if "field_cache_status" in field_df.columns:
        available_rows = field_df[field_df["field_cache_status"] == "available"].copy()

        if available_rows.empty:
            status_summary = (
                field_df[
                    [
                        "target_tournament",
                        "api_start_date",
                        "field_cache_status",
                    ]
                ]
                .drop_duplicates()
                .to_string(index=False)
            )
            raise ValueError(
                "API field artifact exists, but the requested tournament field is not available.\n"
                f"Requested tournament='{tournament}', start='{start_date}'.\n"
                f"Field status:\n{status_summary}"
            )

        field_df = available_rows

    field_df = field_df[
        field_df["player_name_clean"].notna()
        & (field_df["player_name_clean"].astype(str).str.strip() != "")
    ].copy()

    if field_df.empty:
        status_summary = (
            api_fields_df[
                [
                    "target_tournament",
                    "season",
                    "api_start_date",
                    "field_cache_status",
                ]
            ]
            .drop_duplicates()
            .sort_values(["season", "api_start_date", "target_tournament"])
            .to_string(index=False)
        )
        raise ValueError(
            "API field artifact exists, but it contains no player rows for inference.\n"
            "This usually means the API returned only field status metadata.\n"
            f"Available API field records:\n{status_summary}"
        )

    field_df = field_df.sort_values(["player_name_clean"]).drop_duplicates(
        subset=["player_name_clean"],
        keep="first",
    )

    field_df["tournament"] = field_df["target_tournament"]
    field_df["start"] = target_start
    field_df["season"] = target_year

    for round_col in ["round1", "round2", "round3", "round4"]:
        if round_col not in field_df.columns:
            field_df[round_col] = pd.NA

    return field_df


def slugify_tournament_name(tournament: str) -> str:
    slug = tournament.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def manual_field_path_for_tournament(tournament: str, year: int) -> Path:
    slug = slugify_tournament_name(tournament)
    return MANUAL_FIELDS_DIR / f"{slug}_{year}.csv"


def load_manual_field(tournament: str, start_date: str) -> pd.DataFrame:
    target_start = pd.to_datetime(start_date)
    target_year = int(target_start.year)

    manual_field_path = manual_field_path_for_tournament(
        tournament=tournament,
        year=target_year,
    )

    if not manual_field_path.exists():
        raise FileNotFoundError(
            f"Manual field file not found: {manual_field_path}.\n"
            "Create a CSV in data/manual_fields/ with this naming pattern:\n"
            f"{manual_field_path.name}\n"
            "Required column: player_name or player_name_clean."
        )

    manual_df = pd.read_csv(manual_field_path)

    if manual_df.empty:
        raise ValueError(f"Manual field file is empty: {manual_field_path}")

    manual_df = manual_df.copy()

    if "player_name_clean" not in manual_df.columns:
        if "player_name" not in manual_df.columns:
            raise ValueError(
                f"Manual field file must contain either 'player_name' or "
                f"'player_name_clean': {manual_field_path}"
            )

        manual_df["player_name_clean"] = (
            manual_df["player_name"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    if "player_name" not in manual_df.columns:
        manual_df["player_name"] = manual_df["player_name_clean"]

    manual_df = manual_df[
        manual_df["player_name_clean"].notna()
        & (manual_df["player_name_clean"].astype(str).str.strip() != "")
    ].copy()

    if manual_df.empty:
        raise ValueError(
            f"Manual field file contains no usable player rows: {manual_field_path}"
        )

    manual_df["target_tournament"] = tournament
    manual_df["target_tournament_clean"] = slugify_tournament_name(tournament).replace(
        "_",
        " ",
    )
    manual_df["season"] = target_year
    manual_df["round_id"] = 1
    manual_df["field_cache_status"] = "available"
    manual_df["api_record_type"] = "manual_field"
    manual_df["api_start_date"] = target_start
    manual_df["api_end_date"] = pd.NaT

    if "tourn_id" not in manual_df.columns:
        manual_df["tourn_id"] = "manual"

    return manual_df


def get_target_field_from_manual_fields(
    manual_df: pd.DataFrame,
    tournament: str,
    start_date: str,
) -> pd.DataFrame:
    target_start = pd.to_datetime(start_date)
    target_year = int(target_start.year)

    field_df = manual_df.copy()

    if "season" in field_df.columns:
        field_df = field_df[field_df["season"].astype(int) == target_year].copy()

    if "target_tournament" in field_df.columns:
        field_df = field_df[field_df["target_tournament"] == tournament].copy()

    if field_df.empty:
        raise ValueError(
            f"No manual field rows found for tournament='{tournament}' "
            f"and start='{start_date}'."
        )

    field_df = field_df[
        field_df["player_name_clean"].notna()
        & (field_df["player_name_clean"].astype(str).str.strip() != "")
    ].copy()

    if field_df.empty:
        raise ValueError(
            "Manual field artifact exists for the requested tournament, "
            "but it contains no usable player names."
        )

    field_df = field_df.sort_values(["player_name_clean"]).drop_duplicates(
        subset=["player_name_clean"],
        keep="first",
    )

    field_df["tournament"] = tournament
    field_df["start"] = target_start
    field_df["season"] = target_year

    for round_col in ["round1", "round2", "round3", "round4"]:
        field_df[round_col] = pd.NA

    return field_df


def load_live_features() -> pd.DataFrame:
    if not LIVE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Live features file not found: {LIVE_FEATURES_PATH}. "
            "Run the feature pipeline with --fetch-live first."
        )

    live_df = pd.read_parquet(LIVE_FEATURES_PATH)

    required_columns = {
        "target_tournament",
        "season",
        "round_id",
        "player_name_clean",
    }
    missing = required_columns - set(live_df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in live features file: {sorted(missing)}"
        )

    live_df = live_df.copy()

    live_df["season"] = pd.to_numeric(live_df["season"], errors="coerce")
    live_df["round_id"] = pd.to_numeric(live_df["round_id"], errors="coerce")

    live_df = live_df[
        live_df["player_name_clean"].notna()
        & (live_df["player_name_clean"].astype(str).str.strip() != "")
    ].copy()

    if live_df.empty:
        raise ValueError(
            "Live features file exists, but it contains no usable player rows."
        )

    return live_df


def get_target_field_from_live_features(
    live_df: pd.DataFrame,
    tournament: str,
    start_date: str,
) -> pd.DataFrame:
    target_start = pd.to_datetime(start_date)
    target_year = int(target_start.year)

    field_df = live_df.copy()

    field_df = field_df[field_df["season"].astype(int) == target_year].copy()
    field_df = field_df[field_df["target_tournament"] == tournament].copy()

    if field_df.empty:
        available = (
            live_df[["target_tournament", "season", "round_id"]]
            .drop_duplicates()
            .sort_values(["season", "target_tournament", "round_id"])
        )
        raise ValueError(
            f"No live feature rows found for tournament='{tournament}' and season='{target_year}'.\n"
            f"Available live feature records:\n{available.to_string(index=False)}"
        )

    field_df = field_df[
        field_df["player_name_clean"].notna()
        & (field_df["player_name_clean"].astype(str).str.strip() != "")
    ].copy()

    if field_df.empty:
        raise ValueError(
            "Live features artifact exists for the requested tournament, "
            "but it contains no usable player names."
        )

    field_df = field_df.sort_values(["player_name_clean"]).drop_duplicates(
        subset=["player_name_clean"],
        keep="first",
    )

    # Convert live leaderboard rows into the same field shape expected by
    # build_pre_tournament_feature_rows().
    #
    # Important:
    # For now, live_features is used only as a player-field source.
    # We do not inject completed live round scores into actual_round columns yet.
    # That keeps the inference path leakage-safe and avoids mixing partial rounds.
    field_df["tournament"] = field_df["target_tournament"]
    field_df["start"] = target_start
    field_df["season"] = target_year

    for round_col in ["round1", "round2", "round3", "round4"]:
        field_df[round_col] = pd.NA

    return field_df


def validate_field_source(field_source: str) -> None:
    if field_source not in VALID_FIELD_SOURCES:
        raise ValueError(
            f"Invalid FIELD_SOURCE='{field_source}'. "
            f"Expected one of: {sorted(VALID_FIELD_SOURCES)}."
        )


def get_target_field_for_source(
    historical_df: pd.DataFrame,
    tournament: str,
    start_date: str,
    field_source: str,
) -> pd.DataFrame:
    validate_field_source(field_source)

    if field_source == "historical":
        return get_target_field(
            df=historical_df,
            tournament=tournament,
            start_date=start_date,
        )

    if field_source == "api_fields":
        target_year = int(pd.to_datetime(start_date).year)

        try:
            api_fields_df = load_api_fields(year=target_year)

            return get_target_field_from_api_fields(
                api_fields_df=api_fields_df,
                tournament=tournament,
                start_date=start_date,
            )
        except (FileNotFoundError, ValueError) as api_error:
            print(
                "\nWarning: API field source is unavailable for this tournament."
            )
            print(str(api_error))
            print(
                "\nTrying manual field fallback from data/manual_fields/..."
            )

            try:
                manual_df = load_manual_field(
                    tournament=tournament,
                    start_date=start_date,
                )

                return get_target_field_from_manual_fields(
                    manual_df=manual_df,
                    tournament=tournament,
                    start_date=start_date,
                )
            except (FileNotFoundError, ValueError) as manual_error:
                raise ValueError(
                    "Could not build tournament field from API fields or manual fallback.\n"
                    f"API field error:\n{api_error}\n\n"
                    f"Manual field fallback error:\n{manual_error}"
                ) from manual_error

    if field_source == "manual_fields":
        manual_df = load_manual_field(
            tournament=tournament,
            start_date=start_date,
        )

        return get_target_field_from_manual_fields(
            manual_df=manual_df,
            tournament=tournament,
            start_date=start_date,
        )

    if field_source == "live_features":
        live_df = load_live_features()

        return get_target_field_from_live_features(
            live_df=live_df,
            tournament=tournament,
            start_date=start_date,
        )

    raise ValueError(
        f"Invalid FIELD_SOURCE='{field_source}'. "
        f"Expected one of: {sorted(VALID_FIELD_SOURCES)}."
    )


def has_backtest_actuals(predictions_df: pd.DataFrame) -> bool:
    required_actual_cols = ["actual_round1", "actual_round2", "actual_round3", "actual_round4"]

    if not set(required_actual_cols).issubset(predictions_df.columns):
        return False

    actuals = predictions_df[required_actual_cols].apply(pd.to_numeric, errors="coerce")
    return actuals.notna().any().any()


def build_empty_backtest_output() -> pd.DataFrame:
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

    return pd.DataFrame(columns=output_columns)


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

    for actual_col in ["actual_round1", "actual_round2", "actual_round3", "actual_round4"]:
        field_meta[actual_col] = pd.to_numeric(field_meta[actual_col], errors="coerce")

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
    if mode not in VALID_INFERENCE_MODES:
        raise ValueError(
            f"Invalid inference mode: {mode}. Expected one of {sorted(VALID_INFERENCE_MODES)}."
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
        .dropna(subset=["actual_round1"])
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
            .dropna(subset=["actual_total_through_round2"])
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
    validate_prediction_artifact(prediction_df)
    validate_backtest_artifact(backtest_df)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_df.to_parquet(PREDICTION_OUTPUT_PATH, index=False)
    backtest_df.to_parquet(BACKTEST_OUTPUT_PATH, index=False)


def main(
    target_tournament: str = TARGET_TOURNAMENT,
    target_start_date: str = TARGET_START_DATE,
    inference_mode: str = INFERENCE_MODE,
    field_source: str = FIELD_SOURCE,
) -> None:
    validate_inference_mode(inference_mode)
    validate_field_source(field_source)

    target_start = pd.to_datetime(target_start_date, errors="coerce")
    if pd.isna(target_start):
        raise ValueError(
            f"Invalid target_start_date='{target_start_date}'. "
            "Expected format: YYYY-MM-DD."
        )

    print("Loading historical features...")
    df = load_features()

    print(
        f"Selecting field for: {target_tournament} ({target_start_date}) "
        f"using FIELD_SOURCE='{field_source}'"
    )
    field_df = get_target_field_for_source(
        historical_df=df,
        tournament=target_tournament,
        start_date=target_start_date,
        field_source=field_source,
    )

    print(f"Players in field: {len(field_df)}")

    print("Building pre-tournament feature rows...")
    inference_df = build_pre_tournament_feature_rows(df, field_df, target_start)

    print(f"Players with usable history: {len(inference_df)}")

    print("Loading models...")
    round1_model = load_model(ROUND1_MODEL_PATH)
    round2_model = load_model(ROUND2_MODEL_PATH)
    round3_model = load_model(ROUND3_MODEL_PATH)
    round4_model = load_model(ROUND4_MODEL_PATH)

    print("Running Round 1 predictions...")
    predictions_df = predict_round1(round1_model, inference_df)

    print(f"Running Round 2 {inference_mode} predictions for prediction artifact...")
    prediction_mode_df = predict_round2(round2_model, predictions_df, mode=inference_mode)

    print("Applying cut logic to prediction artifact...")
    prediction_mode_df = apply_cut(prediction_mode_df, tournament_name=target_tournament)

    print(f"Running Round 3 {inference_mode} predictions for prediction artifact...")
    prediction_mode_df = predict_round3(round3_model, prediction_mode_df, mode=inference_mode)

    print(f"Running Round 4 {inference_mode} predictions for prediction artifact...")
    prediction_mode_df = predict_round4(round4_model, prediction_mode_df, mode=inference_mode)

    prediction_output_df = build_prediction_output(prediction_mode_df)

    if has_backtest_actuals(predictions_df):
        print("Running Round 2 backtest predictions for evaluation artifact...")
        backtest_mode_df = predict_round2(round2_model, predictions_df, mode="backtest")

        print("Applying cut logic to backtest artifact...")
        backtest_mode_df = apply_cut(backtest_mode_df, tournament_name=target_tournament)

        print("Running Round 3 backtest predictions for evaluation artifact...")
        backtest_mode_df = predict_round3(round3_model, backtest_mode_df, mode="backtest")

        print("Running Round 4 backtest predictions for evaluation artifact...")
        backtest_mode_df = predict_round4(round4_model, backtest_mode_df, mode="backtest")

        backtest_output_df = build_backtest_output(backtest_mode_df)
    else:
        print(
            "Skipping backtest artifact because the selected field source "
            "does not contain actual round scores."
        )
        backtest_output_df = build_empty_backtest_output()

    print(f"Saving prediction artifact to: {PREDICTION_OUTPUT_PATH}")
    print(f"Saving backtest artifact to:   {BACKTEST_OUTPUT_PATH}")
    save_outputs(prediction_output_df, backtest_output_df)

    print("\nDone.")


if __name__ == "__main__":
    args = parse_args()

    main(
        target_tournament=args.target_tournament,
        target_start_date=args.target_start_date,
        inference_mode=args.inference_mode,
        field_source=args.field_source,
    )