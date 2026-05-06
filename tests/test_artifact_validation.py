import pandas as pd
import pytest

from src.artifact_validation import (
    validate_backtest_artifact,
    validate_prediction_artifact,
)


def make_valid_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name_clean": ["player one", "player two"],
            "target_tournament": ["Test Open", "Test Open"],
            "target_start": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "inference_mode": ["live", "live"],
            "predicted_round1": [70.1, 71.2],
            "predicted_round2": [69.8, 72.0],
            "predicted_round3": [70.5, 71.5],
            "predicted_round4": [69.9, 70.8],
            "predicted_total": [280.3, 285.5],
            "predicted_rank_final": [1, 2],
            "made_cut_predicted": [True, True],
        }
    )


def test_validate_prediction_artifact_accepts_valid_df() -> None:
    df = make_valid_prediction_df()

    validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_empty_df() -> None:
    df = make_valid_prediction_df().iloc[0:0]

    with pytest.raises(ValueError, match="Prediction artifact is empty"):
        validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_missing_required_column() -> None:
    df = make_valid_prediction_df().drop(columns=["predicted_total"])

    with pytest.raises(ValueError, match="missing required columns"):
        validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_missing_player_name() -> None:
    df = make_valid_prediction_df()
    df.loc[0, "player_name_clean"] = None

    with pytest.raises(ValueError, match="missing player_name_clean"):
        validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_missing_predicted_total() -> None:
    df = make_valid_prediction_df()
    df.loc[0, "predicted_total"] = None

    with pytest.raises(ValueError, match="missing predicted_total"):
        validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_missing_final_rank() -> None:
    df = make_valid_prediction_df()
    df.loc[0, "predicted_rank_final"] = None

    with pytest.raises(ValueError, match="missing predicted_rank_final"):
        validate_prediction_artifact(df)


def test_validate_prediction_artifact_rejects_missing_cut_status() -> None:
    df = make_valid_prediction_df()
    df["made_cut_predicted"] = df["made_cut_predicted"].astype("object")
    df.loc[0, "made_cut_predicted"] = None

    with pytest.raises(ValueError, match="missing made_cut_predicted"):
        validate_prediction_artifact(df)


def test_validate_backtest_artifact_accepts_empty_df_with_required_columns() -> None:
    df = pd.DataFrame(
        columns=[
            "player_name_clean",
            "target_tournament",
            "target_start",
            "inference_mode",
        ]
    )

    validate_backtest_artifact(df)


def test_validate_backtest_artifact_accepts_valid_df() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": ["player one"],
            "target_tournament": ["Test Open"],
            "target_start": pd.to_datetime(["2026-01-01"]),
            "inference_mode": ["backtest"],
        }
    )

    validate_backtest_artifact(df)


def test_validate_backtest_artifact_rejects_missing_required_column() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": ["player one"],
            "target_tournament": ["Test Open"],
            "target_start": pd.to_datetime(["2026-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="Backtest artifact is missing required columns"):
        validate_backtest_artifact(df)


def test_validate_backtest_artifact_rejects_missing_player_name() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": [None],
            "target_tournament": ["Test Open"],
            "target_start": pd.to_datetime(["2026-01-01"]),
            "inference_mode": ["backtest"],
        }
    )

    with pytest.raises(ValueError, match="missing player_name_clean"):
        validate_backtest_artifact(df)