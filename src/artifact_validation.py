from __future__ import annotations

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "player_name_clean",
    "target_tournament",
    "target_start",
    "inference_mode",
    "predicted_round1",
    "predicted_round2",
    "predicted_round3",
    "predicted_round4",
    "predicted_total",
    "predicted_rank_final",
    "made_cut_predicted",
}


REQUIRED_BACKTEST_COLUMNS = {
    "player_name_clean",
    "target_tournament",
    "target_start",
    "inference_mode",
}


def validate_prediction_artifact(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Prediction artifact is empty.")

    missing_columns = REQUIRED_PREDICTION_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Prediction artifact is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df["player_name_clean"].isna().any():
        raise ValueError("Prediction artifact contains missing player_name_clean values.")

    if df["predicted_total"].isna().any():
        raise ValueError("Prediction artifact contains missing predicted_total values.")

    if df["predicted_rank_final"].isna().any():
        raise ValueError("Prediction artifact contains missing predicted_rank_final values.")

    if df["made_cut_predicted"].isna().any():
        raise ValueError("Prediction artifact contains missing made_cut_predicted values.")


def validate_backtest_artifact(df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_BACKTEST_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Backtest artifact is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df.empty:
        return

    if df["player_name_clean"].isna().any():
        raise ValueError("Backtest artifact contains missing player_name_clean values.")