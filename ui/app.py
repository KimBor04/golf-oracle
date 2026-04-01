from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.paths import PREDICTIONS_DIR

PREDICTION_PATH = PREDICTIONS_DIR / "leaderboard_predictions.parquet"
BACKTEST_PATH = PREDICTIONS_DIR / "leaderboard_backtest.parquet"


st.set_page_config(
    page_title="Golf Oracle",
    page_icon="🏌️",
    layout="wide",
)


@st.cache_data
def load_prediction_df() -> pd.DataFrame:
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_PATH}")

    df = pd.read_parquet(PREDICTION_PATH)

    if "target_start" in df.columns:
        df["target_start"] = pd.to_datetime(df["target_start"], errors="coerce")

    if "feature_source_start" in df.columns:
        df["feature_source_start"] = pd.to_datetime(df["feature_source_start"], errors="coerce")

    return df


@st.cache_data
def load_backtest_df() -> pd.DataFrame:
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Backtest file not found: {BACKTEST_PATH}")

    df = pd.read_parquet(BACKTEST_PATH)

    if "target_start" in df.columns:
        df["target_start"] = pd.to_datetime(df["target_start"], errors="coerce")

    if "feature_source_start" in df.columns:
        df["feature_source_start"] = pd.to_datetime(df["feature_source_start"], errors="coerce")

    return df


def format_metrics(df: pd.DataFrame) -> tuple[float, float, int]:
    mae = df["abs_error"].mean() if "abs_error" in df.columns else float("nan")
    rmse = (
        ((df["predicted_round1"] - df["actual_round1"]) ** 2).mean() ** 0.5
        if {"predicted_round1", "actual_round1"}.issubset(df.columns)
        else float("nan")
    )
    n_players = len(df)
    return mae, rmse, n_players


def main() -> None:
    st.title("🏌️ Golf Oracle")
    st.subheader("Round 1 leaderboard and backtest")

    try:
        prediction_df = load_prediction_df()
    except Exception as e:
        st.error(f"Could not load prediction artifact: {e}")
        st.stop()

    if prediction_df.empty:
        st.warning("The prediction artifact is empty.")
        st.stop()

    backtest_df = None
    backtest_error = None
    try:
        backtest_df = load_backtest_df()
    except Exception as e:
        backtest_error = str(e)

    tournament_name = (
        prediction_df["target_tournament"].iloc[0]
        if "target_tournament" in prediction_df.columns
        else "Unknown Tournament"
    )
    target_start = prediction_df["target_start"].iloc[0] if "target_start" in prediction_df.columns else None

    if pd.notna(target_start):
        st.caption(f"{tournament_name} — {target_start.date()}")
    else:
        st.caption(tournament_name)

    if backtest_df is not None and not backtest_df.empty:
        mae, rmse, n_players = format_metrics(backtest_df)
    else:
        mae, rmse, n_players = float("nan"), float("nan"), len(prediction_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Players", f"{n_players}")
    col2.metric("Event MAE", f"{mae:.3f}" if pd.notna(mae) else "N/A")
    col3.metric("Event RMSE", f"{rmse:.3f}" if pd.notna(rmse) else "N/A")

    st.markdown("---")

    player_search = st.text_input("Search player", "")
    show_top_n = st.slider("Show top N rows", min_value=5, max_value=50, value=20, step=5)

    filtered_prediction_df = prediction_df.copy()
    filtered_backtest_df = backtest_df.copy() if backtest_df is not None else None

    if player_search.strip():
        filtered_prediction_df = filtered_prediction_df[
            filtered_prediction_df["player_name_clean"].str.contains(player_search, case=False, na=False)
        ].copy()

        if filtered_backtest_df is not None:
            filtered_backtest_df = filtered_backtest_df[
                filtered_backtest_df["player_name_clean"].str.contains(player_search, case=False, na=False)
            ].copy()

    st.markdown("### Predicted leaderboard")

    predicted_cols = [
        "predicted_rank",
        "player_name_clean",
        "predicted_round1",
        "feature_source_tournament",
        "feature_source_start",
        "rolling_avg_last_3",
        "rolling_avg_last_5",
        "made_cut_rate_last_5",
        "career_tournament_count",
    ]
    predicted_cols = [col for col in predicted_cols if col in filtered_prediction_df.columns]

    st.dataframe(
        filtered_prediction_df[predicted_cols].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Prediction artifact")
    st.dataframe(filtered_prediction_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("## Backtest view")

    if backtest_error:
        st.info(f"Backtest artifact not available: {backtest_error}")
        return

    if filtered_backtest_df is None or filtered_backtest_df.empty:
        st.warning("Backtest artifact is empty.")
        return

    st.markdown("### Actual leaderboard")

    actual_df = filtered_backtest_df.sort_values(["actual_rank", "player_name_clean"]).copy()

    actual_cols = [
        "actual_rank",
        "predicted_rank",
        "player_name_clean",
        "actual_round1",
        "predicted_round1",
        "abs_error",
    ]
    actual_cols = [col for col in actual_cols if col in actual_df.columns]

    st.dataframe(
        actual_df[actual_cols].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Full backtest table")
    st.dataframe(filtered_backtest_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()