from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.paths import PREDICTIONS_DIR

PREDICTIONS_PATH = PREDICTIONS_DIR / "leaderboard.parquet"


st.set_page_config(
    page_title="Golf Oracle",
    page_icon="🏌️",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTIONS_PATH}")

    df = pd.read_parquet(PREDICTIONS_PATH)

    if "target_start" in df.columns:
        df["target_start"] = pd.to_datetime(df["target_start"], errors="coerce")

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
    st.subheader("Round 1 tournament backtest")

    try:
        df = load_predictions()
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        st.stop()

    if df.empty:
        st.warning("The prediction file is empty.")
        st.stop()

    tournament_name = df["target_tournament"].iloc[0] if "target_tournament" in df.columns else "Unknown Tournament"
    target_start = df["target_start"].iloc[0] if "target_start" in df.columns else None

    if pd.notna(target_start):
        st.caption(f"{tournament_name} — {target_start.date()}")
    else:
        st.caption(tournament_name)

    mae, rmse, n_players = format_metrics(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Players evaluated", f"{n_players}")
    col2.metric("Event MAE", f"{mae:.3f}" if pd.notna(mae) else "N/A")
    col3.metric("Event RMSE", f"{rmse:.3f}" if pd.notna(rmse) else "N/A")

    st.markdown("---")

    player_search = st.text_input("Search player", "")
    show_top_n = st.slider("Show top N rows", min_value=5, max_value=50, value=20, step=5)

    filtered_df = df.copy()

    if player_search.strip():
        filtered_df = filtered_df[
            filtered_df["player_name_clean"].str.contains(player_search, case=False, na=False)
        ].copy()

    st.markdown("### Predicted leaderboard")

    predicted_cols = [
        "predicted_rank",
        "actual_rank",
        "player_name_clean",
        "predicted_round1",
        "actual_round1",
        "abs_error",
    ]
    predicted_cols = [col for col in predicted_cols if col in filtered_df.columns]

    st.dataframe(
        filtered_df[predicted_cols].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Actual leaderboard")

    if "actual_rank" in filtered_df.columns:
        actual_df = filtered_df.sort_values(["actual_rank", "player_name_clean"]).copy()

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
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()