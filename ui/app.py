from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.paths import (
    LEADERBOARD_PREDICTIONS_PATH,
    LEADERBOARD_BACKTEST_PATH,
)

PREDICTION_PATH = LEADERBOARD_PREDICTIONS_PATH
BACKTEST_PATH = LEADERBOARD_BACKTEST_PATH


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


def format_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    metrics = {
        "n_players": len(df),
        "predicted_cut_count": int(df["made_cut_predicted"].sum()) if "made_cut_predicted" in df.columns else 0,
        "mae_round1": float("nan"),
        "rmse_round1": float("nan"),
        "mae_round2": float("nan"),
        "rmse_round2": float("nan"),
        "mae_round3": float("nan"),
        "rmse_round3": float("nan"),
        "mae_round4": float("nan"),
        "rmse_round4": float("nan"),
        "mae_total_r2": float("nan"),
        "rmse_total_r2": float("nan"),
        "mae_total_r3": float("nan"),
        "rmse_total_r3": float("nan"),
        "mae_total_final": float("nan"),
        "rmse_total_final": float("nan"),
    }

    if "abs_error_round1" in df.columns:
        metrics["mae_round1"] = df["abs_error_round1"].mean()

    if {"predicted_round1", "actual_round1"}.issubset(df.columns):
        metrics["rmse_round1"] = (
            ((df["predicted_round1"] - df["actual_round1"]) ** 2).mean() ** 0.5
        )

    if "abs_error_round2" in df.columns:
        metrics["mae_round2"] = df["abs_error_round2"].mean()

    if {"predicted_round2", "actual_round2"}.issubset(df.columns):
        metrics["rmse_round2"] = (
            ((df["predicted_round2"] - df["actual_round2"]) ** 2).mean() ** 0.5
        )

    if "abs_error_round3" in df.columns:
        metrics["mae_round3"] = df["abs_error_round3"].mean()

    if {"predicted_round3", "actual_round3"}.issubset(df.columns):
        valid = df[["predicted_round3", "actual_round3"]].dropna()
        if not valid.empty:
            metrics["rmse_round3"] = (
                ((valid["predicted_round3"] - valid["actual_round3"]) ** 2).mean() ** 0.5
            )

    if "abs_error_round4" in df.columns:
        metrics["mae_round4"] = df["abs_error_round4"].mean()

    if {"predicted_round4", "actual_round4"}.issubset(df.columns):
        valid = df[["predicted_round4", "actual_round4"]].dropna()
        if not valid.empty:
            metrics["rmse_round4"] = (
                ((valid["predicted_round4"] - valid["actual_round4"]) ** 2).mean() ** 0.5
            )

    if "abs_error_total_through_round2" in df.columns:
        metrics["mae_total_r2"] = df["abs_error_total_through_round2"].mean()

    if {"predicted_total_through_round2", "actual_total_through_round2"}.issubset(df.columns):
        metrics["rmse_total_r2"] = (
            (
                (
                    df["predicted_total_through_round2"]
                    - df["actual_total_through_round2"]
                ) ** 2
            ).mean() ** 0.5
        )

    if "abs_error_total_through_round3" in df.columns:
        metrics["mae_total_r3"] = df["abs_error_total_through_round3"].mean()

    if {"predicted_total_through_round3", "actual_total_through_round3"}.issubset(df.columns):
        valid = df[["predicted_total_through_round3", "actual_total_through_round3"]].dropna()
        if not valid.empty:
            metrics["rmse_total_r3"] = (
                ((valid["predicted_total_through_round3"] - valid["actual_total_through_round3"]) ** 2).mean() ** 0.5
            )

    if "abs_error_total" in df.columns:
        metrics["mae_total_final"] = df["abs_error_total"].mean()

    if {"predicted_total", "actual_total"}.issubset(df.columns):
        valid = df[["predicted_total", "actual_total"]].dropna()
        if not valid.empty:
            metrics["rmse_total_final"] = (
                ((valid["predicted_total"] - valid["actual_total"]) ** 2).mean() ** 0.5
            )

    return metrics


def main() -> None:
    st.title("🏌️ Golf Oracle")
    st.subheader("Full round-by-round tournament projection through Round 4")

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

    inference_mode = (
        prediction_df["inference_mode"].iloc[0]
        if "inference_mode" in prediction_df.columns
        else "unknown"
    )
    round1_input_source = (
        prediction_df["round1_input_source"].iloc[0]
        if "round1_input_source" in prediction_df.columns
        else "unknown"
    )
    round3_input_source = (
        prediction_df["round3_input_source"].iloc[0]
        if "round3_input_source" in prediction_df.columns
        else "unknown"
    )
    round4_input_source = (
        prediction_df["round4_input_source"].iloc[0]
        if "round4_input_source" in prediction_df.columns
        else "unknown"
    )

    st.caption(
        f"Prediction mode: {inference_mode} | "
        f"Round 2 input: {round1_input_source} | "
        f"Round 3 input: {round3_input_source} | "
        f"Round 4 input: {round4_input_source}"
    )

    if backtest_df is not None and not backtest_df.empty:
        metrics = format_metrics(backtest_df)
    else:
        metrics = {
            "n_players": len(prediction_df),
            "predicted_cut_count": int(prediction_df["made_cut_predicted"].sum()) if "made_cut_predicted" in prediction_df.columns else 0,
            "mae_round1": float("nan"),
            "rmse_round1": float("nan"),
            "mae_round2": float("nan"),
            "rmse_round2": float("nan"),
            "mae_round3": float("nan"),
            "rmse_round3": float("nan"),
            "mae_round4": float("nan"),
            "rmse_round4": float("nan"),
            "mae_total_r2": float("nan"),
            "rmse_total_r2": float("nan"),
            "mae_total_r3": float("nan"),
            "rmse_total_r3": float("nan"),
            "mae_total_final": float("nan"),
            "rmse_total_final": float("nan"),
        }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players", f"{metrics['n_players']}")
    col2.metric("Predicted made cut", f"{metrics['predicted_cut_count']}")
    col3.metric(
        "Final total MAE",
        f"{metrics['mae_total_final']:.3f}" if pd.notna(metrics["mae_total_final"]) else "N/A",
    )
    col4.metric(
        "Final total RMSE",
        f"{metrics['rmse_total_final']:.3f}" if pd.notna(metrics["rmse_total_final"]) else "N/A",
    )

    col5, col6, col7, col8 = st.columns(4)
    col5.metric(
        "Round 1 MAE",
        f"{metrics['mae_round1']:.3f}" if pd.notna(metrics["mae_round1"]) else "N/A",
    )
    col6.metric(
        "Round 2 MAE",
        f"{metrics['mae_round2']:.3f}" if pd.notna(metrics["mae_round2"]) else "N/A",
    )
    col7.metric(
        "Round 3 MAE",
        f"{metrics['mae_round3']:.3f}" if pd.notna(metrics["mae_round3"]) else "N/A",
    )
    col8.metric(
        "Round 4 MAE",
        f"{metrics['mae_round4']:.3f}" if pd.notna(metrics["mae_round4"]) else "N/A",
    )

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

    st.markdown("### Predicted final leaderboard")

    predicted_cols = [
        "predicted_rank_final",
        "predicted_rank_through_round3",
        "predicted_rank_through_round2",
        "player_name_clean",
        "made_cut_predicted",
        "predicted_round1",
        "predicted_round2",
        "predicted_round3",
        "predicted_round4",
        "predicted_total",
        "cut_line",
        "feature_source_tournament",
        "feature_source_start",
        "rolling_avg_last_3",
        "rolling_avg_last_5",
        "made_cut_rate_last_5",
        "career_tournament_count",
    ]
    predicted_cols = [col for col in predicted_cols if col in filtered_prediction_df.columns]

    sort_cols = ["predicted_rank_final", "player_name_clean"]
    sort_cols = [col for col in sort_cols if col in filtered_prediction_df.columns]

    predicted_view_df = filtered_prediction_df.sort_values(sort_cols).copy()

    st.dataframe(
        predicted_view_df[predicted_cols].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Prediction artifact")
    st.dataframe(
        filtered_prediction_df.sort_values(sort_cols),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("## Backtest view")

    if backtest_error:
        st.info(f"Backtest artifact not available: {backtest_error}")
        return

    if filtered_backtest_df is None or filtered_backtest_df.empty:
        st.warning("Backtest artifact is empty.")
        return

    st.markdown("### Actual final leaderboard")

    actual_sort_cols = ["actual_rank_final", "actual_total", "player_name_clean"]
    actual_sort_cols = [col for col in actual_sort_cols if col in filtered_backtest_df.columns]

    actual_df = filtered_backtest_df.sort_values(actual_sort_cols).copy()

    actual_cols = [
        "actual_rank_final",
        "predicted_rank_final",
        "player_name_clean",
        "actual_round1",
        "predicted_round1",
        "actual_round2",
        "predicted_round2",
        "actual_round3",
        "predicted_round3",
        "actual_round4",
        "predicted_round4",
        "actual_total",
        "predicted_total",
        "abs_error_total",
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