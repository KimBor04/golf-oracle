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
        df["feature_source_start"] = pd.to_datetime(
            df["feature_source_start"],
            errors="coerce",
        )

    return df


@st.cache_data
def load_backtest_df() -> pd.DataFrame:
    if not BACKTEST_PATH.exists():
        raise FileNotFoundError(f"Backtest file not found: {BACKTEST_PATH}")

    df = pd.read_parquet(BACKTEST_PATH)

    if "target_start" in df.columns:
        df["target_start"] = pd.to_datetime(df["target_start"], errors="coerce")

    if "feature_source_start" in df.columns:
        df["feature_source_start"] = pd.to_datetime(
            df["feature_source_start"],
            errors="coerce",
        )

    return df


def format_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    metrics = {
        "n_players": len(df),
        "predicted_cut_count": int(df["made_cut_predicted"].sum())
        if "made_cut_predicted" in df.columns
        else 0,
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

    if df.empty:
        return metrics

    if "abs_error_round1" in df.columns:
        metrics["mae_round1"] = df["abs_error_round1"].mean()

    if {"predicted_round1", "actual_round1"}.issubset(df.columns):
        valid = df[["predicted_round1", "actual_round1"]].dropna()
        if not valid.empty:
            metrics["rmse_round1"] = (
                ((valid["predicted_round1"] - valid["actual_round1"]) ** 2).mean()
                ** 0.5
            )

    if "abs_error_round2" in df.columns:
        metrics["mae_round2"] = df["abs_error_round2"].mean()

    if {"predicted_round2", "actual_round2"}.issubset(df.columns):
        valid = df[["predicted_round2", "actual_round2"]].dropna()
        if not valid.empty:
            metrics["rmse_round2"] = (
                ((valid["predicted_round2"] - valid["actual_round2"]) ** 2).mean()
                ** 0.5
            )

    if "abs_error_round3" in df.columns:
        metrics["mae_round3"] = df["abs_error_round3"].mean()

    if {"predicted_round3", "actual_round3"}.issubset(df.columns):
        valid = df[["predicted_round3", "actual_round3"]].dropna()
        if not valid.empty:
            metrics["rmse_round3"] = (
                ((valid["predicted_round3"] - valid["actual_round3"]) ** 2).mean()
                ** 0.5
            )

    if "abs_error_round4" in df.columns:
        metrics["mae_round4"] = df["abs_error_round4"].mean()

    if {"predicted_round4", "actual_round4"}.issubset(df.columns):
        valid = df[["predicted_round4", "actual_round4"]].dropna()
        if not valid.empty:
            metrics["rmse_round4"] = (
                ((valid["predicted_round4"] - valid["actual_round4"]) ** 2).mean()
                ** 0.5
            )

    if "abs_error_total_through_round2" in df.columns:
        metrics["mae_total_r2"] = df["abs_error_total_through_round2"].mean()

    if {"predicted_total_through_round2", "actual_total_through_round2"}.issubset(
        df.columns
    ):
        valid = df[
            ["predicted_total_through_round2", "actual_total_through_round2"]
        ].dropna()
        if not valid.empty:
            metrics["rmse_total_r2"] = (
                (
                    (
                        valid["predicted_total_through_round2"]
                        - valid["actual_total_through_round2"]
                    )
                    ** 2
                ).mean()
                ** 0.5
            )

    if "abs_error_total_through_round3" in df.columns:
        metrics["mae_total_r3"] = df["abs_error_total_through_round3"].mean()

    if {"predicted_total_through_round3", "actual_total_through_round3"}.issubset(
        df.columns
    ):
        valid = df[
            ["predicted_total_through_round3", "actual_total_through_round3"]
        ].dropna()
        if not valid.empty:
            metrics["rmse_total_r3"] = (
                (
                    (
                        valid["predicted_total_through_round3"]
                        - valid["actual_total_through_round3"]
                    )
                    ** 2
                ).mean()
                ** 0.5
            )

    if "abs_error_total" in df.columns:
        metrics["mae_total_final"] = df["abs_error_total"].mean()

    if {"predicted_total", "actual_total"}.issubset(df.columns):
        valid = df[["predicted_total", "actual_total"]].dropna()
        if not valid.empty:
            metrics["rmse_total_final"] = (
                ((valid["predicted_total"] - valid["actual_total"]) ** 2).mean()
                ** 0.5
            )

    return metrics


def empty_metrics(prediction_df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n_players": len(prediction_df),
        "predicted_cut_count": int(prediction_df["made_cut_predicted"].sum())
        if "made_cut_predicted" in prediction_df.columns
        else 0,
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


def metric_value(value: float | int) -> str:
    if pd.notna(value):
        if isinstance(value, int):
            return str(value)
        return f"{value:.3f}"

    return "N/A"


def get_first_value(df: pd.DataFrame, column: str, default: str = "unknown") -> str:
    if column not in df.columns or df.empty:
        return default

    value = df[column].iloc[0]
    if pd.isna(value):
        return default

    return str(value)


def filter_by_player(df: pd.DataFrame, player_search: str) -> pd.DataFrame:
    if not player_search.strip():
        return df.copy()

    if "player_name_clean" not in df.columns:
        return df.copy()

    return df[
        df["player_name_clean"].str.contains(player_search, case=False, na=False)
    ].copy()


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def sort_dataframe(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    available_sort_columns = available_columns(df, sort_columns)

    if not available_sort_columns:
        return df.copy()

    return df.sort_values(available_sort_columns).copy()


def render_header(prediction_df: pd.DataFrame) -> None:
    st.title("🏌️ Golf Oracle")
    st.subheader("Full round-by-round tournament projection through Round 4")

    tournament_name = get_first_value(
        prediction_df,
        "target_tournament",
        "Unknown Tournament",
    )

    target_start = (
        prediction_df["target_start"].iloc[0]
        if "target_start" in prediction_df.columns
        else None
    )

    if pd.notna(target_start):
        st.caption(f"{tournament_name} — {target_start.date()}")
    else:
        st.caption(tournament_name)

    inference_mode = get_first_value(prediction_df, "inference_mode")
    round1_input_source = get_first_value(prediction_df, "round1_input_source")
    round3_input_source = get_first_value(prediction_df, "round3_input_source")
    round4_input_source = get_first_value(prediction_df, "round4_input_source")

    st.caption(
        f"Prediction mode: {inference_mode} | "
        f"Round 2 input: {round1_input_source} | "
        f"Round 3 input: {round3_input_source} | "
        f"Round 4 input: {round4_input_source}"
    )


def render_top_metrics(
    prediction_df: pd.DataFrame,
    backtest_df: pd.DataFrame | None,
) -> dict[str, float | int]:
    if backtest_df is not None and not backtest_df.empty:
        metrics = format_metrics(backtest_df)
    else:
        metrics = empty_metrics(prediction_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Players", f"{metrics['n_players']}")
    col2.metric("Predicted made cut", f"{metrics['predicted_cut_count']}")
    col3.metric("Final total MAE", metric_value(metrics["mae_total_final"]))
    col4.metric("Final total RMSE", metric_value(metrics["rmse_total_final"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Round 1 MAE", metric_value(metrics["mae_round1"]))
    col6.metric("Round 2 MAE", metric_value(metrics["mae_round2"]))
    col7.metric("Round 3 MAE", metric_value(metrics["mae_round3"]))
    col8.metric("Round 4 MAE", metric_value(metrics["mae_round4"]))

    return metrics


def render_leaderboard_tab(
    prediction_df: pd.DataFrame,
    player_search: str,
    show_top_n: int,
) -> None:
    st.markdown("### Predicted final leaderboard")

    filtered_prediction_df = filter_by_player(prediction_df, player_search)

    leaderboard_columns = [
        "predicted_rank_final",
        "player_name_clean",
        "predicted_total",
        "predicted_round1",
        "predicted_round2",
        "predicted_round3",
        "predicted_round4",
        "made_cut_predicted",
        "feature_source_tournament",
        "feature_source_start",
    ]
    leaderboard_columns = available_columns(filtered_prediction_df, leaderboard_columns)

    predicted_view_df = sort_dataframe(
        filtered_prediction_df,
        ["predicted_rank_final", "player_name_clean"],
    )

    st.dataframe(
        predicted_view_df[leaderboard_columns].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show full prediction artifact"):
        st.dataframe(
            predicted_view_df,
            use_container_width=True,
            hide_index=True,
        )


def render_cut_summary_tab(prediction_df: pd.DataFrame) -> None:
    st.markdown("### Predicted cut summary")

    players_total = len(prediction_df)

    if "made_cut_predicted" in prediction_df.columns:
        players_made_cut = int(prediction_df["made_cut_predicted"].sum())
        players_missed_cut = players_total - players_made_cut
    else:
        players_made_cut = 0
        players_missed_cut = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Players in field", players_total)
    col2.metric("Predicted made cut", players_made_cut)
    col3.metric("Predicted missed cut", players_missed_cut)

    cut_columns = [
        "predicted_rank_through_round2",
        "player_name_clean",
        "made_cut_predicted",
        "predicted_total_through_round2",
        "cut_line",
        "leader_score_r2",
        "cut_rule_top_n",
        "cut_rule_ties",
        "cut_rule_within_leader_strokes",
    ]
    cut_columns = available_columns(prediction_df, cut_columns)

    if not cut_columns:
        st.info("No cut metadata is available in the current prediction artifact.")
        return

    cut_df = sort_dataframe(
        prediction_df,
        ["made_cut_predicted", "predicted_total_through_round2", "player_name_clean"],
    )

    st.markdown("### Cut detail table")
    st.dataframe(
        cut_df[cut_columns],
        use_container_width=True,
        hide_index=True,
    )


def render_backtest_tab(
    backtest_df: pd.DataFrame | None,
    backtest_error: str | None,
    player_search: str,
    show_top_n: int,
) -> None:
    st.markdown("### Backtest evaluation")

    if backtest_error:
        st.info(f"Backtest artifact not available: {backtest_error}")
        return

    if backtest_df is None or backtest_df.empty:
        st.info(
            "No backtest rows are available for this run. "
            "This is expected for API-field inference when actual round scores are not available."
        )
        return

    filtered_backtest_df = filter_by_player(backtest_df, player_search)

    if filtered_backtest_df.empty:
        st.warning("No backtest rows match the current player search.")
        return

    metrics = format_metrics(filtered_backtest_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final total MAE", metric_value(metrics["mae_total_final"]))
    col2.metric("Final total RMSE", metric_value(metrics["rmse_total_final"]))
    col3.metric("Round 1 MAE", metric_value(metrics["mae_round1"]))
    col4.metric("Round 2 MAE", metric_value(metrics["mae_round2"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Round 3 MAE", metric_value(metrics["mae_round3"]))
    col6.metric("Round 4 MAE", metric_value(metrics["mae_round4"]))
    col7.metric("Total through R2 MAE", metric_value(metrics["mae_total_r2"]))
    col8.metric("Total through R3 MAE", metric_value(metrics["mae_total_r3"]))

    actual_df = sort_dataframe(
        filtered_backtest_df,
        ["actual_rank_final", "actual_total", "player_name_clean"],
    )

    actual_columns = [
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
    actual_columns = available_columns(actual_df, actual_columns)

    st.markdown("### Actual vs predicted final leaderboard")
    st.dataframe(
        actual_df[actual_columns].head(show_top_n),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show full backtest artifact"):
        st.dataframe(
            filtered_backtest_df,
            use_container_width=True,
            hide_index=True,
        )


def render_model_info_tab(
    prediction_df: pd.DataFrame,
    backtest_df: pd.DataFrame | None,
    metrics: dict[str, float | int],
) -> None:
    st.markdown("### Model and artifact information")

    st.markdown(
        """
        The current baseline uses one XGBoost regression model per round.
        The UI does not run inference directly. It only reads precomputed
        Parquet artifacts from the `predictions/` folder.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Prediction artifact")
        st.write(f"`{PREDICTION_PATH}`")
        st.metric("Prediction rows", len(prediction_df))

        if "inference_mode" in prediction_df.columns:
            st.write(f"**Inference mode:** {get_first_value(prediction_df, 'inference_mode')}")

        if "target_tournament" in prediction_df.columns:
            st.write(
                f"**Target tournament:** "
                f"{get_first_value(prediction_df, 'target_tournament')}"
            )

        if "target_start" in prediction_df.columns:
            target_start = prediction_df["target_start"].iloc[0]
            if pd.notna(target_start):
                st.write(f"**Target start:** {target_start.date()}")

    with col2:
        st.markdown("#### Backtest artifact")
        st.write(f"`{BACKTEST_PATH}`")

        if backtest_df is not None:
            st.metric("Backtest rows", len(backtest_df))
        else:
            st.metric("Backtest rows", 0)

        st.write(f"**Final total MAE:** {metric_value(metrics['mae_total_final'])}")
        st.write(f"**Final total RMSE:** {metric_value(metrics['rmse_total_final'])}")

    st.markdown("### Current model artifacts")

    model_info = pd.DataFrame(
        [
            {
                "round": "Round 1",
                "target": "round1",
                "artifact": "models/xgb_round1_baseline.joblib",
                "live_inputs": "historical pre-tournament features",
            },
            {
                "round": "Round 2",
                "target": "round2",
                "artifact": "models/xgb_round2_baseline.joblib",
                "live_inputs": "historical features + predicted_round1",
            },
            {
                "round": "Round 3",
                "target": "round3",
                "artifact": "models/xgb_round3_baseline.joblib",
                "live_inputs": "historical features + predicted_round1 + predicted_round2",
            },
            {
                "round": "Round 4",
                "target": "round4",
                "artifact": "models/xgb_round4_baseline.joblib",
                "live_inputs": "historical features + predicted_round1 + predicted_round2 + predicted_round3",
            },
        ]
    )

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
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

    render_header(prediction_df)
    metrics = render_top_metrics(prediction_df, backtest_df)

    st.markdown("---")

    player_search = st.text_input("Search player", "")
    show_top_n = st.slider(
        "Show top N rows",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )

    tab_leaderboard, tab_cut, tab_backtest, tab_model = st.tabs(
        [
            "Final Leaderboard",
            "Cut Summary",
            "Backtest Evaluation",
            "Model Info",
        ]
    )

    with tab_leaderboard:
        render_leaderboard_tab(
            prediction_df=prediction_df,
            player_search=player_search,
            show_top_n=show_top_n,
        )

    with tab_cut:
        render_cut_summary_tab(prediction_df=prediction_df)

    with tab_backtest:
        render_backtest_tab(
            backtest_df=backtest_df,
            backtest_error=backtest_error,
            player_search=player_search,
            show_top_n=show_top_n,
        )

    with tab_model:
        render_model_info_tab(
            prediction_df=prediction_df,
            backtest_df=backtest_df,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()