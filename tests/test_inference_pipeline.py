import pandas as pd
import pytest

from pipelines.inference_pipeline import (
    FEATURE_COLUMNS,
    build_backtest_output,
    build_prediction_output,
    build_pre_tournament_feature_rows,
    get_target_field,
    predict_round1,
)


class DummyModel:
    def predict(self, X: pd.DataFrame):
        # deterministic fake predictions for testing
        return X["prev_tournament_avg_score"].to_numpy()


@pytest.fixture
def sample_inference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name_clean": [
                "alice", "alice",
                "bob", "bob",
                "charlie",
            ],
            "tournament": [
                "Event A", "Masters Tournament",
                "Event B", "Masters Tournament",
                "Masters Tournament",
            ],
            "start": pd.to_datetime(
                [
                    "2025-03-01",
                    "2025-04-10",
                    "2025-03-15",
                    "2025-04-10",
                    "2025-04-10",
                ]
            ),
            "season": [2025, 2025, 2025, 2025, 2025],
            "round1": [70, 72, 68, 69, 71],
            "prev_tournament_avg_score": [70.5, 70.5, 68.5, 68.5, None],
            "prev_tournament_total": [282, 282, 274, 274, None],
            "prev_tournament_made_cut": [1, 1, 1, 1, None],
            "prev_tournament_earnings": [1000, 1000, 1500, 1500, None],
            "rolling_avg_last_3": [70.5, 70.5, 68.5, 68.5, None],
            "rolling_avg_last_5": [70.5, 70.5, 68.5, 68.5, None],
            "rolling_total_last_3": [282.0, 282.0, 274.0, 274.0, None],
            "made_cut_rate_last_5": [1.0, 1.0, 1.0, 1.0, None],
            "form_index_last_3": [70.5, 70.5, 68.5, 68.5, None],
            "career_tournament_count": [1, 1, 1, 1, None],
        }
    )


def test_get_target_field_selects_requested_event_and_date(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")

    assert len(field_df) == 3
    assert set(field_df["player_name_clean"]) == {"alice", "bob", "charlie"}
    assert (field_df["tournament"] == "Masters Tournament").all()
    assert (field_df["start"] == pd.Timestamp("2025-04-10")).all()


def test_get_target_field_raises_for_missing_event(sample_inference_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No rows found for tournament"):
        get_target_field(sample_inference_df, "Not A Real Event", "2025-04-10")


def test_build_pre_tournament_feature_rows_drops_players_without_history(
    sample_inference_df: pd.DataFrame,
) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")

    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    assert set(inference_df["player_name_clean"]) == {"alice", "bob"}
    assert "charlie" not in set(inference_df["player_name_clean"])


def test_build_pre_tournament_feature_rows_uses_latest_history_row(
    sample_inference_df: pd.DataFrame,
) -> None:
    extra_row = pd.DataFrame(
        {
            "player_name_clean": ["alice"],
            "tournament": ["Event Older"],
            "start": [pd.Timestamp("2025-02-01")],
            "season": [2025],
            "round1": [74],
            "prev_tournament_avg_score": [74.5],
            "prev_tournament_total": [298],
            "prev_tournament_made_cut": [0],
            "prev_tournament_earnings": [200],
            "rolling_avg_last_3": [74.5],
            "rolling_avg_last_5": [74.5],
            "rolling_total_last_3": [298.0],
            "made_cut_rate_last_5": [0.0],
            "form_index_last_3": [74.5],
            "career_tournament_count": [0],
        }
    )

    df = pd.concat([sample_inference_df, extra_row], ignore_index=True)
    field_df = get_target_field(df, "Masters Tournament", "2025-04-10")

    inference_df = build_pre_tournament_feature_rows(
        df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    alice_row = inference_df[inference_df["player_name_clean"] == "alice"].iloc[0]

    assert alice_row["feature_source_start"] == pd.Timestamp("2025-03-01")
    assert alice_row["feature_source_tournament"] == "Event A"
    assert alice_row["feature_source_season"] == 2025


def test_predict_round1_adds_prediction_columns_and_ranks(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    model = DummyModel()
    predictions_df = predict_round1(model, inference_df)

    required_columns = {
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
        "predicted_round1",
        "actual_round1",
        "abs_error",
        "predicted_rank",
        "actual_rank",
        *FEATURE_COLUMNS,
    }

    assert required_columns.issubset(predictions_df.columns)
    assert len(predictions_df) == 2

    # bob should rank ahead of alice because 68.5 < 70.5
    assert predictions_df.iloc[0]["player_name_clean"] == "bob"
    assert predictions_df.iloc[0]["predicted_rank"] == 1
    assert predictions_df.iloc[1]["player_name_clean"] == "alice"
    assert predictions_df.iloc[1]["predicted_rank"] == 2

    bob_row = predictions_df[predictions_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = predictions_df[predictions_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["actual_rank"] == 1
    assert alice_row["actual_rank"] == 2


def test_build_prediction_output_returns_expected_columns(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    model = DummyModel()
    predictions_df = predict_round1(model, inference_df)
    prediction_output_df = build_prediction_output(predictions_df)

    expected_columns = [
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

    assert prediction_output_df.columns.tolist() == expected_columns
    assert len(prediction_output_df) == 2


def test_build_backtest_output_returns_expected_columns(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    model = DummyModel()
    predictions_df = predict_round1(model, inference_df)
    backtest_output_df = build_backtest_output(predictions_df)

    expected_columns = [
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

    assert backtest_output_df.columns.tolist() == expected_columns
    assert len(backtest_output_df) == 2


def test_build_pre_tournament_feature_rows_contains_required_feature_columns(
    sample_inference_df: pd.DataFrame,
) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")

    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    for col in FEATURE_COLUMNS:
        assert col in inference_df.columns