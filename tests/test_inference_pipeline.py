import pandas as pd
import pytest

from pipelines.inference_pipeline import (
    ROUND1_FEATURE_COLUMNS,
    apply_cut,
    build_backtest_output,
    build_prediction_output,
    build_pre_tournament_feature_rows,
    get_target_field,
    predict_round1,
    predict_round2,
    prepare_round2_features,
    validate_inference_mode,
)


class DummyRound1Model:
    def predict(self, X: pd.DataFrame):
        # deterministic fake predictions for testing
        return X["prev_tournament_avg_score"].to_numpy()


class DummyRound2Model:
    def predict(self, X: pd.DataFrame):
        # deterministic fake predictions for testing
        return X["round1"].to_numpy() - 1.0


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
            "round2": [71, 73, 67, 68, 72],
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
            "round2": [75],
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

    model = DummyRound1Model()
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
        "abs_error_round1",
        "predicted_rank_round1",
        "actual_rank_round1",
        *ROUND1_FEATURE_COLUMNS,
    }

    assert required_columns.issubset(predictions_df.columns)
    assert len(predictions_df) == 2

    assert predictions_df.iloc[0]["player_name_clean"] == "bob"
    assert predictions_df.iloc[0]["predicted_rank_round1"] == 1
    assert predictions_df.iloc[1]["player_name_clean"] == "alice"
    assert predictions_df.iloc[1]["predicted_rank_round1"] == 2

    bob_row = predictions_df[predictions_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = predictions_df[predictions_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["actual_rank_round1"] == 1
    assert alice_row["actual_rank_round1"] == 2


def test_validate_inference_mode_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="Invalid inference mode"):
        validate_inference_mode("not_a_real_mode")


def test_prepare_round2_features_uses_actual_round1_in_backtest_mode(
    sample_inference_df: pd.DataFrame,
) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    round1_model = DummyRound1Model()
    predictions_df = predict_round1(round1_model, inference_df)

    round2_df = prepare_round2_features(predictions_df, mode="backtest")

    bob_row = round2_df[round2_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = round2_df[round2_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["round1"] == bob_row["actual_round1"]
    assert alice_row["round1"] == alice_row["actual_round1"]
    assert (round2_df["round1_input_source"] == "actual_round1").all()
    assert (round2_df["inference_mode"] == "backtest").all()


def test_prepare_round2_features_uses_predicted_round1_in_live_mode(
    sample_inference_df: pd.DataFrame,
) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    round1_model = DummyRound1Model()
    predictions_df = predict_round1(round1_model, inference_df)

    round2_df = prepare_round2_features(predictions_df, mode="live")

    bob_row = round2_df[round2_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = round2_df[round2_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["round1"] == bob_row["predicted_round1"]
    assert alice_row["round1"] == alice_row["predicted_round1"]
    assert (round2_df["round1_input_source"] == "predicted_round1").all()
    assert (round2_df["inference_mode"] == "live").all()


def test_predict_round2_backtest_adds_round2_and_total_columns(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    round1_model = DummyRound1Model()
    round2_model = DummyRound2Model()

    predictions_df = predict_round1(round1_model, inference_df)
    predictions_df = predict_round2(round2_model, predictions_df, mode="backtest")

    required_columns = {
        "predicted_round2",
        "actual_round2",
        "abs_error_round2",
        "predicted_total_through_round2",
        "actual_total_through_round2",
        "abs_error_total_through_round2",
        "predicted_rank_through_round2",
        "actual_rank_through_round2",
        "inference_mode",
        "round1_input_source",
    }

    assert required_columns.issubset(predictions_df.columns)
    assert len(predictions_df) == 2

    bob_row = predictions_df[predictions_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = predictions_df[predictions_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["predicted_round2"] == 68.0
    assert alice_row["predicted_round2"] == 71.0

    assert bob_row["predicted_total_through_round2"] == bob_row["predicted_round1"] + bob_row["predicted_round2"]
    assert alice_row["predicted_total_through_round2"] == alice_row["predicted_round1"] + alice_row["predicted_round2"]


def test_predict_round2_live_uses_predicted_round1(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    round1_model = DummyRound1Model()
    round2_model = DummyRound2Model()

    predictions_df = predict_round1(round1_model, inference_df)
    predictions_df = predict_round2(round2_model, predictions_df, mode="live")

    bob_row = predictions_df[predictions_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = predictions_df[predictions_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["predicted_round2"] == bob_row["predicted_round1"] - 1.0
    assert alice_row["predicted_round2"] == alice_row["predicted_round1"] - 1.0
    assert (predictions_df["round1_input_source"] == "predicted_round1").all()
    assert (predictions_df["inference_mode"] == "live").all()


def test_build_prediction_output_returns_expected_columns(sample_inference_df: pd.DataFrame) -> None:
    field_df = get_target_field(sample_inference_df, "Masters Tournament", "2025-04-10")
    inference_df = build_pre_tournament_feature_rows(
        sample_inference_df,
        field_df,
        pd.Timestamp("2025-04-10"),
    )

    round1_model = DummyRound1Model()
    round2_model = DummyRound2Model()

    predictions_df = predict_round1(round1_model, inference_df)
    predictions_df = predict_round2(round2_model, predictions_df, mode="live")
    predictions_df = apply_cut(predictions_df, tournament_name="Masters Tournament")
    prediction_output_df = build_prediction_output(predictions_df)

    expected_columns = [
        "predicted_rank_round1",
        "predicted_rank_through_round2",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
        "inference_mode",
        "round1_input_source",
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
        "predicted_round2",
        "predicted_total_through_round2",
        "cut_rule_top_n",
        "cut_rule_ties",
        "cut_rule_within_leader_strokes",
        "leader_score_r2",
        "cut_line",
        "made_cut_predicted",
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

    round1_model = DummyRound1Model()
    round2_model = DummyRound2Model()

    predictions_df = predict_round1(round1_model, inference_df)
    predictions_df = predict_round2(round2_model, predictions_df, mode="backtest")
    predictions_df = apply_cut(predictions_df, tournament_name="Masters Tournament")
    backtest_output_df = build_backtest_output(predictions_df)

    expected_columns = [
        "predicted_rank_round1",
        "actual_rank_round1",
        "predicted_rank_through_round2",
        "actual_rank_through_round2",
        "player_name_clean",
        "target_tournament",
        "target_start",
        "target_season",
        "feature_source_season",
        "feature_source_start",
        "feature_source_tournament",
        "inference_mode",
        "round1_input_source",
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
        "abs_error_round1",
        "predicted_round2",
        "actual_round2",
        "abs_error_round2",
        "predicted_total_through_round2",
        "actual_total_through_round2",
        "abs_error_total_through_round2",
        "cut_rule_top_n",
        "cut_rule_ties",
        "cut_rule_within_leader_strokes",
        "leader_score_r2",
        "cut_line",
        "made_cut_predicted",
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

    for col in ROUND1_FEATURE_COLUMNS:
        assert col in inference_df.columns


def test_apply_cut_default_top_65_and_ties() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": [f"player_{i}" for i in range(1, 71)],
            "predicted_round1": [70.0] * 70,
            "predicted_round2": [float(i) for i in range(70)],
            "predicted_total_through_round2": [70.0 + float(i) for i in range(70)],
        }
    )

    result = apply_cut(df, tournament_name="Sony Open in Hawaii")

    assert "made_cut_predicted" in result.columns
    assert "cut_line" in result.columns
    assert result["made_cut_predicted"].sum() == 65
    assert result["cut_line"].iloc[0] == 134.0


def test_apply_cut_default_top_65_and_ties_includes_ties() -> None:
    totals = [140.0] * 64 + [141.0] * 3 + [150.0] * 3

    df = pd.DataFrame(
        {
            "player_name_clean": [f"player_{i}" for i in range(1, 71)],
            "predicted_round1": [70.0] * 70,
            "predicted_round2": [t - 70.0 for t in totals],
            "predicted_total_through_round2": totals,
        }
    )

    result = apply_cut(df, tournament_name="Sony Open in Hawaii")

    assert result["cut_line"].iloc[0] == 141.0
    assert result["made_cut_predicted"].sum() == 67


def test_apply_cut_masters_includes_players_within_10_of_leader() -> None:
    totals = [140.0] * 50 + [149.0, 150.0] + [151.0, 152.0, 153.0]

    df = pd.DataFrame(
        {
            "player_name_clean": [f"player_{i}" for i in range(1, 56)],
            "predicted_round1": [70.0] * 55,
            "predicted_round2": [t - 70.0 for t in totals],
            "predicted_total_through_round2": totals,
        }
    )

    result = apply_cut(df, tournament_name="Masters Tournament")

    assert result["cut_rule_top_n"].iloc[0] == 50
    assert result["cut_rule_within_leader_strokes"].iloc[0] == 10
    assert result["leader_score_r2"].iloc[0] == 140.0
    assert result["cut_line"].iloc[0] == 140.0
    assert result["made_cut_predicted"].sum() == 52


def test_apply_cut_when_field_smaller_than_cut_number() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": [f"player_{i}" for i in range(1, 11)],
            "predicted_round1": [70.0] * 10,
            "predicted_round2": [float(i) for i in range(10)],
            "predicted_total_through_round2": [70.0 + float(i) for i in range(10)],
        }
    )

    result = apply_cut(df, tournament_name="Sony Open in Hawaii")

    assert len(result) == 10
    assert result["made_cut_predicted"].sum() == 10


def test_apply_cut_raises_on_missing_prediction_columns() -> None:
    df = pd.DataFrame(
        {
            "player_name_clean": ["a", "b", "c"],
            "predicted_round1": [70.0, 71.0, 72.0],
            "predicted_round2": [69.0, 70.0, 71.0],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns for cut logic"):
        apply_cut(df, tournament_name="Masters Tournament")