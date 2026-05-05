import pandas as pd
import pytest

from pipelines.inference_pipeline import (
    ROUND1_FEATURE_COLUMNS,
    apply_cut,
    build_backtest_output,
    build_prediction_output,
    build_pre_tournament_feature_rows,
    filter_players_making_cut,
    get_target_field,
    predict_round1,
    predict_round2,
    predict_round3,
    predict_round4,
    prepare_round2_features,
    prepare_round3_features,
    prepare_round4_features,
    validate_inference_mode,
)


class DummyRound1Model:
    def predict(self, X: pd.DataFrame):
        return X["prev_tournament_avg_score"].to_numpy()


class DummyRound2Model:
    def predict(self, X: pd.DataFrame):
        return X["round1"].to_numpy() - 1.0


class DummyRound3Model:
    def predict(self, X: pd.DataFrame):
        return X["round2"].to_numpy() - 1.0


class DummyRound4Model:
    def predict(self, X: pd.DataFrame):
        return X["round3"].to_numpy() - 1.0


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
            "round3": [72, 74, 66, 67, 73],
            "round4": [73, 75, 65, 66, 74],
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
            "days_since_last_tournament": [14, 14, 21, 21, None],
            "tournaments_last_30": [1, 1, 1, 1, None],
            "tournaments_last_60": [2, 2, 2, 2, None],
            "tournaments_last_90": [3, 3, 3, 3, None],
            "made_cut_streak": [1, 1, 2, 2, None],
            "missed_cut_streak": [0, 0, 0, 0, None],
            "round_std_last_5": [0.8, 0.8, 0.6, 0.6, None],
            "round_std_last_10": [1.0, 1.0, 0.9, 0.9, None],
            "score_range_last_5": [2.5, 2.5, 1.8, 1.8, None],
            "best_round_last_10": [68, 68, 66, 66, None],
            "worst_round_last_10": [74, 74, 70, 70, None],
            "best_total_last_10": [280, 280, 272, 272, None],
            "worst_total_last_10": [292, 292, 279, 279, None],
            "missed_cut_rate_last_10": [0.0, 0.0, 0.1, 0.1, None],
        }
    )


@pytest.fixture
def weekend_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name_clean": ["alice", "bob", "charlie"],
            "target_tournament": ["Masters Tournament"] * 3,
            "target_start": [pd.Timestamp("2025-04-10")] * 3,
            "target_season": [2025] * 3,
            "season": [2025] * 3,
            "feature_source_season": [2025] * 3,
            "feature_source_start": [
                pd.Timestamp("2025-03-01"),
                pd.Timestamp("2025-03-15"),
                pd.Timestamp("2025-03-20"),
            ],
            "feature_source_tournament": ["Event A", "Event B", "Event C"],
            "predicted_rank_round1": [2, 1, 3],
            "actual_rank_round1": [2, 1, 3],
            "predicted_round1": [70.5, 68.5, 71.5],
            "actual_round1": [72.0, 69.0, 74.0],
            "abs_error_round1": [1.5, 0.5, 2.5],
            "round1_input_source": ["predicted_round1"] * 3,
            "predicted_round2": [69.5, 67.5, 72.5],
            "actual_round2": [73.0, 68.0, 75.0],
            "abs_error_round2": [3.5, 0.5, 2.5],
            "predicted_total_through_round2": [140.0, 136.0, 144.0],
            "actual_total_through_round2": [145.0, 137.0, 149.0],
            "abs_error_total_through_round2": [5.0, 1.0, 5.0],
            "predicted_rank_through_round2": [2, 1, 3],
            "actual_rank_through_round2": [2, 1, 3],
            "cut_rule_top_n": [50] * 3,
            "cut_rule_ties": [True] * 3,
            "cut_rule_within_leader_strokes": [10] * 3,
            "leader_score_r2": [136.0] * 3,
            "cut_line": [144.0] * 3,
            "made_cut_predicted": [True, True, False],
            "prev_tournament_avg_score": [70.5, 68.5, 71.5],
            "prev_tournament_total": [282.0, 274.0, 286.0],
            "prev_tournament_made_cut": [1.0, 1.0, 0.0],
            "prev_tournament_earnings": [1000.0, 1500.0, 400.0],
            "rolling_avg_last_3": [70.5, 68.5, 71.5],
            "rolling_avg_last_5": [70.5, 68.5, 71.5],
            "rolling_total_last_3": [282.0, 274.0, 286.0],
            "made_cut_rate_last_5": [1.0, 1.0, 0.4],
            "form_index_last_3": [70.5, 68.5, 71.5],
            "career_tournament_count": [10.0, 8.0, 3.0],
            "days_since_last_tournament": [14.0, 21.0, 7.0],
            "tournaments_last_30": [1.0, 1.0, 2.0],
            "tournaments_last_60": [2.0, 2.0, 3.0],
            "tournaments_last_90": [3.0, 3.0, 4.0],
            "made_cut_streak": [3.0, 2.0, 0.0],
            "missed_cut_streak": [0.0, 0.0, 1.0],
            "actual_round3": [74.0, 67.0, 76.0],
            "actual_round4": [75.0, 66.0, 77.0],
            "inference_mode": ["live"] * 3,
            "round_std_last_5": [0.8, 0.6, 1.2],
            "round_std_last_10": [1.0, 0.9, 1.4],
            "score_range_last_5": [2.5, 1.8, 3.8],
            "best_round_last_10": [68.0, 66.0, 69.0],
            "worst_round_last_10": [74.0, 70.0, 78.0],
            "best_total_last_10": [280.0, 272.0, 284.0],
            "worst_total_last_10": [292.0, 279.0, 300.0],
            "missed_cut_rate_last_10": [0.0, 0.1, 0.5],
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
            "round3": [76],
            "round4": [77],
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
            "days_since_last_tournament": [30],
            "tournaments_last_30": [1],
            "tournaments_last_60": [1],
            "tournaments_last_90": [1],
            "made_cut_streak": [0],
            "missed_cut_streak": [1],
            "round_std_last_5": [1.5],
            "round_std_last_10": [2.0],
            "score_range_last_5": [6.0],
            "best_round_last_10": [72],
            "worst_round_last_10": [78],
            "best_total_last_10": [296],
            "worst_total_last_10": [306],
            "missed_cut_rate_last_10": [1.0],
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

    predictions_df = predict_round1(round1_model, inference_df, apply_calibration=False)
    predictions_df = predict_round2(round2_model, predictions_df, mode="backtest", apply_calibration=False)

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

    predictions_df = predict_round1(round1_model, inference_df, apply_calibration=False)
    predictions_df = predict_round2(round2_model, predictions_df, mode="live", apply_calibration=False)

    bob_row = predictions_df[predictions_df["player_name_clean"] == "bob"].iloc[0]
    alice_row = predictions_df[predictions_df["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["predicted_round2"] == bob_row["predicted_round1"] - 1.0
    assert alice_row["predicted_round2"] == alice_row["predicted_round1"] - 1.0
    assert (predictions_df["round1_input_source"] == "predicted_round1").all()
    assert (predictions_df["inference_mode"] == "live").all()


def test_prepare_round3_features_uses_actual_rounds_in_backtest_mode(
    weekend_predictions_df: pd.DataFrame,
) -> None:
    round3_df = prepare_round3_features(weekend_predictions_df, mode="backtest")

    kept = round3_df[round3_df["player_name_clean"].isin(["alice", "bob"])]

    assert (kept["round1"] == kept["actual_round1"]).all()
    assert (kept["round2"] == kept["actual_round2"]).all()
    assert (round3_df["round3_input_source"] == "actual_round1_actual_round2").all()
    assert (round3_df["inference_mode"] == "backtest").all()


def test_prepare_round3_features_uses_predicted_rounds_in_live_mode(
    weekend_predictions_df: pd.DataFrame,
) -> None:
    round3_df = prepare_round3_features(weekend_predictions_df, mode="live")

    kept = round3_df[round3_df["player_name_clean"].isin(["alice", "bob"])]

    assert (kept["round1"] == kept["predicted_round1"]).all()
    assert (kept["round2"] == kept["predicted_round2"]).all()
    assert (round3_df["round3_input_source"] == "predicted_round1_predicted_round2").all()
    assert (round3_df["inference_mode"] == "live").all()


def test_prepare_round4_features_uses_actual_rounds_in_backtest_mode(
    weekend_predictions_df: pd.DataFrame,
) -> None:
    df = weekend_predictions_df.copy()
    df["predicted_round3"] = [72.0, 66.0, 75.0]

    round4_df = prepare_round4_features(df, mode="backtest")

    kept = round4_df[round4_df["player_name_clean"].isin(["alice", "bob"])]

    assert (kept["round1"] == kept["actual_round1"]).all()
    assert (kept["round2"] == kept["actual_round2"]).all()
    assert (kept["round3"] == kept["actual_round3"]).all()
    assert (round4_df["round4_input_source"] == "actual_round1_actual_round2_actual_round3").all()
    assert (round4_df["inference_mode"] == "backtest").all()


def test_prepare_round4_features_uses_predicted_rounds_in_live_mode(
    weekend_predictions_df: pd.DataFrame,
) -> None:
    df = weekend_predictions_df.copy()
    df["predicted_round3"] = [68.5, 66.5, 73.5]

    round4_df = prepare_round4_features(df, mode="live")

    kept = round4_df[round4_df["player_name_clean"].isin(["alice", "bob"])]

    assert (kept["round1"] == kept["predicted_round1"]).all()
    assert (kept["round2"] == kept["predicted_round2"]).all()
    assert (kept["round3"] == kept["predicted_round3"]).all()
    assert (round4_df["round4_input_source"] == "predicted_round1_predicted_round2_predicted_round3").all()
    assert (round4_df["inference_mode"] == "live").all()


def test_filter_players_making_cut_keeps_only_true_rows(weekend_predictions_df: pd.DataFrame) -> None:
    result = filter_players_making_cut(weekend_predictions_df)

    assert set(result["player_name_clean"]) == {"alice", "bob"}
    assert result["made_cut_predicted"].all()


def test_predict_round3_only_uses_players_making_cut(weekend_predictions_df: pd.DataFrame) -> None:
    round3_model = DummyRound3Model()

    result = predict_round3(round3_model, weekend_predictions_df, mode="live", apply_calibration=False)

    assert set(result["player_name_clean"]) == {"alice", "bob"}
    assert "charlie" not in set(result["player_name_clean"])
    assert "predicted_round3" in result.columns
    assert "predicted_total_through_round3" in result.columns
    assert "predicted_rank_through_round3" in result.columns


def test_predict_round3_live_uses_predicted_round2(weekend_predictions_df: pd.DataFrame) -> None:
    round3_model = DummyRound3Model()

    result = predict_round3(
        round3_model,
        weekend_predictions_df,
        mode="live",
        apply_calibration=False,
    )

    bob_row = result[result["player_name_clean"] == "bob"].iloc[0]
    alice_row = result[result["player_name_clean"] == "alice"].iloc[0]

    assert bob_row["predicted_round3"] == bob_row["predicted_round2"] - 1.0
    assert alice_row["predicted_round3"] == alice_row["predicted_round2"] - 1.0


def test_predict_round3_backtest_adds_actuals_and_errors(weekend_predictions_df: pd.DataFrame) -> None:
    round3_model = DummyRound3Model()

    result = predict_round3(round3_model, weekend_predictions_df, mode="backtest")

    required_columns = {
        "predicted_round3",
        "actual_round3",
        "abs_error_round3",
        "predicted_total_through_round3",
        "actual_total_through_round3",
        "abs_error_total_through_round3",
        "predicted_rank_through_round3",
    }

    assert required_columns.issubset(result.columns)
    assert set(result["player_name_clean"]) == {"alice", "bob"}


def test_predict_round4_live_adds_final_total_and_rank(weekend_predictions_df: pd.DataFrame) -> None:
    round3_model = DummyRound3Model()
    round4_model = DummyRound4Model()

    round3_df = predict_round3(round3_model, weekend_predictions_df, mode="live")
    result = predict_round4(round4_model, round3_df, mode="live")

    required_columns = {
        "predicted_round4",
        "predicted_total",
        "predicted_rank_final",
        "round4_input_source",
    }

    assert required_columns.issubset(result.columns)
    assert set(result["player_name_clean"]) == {"alice", "bob"}


def test_predict_round4_backtest_adds_actuals_and_errors(weekend_predictions_df: pd.DataFrame) -> None:
    round3_model = DummyRound3Model()
    round4_model = DummyRound4Model()

    round3_df = predict_round3(round3_model, weekend_predictions_df, mode="backtest")
    result = predict_round4(round4_model, round3_df, mode="backtest")

    required_columns = {
        "predicted_round4",
        "actual_round4",
        "abs_error_round4",
        "predicted_total",
        "actual_total",
        "abs_error_total",
        "predicted_rank_final",
        "actual_rank_final",
    }

    assert required_columns.issubset(result.columns)
    assert set(result["player_name_clean"]) == {"alice", "bob"}


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

    predictions_df = predictions_df[predictions_df["made_cut_predicted"]].copy()
    predictions_df["round3_input_source"] = "predicted_round1_predicted_round2"
    predictions_df["round4_input_source"] = "predicted_round1_predicted_round2_predicted_round3"
    predictions_df["predicted_round3"] = [66.5, 69.5]
    predictions_df["predicted_round4"] = [65.5, 68.5]
    predictions_df["predicted_total_through_round3"] = (
        predictions_df["predicted_round1"]
        + predictions_df["predicted_round2"]
        + predictions_df["predicted_round3"]
    )
    predictions_df["predicted_total"] = (
        predictions_df["predicted_total_through_round3"]
        + predictions_df["predicted_round4"]
    )
    predictions_df["predicted_rank_through_round3"] = [1, 2]
    predictions_df["predicted_rank_final"] = [1, 2]

    prediction_output_df = build_prediction_output(predictions_df)

    expected_columns = [
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

    assert prediction_output_df.columns.tolist() == expected_columns
    assert len(prediction_output_df) == 2


def test_build_backtest_output_returns_expected_columns(weekend_predictions_df: pd.DataFrame) -> None:
    df = weekend_predictions_df[weekend_predictions_df["made_cut_predicted"]].copy()
    df["round3_input_source"] = "actual_round1_actual_round2"
    df["round4_input_source"] = "actual_round1_actual_round2_actual_round3"
    df["predicted_round3"] = [72.0, 66.0]
    df["abs_error_round3"] = (df["predicted_round3"] - df["actual_round3"]).abs()
    df["predicted_total_through_round3"] = (
        df["predicted_round1"] + df["predicted_round2"] + df["predicted_round3"]
    )
    df["actual_total_through_round3"] = (
        df["actual_round1"] + df["actual_round2"] + df["actual_round3"]
    )
    df["abs_error_total_through_round3"] = (
        df["predicted_total_through_round3"] - df["actual_total_through_round3"]
    ).abs()
    df["predicted_round4"] = [71.0, 65.0]
    df["abs_error_round4"] = (df["predicted_round4"] - df["actual_round4"]).abs()
    df["predicted_total"] = (
        df["predicted_total_through_round3"] + df["predicted_round4"]
    )
    df["actual_total"] = (
        df["actual_total_through_round3"] + df["actual_round4"]
    )
    df["abs_error_total"] = (df["predicted_total"] - df["actual_total"]).abs()
    df["predicted_rank_through_round3"] = [2, 1]
    df["predicted_rank_final"] = [2, 1]
    df["actual_rank_final"] = [2, 1]

    backtest_output_df = build_backtest_output(df)

    expected_columns = [
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