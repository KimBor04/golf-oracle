from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines import inference_pipeline as inference


def make_api_fields_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name_clean": [
                "alice",
                "bob",
                "charlie",
                "duplicate alice",
                "",
                None,
            ],
            "target_tournament": [
                "Cadillac Championship",
                "Cadillac Championship",
                "Cadillac Championship",
                "Other Tournament",
                "Cadillac Championship",
                "Cadillac Championship",
            ],
            "tourn_id": [
                "999",
                "999",
                "999",
                "123",
                "999",
                "999",
            ],
            "season": [
                2026,
                2026,
                2026,
                2026,
                2026,
                2026,
            ],
            "round_id": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "api_start_date": pd.to_datetime(
                [
                    "2026-04-30",
                    "2026-04-30",
                    "2026-04-30",
                    "2026-04-30",
                    "2026-04-30",
                    "2026-04-30",
                ]
            ),
            "api_end_date": pd.to_datetime(
                [
                    "2026-05-03",
                    "2026-05-03",
                    "2026-05-03",
                    "2026-05-03",
                    "2026-05-03",
                    "2026-05-03",
                ]
            ),
            "api_record_type": [
                "future_field",
                "future_field",
                "future_field",
                "future_field",
                "future_field",
                "future_field",
            ],
            "field_cache_status": [
                "available",
                "available",
                "available",
                "available",
                "available",
                "available",
            ],
            "playerid": [
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
            ],
        }
    )


def test_get_target_field_from_api_fields_returns_expected_field() -> None:
    api_fields_df = make_api_fields_df()

    field_df = inference.get_target_field_from_api_fields(
        api_fields_df=api_fields_df,
        tournament="Cadillac Championship",
        start_date="2026-04-30",
    )

    assert len(field_df) == 3
    assert set(field_df["player_name_clean"]) == {"alice", "bob", "charlie"}

    assert (field_df["tournament"] == "Cadillac Championship").all()
    assert (field_df["start"] == pd.Timestamp("2026-04-30")).all()
    assert (field_df["season"] == 2026).all()

    for round_col in ["round1", "round2", "round3", "round4"]:
        assert round_col in field_df.columns
        assert field_df[round_col].isna().all()


def test_get_target_field_from_api_fields_filters_by_start_date() -> None:
    api_fields_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "bob"],
            "target_tournament": ["Cadillac Championship", "Cadillac Championship"],
            "tourn_id": ["999", "999"],
            "season": [2026, 2026],
            "round_id": [1, 1],
            "api_start_date": pd.to_datetime(["2026-04-30", "2026-05-07"]),
            "api_record_type": ["future_field", "future_field"],
            "field_cache_status": ["available", "available"],
        }
    )

    field_df = inference.get_target_field_from_api_fields(
        api_fields_df=api_fields_df,
        tournament="Cadillac Championship",
        start_date="2026-04-30",
    )

    assert len(field_df) == 1
    assert field_df.iloc[0]["player_name_clean"] == "alice"
    assert field_df.iloc[0]["start"] == pd.Timestamp("2026-04-30")


def test_get_target_field_from_api_fields_prefers_available_rows() -> None:
    api_fields_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "bob"],
            "target_tournament": ["Cadillac Championship", "Cadillac Championship"],
            "tourn_id": ["999", "999"],
            "season": [2026, 2026],
            "round_id": [1, 1],
            "api_start_date": pd.to_datetime(["2026-04-30", "2026-04-30"]),
            "api_record_type": ["future_field", "future_field"],
            "field_cache_status": ["available", "incomplete"],
        }
    )

    field_df = inference.get_target_field_from_api_fields(
        api_fields_df=api_fields_df,
        tournament="Cadillac Championship",
        start_date="2026-04-30",
    )

    assert len(field_df) == 1
    assert field_df.iloc[0]["player_name_clean"] == "alice"
    assert field_df.iloc[0]["field_cache_status"] == "available"


def test_get_target_field_from_api_fields_removes_duplicate_players() -> None:
    api_fields_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "alice", "bob"],
            "target_tournament": [
                "Cadillac Championship",
                "Cadillac Championship",
                "Cadillac Championship",
            ],
            "tourn_id": ["999", "999", "999"],
            "season": [2026, 2026, 2026],
            "round_id": [1, 1, 1],
            "api_start_date": pd.to_datetime(
                ["2026-04-30", "2026-04-30", "2026-04-30"]
            ),
            "api_record_type": ["future_field", "future_field", "future_field"],
            "field_cache_status": ["available", "available", "available"],
            "playerid": ["p1", "p1-duplicate", "p2"],
        }
    )

    field_df = inference.get_target_field_from_api_fields(
        api_fields_df=api_fields_df,
        tournament="Cadillac Championship",
        start_date="2026-04-30",
    )

    assert len(field_df) == 2
    assert field_df["player_name_clean"].tolist() == ["alice", "bob"]


def test_get_target_field_from_api_fields_raises_when_no_rows_match() -> None:
    api_fields_df = make_api_fields_df()

    with pytest.raises(ValueError, match="No API field rows found"):
        inference.get_target_field_from_api_fields(
            api_fields_df=api_fields_df,
            tournament="Missing Tournament",
            start_date="2026-04-30",
        )


def test_load_api_fields_reads_expected_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_fields_path = tmp_path / "api_fields_2026.parquet"
    make_api_fields_df().to_parquet(api_fields_path, index=False)

    monkeypatch.setattr(
        inference,
        "api_fields_path_for_year",
        lambda year: api_fields_path,
    )

    loaded_df = inference.load_api_fields(year=2026)

    assert len(loaded_df) == 6
    assert "player_name_clean" in loaded_df.columns
    assert "target_tournament" in loaded_df.columns
    assert pd.api.types.is_datetime64_any_dtype(loaded_df["api_start_date"])


def test_load_api_fields_raises_for_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_path = tmp_path / "api_fields_2026.parquet"

    monkeypatch.setattr(
        inference,
        "api_fields_path_for_year",
        lambda year: missing_path,
    )

    with pytest.raises(FileNotFoundError, match="API fields file not found"):
        inference.load_api_fields(year=2026)


def test_load_api_fields_raises_for_missing_required_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_fields_path = tmp_path / "api_fields_2026.parquet"

    bad_df = pd.DataFrame(
        {
            "player_name_clean": ["alice"],
            "target_tournament": ["Cadillac Championship"],
        }
    )
    bad_df.to_parquet(api_fields_path, index=False)

    monkeypatch.setattr(
        inference,
        "api_fields_path_for_year",
        lambda year: api_fields_path,
    )

    with pytest.raises(ValueError, match="Missing required columns in API fields file"):
        inference.load_api_fields(year=2026)


def test_get_target_field_for_source_uses_historical_source() -> None:
    historical_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "bob"],
            "tournament": ["Masters Tournament", "Masters Tournament"],
            "start": pd.to_datetime(["2025-04-10", "2025-04-10"]),
            "season": [2025, 2025],
            "round1": [70.0, 71.0],
            "round2": [70.0, 71.0],
            "round3": [70.0, 71.0],
            "round4": [70.0, 71.0],
        }
    )

    field_df = inference.get_target_field_for_source(
        historical_df=historical_df,
        tournament="Masters Tournament",
        start_date="2025-04-10",
        field_source="historical",
    )

    assert len(field_df) == 2
    assert set(field_df["player_name_clean"]) == {"alice", "bob"}


def test_get_target_field_for_source_uses_api_fields_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_fields_path = tmp_path / "api_fields_2026.parquet"
    make_api_fields_df().to_parquet(api_fields_path, index=False)

    monkeypatch.setattr(
        inference,
        "api_fields_path_for_year",
        lambda year: api_fields_path,
    )

    historical_df = pd.DataFrame()

    field_df = inference.get_target_field_for_source(
        historical_df=historical_df,
        tournament="Cadillac Championship",
        start_date="2026-04-30",
        field_source="api_fields",
    )

    assert len(field_df) == 3
    assert set(field_df["player_name_clean"]) == {"alice", "bob", "charlie"}


def test_get_target_field_for_source_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="Invalid FIELD_SOURCE"):
        inference.get_target_field_for_source(
            historical_df=pd.DataFrame(),
            tournament="Cadillac Championship",
            start_date="2026-04-30",
            field_source="not_real",
        )


def test_has_backtest_actuals_returns_true_when_any_actual_exists() -> None:
    predictions_df = pd.DataFrame(
        {
            "actual_round1": [pd.NA, 70.0],
            "actual_round2": [pd.NA, pd.NA],
            "actual_round3": [pd.NA, pd.NA],
            "actual_round4": [pd.NA, pd.NA],
        }
    )

    assert inference.has_backtest_actuals(predictions_df)


def test_has_backtest_actuals_returns_false_when_all_actuals_missing() -> None:
    predictions_df = pd.DataFrame(
        {
            "actual_round1": [pd.NA, pd.NA],
            "actual_round2": [pd.NA, pd.NA],
            "actual_round3": [pd.NA, pd.NA],
            "actual_round4": [pd.NA, pd.NA],
        }
    )

    assert not inference.has_backtest_actuals(predictions_df)


def test_has_backtest_actuals_returns_false_when_columns_are_missing() -> None:
    predictions_df = pd.DataFrame(
        {
            "actual_round1": [70.0],
        }
    )

    assert inference.has_backtest_actuals(predictions_df) is False


def test_build_empty_backtest_output_returns_expected_columns() -> None:
    empty_df = inference.build_empty_backtest_output()

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

    assert empty_df.empty
    assert empty_df.columns.tolist() == expected_columns