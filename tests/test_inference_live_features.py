from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines import inference_pipeline as inference


def make_live_features_df() -> pd.DataFrame:
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
                "Truist Championship",
                "Truist Championship",
                "Truist Championship",
                "Other Tournament",
                "Truist Championship",
                "Truist Championship",
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
            "live_total_to_par": [
                0,
                -1,
                2,
                3,
                4,
                5,
            ],
            "live_current_round_to_par": [
                0,
                -1,
                2,
                3,
                4,
                5,
            ],
            "live_position_numeric": [
                10,
                5,
                30,
                40,
                50,
                60,
            ],
            "live_completed_round_strokes": [
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "live_thru": [
                "5",
                "6",
                "4",
                "3",
                "2",
                "1",
            ],
            "live_round_complete": [
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_get_target_field_from_live_features_returns_expected_field() -> None:
    live_df = make_live_features_df()

    field_df = inference.get_target_field_from_live_features(
        live_df,
        tournament="Truist Championship",
        start_date="2026-05-07",
    )

    assert len(field_df) == 3
    assert set(field_df["player_name_clean"]) == {"alice", "bob", "charlie"}

    assert (field_df["tournament"] == "Truist Championship").all()
    assert (field_df["start"] == pd.Timestamp("2026-05-07")).all()
    assert (field_df["season"] == 2026).all()

    for round_col in ["round1", "round2", "round3", "round4"]:
        assert round_col in field_df.columns
        assert field_df[round_col].isna().all()


def test_get_target_field_from_live_features_removes_duplicate_players() -> None:
    live_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "alice", "bob"],
            "target_tournament": [
                "Truist Championship",
                "Truist Championship",
                "Truist Championship",
            ],
            "season": [2026, 2026, 2026],
            "round_id": [1, 1, 1],
            "live_total_to_par": [0, -1, 2],
            "live_current_round_to_par": [0, -1, 2],
            "live_position_numeric": [10, 5, 30],
            "live_completed_round_strokes": [pd.NA, pd.NA, pd.NA],
            "live_thru": ["5", "6", "4"],
            "live_round_complete": [False, False, False],
        }
    )

    field_df = inference.get_target_field_from_live_features(
        live_df,
        tournament="Truist Championship",
        start_date="2026-05-07",
    )

    assert len(field_df) == 2
    assert field_df["player_name_clean"].tolist() == ["alice", "bob"]


def test_get_target_field_from_live_features_filters_by_tournament() -> None:
    live_df = pd.DataFrame(
        {
            "player_name_clean": ["alice", "bob"],
            "target_tournament": ["Truist Championship", "Other Tournament"],
            "season": [2026, 2026],
            "round_id": [1, 1],
            "live_total_to_par": [0, -1],
            "live_current_round_to_par": [0, -1],
            "live_position_numeric": [10, 5],
            "live_completed_round_strokes": [pd.NA, pd.NA],
            "live_thru": ["5", "6"],
            "live_round_complete": [False, False],
        }
    )

    field_df = inference.get_target_field_from_live_features(
        live_df,
        tournament="Truist Championship",
        start_date="2026-05-07",
    )

    assert len(field_df) == 1
    assert field_df.iloc[0]["player_name_clean"] == "alice"
    assert field_df.iloc[0]["tournament"] == "Truist Championship"


def test_get_target_field_from_live_features_raises_when_no_rows_match() -> None:
    live_df = make_live_features_df()

    with pytest.raises(ValueError, match="No live feature rows found"):
        inference.get_target_field_from_live_features(
            live_df,
            tournament="Missing Tournament",
            start_date="2026-05-07",
        )


def test_load_live_features_reads_expected_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_features_path = tmp_path / "live_features.parquet"
    make_live_features_df().to_parquet(live_features_path, index=False)

    monkeypatch.setattr(
        inference,
        "LIVE_FEATURES_PATH",
        live_features_path,
    )

    loaded_df = inference.load_live_features()

    assert len(loaded_df) == 4
    assert loaded_df["player_name_clean"].tolist() == [
        "alice",
        "bob",
        "charlie",
        "duplicate alice",
    ]
    assert "player_name_clean" in loaded_df.columns
    assert "target_tournament" in loaded_df.columns
    assert "live_round_complete" in loaded_df.columns


def test_load_live_features_raises_for_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_path = tmp_path / "live_features.parquet"

    monkeypatch.setattr(
        inference,
        "LIVE_FEATURES_PATH",
        missing_path,
    )

    with pytest.raises(FileNotFoundError, match="Live features file not found"):
        inference.load_live_features()


def test_load_live_features_raises_for_missing_required_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_features_path = tmp_path / "live_features.parquet"

    bad_df = pd.DataFrame(
        {
            "player_name_clean": ["alice"],
            "tournament": ["Truist Championship"],
        }
    )
    bad_df.to_parquet(live_features_path, index=False)

    monkeypatch.setattr(
        inference,
        "LIVE_FEATURES_PATH",
        live_features_path,
    )

    with pytest.raises(ValueError, match="Missing required columns in live features file"):
        inference.load_live_features()


def test_get_target_field_for_source_uses_live_features_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_features_path = tmp_path / "live_features.parquet"
    make_live_features_df().to_parquet(live_features_path, index=False)

    monkeypatch.setattr(
        inference,
        "LIVE_FEATURES_PATH",
        live_features_path,
    )

    historical_df = pd.DataFrame()

    field_df = inference.get_target_field_for_source(
        historical_df=historical_df,
        tournament="Truist Championship",
        start_date="2026-05-07",
        field_source="live_features",
    )

    assert len(field_df) == 3
    assert set(field_df["player_name_clean"]) == {"alice", "bob", "charlie"}

    for round_col in ["round1", "round2", "round3", "round4"]:
        assert round_col in field_df.columns
        assert field_df[round_col].isna().all()


def test_live_features_field_source_keeps_backtest_actuals_empty() -> None:
    field_df = inference.get_target_field_from_live_features(
        make_live_features_df(),
        tournament="Truist Championship",
        start_date="2026-05-07",
    )

    actual_cols = ["round1", "round2", "round3", "round4"]

    assert not field_df[actual_cols].notna().any().any()