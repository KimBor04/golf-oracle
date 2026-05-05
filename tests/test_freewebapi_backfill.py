from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipelines import freewebapi_backfill as backfill


class FailingClient:
    def get_leaderboard(self, *args, **kwargs):
        raise AssertionError("The test should not make a real API call.")

    def get_schedule(self, *args, **kwargs):
        raise AssertionError("The test should not make a real API call.")


# ──────────────────────────────────────────────
# API budget
# ──────────────────────────────────────────────


def test_api_call_budget_tracks_remaining_calls() -> None:
    budget = backfill.ApiCallBudget(max_calls=2)

    assert budget.can_call() is True
    assert budget.remaining == 2

    budget.consume()

    assert budget.used_calls == 1
    assert budget.remaining == 1

    budget.consume()

    assert budget.used_calls == 2
    assert budget.remaining == 0
    assert budget.can_call() is False

    with pytest.raises(RuntimeError, match="API call budget exhausted"):
        budget.consume()


# ──────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────


def test_fetch_leaderboard_round_if_missing_uses_cached_file_without_api_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill, "API_CACHE_DIR", tmp_path)

    cached_payload = {"leaderboardRows": [{"firstname": "Scottie", "lastname": "Scheffler"}]}
    cache_path = backfill.leaderboard_cache_path(
        org_id=backfill.ORG_ID,
        tourn_id="014",
        year=2026,
        round_id=1,
    )
    backfill.write_json(cache_path, cached_payload)

    budget = backfill.ApiCallBudget(max_calls=0)

    result = backfill.fetch_leaderboard_round_if_missing(
        client=FailingClient(),
        tourn_id="014",
        year=2026,
        round_id=1,
        budget=budget,
        refresh_cache=False,
    )

    assert result == cached_payload
    assert budget.used_calls == 0


def test_fetch_leaderboard_round_if_missing_returns_none_when_budget_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill, "API_CACHE_DIR", tmp_path)

    budget = backfill.ApiCallBudget(max_calls=0)

    result = backfill.fetch_leaderboard_round_if_missing(
        client=FailingClient(),
        tourn_id="999",
        year=2026,
        round_id=1,
        budget=budget,
        refresh_cache=False,
    )

    assert result is None
    assert budget.used_calls == 0


def test_result_cache_status_detects_missing_partial_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill, "API_CACHE_DIR", tmp_path)

    # missing: no cached rounds for tournament A
    assert backfill.result_cache_status_for_tournament("A", 2026) == "missing"

    # partial: only rounds 1 and 2 cached for tournament B
    for round_id in [1, 2]:
        path = backfill.leaderboard_cache_path(backfill.ORG_ID, "B", 2026, round_id)
        backfill.write_json(path, {"round": round_id})

    assert backfill.cached_round_ids_for_tournament("B", 2026) == {1, 2}
    assert backfill.result_cache_status_for_tournament("B", 2026) == "partial"

    # complete: all configured rounds cached for tournament C
    for round_id in backfill.ROUND_IDS:
        path = backfill.leaderboard_cache_path(backfill.ORG_ID, "C", 2026, round_id)
        backfill.write_json(path, {"round": round_id})

    assert backfill.cached_round_ids_for_tournament("C", 2026) == set(backfill.ROUND_IDS)
    assert backfill.result_cache_status_for_tournament("C", 2026) == "complete"


# ──────────────────────────────────────────────
# Tournament selection logic
# ──────────────────────────────────────────────


def test_select_completed_tournaments_prioritizes_partial_cache_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backfill, "API_CACHE_DIR", tmp_path)

    # Tournament 100 is partial.
    for round_id in [1, 2]:
        backfill.write_json(
            backfill.leaderboard_cache_path(backfill.ORG_ID, "100", 2026, round_id),
            {"round": round_id},
        )

    # Tournament 300 is complete.
    for round_id in backfill.ROUND_IDS:
        backfill.write_json(
            backfill.leaderboard_cache_path(backfill.ORG_ID, "300", 2026, round_id),
            {"round": round_id},
        )

    target_tournaments = pd.DataFrame(
        {
            "tourn_id": ["200", "300", "100"],
            "target_tournament": ["Missing Event", "Complete Event", "Partial Event"],
            "api_start_date": pd.to_datetime(["2026-01-08", "2026-01-15", "2026-01-22"]),
            "api_end_date": pd.to_datetime(["2026-01-11", "2026-01-18", "2026-01-25"]),
        }
    )

    selected = backfill.select_completed_tournaments(
        target_tournaments_df=target_tournaments,
        as_of_date=pd.Timestamp("2026-02-01"),
        year=2026,
    )

    assert selected["result_cache_status"].tolist() == ["partial", "missing", "complete"]
    assert selected["tourn_id"].tolist() == ["100", "200", "300"]
    assert selected.loc[0, "missing_round_ids"] == [3, 4]


def test_select_next_tournaments_respects_lookahead_window() -> None:
    target_tournaments = pd.DataFrame(
        {
            "tourn_id": ["A", "B", "C"],
            "target_tournament": ["Today Event", "Soon Event", "Too Far Event"],
            "api_start_date": pd.to_datetime(["2026-05-05", "2026-05-10", "2026-05-20"]),
        }
    )

    selected = backfill.select_next_tournaments(
        target_tournaments_df=target_tournaments,
        as_of_date=pd.Timestamp("2026-05-05"),
        next_events=5,
        field_lookahead_days=7,
    )

    assert selected["tourn_id"].tolist() == ["A", "B"]


def test_select_next_tournaments_respects_next_events_limit() -> None:
    target_tournaments = pd.DataFrame(
        {
            "tourn_id": ["A", "B", "C"],
            "target_tournament": ["Event A", "Event B", "Event C"],
            "api_start_date": pd.to_datetime(["2026-05-06", "2026-05-07", "2026-05-08"]),
        }
    )

    selected = backfill.select_next_tournaments(
        target_tournaments_df=target_tournaments,
        as_of_date=pd.Timestamp("2026-05-05"),
        next_events=2,
        field_lookahead_days=10,
    )

    assert selected["tourn_id"].tolist() == ["A", "B"]


# ──────────────────────────────────────────────
# Field status logic
# ──────────────────────────────────────────────


def test_classify_field_cache_status() -> None:
    assert backfill.classify_field_cache_status(pd.DataFrame()) == "not_available_yet"

    no_player_name_column = pd.DataFrame({"playerid": ["1", "2"]})
    assert backfill.classify_field_cache_status(no_player_name_column) == "incomplete"

    incomplete_field = pd.DataFrame(
        {
            "player_name_clean": ["player one", "player two", ""],
        }
    )
    assert backfill.classify_field_cache_status(incomplete_field) == "incomplete"

    available_field = pd.DataFrame(
        {
            "player_name_clean": [f"player {idx}" for idx in range(25)],
        }
    )
    assert backfill.classify_field_cache_status(available_field) == "available"


def test_load_existing_field_statuses_reads_latest_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields_path = tmp_path / "api_fields_2026.parquet"
    monkeypatch.setattr(backfill, "api_fields_path", lambda year: fields_path)

    fields_df = pd.DataFrame(
        {
            "tourn_id": ["014", "014", "999"],
            "season": [2026, 2026, 2026],
            "round_id": [1, 1, 1],
            "playerid": ["a", "b", "c"],
            "field_cache_status": ["incomplete", "available", "failed"],
        }
    )
    fields_df.to_parquet(fields_path, index=False)

    statuses = backfill.load_existing_field_statuses(2026)

    assert statuses[("014", 2026)] == "available"
    assert statuses[("999", 2026)] == "failed"


def test_fetch_next_fields_skips_available_field_without_api_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fields_path = tmp_path / "api_fields_2026.parquet"
    monkeypatch.setattr(backfill, "api_fields_path", lambda year: fields_path)
    monkeypatch.setattr(backfill, "API_CACHE_DIR", tmp_path / "api_cache")

    existing_fields = pd.DataFrame(
        {
            "target_tournament": ["Already Available Event"],
            "target_tournament_clean": ["already available event"],
            "tourn_id": ["014"],
            "season": [2026],
            "round_id": [1],
            "playerid": ["player-1"],
            "player_name_clean": ["player one"],
            "field_cache_status": ["available"],
            "api_record_type": ["future_field"],
        }
    )
    existing_fields.to_parquet(fields_path, index=False)

    target_tournaments = pd.DataFrame(
        {
            "tourn_id": ["014"],
            "target_tournament": ["Already Available Event"],
            "api_start_date": pd.to_datetime(["2026-05-07"]),
            "api_end_date": pd.to_datetime(["2026-05-10"]),
        }
    )

    budget = backfill.ApiCallBudget(max_calls=0)

    result = backfill.fetch_next_fields(
        client=FailingClient(),
        target_tournaments_df=target_tournaments,
        year=2026,
        budget=budget,
        as_of_date=pd.Timestamp("2026-05-05"),
        next_events=1,
        field_lookahead_days=7,
        refresh_cache=False,
    )

    assert budget.used_calls == 0
    assert len(result) == 1
    assert result.loc[0, "field_cache_status"] == "available"


# ──────────────────────────────────────────────
# DataFrame utility helpers
# ──────────────────────────────────────────────


def test_concat_and_dedupe_keeps_latest_matching_row() -> None:
    existing = pd.DataFrame(
        {
            "tourn_id": ["014"],
            "season": [2026],
            "round_id": [1],
            "playerid": ["p1"],
            "value": ["old"],
        }
    )
    new = pd.DataFrame(
        {
            "tourn_id": ["014", "014"],
            "season": [2026, 2026],
            "round_id": [1, 1],
            "playerid": ["p1", "p2"],
            "value": ["new", "other"],
        }
    )

    combined = backfill.concat_and_dedupe(
        existing_df=existing,
        new_df=new,
        subset=["tourn_id", "season", "round_id", "playerid"],
    )

    assert len(combined) == 2
    assert combined.loc[combined["playerid"] == "p1", "value"].iloc[0] == "new"
    assert combined.loc[combined["playerid"] == "p2", "value"].iloc[0] == "other"


# ──────────────────────────────────────────────
# Schedule/date helpers
# ──────────────────────────────────────────────


def test_parse_api_date_handles_milliseconds_seconds_and_strings() -> None:
    assert backfill.parse_api_date(1767225600000) == pd.Timestamp("2026-01-01")
    assert backfill.parse_api_date(1767225600) == pd.Timestamp("2026-01-01")
    assert backfill.parse_api_date("2026-01-01") == pd.Timestamp("2026-01-01")
    assert pd.isna(backfill.parse_api_date(""))