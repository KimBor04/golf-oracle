from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from src.freewebapi_golf_client import (
    FreeWebAPIGolfClient,
    find_first_existing_column,
    normalize_leaderboard_response,
    normalize_name,
    normalize_schedule_response,
)
from src.paths import FEATURES_DIR


# ──────────────────────────────────────────────
# Project / API settings
# ──────────────────────────────────────────────

ORG_ID = 1  # PGA TOUR
DEFAULT_YEAR = 2026
DEFAULT_MAX_API_CALLS = 5
DEFAULT_NEXT_EVENTS = 3
DEFAULT_FIELD_LOOKAHEAD_DAYS = 7
KAGGLE_CUTOFF_DATE = pd.Timestamp("2025-12-31")

API_CACHE_DIR = FEATURES_DIR / "api_cache"

ROUND_IDS = [1, 2, 3, 4]


# ──────────────────────────────────────────────
# Output paths
# ──────────────────────────────────────────────

def api_schedule_path(year: int) -> Path:
    return FEATURES_DIR / f"api_schedule_{year}.parquet"


def api_target_tournaments_path(year: int) -> Path:
    return FEATURES_DIR / f"api_target_tournaments_{year}.parquet"


def api_results_path(year: int) -> Path:
    return FEATURES_DIR / f"api_results_{year}.parquet"


def api_fields_path(year: int) -> Path:
    return FEATURES_DIR / f"api_fields_{year}.parquet"


def schedule_cache_path(org_id: int, year: int) -> Path:
    return API_CACHE_DIR / f"schedule_org{org_id}_{year}.json"


def leaderboard_cache_path(
    org_id: int,
    tourn_id: str | int,
    year: int,
    round_id: int,
) -> Path:
    return API_CACHE_DIR / f"leaderboard_org{org_id}_tourn{tourn_id}_{year}_round{round_id}.json"


# ──────────────────────────────────────────────
# API call budget
# ──────────────────────────────────────────────

@dataclass
class ApiCallBudget:
    max_calls: int
    used_calls: int = 0

    def can_call(self) -> bool:
        return self.used_calls < self.max_calls

    def consume(self) -> None:
        if not self.can_call():
            raise RuntimeError(
                f"API call budget exhausted: used {self.used_calls}/{self.max_calls}"
            )

        self.used_calls += 1

    @property
    def remaining(self) -> int:
        return self.max_calls - self.used_calls


# ──────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────

def ensure_directories() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    API_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def read_cached_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    return read_json(path)


def parse_api_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT

    value_str = str(value).strip()

    if value_str in {"", "-", "None", "null"}:
        return pd.NaT

    try:
        numeric_value = float(value_str)

        # FreeWebAPI schedule dates are Unix timestamps in milliseconds.
        if numeric_value > 10_000_000_000:
            return pd.to_datetime(numeric_value, unit="ms", errors="coerce")

        # Fallback for Unix timestamps in seconds.
        if numeric_value > 1_000_000_000:
            return pd.to_datetime(numeric_value, unit="s", errors="coerce")

    except ValueError:
        pass

    return pd.to_datetime(value_str, errors="coerce")


def parse_schedule_dates(schedule_df: pd.DataFrame) -> pd.DataFrame:
    df = schedule_df.copy()

    start_col = find_first_existing_column(
        df,
        [
            "date_start_$date_$numberlong",
            "date_start",
            "startdate",
            "start_date",
            "start",
            "date",
            "tournament_date",
            "event_date",
        ],
    )

    end_col = find_first_existing_column(
        df,
        [
            "date_end_$date_$numberlong",
            "date_end",
            "enddate",
            "end_date",
            "end",
            "tournament_end_date",
            "event_end_date",
        ],
    )

    if start_col is not None:
        df["api_start_date"] = df[start_col].apply(parse_api_date)
    else:
        df["api_start_date"] = pd.NaT

    if end_col is not None:
        df["api_end_date"] = df[end_col].apply(parse_api_date)
    else:
        df["api_end_date"] = df["api_start_date"]

    return df


def get_tournament_name_column(df: pd.DataFrame) -> str:
    tournament_col = find_first_existing_column(
        df,
        [
            "tournamentname",
            "tournament_name",
            "eventname",
            "event_name",
            "name",
            "tournament",
        ],
    )

    if tournament_col is None:
        raise ValueError(
            f"Could not find tournament name column. Available columns: {df.columns.tolist()}"
        )

    return tournament_col


def get_tourn_id_column(df: pd.DataFrame) -> str:
    tourn_id_col = find_first_existing_column(
        df,
        [
            "tournid",
            "tourn_id",
            "tournamentid",
            "tournament_id",
            "eventid",
            "event_id",
            "id",
        ],
    )

    if tourn_id_col is None:
        raise ValueError(
            f"Could not find tournament ID column. Available columns: {df.columns.tolist()}"
        )

    return tourn_id_col


def add_standard_schedule_columns(schedule_df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = parse_schedule_dates(schedule_df)

    tournament_col = get_tournament_name_column(df)
    tourn_id_col = get_tourn_id_column(df)

    df["target_tournament"] = df[tournament_col].astype(str)
    df["target_tournament_clean"] = df["target_tournament"].apply(normalize_name)
    df["tourn_id"] = df[tourn_id_col].astype(str)
    df["season"] = year
    df["org_id"] = ORG_ID

    return df


def cached_round_ids_for_tournament(
    tourn_id: str | int,
    year: int,
    org_id: int = ORG_ID,
) -> set[int]:
    cached_rounds: set[int] = set()

    for round_id in ROUND_IDS:
        if leaderboard_cache_path(org_id, tourn_id, year, round_id).exists():
            cached_rounds.add(round_id)

    return cached_rounds


def result_cache_status_for_tournament(
    tourn_id: str | int,
    year: int,
    org_id: int = ORG_ID,
) -> str:
    cached_rounds = cached_round_ids_for_tournament(
        tourn_id=tourn_id,
        year=year,
        org_id=org_id,
    )

    if len(cached_rounds) == 0:
        return "missing"

    if len(cached_rounds) == len(ROUND_IDS):
        return "complete"

    return "partial"


def add_result_cache_status(
    tournaments_df: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    df = tournaments_df.copy()

    df["cached_round_ids"] = df["tourn_id"].apply(
        lambda tourn_id: sorted(
            cached_round_ids_for_tournament(
                tourn_id=tourn_id,
                year=year,
            )
        )
    )

    df["missing_round_ids"] = df["cached_round_ids"].apply(
        lambda cached_rounds: [
            round_id for round_id in ROUND_IDS if round_id not in cached_rounds
        ]
    )

    df["result_cache_status"] = df["tourn_id"].apply(
        lambda tourn_id: result_cache_status_for_tournament(
            tourn_id=tourn_id,
            year=year,
        )
    )

    return df


def is_cache_complete_for_tournament(
    tourn_id: str | int,
    year: int,
    org_id: int = ORG_ID,
) -> bool:
    return result_cache_status_for_tournament(
        tourn_id=tourn_id,
        year=year,
        org_id=org_id,
    ) == "complete"


def load_existing_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)

    return pd.DataFrame()


def concat_and_dedupe(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    subset: list[str],
) -> pd.DataFrame:
    if existing_df.empty:
        combined = new_df.copy()
    elif new_df.empty:
        combined = existing_df.copy()
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)

    if combined.empty:
        return combined

    existing_subset = [col for col in subset if col in combined.columns]

    if existing_subset:
        combined = combined.drop_duplicates(subset=existing_subset, keep="last")

    return combined.reset_index(drop=True)


# ──────────────────────────────────────────────
# Fetch schedule
# ──────────────────────────────────────────────

def fetch_schedule_once(
    client: FreeWebAPIGolfClient,
    year: int,
    budget: ApiCallBudget,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache_path = schedule_cache_path(ORG_ID, year)

    cached_data = None if refresh_cache else read_cached_json(cache_path)

    if cached_data is not None:
        print(f"Using cached schedule: {cache_path}")
        schedule_data = cached_data
    else:
        if not budget.can_call():
            raise RuntimeError("No API calls left to fetch schedule.")

        print(f"Fetching schedule from API: orgId={ORG_ID}, year={year}")
        budget.consume()

        schedule_data = client.get_schedule(
            org_id=ORG_ID,
            year=year,
            use_cache=False,
        )

        write_json(cache_path, schedule_data)

    schedule_df = normalize_schedule_response(schedule_data)
    schedule_df = add_standard_schedule_columns(schedule_df, year=year)

    output_path = api_schedule_path(year)
    schedule_df.to_parquet(output_path, index=False)

    print(f"Saved normalized schedule: {output_path}")
    print(f"Schedule rows: {len(schedule_df)}")

    return schedule_df


def build_target_tournaments(
    schedule_df: pd.DataFrame,
    year: int,
    cutoff_date: pd.Timestamp = KAGGLE_CUTOFF_DATE,
) -> pd.DataFrame:
    df = schedule_df.copy()

    if "api_start_date" in df.columns and df["api_start_date"].notna().any():
        target_df = df[df["api_start_date"] > cutoff_date].copy()
    else:
        target_df = df.copy()

    target_df["needs_api_after_kaggle_cutoff"] = True

    output_path = api_target_tournaments_path(year)
    target_df.to_parquet(output_path, index=False)

    print(f"Saved API target tournament list: {output_path}")
    print(f"Target tournaments: {len(target_df)}")

    return target_df


# ──────────────────────────────────────────────
# Fetch completed tournament results
# ──────────────────────────────────────────────

def fetch_leaderboard_round_if_missing(
    client: FreeWebAPIGolfClient,
    tourn_id: str | int,
    year: int,
    round_id: int,
    budget: ApiCallBudget,
    refresh_cache: bool = False,
) -> dict[str, Any] | None:
    cache_path = leaderboard_cache_path(ORG_ID, tourn_id, year, round_id)

    if cache_path.exists() and not refresh_cache:
        print(f"Using cached leaderboard: {cache_path.name}")
        return read_json(cache_path)

    if not budget.can_call():
        print(
            f"Skipping API call because budget is exhausted: "
            f"tournId={tourn_id}, roundId={round_id}"
        )
        return None

    print(
        f"Fetching leaderboard from API: "
        f"orgId={ORG_ID}, tournId={tourn_id}, year={year}, roundId={round_id}"
    )

    budget.consume()

    try:
        data = client.get_leaderboard(
            org_id=ORG_ID,
            tourn_id=tourn_id,
            year=year,
            round_id=round_id,
            use_cache=False,
        )
    except requests.HTTPError as error:
        print(
            f"WARNING: leaderboard request failed for "
            f"tournId={tourn_id}, roundId={round_id}: {error}"
        )
        return None
    except requests.RequestException as error:
        print(
            f"WARNING: leaderboard request failed for "
            f"tournId={tourn_id}, roundId={round_id}: {error}"
        )
        return None

    write_json(cache_path, data)
    return data


def select_completed_tournaments(
    target_tournaments_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    year: int,
) -> pd.DataFrame:
    df = target_tournaments_df.copy()

    if "api_end_date" in df.columns and df["api_end_date"].notna().any():
        completed = df[df["api_end_date"] < as_of_date].copy()
    elif "api_start_date" in df.columns and df["api_start_date"].notna().any():
        completed = df[df["api_start_date"] < as_of_date].copy()
    else:
        completed = pd.DataFrame(columns=df.columns)

    if completed.empty:
        return completed.reset_index(drop=True)

    completed = add_result_cache_status(
        tournaments_df=completed,
        year=year,
    )

    status_priority = {
        "partial": 0,
        "missing": 1,
        "complete": 2,
    }

    completed["result_cache_priority"] = (
        completed["result_cache_status"]
        .map(status_priority)
        .fillna(99)
    )

    completed = completed.sort_values(
        [
            "result_cache_priority",
            "api_start_date",
            "target_tournament",
        ],
        na_position="last",
    )

    return completed.reset_index(drop=True)


def fetch_completed_results(
    client: FreeWebAPIGolfClient,
    target_tournaments_df: pd.DataFrame,
    year: int,
    budget: ApiCallBudget,
    as_of_date: pd.Timestamp,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    completed_tournaments = select_completed_tournaments(
        target_tournaments_df=target_tournaments_df,
        as_of_date=as_of_date,
        year=year,
    )

    print(f"Completed tournaments eligible for API result backfill: {len(completed_tournaments)}")

    if not completed_tournaments.empty and "result_cache_status" in completed_tournaments.columns:
        status_counts = completed_tournaments["result_cache_status"].value_counts().to_dict()
        print(f"Result cache status counts: {status_counts}")

    normalized_rounds: list[pd.DataFrame] = []

    for _, tournament in completed_tournaments.iterrows():
        tourn_id = tournament["tourn_id"]
        tournament_name = tournament["target_tournament"]

        missing_round_ids = tournament.get("missing_round_ids", ROUND_IDS)

        if not missing_round_ids:
            print(
                f"Skipping completed tournament because all rounds are cached: "
                f"{tournament_name} ({year}, tournId={tourn_id})"
            )
            continue

        print(
            f"Backfilling {tournament_name} ({year}, tournId={tourn_id}) "
            f"missing rounds: {missing_round_ids}"
        )

        for round_id in missing_round_ids:
            data = fetch_leaderboard_round_if_missing(
                client=client,
                tourn_id=tourn_id,
                year=year,
                round_id=round_id,
                budget=budget,
                refresh_cache=refresh_cache,
            )

            if data is None:
                continue

            try:
                round_df = normalize_leaderboard_response(
                    data=data,
                    tournament_name=tournament_name,
                    tourn_id=tourn_id,
                    year=year,
                    round_id=round_id,
                )
            except ValueError as error:
                print(
                    f"WARNING: could not normalize leaderboard for "
                    f"{tournament_name}, round {round_id}: {error}"
                )
                continue

            round_df["api_record_type"] = "completed_result"
            round_df["api_start_date"] = tournament.get("api_start_date", pd.NaT)
            round_df["api_end_date"] = tournament.get("api_end_date", pd.NaT)

            normalized_rounds.append(round_df)

            if not budget.can_call():
                print("API call budget exhausted during results backfill.")
                break

        if not budget.can_call():
            break

    new_results_df = (
        pd.concat(normalized_rounds, ignore_index=True)
        if normalized_rounds
        else pd.DataFrame()
    )

    output_path = api_results_path(year)
    existing_results_df = load_existing_parquet(output_path)

    final_results_df = concat_and_dedupe(
        existing_df=existing_results_df,
        new_df=new_results_df,
        subset=["tourn_id", "season", "round_id", "playerid"],
    )

    final_results_df.to_parquet(output_path, index=False)

    print(f"Saved API results: {output_path}")
    print(f"API result rows: {len(final_results_df)}")

    return final_results_df


# ──────────────────────────────────────────────
# Fetch next tournament fields safely
# ──────────────────────────────────────────────

def select_next_tournaments(
    target_tournaments_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    next_events: int,
    field_lookahead_days: int,
) -> pd.DataFrame:
    df = target_tournaments_df.copy()

    if "api_start_date" in df.columns and df["api_start_date"].notna().any():
        lookahead_end_date = as_of_date + pd.Timedelta(days=field_lookahead_days)

        upcoming = df[
            (df["api_start_date"] >= as_of_date)
            & (df["api_start_date"] <= lookahead_end_date)
        ].copy()

        upcoming = upcoming.sort_values(["api_start_date", "target_tournament"])
    else:
        upcoming = df.copy().sort_values(["target_tournament"])

    return upcoming.head(next_events).reset_index(drop=True)


def load_existing_field_statuses(year: int) -> dict[tuple[str, int], str]:
    fields_df = load_existing_parquet(api_fields_path(year))

    if fields_df.empty:
        return {}

    required_cols = {"tourn_id", "season", "field_cache_status"}

    if not required_cols.issubset(fields_df.columns):
        return {}

    status_df = (
        fields_df[["tourn_id", "season", "field_cache_status"]]
        .dropna(subset=["tourn_id", "season", "field_cache_status"])
        .drop_duplicates(subset=["tourn_id", "season"], keep="last")
    )

    return {
        (str(row["tourn_id"]), int(row["season"])): str(row["field_cache_status"])
        for _, row in status_df.iterrows()
    }


def classify_field_cache_status(field_df: pd.DataFrame) -> str:
    if field_df.empty:
        return "not_available_yet"

    if "player_name_clean" not in field_df.columns:
        return "incomplete"

    player_count = field_df["player_name_clean"].replace("", pd.NA).dropna().nunique()

    if player_count >= 20:
        return "available"

    if player_count > 0:
        return "incomplete"

    return "not_available_yet"


def fetch_next_fields(
    client: FreeWebAPIGolfClient,
    target_tournaments_df: pd.DataFrame,
    year: int,
    budget: ApiCallBudget,
    as_of_date: pd.Timestamp,
    next_events: int,
    field_lookahead_days: int,
    refresh_cache: bool = False,
) -> pd.DataFrame:
    next_tournaments = select_next_tournaments(
        target_tournaments_df=target_tournaments_df,
        as_of_date=as_of_date,
        next_events=next_events,
        field_lookahead_days=field_lookahead_days,
    )

    print(
        f"Next tournaments considered for field import within "
        f"{field_lookahead_days} days: {len(next_tournaments)}"
    )

    existing_field_statuses = load_existing_field_statuses(year)

    normalized_fields: list[pd.DataFrame] = []

    for _, tournament in next_tournaments.iterrows():
        tourn_id = str(tournament["tourn_id"])
        tournament_name = tournament["target_tournament"]
        field_key = (tourn_id, int(year))

        existing_status = existing_field_statuses.get(field_key)

        if existing_status == "available" and not refresh_cache:
            print(
                f"Skipping field fetch because field is already available: "
                f"{tournament_name} ({year}, tournId={tourn_id})"
            )
            continue

        if existing_status in {"failed", "not_available_yet", "incomplete"} and not refresh_cache:
            print(
                f"Retrying field fetch because tournament is within "
                f"{field_lookahead_days} days and previous status was "
                f"'{existing_status}': {tournament_name} "
                f"({year}, tournId={tourn_id})"
            )

        data = fetch_leaderboard_round_if_missing(
            client=client,
            tourn_id=tourn_id,
            year=year,
            round_id=1,
            budget=budget,
            refresh_cache=refresh_cache,
        )

        if data is None:
            status_df = pd.DataFrame(
                [
                    {
                        "target_tournament": tournament_name,
                        "target_tournament_clean": normalize_name(tournament_name),
                        "tourn_id": tourn_id,
                        "season": year,
                        "round_id": 1,
                        "field_cache_status": "failed",
                        "api_record_type": "future_field",
                        "api_start_date": tournament.get("api_start_date", pd.NaT),
                        "api_end_date": tournament.get("api_end_date", pd.NaT),
                    }
                ]
            )
            normalized_fields.append(status_df)
            continue

        try:
            field_df = normalize_leaderboard_response(
                data=data,
                tournament_name=tournament_name,
                tourn_id=tourn_id,
                year=year,
                round_id=1,
            )
        except ValueError as error:
            print(
                f"WARNING: could not normalize future field for "
                f"{tournament_name}: {error}"
            )
            field_df = pd.DataFrame()

        field_status = classify_field_cache_status(field_df)

        if field_df.empty:
            field_df = pd.DataFrame(
                [
                    {
                        "target_tournament": tournament_name,
                        "target_tournament_clean": normalize_name(tournament_name),
                        "tourn_id": tourn_id,
                        "season": year,
                        "round_id": 1,
                    }
                ]
            )

        field_df["field_cache_status"] = field_status
        field_df["api_record_type"] = "future_field"
        field_df["api_start_date"] = tournament.get("api_start_date", pd.NaT)
        field_df["api_end_date"] = tournament.get("api_end_date", pd.NaT)

        normalized_fields.append(field_df)

        print(
            f"Field status for {tournament_name} ({year}, tournId={tourn_id}): "
            f"{field_status}"
        )

        if not budget.can_call():
            print("API call budget exhausted during field import.")
            break

    new_fields_df = (
        pd.concat(normalized_fields, ignore_index=True)
        if normalized_fields
        else pd.DataFrame()
    )

    output_path = api_fields_path(year)
    existing_fields_df = load_existing_parquet(output_path)

    final_fields_df = concat_and_dedupe(
        existing_df=existing_fields_df,
        new_df=new_fields_df,
        subset=["tourn_id", "season", "round_id", "playerid"],
    )

    final_fields_df.to_parquet(output_path, index=False)

    print(f"Saved API fields: {output_path}")
    print(f"API field rows: {len(final_fields_df)}")

    return final_fields_df


# ──────────────────────────────────────────────
# Main orchestration
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache-first FreeWebAPI PGA TOUR backfill pipeline. "
            "Designed for limited API quotas and later GitHub Actions automation."
        )
    )

    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="Season year to backfill.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["schedule", "results", "fields", "all"],
        default="all",
        help=(
            "schedule = fetch/save schedule only; "
            "results = fetch completed tournament rounds; "
            "fields = fetch next tournament fields; "
            "all = run schedule, results, and fields."
        ),
    )

    parser.add_argument(
        "--max-api-calls",
        type=int,
        default=DEFAULT_MAX_API_CALLS,
        help="Maximum real API requests allowed in this run.",
    )

    parser.add_argument(
        "--next-events",
        type=int,
        default=DEFAULT_NEXT_EVENTS,
        help="Number of upcoming tournaments to consider for field import.",
    )

    parser.add_argument(
        "--field-lookahead-days",
        type=int,
        default=DEFAULT_FIELD_LOOKAHEAD_DAYS,
        help=(
            "Only try to fetch future fields for tournaments starting within this many days. "
            "Default: 7."
        ),
    )

    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help=(
            "Date used to decide completed/upcoming tournaments. "
            "Format: YYYY-MM-DD. Defaults to today."
        ),
    )

    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help=(
            "Force refetching API cache files. Use carefully because API calls are limited."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_directories()

    as_of_date = (
        pd.Timestamp(args.as_of_date)
        if args.as_of_date is not None
        else pd.Timestamp.today().normalize()
    )

    budget = ApiCallBudget(max_calls=args.max_api_calls)
    client = FreeWebAPIGolfClient()

    print("\n=== FREEWEBAPI PGA TOUR BACKFILL ===")
    print(f"Year: {args.year}")
    print(f"Mode: {args.mode}")
    print(f"Org ID: {ORG_ID} (PGA TOUR)")
    print(f"As-of date: {as_of_date.date()}")
    print(f"Max API calls this run: {args.max_api_calls}")
    print(f"Field lookahead days: {args.field_lookahead_days}")
    print(f"Refresh cache: {args.refresh_cache}")

    schedule_df = fetch_schedule_once(
        client=client,
        year=args.year,
        budget=budget,
        refresh_cache=args.refresh_cache,
    )

    target_tournaments_df = build_target_tournaments(
        schedule_df=schedule_df,
        year=args.year,
        cutoff_date=KAGGLE_CUTOFF_DATE,
    )

    if args.mode in {"results", "all"}:
        fetch_completed_results(
            client=client,
            target_tournaments_df=target_tournaments_df,
            year=args.year,
            budget=budget,
            as_of_date=as_of_date,
            refresh_cache=args.refresh_cache,
        )

    if args.mode in {"fields", "all"}:
        fetch_next_fields(
            client=client,
            target_tournaments_df=target_tournaments_df,
            year=args.year,
            budget=budget,
            as_of_date=as_of_date,
            next_events=args.next_events,
            field_lookahead_days=args.field_lookahead_days,
            refresh_cache=args.refresh_cache,
        )

    print("\n=== BACKFILL COMPLETE ===")
    print(f"API calls used: {budget.used_calls}/{budget.max_calls}")
    print(f"API calls remaining in this run: {budget.remaining}")


if __name__ == "__main__":
    main()