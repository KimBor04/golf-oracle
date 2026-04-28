from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.paths import API_CACHE_DIR


DEFAULT_API_HOST = "live-golf-data.p.rapidapi.com"
BASE_URL = "https://live-golf-data.p.rapidapi.com"


class FreeWebAPIGolfClient:
    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.getenv("FREEWEBAPI_GOLF_API_KEY")
        self.api_host = os.getenv("FREEWEBAPI_GOLF_API_HOST", DEFAULT_API_HOST)

        if not self.api_key:
            raise ValueError(
                "Missing FREEWEBAPI_GOLF_API_KEY. Add it to your local .env file."
            )

        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.api_host,
            "Content-Type": "application/json",
        }

    def get_schedule(
        self,
        org_id: int = 1,
        year: int = 2024,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_path = API_CACHE_DIR / f"schedule_org{org_id}_{year}.json"

        if use_cache and cache_path.exists():
            return self._read_json(cache_path)

        data = self._get(
            endpoint="/schedule",
            params={
                "orgId": org_id,
                "year": year,
            },
        )

        self._write_json(cache_path, data)
        return data

    def get_leaderboard(
        self,
        tourn_id: str | int,
        year: int,
        round_id: int,
        org_id: int = 1,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        cache_path = API_CACHE_DIR / (
            f"leaderboard_org{org_id}_tourn{tourn_id}_{year}_round{round_id}.json"
        )

        if use_cache and cache_path.exists():
            return self._read_json(cache_path)

        data = self._get(
            endpoint="/leaderboard",
            params={
                "orgId": org_id,
                "tournId": tourn_id,
                "year": year,
                "roundId": round_id,
            },
        )

        self._write_json(cache_path, data)
        return data

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        response = requests.get(
            f"{BASE_URL}{endpoint}",
            headers=self.headers,
            params=params,
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def extract_list_from_response(
    data: dict[str, Any],
    possible_keys: list[str],
) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data

    for key in possible_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    raise ValueError(
        f"Could not find list in API response. Top-level keys: {list(data.keys())}"
    )


def clean_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(".", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def parse_score_to_par(value: object) -> float | None:
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if value_str in {"", "-", "null", "None"}:
        return None

    if value_str.upper() == "E":
        return 0.0

    try:
        return float(value_str.replace("+", ""))
    except ValueError:
        return None


def parse_position(value: object) -> float | None:
    if pd.isna(value):
        return None

    value_str = str(value).strip().upper().replace("T", "")

    if value_str in {"", "-", "WD", "CUT"}:
        return None

    try:
        return float(value_str)
    except ValueError:
        return None


def parse_strokes(value: object) -> float | None:
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if value_str in {"", "-", "null", "None"}:
        return None

    try:
        return float(value_str)
    except ValueError:
        return None


def normalize_schedule_response(data: dict[str, Any]) -> pd.DataFrame:
    events = extract_list_from_response(
        data,
        possible_keys=[
            "schedule",
            "data",
            "events",
            "tournaments",
        ],
    )

    df = pd.json_normalize(events)
    df = clean_api_columns(df)

    return df


def find_tournament_id(schedule_df: pd.DataFrame, tournament_name: str) -> str:
    tournament_col = find_first_existing_column(
        schedule_df,
        [
            "name",
            "tournament",
            "tournament_name",
            "event_name",
        ],
    )

    tourn_id_col = find_first_existing_column(
        schedule_df,
        [
            "tournid",
            "tourn_id",
            "id",
            "tournament_id",
            "event_id",
        ],
    )

    if tournament_col is None:
        raise ValueError(
            f"Could not find tournament name column in schedule. "
            f"Available columns: {schedule_df.columns.tolist()}"
        )

    if tourn_id_col is None:
        raise ValueError(
            f"Could not find tournament id column in schedule. "
            f"Available columns: {schedule_df.columns.tolist()}"
        )

    schedule_df = schedule_df.copy()
    schedule_df["tournament_name_clean"] = schedule_df[tournament_col].apply(normalize_name)
    target_clean = normalize_name(tournament_name)

    exact_matches = schedule_df[schedule_df["tournament_name_clean"] == target_clean]

    if not exact_matches.empty:
        return str(exact_matches.iloc[0][tourn_id_col])

    partial_matches = schedule_df[
        schedule_df["tournament_name_clean"].str.contains(target_clean, na=False)
    ]

    if not partial_matches.empty:
        return str(partial_matches.iloc[0][tourn_id_col])

    reverse_partial_matches = schedule_df[
        schedule_df["tournament_name_clean"].apply(
            lambda value: target_clean in value or value in target_clean
        )
    ]

    if not reverse_partial_matches.empty:
        return str(reverse_partial_matches.iloc[0][tourn_id_col])

    available_tournaments = schedule_df[tournament_col].dropna().head(25).tolist()

    raise ValueError(
        f"Could not find tournament '{tournament_name}' in schedule. "
        f"Available tournaments: {available_tournaments}"
    )


def normalize_leaderboard_response(
    data: dict[str, Any],
    tournament_name: str,
    tourn_id: str | int,
    year: int,
    round_id: int,
) -> pd.DataFrame:
    players = extract_list_from_response(
        data,
        possible_keys=[
            "leaderboardRows",
            "leaderboard",
            "data",
            "players",
            "results",
        ],
    )

    df = pd.json_normalize(players)
    df = clean_api_columns(df)

    df["api_source"] = "freewebapi_golf_leaderboard"
    df["target_tournament"] = tournament_name
    df["tourn_id"] = str(tourn_id)
    df["season"] = year
    df["round_id"] = round_id

    if "firstname" in df.columns and "lastname" in df.columns:
        df["player_name"] = (
            df["firstname"].astype(str).str.strip()
            + " "
            + df["lastname"].astype(str).str.strip()
        )
        df["player_name_clean"] = df["player_name"].apply(normalize_name)

    elif "player_first_name" in df.columns and "player_last_name" in df.columns:
        df["player_name"] = (
            df["player_first_name"].astype(str).str.strip()
            + " "
            + df["player_last_name"].astype(str).str.strip()
        )
        df["player_name_clean"] = df["player_name"].apply(normalize_name)

    elif "first_name" in df.columns and "last_name" in df.columns:
        df["player_name"] = (
            df["first_name"].astype(str).str.strip()
            + " "
            + df["last_name"].astype(str).str.strip()
        )
        df["player_name_clean"] = df["player_name"].apply(normalize_name)

    else:
        player_col = find_first_existing_column(
            df,
            [
                "player_name",
                "player",
                "name",
                "player_full_name",
                "display_name",
                "player_display_name",
            ],
        )

        if player_col is not None:
            df["player_name"] = df[player_col]
            df["player_name_clean"] = df[player_col].apply(normalize_name)
        else:
            df["player_name"] = ""
            df["player_name_clean"] = ""

    if "total" in df.columns:
        df["live_total_to_par"] = df["total"].apply(parse_score_to_par)

    if "currentroundscore" in df.columns:
        df["live_current_round_to_par"] = df["currentroundscore"].apply(parse_score_to_par)

    if "position" in df.columns:
        df["live_position_numeric"] = df["position"].apply(parse_position)

    if "totalstrokesfromcompletedrounds" in df.columns:
        df["live_completed_round_strokes"] = df["totalstrokesfromcompletedrounds"].apply(parse_strokes)

    if "thru" in df.columns:
        df["live_thru"] = df["thru"].replace("", pd.NA)

    if "roundcomplete" in df.columns:
        df["live_round_complete"] = df["roundcomplete"].astype(bool)

    return df


def fetch_live_leaderboard_features(
    tournament_name: str,
    year: int,
    round_id: int,
    org_id: int = 1,
    use_cache: bool = True,
) -> pd.DataFrame:
    client = FreeWebAPIGolfClient()

    schedule_data = client.get_schedule(
        org_id=org_id,
        year=year,
        use_cache=use_cache,
    )
    schedule_df = normalize_schedule_response(schedule_data)

    tourn_id = find_tournament_id(
        schedule_df=schedule_df,
        tournament_name=tournament_name,
    )

    leaderboard_data = client.get_leaderboard(
        org_id=org_id,
        tourn_id=tourn_id,
        year=year,
        round_id=round_id,
        use_cache=use_cache,
    )

    leaderboard_df = normalize_leaderboard_response(
        leaderboard_data,
        tournament_name=tournament_name,
        tourn_id=tourn_id,
        year=year,
        round_id=round_id,
    )

    return leaderboard_df

def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name