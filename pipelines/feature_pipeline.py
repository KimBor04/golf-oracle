from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "features"

RESULTS_PATH = RAW_DIR / "pga_results.tsv"
STATS_PATH = RAW_DIR / "pga_stats.csv"

CLEAN_RESULTS_PATH = FEATURES_DIR / "historical_results_clean.parquet"
CLEAN_STATS_PATH = FEATURES_DIR / "historical_stats_clean.parquet"
HISTORICAL_FEATURES_PATH = FEATURES_DIR / "historical_features.parquet"


def ensure_directories() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""

    name = str(name).strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "season" in df.columns:
        return df

    if "year" in df.columns:
        df["season"] = df["year"]
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["season"] = df["date"].dt.year
        return df

    raise ValueError("No 'season', 'year', or 'date' column found.")


def find_player_column(df: pd.DataFrame) -> str:
    candidates = ["player_name", "player", "name", "golfer"]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find a player column. Available columns: {df.columns.tolist()}")


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, sep="\t")
    df = clean_columns(df)
    df = add_season_column(df)

    player_col = find_player_column(df)
    df["player_name_clean"] = df[player_col].apply(normalize_name)

    return df


def load_stats() -> pd.DataFrame:
    df = pd.read_csv(STATS_PATH)
    df = clean_columns(df)
    df = add_season_column(df)

    player_col = find_player_column(df)
    df["player_name_clean"] = df[player_col].apply(normalize_name)

    return df


def save_clean_inputs(results: pd.DataFrame, stats: pd.DataFrame) -> None:
    results.to_parquet(CLEAN_RESULTS_PATH, index=False)
    stats.to_parquet(CLEAN_STATS_PATH, index=False)

    print(f"\nSaved cleaned results to: {CLEAN_RESULTS_PATH}")
    print(f"Saved cleaned stats to: {CLEAN_STATS_PATH}")


def prepare_results_features(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    numeric_cols = ["round1", "round2", "round3", "round4", "total", "earnings", "fedex_points"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rounds_played"] = df[["round1", "round2", "round3", "round4"]].notna().sum(axis=1)

    df["made_cut"] = np.where(df["round3"].notna() | df["round4"].notna(), 1, 0)

    df["avg_score_played_rounds"] = df[["round1", "round2", "round3", "round4"]].mean(axis=1)

    # event-level ceiling / floor summaries
    df["best_round_in_event"] = df[["round1", "round2", "round3", "round4"]].min(axis=1)
    df["worst_round_in_event"] = df[["round1", "round2", "round3", "round4"]].max(axis=1)

    df = df.sort_values(["player_name_clean", "start", "tournament"]).reset_index(drop=True)

    player_group = df.groupby("player_name_clean", group_keys=False)

    # shift by 1 to avoid leakage
    df["prev_tournament_avg_score"] = player_group["avg_score_played_rounds"].shift(1)
    df["prev_tournament_total"] = player_group["total"].shift(1)
    df["prev_tournament_made_cut"] = player_group["made_cut"].shift(1)
    df["prev_tournament_earnings"] = player_group["earnings"].shift(1)

    df["rolling_avg_last_3"] = (
        player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["rolling_avg_last_5"] = (
        player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    df["rolling_total_last_3"] = (
        player_group["total"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    df["made_cut_rate_last_5"] = (
        player_group["made_cut"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # weighted recent form from previous tournaments only
    def weighted_form(series: pd.Series) -> pd.Series:
        shifted = series.shift(1)
        values = []

        for i in range(len(shifted)):
            window = shifted.iloc[max(0, i - 2):i + 1].dropna().tolist()

            if len(window) == 0:
                values.append(np.nan)
            elif len(window) == 1:
                values.append(window[-1])
            elif len(window) == 2:
                values.append((window[-2] + 2 * window[-1]) / 3)
            else:
                values.append((window[-3] + 2 * window[-2] + 3 * window[-1]) / 6)

        return pd.Series(values, index=series.index)

    df["form_index_last_3"] = player_group["avg_score_played_rounds"].transform(weighted_form)

    df["career_tournament_count"] = player_group.cumcount()

    # --- realism / dispersion features ---
    df["round_std_last_5"] = (
        player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    )

    df["round_std_last_10"] = (
        player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=2).std())
    )

    df["score_range_last_5"] = (
        player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).max())
        - player_group["avg_score_played_rounds"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).min())
    )

    df["best_round_last_10"] = (
        player_group["best_round_in_event"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).min())
    )

    df["worst_round_last_10"] = (
        player_group["worst_round_in_event"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).max())
    )

    df["best_total_last_10"] = (
        player_group["total"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).min())
    )

    df["worst_total_last_10"] = (
        player_group["total"]
        .transform(lambda s: s.shift(1).rolling(10, min_periods=1).max())
    )

    df["missed_cut_rate_last_10"] = (
        player_group["made_cut"]
        .transform(lambda s: 1.0 - s.shift(1).rolling(10, min_periods=1).mean())
    )

    return df


def print_feature_overview(df: pd.DataFrame) -> None:
    print("\n=== HISTORICAL FEATURES OVERVIEW ===")
    print("Shape:", df.shape)

    cols_to_show = [
        "season",
        "start",
        "tournament",
        "name",
        "round1",
        "round2",
        "round3",
        "round4",
        "total",
        "rounds_played",
        "made_cut",
        "avg_score_played_rounds",
        "prev_tournament_avg_score",
        "rolling_avg_last_3",
        "rolling_avg_last_5",
        "made_cut_rate_last_5",
        "form_index_last_3",
        "career_tournament_count",
        "round_std_last_5",
        "round_std_last_10",
        "score_range_last_5",
        "best_round_last_10",
        "worst_round_last_10",
        "best_total_last_10",
        "worst_total_last_10",
        "missed_cut_rate_last_10",
    ]

    existing_cols = [c for c in cols_to_show if c in df.columns]
    print("Columns preview:", existing_cols)

    print("\nSample rows:")
    print(df[existing_cols].head(10))

    print("\nMissing values in key engineered features:")
    key_features = [
        "prev_tournament_avg_score",
        "rolling_avg_last_3",
        "rolling_avg_last_5",
        "made_cut_rate_last_5",
        "form_index_last_3",
        "round_std_last_5",
        "round_std_last_10",
        "score_range_last_5",
        "best_round_last_10",
        "worst_round_last_10",
        "best_total_last_10",
        "worst_total_last_10",
        "missed_cut_rate_last_10",
    ]
    print(df[key_features].isna().mean().sort_values())


def main() -> None:
    ensure_directories()

    results = load_results()
    stats = load_stats()

    print("\n=== RESULTS OVERVIEW ===")
    print("Shape:", results.shape)
    print("Columns:", results.columns.tolist())

    print("\n=== STATS OVERVIEW ===")
    print("Shape:", stats.shape)
    print("Columns:", stats.columns.tolist())

    save_clean_inputs(results, stats)

    historical_features = prepare_results_features(results)
    print_feature_overview(historical_features)

    historical_features.to_parquet(HISTORICAL_FEATURES_PATH, index=False)
    print(f"\nSaved historical features to: {HISTORICAL_FEATURES_PATH}")


if __name__ == "__main__":
    main()