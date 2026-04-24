import pandas as pd
import pytest

from pipelines.feature_pipeline import prepare_results_features


@pytest.fixture
def sample_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name_clean": [
                "player_a", "player_a", "player_a",
                "player_b", "player_b",
            ],
            "start": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-15",
                    "2024-02-01",
                    "2024-01-05",
                    "2024-01-20",
                ]
            ),
            "end": pd.to_datetime(
                [
                    "2024-01-04",
                    "2024-01-18",
                    "2024-02-04",
                    "2024-01-08",
                    "2024-01-23",
                ]
            ),
            "tournament": [
                "event_1", "event_2", "event_3",
                "event_1", "event_2",
            ],
            "round1": [70, 68, 72, 71, 69],
            "round2": [71, 69, 73, 72, 70],
            "round3": [72, 70, 74, None, 71],
            "round4": [73, 71, 75, None, 72],
            "total": [286, 278, 294, 143, 282],
            "earnings": [1000, 2000, 1500, 500, 1200],
            "fedex_points": [10, 20, 15, 5, 12],
        }
    )


def test_prev_tournament_features_use_only_previous_event(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_a.loc[0, "prev_tournament_total"])
    assert pd.isna(player_a.loc[0, "prev_tournament_avg_score"])
    assert pd.isna(player_a.loc[0, "prev_tournament_earnings"])

    assert player_a.loc[1, "prev_tournament_total"] == 286
    assert player_a.loc[1, "prev_tournament_earnings"] == 1000
    assert player_a.loc[1, "prev_tournament_avg_score"] == pytest.approx(71.5)

    assert player_a.loc[2, "prev_tournament_total"] == 278
    assert player_a.loc[2, "prev_tournament_earnings"] == 2000
    assert player_a.loc[2, "prev_tournament_avg_score"] == pytest.approx(69.5)


def test_rolling_features_are_shifted_and_exclude_current_event(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    avg_event_1 = 71.5
    avg_event_2 = 69.5
    avg_event_3 = 73.5

    assert pd.isna(player_a.loc[0, "rolling_avg_last_3"])
    assert player_a.loc[1, "rolling_avg_last_3"] == pytest.approx(avg_event_1)
    assert player_a.loc[2, "rolling_avg_last_3"] == pytest.approx((avg_event_1 + avg_event_2) / 2)

    assert player_a.loc[2, "rolling_avg_last_3"] != pytest.approx(
        (avg_event_1 + avg_event_2 + avg_event_3) / 3
    )


def test_career_tournament_count_excludes_current_event(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert player_a.loc[0, "career_tournament_count"] == 0
    assert player_a.loc[1, "career_tournament_count"] == 1
    assert player_a.loc[2, "career_tournament_count"] == 2


def test_prev_tournament_made_cut_uses_previous_event(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_b = (
        features[features["player_name_clean"] == "player_b"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_b.loc[0, "prev_tournament_made_cut"])
    assert player_b.loc[1, "prev_tournament_made_cut"] == 0


def test_made_cut_is_derived_correctly(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_b = (
        features[features["player_name_clean"] == "player_b"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert player_b.loc[0, "made_cut"] == 0
    assert player_b.loc[1, "made_cut"] == 1


def test_days_since_last_tournament_uses_previous_event_date(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_a.loc[0, "days_since_last_tournament"])
    assert player_a.loc[1, "days_since_last_tournament"] == 14
    assert player_a.loc[2, "days_since_last_tournament"] == 17


def test_tournaments_last_30_60_90_use_only_prior_events(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_a.loc[0, "tournaments_last_30"])
    assert pd.isna(player_a.loc[0, "tournaments_last_60"])
    assert pd.isna(player_a.loc[0, "tournaments_last_90"])

    assert player_a.loc[1, "tournaments_last_30"] == 1
    assert player_a.loc[1, "tournaments_last_60"] == 1
    assert player_a.loc[1, "tournaments_last_90"] == 1

    assert player_a.loc[2, "tournaments_last_30"] == 1
    assert player_a.loc[2, "tournaments_last_60"] == 2
    assert player_a.loc[2, "tournaments_last_90"] == 2


def test_made_cut_streak_uses_only_previous_events(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_a = (
        features[features["player_name_clean"] == "player_a"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_a.loc[0, "made_cut_streak"])
    assert player_a.loc[1, "made_cut_streak"] == 1
    assert player_a.loc[2, "made_cut_streak"] == 2


def test_missed_cut_streak_uses_only_previous_events(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_b = (
        features[features["player_name_clean"] == "player_b"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_b.loc[0, "missed_cut_streak"])
    assert player_b.loc[1, "missed_cut_streak"] == 1


def test_streaks_reset_correctly_when_previous_event_changes_outcome(sample_results: pd.DataFrame) -> None:
    features = prepare_results_features(sample_results).copy()

    player_b = (
        features[features["player_name_clean"] == "player_b"]
        .sort_values("start")
        .reset_index(drop=True)
    )

    assert pd.isna(player_b.loc[0, "made_cut_streak"])
    assert player_b.loc[1, "made_cut_streak"] == 0

    assert pd.isna(player_b.loc[0, "missed_cut_streak"])
    assert player_b.loc[1, "missed_cut_streak"] == 1