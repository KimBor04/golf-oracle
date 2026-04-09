import pandas as pd
import pytest

from pipelines.training_pipeline import (
    BASE_FEATURE_COLUMNS,
    prepare_training_data,
    time_split,
    ROUND_FEATURE_CONFIG,
)

def make_sample_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2024, 2024, 2024, 2024],
            "start": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-10",
                    "2024-01-20",
                    "2024-02-01",
                    "2024-02-10",
                ]
            ),
            "tournament": ["A", "B", "C", "D", "E"],
            "name": ["p1", "p2", "p3", "p4", "p5"],
            "player_name_clean": ["p1", "p2", "p3", "p4", "p5"],
            "round1": [70, 71, 69, 72, 68],
            "round2": [69, 70, 68, 73, 67],
            "round3": [68, 72, 70, None, 69],
            "round4": [70, 71, 69, None, 68],
            "prev_tournament_avg_score": [71.5, 70.2, 69.8, 72.1, 68.9],
            "prev_tournament_total": [286, 281, 279, 288, 276],
            "prev_tournament_made_cut": [1, 1, 1, 0, 1],
            "prev_tournament_earnings": [1000, 1200, 1500, 400, 1800],
            "rolling_avg_last_3": [71.5, 70.8, 70.1, 71.2, 69.0],
            "rolling_avg_last_5": [71.5, 70.8, 70.1, 71.0, 69.5],
            "rolling_total_last_3": [286, 283.5, 282.0, 284.0, 279.0],
            "made_cut_rate_last_5": [1.0, 1.0, 1.0, 0.75, 0.8],
            "form_index_last_3": [71.5, 70.4, 69.9, 71.3, 68.8],
            "career_tournament_count": [1, 2, 3, 4, 5],
        }
    )


def test_prepare_training_data_round1_returns_expected_feature_columns() -> None:
    df = make_sample_training_df()

    X, y, meta = prepare_training_data(df, "round1")

    assert X.columns.tolist() == BASE_FEATURE_COLUMNS
    assert y.name == "round1"
    assert meta.columns.tolist() == ["start", "tournament", "name", "player_name_clean", "round1"]


def test_prepare_training_data_round2_adds_round1_as_feature() -> None:
    df = make_sample_training_df()

    X, y, meta = prepare_training_data(df, "round2")

    expected_feature_cols = BASE_FEATURE_COLUMNS + ["round1"]

    assert X.columns.tolist() == expected_feature_cols
    assert y.name == "round2"
    assert meta.columns.tolist() == ["start", "tournament", "name", "player_name_clean", "round2"]


def test_prepare_training_data_round1_excludes_target_and_metadata_from_features() -> None:
    df = make_sample_training_df()

    X, _, _ = prepare_training_data(df, "round1")

    forbidden_cols = {"round1", "start", "tournament", "name", "player_name_clean"}
    assert forbidden_cols.isdisjoint(set(X.columns))


def test_prepare_training_data_round2_excludes_future_rounds_and_metadata_from_features() -> None:
    df = make_sample_training_df()

    X, _, _ = prepare_training_data(df, "round2")

    forbidden_cols = {"round2", "round3", "round4", "start", "tournament", "name", "player_name_clean"}
    assert forbidden_cols.isdisjoint(set(X.columns))

    assert "round1" in X.columns


def test_prepare_training_data_round1_drops_rows_without_target_or_history() -> None:
    df = make_sample_training_df().copy()

    df.loc[0, "round1"] = None
    df.loc[1, "prev_tournament_avg_score"] = None

    X, y, meta = prepare_training_data(df, "round1")

    assert len(X) == 3
    assert len(y) == 3
    assert len(meta) == 3


def test_prepare_training_data_round2_drops_rows_without_target_or_required_features() -> None:
    df = make_sample_training_df().copy()

    df.loc[0, "round2"] = None
    df.loc[1, "round1"] = None

    X, y, meta = prepare_training_data(df, "round2")

    assert len(X) == 3
    assert len(y) == 3
    assert len(meta) == 3


def test_prepare_training_data_raises_for_unsupported_round() -> None:
    df = make_sample_training_df()

    with pytest.raises(ValueError, match="Unsupported round_name"):
        prepare_training_data(df, "round5")


def test_time_split_is_strictly_chronological() -> None:
    df = make_sample_training_df()

    X, y, meta = prepare_training_data(df, "round1")

    X_train, X_test, y_train, y_test, meta_train, meta_test = time_split(
        X, y, meta, test_size=0.4
    )

    assert len(X_train) == 3
    assert len(X_test) == 2

    assert meta_train["start"].max() < meta_test["start"].min()

    assert meta_train["start"].tolist() == sorted(meta_train["start"].tolist())
    assert meta_test["start"].tolist() == sorted(meta_test["start"].tolist())


def test_time_split_sorts_even_if_input_is_unsorted() -> None:
    df = make_sample_training_df().sample(frac=1, random_state=42).reset_index(drop=True)

    X, y, meta = prepare_training_data(df, "round1")

    _, _, _, _, meta_train, meta_test = time_split(X, y, meta, test_size=0.4)

    combined_dates = meta_train["start"].tolist() + meta_test["start"].tolist()
    assert combined_dates == sorted(combined_dates)

def test_prepare_training_data_round2_matches_configured_feature_columns() -> None:
    df = make_sample_training_df()

    X, _, _ = prepare_training_data(df, "round2")

    assert X.columns.tolist() == ROUND_FEATURE_CONFIG["round2"]["feature_cols"]

def test_prepare_training_data_round1_does_not_include_round1_as_feature() -> None:
    df = make_sample_training_df()

    X, _, _ = prepare_training_data(df, "round1")

    assert "round1" not in X.columns

def test_time_split_raises_for_invalid_test_size() -> None:
    df = make_sample_training_df()
    X, y, meta = prepare_training_data(df, "round1")

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        time_split(X, y, meta, test_size=0)

    with pytest.raises(ValueError, match="test_size must be between 0 and 1"):
        time_split(X, y, meta, test_size=1)

def test_time_split_raises_when_split_would_create_empty_partition() -> None:
    df = make_sample_training_df().head(1).copy()

    X, y, meta = prepare_training_data(df, "round1")

    with pytest.raises(ValueError, match="Invalid split_idx"):
        time_split(X, y, meta, test_size=0.2)