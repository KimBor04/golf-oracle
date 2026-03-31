import pandas as pd

from pipelines.training_pipeline import prepare_training_data, time_split


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


def test_prepare_training_data_returns_expected_feature_columns() -> None:
    df = make_sample_training_df()

    X, y, meta = prepare_training_data(df)

    expected_feature_cols = [
        "season",
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
    ]

    assert X.columns.tolist() == expected_feature_cols
    assert y.name == "round1"
    assert meta.columns.tolist() == ["start", "tournament", "name", "player_name_clean", "round1"]


def test_prepare_training_data_excludes_target_and_metadata_from_features() -> None:
    df = make_sample_training_df()

    X, _, _ = prepare_training_data(df)

    forbidden_cols = {"round1", "start", "tournament", "name", "player_name_clean"}
    assert forbidden_cols.isdisjoint(set(X.columns))


def test_prepare_training_data_drops_rows_without_target_or_history() -> None:
    df = make_sample_training_df().copy()

    df.loc[0, "round1"] = None
    df.loc[1, "prev_tournament_avg_score"] = None

    X, y, meta = prepare_training_data(df)

    assert len(X) == 3
    assert len(y) == 3
    assert len(meta) == 3


def test_time_split_is_strictly_chronological() -> None:
    df = make_sample_training_df()

    X, y, meta = prepare_training_data(df)

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

    X, y, meta = prepare_training_data(df)

    _, _, _, _, meta_train, meta_test = time_split(X, y, meta, test_size=0.4)

    combined_dates = meta_train["start"].tolist() + meta_test["start"].tolist()
    assert combined_dates == sorted(combined_dates)