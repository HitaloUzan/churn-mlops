import pandas as pd

from mlops.data_splits import X_y, make_splits


def _synthetic_df(n=1000):
    return pd.DataFrame({"feature_a": range(n), "Churn": [i % 2 for i in range(n)]})


def test_make_splits_produces_no_row_overlap():
    df = _synthetic_df()
    splits = make_splits(df)

    all_a = set(splits["eval_holdout"]["feature_a"])
    ref_a = set(splits["reference"]["feature_a"])
    cur_a = set(splits["current_raw"]["feature_a"])

    assert all_a.isdisjoint(ref_a)
    assert all_a.isdisjoint(cur_a)
    assert ref_a.isdisjoint(cur_a)


def test_make_splits_covers_the_full_dataset():
    df = _synthetic_df()
    splits = make_splits(df)

    total = len(splits["eval_holdout"]) + len(splits["reference"]) + len(splits["current_raw"])
    assert total == len(df)


def test_make_splits_is_deterministic():
    df = _synthetic_df()
    splits1 = make_splits(df)
    splits2 = make_splits(df)

    assert list(splits1["reference"]["feature_a"]) == list(splits2["reference"]["feature_a"])


def test_x_y_separates_target_column():
    df = pd.DataFrame({"a": [1, 2], "Churn": [0, 1]})
    X, y = X_y(df)

    assert "Churn" not in X.columns
    assert list(y) == [0, 1]
