import pandas as pd
import mrmr

columns = ["target_classif", "target_regression", "some_null", "feature_a", "constant", "feature_b"]
target_column_classif = "target_classif"
target_column_regression = "target_regression"
features = ["some_null", "feature_a", "constant", "feature_b"]

data = [
    ('a', 1.0, 1.0,          2.0, 7.0, 3.0),
    ('a', 2.0, float('NaN'), 2.0, 7.0, 2.0),
    ('b', 3.0, float('NaN'), 3.0, 7.0, 1.0),
    ('b', 4.0, 4.0,          3.0, 7.0, 2.0),
    ('b', 5.0, 5.0,          4.0, 7.0, 3.0),
]

df_pandas = pd.DataFrame(data=data, columns=columns)


def test_mrmr_classif_without_scores():
    selected_features = mrmr.pandas.mrmr_classif(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_classif],
        K=4,
        relevance="f",
        redundancy="c",
        denominator="mean",
        cat_features= [],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=False,
        n_jobs=1000,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])


def test_mrmr_classif_ks():
    selected_features = mrmr.pandas.mrmr_classif(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_classif],
        K=4,
        relevance="ks",
        redundancy="c",
        denominator="mean",
        cat_features= [],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=False,
        n_jobs=1000,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])


def test_mrmr_classif_rf():
    selected_features = mrmr.pandas.mrmr_classif(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_classif],
        K=4,
        relevance="rf",
        redundancy="c",
        denominator="mean",
        cat_features= [],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=False,
        n_jobs=1000,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])


def test_mrmr_classif_with_scores():
    selected_features, relevance, redundancy = mrmr.pandas.mrmr_classif(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_classif],
        K=4,
        relevance="f",
        redundancy="c",
        denominator="mean",
        cat_features=[],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=True,
        n_jobs=1000,
        show_progress=True
    )

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)


def test_mrmr_regression_without_scores():
    selected_features = mrmr.pandas.mrmr_regression(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_regression],
        K=4,
        relevance="f",
        redundancy="c",
        denominator="mean",
        cat_features= [],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=False,
        n_jobs=1000,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a"])


def test_mrmr_regression_with_scores():
    selected_features, relevance, redundancy = mrmr.pandas.mrmr_regression(
        X=df_pandas.loc[:, features],
        y=df_pandas.loc[:, target_column_regression],
        K=4,
        relevance="f",
        redundancy="c",
        denominator="mean",
        cat_features= [],
        cat_encoding="leave_one_out",
        only_same_domain=False,
        return_scores=True,
        n_jobs=1000,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)