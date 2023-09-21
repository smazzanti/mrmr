import pandas as pd
import polars
import mrmr

columns = ["target_classif", "target_regression", "some_null", "feature_a", "constant", "feature_b"]
target_column_classif = "target_classif"
target_column_regression = "target_regression"
features = ["some_null", "feature_a", "constant", "feature_b"]

df_polars = polars.DataFrame()
df_polars = df_polars.with_columns(polars.Series(['a','a','b','b','b']).alias("target_classif")) 
df_polars = df_polars.with_columns(polars.Series([1.0,2.0,3.0,4.0,5.0]).alias("target_regression"))
df_polars = df_polars.with_columns(polars.Series([1.0,None,None,4.0,5.0]).alias("some_null"))
df_polars = df_polars.with_columns(polars.Series([2.0,2.0,3.0,3.0,4.0]).alias("feature_a"))
df_polars = df_polars.with_columns(polars.Series([7.0]*5).alias("constant"))
df_polars = df_polars.with_columns(polars.Series([3.0,2.0,1.0,2.0,3.0]).alias("feature_b"))


def test_mrmr_classif_without_scores():
    selected_features = mrmr.polars.mrmr_classif(
        df=df_polars,
        K=4,
        target_column=target_column_classif,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=False,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])


def test_mrmr_classif_with_scores():
    selected_features, relevance, redundancy = mrmr.polars.mrmr_classif(
        df=df_polars,
        K=4,
        target_column=target_column_classif,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=True,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)


def test_mrmr_regression_without_scores():
    selected_features = mrmr.polars.mrmr_regression(
        df=df_polars,
        K=4,
        target_column=target_column_regression,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=False,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a"])


def test_mrmr_regression_with_scores():
    selected_features, relevance, redundancy = mrmr.polars.mrmr_regression(
        df=df_polars,
        K=4,
        target_column=target_column_regression,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=True,
        show_progress=True)

    assert set(selected_features) == set(["some_null", "feature_a"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)