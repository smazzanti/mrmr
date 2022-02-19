import pandas as pd
import google.cloud.bigquery as google_cloud_bigquery
from tests.utils import pd_dataframe2bq_query
import mrmr

bq_client = google_cloud_bigquery.Client.from_service_account_json(json_credentials_path='google_credentials.json')

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
df_bigquery = f"({pd_dataframe2bq_query(df_pandas)})"


def test_mrmr_classif_without_scores():
    selected_features = mrmr.bigquery.mrmr_classif(
        bq_client=bq_client,
        table_id=df_bigquery,
        K=4,
        target_column=target_column_classif,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=False
    )

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])


def test_mrmr_classif_with_scores():
    selected_features, relevance, redundancy = mrmr.bigquery.mrmr_classif(
        bq_client=bq_client,
        table_id=df_bigquery,
        K=4,
        target_column=target_column_classif,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=True
    )

    assert set(selected_features) == set(["some_null", "feature_a", "feature_b"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)


def test_mrmr_regression_without_scores():
    selected_features = mrmr.bigquery.mrmr_regression(
        bq_client=bq_client,
        table_id=df_bigquery,
        K=4,
        target_column=target_column_regression,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=False
    )

    assert set(selected_features) == set(["some_null", "feature_a"])


def test_mrmr_regression_with_scores():
    selected_features, relevance, redundancy = mrmr.bigquery.mrmr_regression(
        bq_client=bq_client,
        table_id=df_bigquery,
        K=4,
        target_column=target_column_regression,
        features=features,
        denominator="mean",
        only_same_domain=False,
        return_scores=True
    )

    assert set(selected_features) == set(["some_null", "feature_a"])
    assert isinstance(relevance, pd.Series)
    assert isinstance(redundancy, pd.DataFrame)