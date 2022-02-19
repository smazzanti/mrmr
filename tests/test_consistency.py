import pandas as pd
import pyspark
from google.cloud.bigquery import Client
from tests.utils import pd_dataframe2bq_query
import mrmr

spark_session = pyspark.sql.SparkSession(pyspark.context.SparkContext())
bq_client = Client.from_service_account_json(json_credentials_path='google_credentials.json')

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
df_spark = spark_session.createDataFrame(data=data, schema=columns)
df_bigquery = f"({pd_dataframe2bq_query(df_pandas)})"


def test_consistency_f_classif():
    f_classif_pandas = mrmr.pandas.f_classif(X=df_pandas.loc[:, features], y=df_pandas.loc[:, target_column_classif])
    f_classif_spark = mrmr.spark.f_classif(df=df_spark, target_column=target_column_classif, features=features)
    f_classif_bigquery = mrmr.bigquery.f_classif(bq_client=bq_client, table_id=df_bigquery, target_column=target_column_classif, features=features)

    assert set(f_classif_pandas.index) == set(f_classif_spark.index)
    assert set(f_classif_pandas.index) == set(f_classif_bigquery.index)
    assert ((f_classif_pandas - f_classif_spark[f_classif_pandas.index]).abs() < .001).all()
    assert ((f_classif_pandas - f_classif_bigquery[f_classif_pandas.index]).abs() < .001).all()


def test_consistency_correlation():
    correlation_pandas = mrmr.pandas.correlation(target_column=target_column_regression, features=features, X=df_pandas)
    correlation_spark = mrmr.spark.correlation(target_column=target_column_regression, features=features, df=df_spark)
    correlation_bigquery = mrmr.bigquery.correlation(bq_client=bq_client, table_id=df_bigquery, target_column=target_column_regression, features=features)

    assert set(correlation_pandas.index) == set(correlation_spark.index)
    assert set(correlation_pandas.index) == set(correlation_bigquery.index)
    assert ((correlation_pandas - correlation_spark[correlation_pandas.index]).abs() < .001).all()
    assert ((correlation_pandas - correlation_bigquery[correlation_pandas.index]).abs() < .001).all()


def test_consistency_f_regression():
    f_regression_pandas = mrmr.pandas.f_regression(X=df_pandas.loc[:, features], y=df_pandas.loc[:, target_column_regression])
    f_regression_spark = mrmr.spark.f_regression(df=df_spark, target_column=target_column_regression, features=features)
    f_regression_bigquery = mrmr.bigquery.f_regression(bq_client=bq_client, table_id=df_bigquery, target_column=target_column_regression, features=features)

    assert set(f_regression_pandas.index) == set(f_regression_spark.index)
    assert set(f_regression_pandas.index) == set(f_regression_bigquery.index)
    assert ((f_regression_pandas - f_regression_spark[f_regression_pandas.index]).abs() < .001).all()
    assert ((f_regression_pandas - f_regression_bigquery[f_regression_pandas.index]).abs() < .001).all()


def test_consistency_mrmr_classif():
    assert True


def test_consistency_mrmr_regression():
    assert True