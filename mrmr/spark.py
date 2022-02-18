import numpy as np
import pandas as pd
from .main import groupstats2fstat, mrmr_base


def get_numeric_features(df, target_column):
    """Get all numeric column names from a Spark DataFrame

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    target_column: str
        Name of target column.

    Returns
    -------
    numeric_features : list of str
        List of numeric column names.
    """
    numeric_dtypes = ['int', 'bigint', 'long', 'float', 'double', 'decimal']
    numeric_features = [column_name for column_name, column_type in df.dtypes if column_type in numeric_dtypes and column_name != target_column]
    return numeric_features


def correlation(target_column, features, df):
    out = pd.Series(features, index=features).apply(
        lambda feature: df.select([feature, target_column]).na.drop("any").corr(feature, target_column)
    ).astype(float).fillna(0.0)
    return out


def notna(target_column, features, df):
    out = pd.Series(features, index=features).apply(
        lambda feature: df.select([feature, target_column]).na.drop("any").count()
    ).astype(float)
    return out


def f_regression(target_column, features, df):
    """F-statistic between one numeric target column and many numeric columns of a Spark DataFrame

    F-statistic is actually obtained from the Pearson's correlation coefficient through the following formula:
    corr_coef ** 2 / (1 - corr_coef ** 2) * degrees_of_freedom
    where degrees_of_freedom = n_instances - 1.

    Parameters
    ----------
    target_column: str
        Name of target column.

    features: list of str
        List of numeric column names.

    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic between each column and the target column.
    """

    corr_coef = correlation(target_column=target_column, features=features, df=df)
    n = notna(target_column=target_column, features=features, df=df)

    deg_of_freedom = n - 2
    corr_coef_squared = corr_coef ** 2
    f = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    return f


def f_classif(target_column, features, df):
    groupby = df.replace(float('nan'), None).groupBy(target_column)

    avg = groupby.agg({feature: 'mean' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[4:-1], axis=1)

    var = groupby.agg({feature: 'var_pop' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[8:-1], axis=1)

    n = groupby.agg({feature: 'count' for feature in features}).toPandas().set_index(target_column).rename(
        lambda colname: colname[6:-1], axis=1)

    f = groupstats2fstat(avg=avg, var=var, n=n)
    f.name = target_column

    return f


def mrmr_classif(df, K, target_column, features=None, denominator='mean', only_same_domain=False,
                 show_progress=True):
    """MRMR feature selection for a classification task

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    K: int
        Number of features to select.

    target_column: str
        Name of target column.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    if features is None:
        features = get_numeric_features(df=df, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'target_column': target_column, 'features': features, 'df': df}
    redundancy_args = {'df': df}

    selected_features = mrmr_base(K=K, relevance_func=f_classif, redundancy_func=correlation,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain,
                                  show_progress=show_progress)
    return selected_features


def mrmr_regression(df, target_column, K, features=None, denominator='mean', only_same_domain=False,
                    show_progress=True):
    """MRMR feature selection for a regression task

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        Spark DataFrame.

    target_column: str
        Name of target column.

    K: int
        Number of features to select.

    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.

    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.

    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.

    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.

    Returns
    -------
    selected: list of str
        List of selected features.
    """

    if features is None:
        features = get_numeric_features(df=df, target_column=target_column)

    if type(denominator) == str and denominator == 'mean':
        denominator_func = np.mean
    elif type(denominator) == str and denominator == 'max':
        denominator_func = np.max
    elif type(denominator) == str:
        raise ValueError("Invalid denominator function. It should be one of ['mean', 'max'].")
    else:
        denominator_func = denominator

    relevance_args = {'target_column': target_column, 'features': features, 'df': df}
    redundancy_args = {'df': df}

    selected_features = mrmr_base(K=K, relevance_func=f_regression, redundancy_func=correlation,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain,
                                  show_progress=show_progress)

    return selected_features