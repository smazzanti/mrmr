import functools
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from mrmr.main import mrmr_base


def parallel_df(func, df, series, n_jobs):
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == -1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def _f_classif(X, y):
    def _f_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_classif(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_classif_series(col, y)).fillna(0.0)


def _f_regression(X, y):
    def _f_regression_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return sklearn_f_regression(x[x_not_na].to_frame(), y[x_not_na])[0][0]

    return X.apply(lambda col: _f_regression_series(col, y)).fillna(0.0)


def f_classif(X, y, n_jobs):
    return parallel_df(_f_classif, X, y, n_jobs=n_jobs)


def f_regression(X, y, n_jobs):
    return parallel_df(_f_regression, X, y, n_jobs=n_jobs)


def _ks_classif(X, y):
    def _ks_classif_series(x, y):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        x = x[x_not_na]
        y = y[x_not_na]
        return x.groupby(y).apply(lambda s: ks_2samp(s, x[y != s.name])[0]).mean()

    return X.apply(lambda col: _ks_classif_series(col, y)).fillna(0.0)


def ks_classif(X, y, n_jobs):
    return parallel_df(_ks_classif, X, y, n_jobs=n_jobs)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)

    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)


def encode_df(X, y, cat_features, cat_encoding):
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols=cat_features, handle_missing='return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols=cat_features, handle_missing='return_nan'),
        'target': ce.TargetEncoder(cols=cat_features, handle_missing='return_nan')
    }
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    return X


def mrmr_classif(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a classification task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "ks" (kolmogorov-smirnov), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    if relevance == "f":
        relevance_func = functools.partial(f_classif, n_jobs=n_jobs)
    elif relevance == "ks":
        relevance_func = functools.partial(ks_classif, n_jobs=n_jobs)
    elif relevance == "rf":
        relevance_func = random_forest_classif
    else:
        relevance_func = relevance

    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)


def mrmr_regression(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=None, cat_encoding='leave_one_out',
        only_same_domain=False, return_scores=False,
        n_jobs=-1, show_progress=True
):
    """MRMR feature selection for a regression task
    Parameters
    ----------
    X: pandas.DataFrame
        A DataFrame containing all the features.
    y: pandas.Series
        A Series containing the (categorical) target variable.
    K: int
        Number of features to select.
    features: list of str (optional, default=None)
        List of numeric column names. If not specified, all numeric columns (integer and float) are used.
    relevance: str or callable
        Relevance method.
        If string, name of method, supported: "f" (f-statistic), "rf" (random forest).
        If callable, it should take "X" and "y" as input and return a pandas.Series containing a (non-negative)
        score of relevance for each feature.
    redundancy: str or callable
        Redundancy method.
        If string, name of method, supported: "c" (Pearson correlation).
        If callable, it should take "X", "target_column" and "features" as input and return a pandas.Series
        containing a score of redundancy for each feature.
    denominator: str or callable (optional, default='mean')
        Synthesis function to apply to the denominator of MRMR score.
        If string, name of method. Supported: 'max', 'mean'.
        If callable, it should take an iterable as input and return a scalar.
    cat_features: list (optional, default=None)
        List of categorical features. If None, all string columns will be encoded.
    cat_encoding: str
        Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'.
    only_same_domain: bool (optional, default=False)
        If False, all the necessary correlation coefficients are computed.
        If True, only features belonging to the same domain are compared.
        Domain is defined by the string preceding the first underscore:
        for instance "cusinfo_age" and "cusinfo_income" belong to the same domain, whereas "age" and "income" don't.
    return_scores: bool (optional, default=False)
        If False, only the list of selected features is returned.
        If True, a tuple containing (list of selected features, relevance, redundancy) is returned.

    n_jobs: int (optional, default=-1)
        Maximum number of workers to use. Only used when relevance = "f" or redundancy = "corr".
        If -1, use as many workers as min(cpu count, number of features).
    show_progress: bool (optional, default=True)
        If False, no progress bar is displayed.
        If True, a TQDM progress bar shows the number of features processed.
    Returns
    -------
    selected_features: list of str
        List of selected features.
    """
    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = functools.partial(f_regression, n_jobs=n_jobs) if relevance == 'f' else (
        random_forest_regression if relevance == 'rf' else relevance)
    redundancy_func = functools.partial(correlation, n_jobs=n_jobs) if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
        np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    return mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                     relevance_args=relevance_args, redundancy_args=redundancy_args,
                     denominator_func=denominator_func, only_same_domain=only_same_domain,
                     return_scores=return_scores, show_progress=show_progress)