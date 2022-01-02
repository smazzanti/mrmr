from joblib import Parallel, delayed
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .main import mrmr_base


def parallel_df(func, df, series):
    n_jobs = min(cpu_count(), len(df.columns))
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


def f_classif(X, y):
    return parallel_df(_f_classif, X, y)


def f_regression(X, y):
    return parallel_df(_f_regression, X, y)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def random_forest_regression(X, y):
    forest = RandomForestRegressor(max_depth=5, random_state=0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X):
    def _correlation(X, y):
        return X.corrwith(y).fillna(0.0)
    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column])


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
        cat_features=[], cat_encoding='leave_one_out',
        only_same_domain=False
):
    '''
    Do MRMR feature selection on classification task.

    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the (categorical) target variable.
        K: (int) Number of features to select.
        relevance: (str or function) Relevance method.
            If function, it should take X and y as input and return a pandas.Series containing a (non-negative) score of relevance for each feature of X.
            If string, name of method, supported: 'f' (f-statistic), 'rf' (random forest).
        redundancy: (str or function) Redundancy method.
            If function, it should take A and b as input and return a pandas.Series containing a (non-negative) score of redundancy for each feature of A.
            If string, name of method, supported: 'c' (Pearson correlation)
        denominator: (str or function) Synthesis function to apply to the denominator of MRMR score.
            If function, it should take an iterable as input and return a scalar.
            If string, name of method, supported: 'max', 'mean'
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,
            whereas "age" and "income" don't.

    Returns:
        (list) List of K names of selected features (sorted by importance).
    '''

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = f_classif if relevance=='f' else (
                     random_forest_classif if relevance=='rf' else relevance)
    redundancy_func = correlation if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
                       np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    selected_features = mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain)

    return selected_features


def mrmr_regression(
        X, y, K,
        relevance='f', redundancy='c', denominator='mean',
        cat_features=[], cat_encoding='leave_one_out',
        only_same_domain=False
):
    '''
    Do MRMR feature selection on regression task.

    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the (numerical) target variable.
        K: (int) Number of features to select.
        relevance: (str or function) Relevance method.
            If function, it should take X and y as input and return a pandas.Series containing a (non-negative) score of relevance for each feature of X.
            If string, name of method, supported: 'f' (f-statistic), 'rf' (random forest).
        redundancy: (str or function) Redundancy method.
            If function, it should take A and b as input and return a pandas.Series containing a (non-negative) score of redundancy for each feature of A.
            If string, name of method, supported: 'c' (Pearson correlation)
        denominator: (str or function) Synthesis function to apply to the denominator of MRMR score.
            If function, it should take an iterable as input and return a scalar.
            If string, name of method, supported: 'max', 'mean'
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,
            whereas "age" and "income" don't.

    Returns:
        (list) List of K names of selected features (sorted by importance).
    '''

    if cat_features:
        X = encode_df(X=X, y=y, cat_features=cat_features, cat_encoding=cat_encoding)

    relevance_func = f_classif if relevance=='f' else (
                     random_forest_classif if relevance=='rf' else relevance)
    redundancy_func = correlation if redundancy == 'c' else redundancy
    denominator_func = np.mean if denominator == 'mean' else (
                       np.max if denominator == 'max' else denominator)

    relevance_args = {'X': X, 'y': y}
    redundancy_args = {'X': X}

    selected_features = mrmr_base(K=K, relevance_func=relevance_func, redundancy_func=redundancy_func,
                                  relevance_args=relevance_args, redundancy_args=redundancy_args,
                                  denominator_func=denominator_func, only_same_domain=only_same_domain)

    return selected_features