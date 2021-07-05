from joblib import Parallel, delayed
from multiprocessing import cpu_count
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.feature_selection import f_classif as sklearn_f_classif
from sklearn.feature_selection import f_regression as sklearn_f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings; warnings.filterwarnings("ignore")
import tqdm

FLOOR = .00001

#####################################################################
# Functions for parallelization

def parallel_df(func, df, series):
    n_jobs = min(cpu_count(), len(df.columns))
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs = n_jobs)(
        delayed (func) (df.iloc[:, col_chunk], series) 
        for col_chunk in col_chunks
    )
    return pd.concat(lst)

#####################################################################
# Functions for computing relevance and redundancy

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

def _corr_pearson(A, b):
    return A.corrwith(b).fillna(0.0).abs().clip(FLOOR)

#####################################################################
# Functions for computing relevance and redundancy
# Parallelized versions (except random_forest_classif which cannot be parallelized)

def f_classif(X, y):
    '''Compute F-statistic between DataFrame X and Series y'''
    return parallel_df(_f_classif, X, y)

def f_regression(X, y):
    '''Compute F-statistic between DataFrame X and Series y'''
    return parallel_df(_f_regression, X, y)

def random_forest_classif(X, y):
    '''Compute feature importance of each column of DataFrame X after fitting a random forest on Series y'''
    forest = RandomForestClassifier(max_depth = 5, random_state = 0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index = X.columns)

def random_forest_regression(X, y):
    '''Compute feature importance of each column of DataFrame X after fitting a random forest on Series y'''
    forest = RandomForestRegressor(max_depth = 5, random_state = 0).fit(X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index = X.columns)

def corr_pearson(A, b):
    '''Compute Pearson correlation between DataFrame A and Series b'''
    return parallel_df(_corr_pearson, A, b)

#####################################################################

def encode_df(X, y, cat_features, cat_encoding):
    
    ENCODERS = {
        'leave_one_out': ce.LeaveOneOutEncoder(cols = cat_features, handle_missing = 'return_nan'),
        'james_stein': ce.JamesSteinEncoder(cols = cat_features, handle_missing = 'return_nan'),
        'target': ce.TargetEncoder(cols = cat_features, handle_missing = 'return_nan')
    }
    
    X = ENCODERS[cat_encoding].fit_transform(X, y)
    
    return X
    
#####################################################################
# MRMR selection

def _mrmr_base(
    X, y, K, 
    func_relevance, func_redundancy, func_denominator, 
    cat_features = None, cat_encoding = 'leave_one_out',
    only_same_domain = False
):
    '''
    Do MRMR selection.
    
    Args:
        X: (pandas.DataFrame) A Dataframe consisting of numeric features only.
        y: (pandas.Series) A Series containing the target variable.
        K: (int) Number of features to select.
        func_relevance: (func) Relevance function.
        func_redundancy: (func) Redundancy function.
        func_denominator: (func) Synthesis function to apply to the denominator.
        cat_features: (list) List of categorical features. If None, all string columns will be encoded
        cat_encoding: (str) Name of categorical encoding. Supported: 'leave_one_out', 'james_stein', 'target'
        only_same_domain: (bool) If False, all the necessary correlation coefficients are computed.
            If True, only features belonging to the same domain are compared.
            Domain is defined by the string preceding the first underscore:
            for instance "cusinfo_age" and "cusinfo_income" belong to the same domain,whereas "age" and "income" don't.
        
    Returns:
        (list) List of K names of selected features (sorted by importance).
    '''
    
    # encode categorical features
    X = encode_df(X, y, cat_features = cat_features, cat_encoding = cat_encoding)
            
    # compute relevance
    rel = func_relevance(X, y)
            
    # keep only columns that have positive relevance
    columns = rel[rel.fillna(0) > 0].index.to_list()
    K = min(K, len(columns))
    rel = rel.loc[columns]
    
    # init
    red = pd.DataFrame(FLOOR, index = columns, columns = columns)
    selected = []
    not_selected = columns.copy()
    
    for i in tqdm.tqdm(range(K)):
        
        # compute score numerator
        score_numerator = rel.loc[not_selected]

        # compute score denominator
        if i > 0:

            last_selected = selected[-1]
            
            if only_same_domain:
                not_selected_subset = [c for c in not_selected if c.split('_')[0] == last_selected.split('_')[0]]
            else:
                not_selected_subset = not_selected
                                
            if not_selected_subset:
                red.loc[not_selected_subset, last_selected] = func_redundancy(X[not_selected_subset], X[last_selected]).abs().clip(FLOOR).fillna(FLOOR)
                score_denominator = red.loc[not_selected, selected].apply(func_denominator, axis = 1).round(5).replace(1.0, float('Inf'))
                
        else:
            score_denominator = pd.Series(1, index = columns)
            
        # compute score and select best
        score = score_numerator / score_denominator
        best = score.index[score.argmax()]
        selected.append(best)
        not_selected.remove(best)
        
    return selected
    
def mrmr_classif(
    X, y, K, 
    relevance = 'f', redundancy = 'c', denominator = 'mean', 
    cat_features = None, cat_encoding = 'leave_one_out',
    only_same_domain = False
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
    
    FUNCS = {
        'f': f_classif, 
        'rf': random_forest_classif,
        'c': corr_pearson,
        'mean': np.mean,
        'max': np.max
    }
    
    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    
    return _mrmr_base(
        X, y, K, 
        func_relevance, func_redundancy, func_denominator, 
        cat_features, cat_encoding,
        only_same_domain
    )
        
def mrmr_regression(
    X, y, K, 
    relevance = 'f', redundancy = 'c', denominator = 'mean', 
    cat_features = None, cat_encoding = 'leave_one_out',
    only_same_domain = False
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
    
    FUNCS = {
        'f': f_regression, 
        'rf': random_forest_regression,
        'c': corr_pearson,
        'mean': np.mean,
        'max': np.max
    }
    
    func_relevance = FUNCS[relevance] if relevance in FUNCS.keys() else relevance
    func_redundancy = FUNCS[redundancy] if redundancy in FUNCS.keys() else redundancy
    func_denominator = FUNCS[denominator] if denominator in FUNCS.keys() else denominator
    
    return _mrmr_base(
        X, y, K, 
        func_relevance, func_redundancy, func_denominator, 
        cat_features, cat_encoding,
        only_same_domain
    )
