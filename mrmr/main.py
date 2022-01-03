import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings("ignore")
from tqdm import tqdm

FLOOR = .001

def groupstats2fstat(avg, var, n):
    """Compute F-statistic of some variables across groups

    Compute F-statistic of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.

    Parameters
    ----------
    avg: pandas.DataFrame of shape (n_groups, n_variables)
        Simple average of variables within groups. Each row is a group, each column is a variable.

    var: pandas.DataFrame of shape (n_groups, n_variables)
        Variance of variables within groups. Each row is a group, each column is a variable.

    n: pandas.DataFrame of shape (n_groups, n_variables)
        Count of instances for whom variable is not null. Each row is a group, each column is a variable.

    Returns
    -------
    f: pandas.Series of shape (n_variables, )
        F-statistic of each variable, based on group statistics.

    Reference
    ---------
    https://en.wikipedia.org/wiki/F-test
    """
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f.fillna(0.0)


def mrmr_base(K, relevance_func, redundancy_func,
              relevance_args={}, redundancy_args={},
              denominator_func=np.mean, only_same_domain=False):

    relevance = relevance_func(**relevance_args)
    features = relevance[relevance.fillna(0) > 0].index.to_list()
    relevance = relevance.loc[features]
    redundancy = pd.DataFrame(FLOOR, index=features, columns=features)
    K = min(K, len(features))
    selected_features = []
    not_selected_features = features.copy()

    for i in tqdm(range(K)):

        score_numerator = relevance.loc[not_selected_features]

        if i > 0:

            last_selected_feature = selected_features[-1]

            if only_same_domain:
                not_selected_features_sub = [c for c in not_selected_features if c.split('_')[0] == last_selected_feature.split('_')[0]]
            else:
                not_selected_features_sub = not_selected_features

            if not_selected_features_sub:
                redundancy.loc[not_selected_features_sub, last_selected_feature] = redundancy_func(
                    target_column=last_selected_feature,
                    features=not_selected_features_sub,
                    **redundancy_args
                ).fillna(FLOOR).abs().clip(FLOOR)
                score_denominator = redundancy.loc[not_selected_features, selected_features].apply(
                    denominator_func, axis=1).replace(1.0, float('Inf'))

        else:
            score_denominator = pd.Series(1, index=features)

        score = score_numerator / score_denominator

        best_feature = score.index[score.argmax()]
        selected_features.append(best_feature)
        not_selected_features.remove(best_feature)

    return selected_features