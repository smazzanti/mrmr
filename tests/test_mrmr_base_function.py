import mrmr
import numpy as np
import pandas as pd
import pytest

variables = ['drop', 'second', 'third', 'first']

relevance = pd.Series(
    [0, 1, 2, 3], index=variables)

redundancy = pd.DataFrame([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.2, 0.1],
    [0.0, 0.2, 1.0, 0.8],
    [0.0, 0.1, 0.8, 1.0]], index=variables, columns=variables)

def relevance_func():
    return relevance

def redundancy_func(target_column,features):
    return redundancy.loc[features, target_column]

def test_mrmr_base():
    selected_features = mrmr.mrmr_base(
        K=100, relevance_func=relevance_func, redundancy_func=redundancy_func,
        relevance_args={}, redundancy_args={},
        denominator_func=np.mean, only_same_domain=False)

    assert selected_features == ['first', 'second', 'third']