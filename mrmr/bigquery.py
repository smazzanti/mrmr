import jinja2
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_numeric_columns(bq_client, table_id):
    schema = bq_client.get_table(table_id).schema
    return [field.name for field in schema if field.field_type in ['INTEGER', 'FLOAT']]


def groupstats2fstat(avg, var, n):
    '''
    Compute F-statistic (https://en.wikipedia.org/wiki/F-test) of many variables, with respect to some groups of instances.
    For each group, the input consists of the simple average, variance and count with respect to each variable.

    Args:
        avg: (pd.DataFrame) Simple average of variable within a group. Each row is a group, each column is a variable.
        var: (pd.DataFrame) Variance of a variable within a group. Each row is a group, each column is a variable.
        n: (pd.DataFrame) Count of instances for whom variable is not null. Each row is a group, each column is a variable.

    Returns:
        f: pd.Series. F-statistic of each variable.
    '''
    avg_global = (avg * n).sum() / n.sum()  # global average of each variable
    numerator = (n * ((avg - avg_global) ** 2)).sum() / (len(n) - 1)  # between group variability
    denominator = (var * n).sum() / (n.sum() - len(n))  # within group variability
    f = numerator / denominator
    return f


def f_classif(bq_client, table_id, target_column, numeric_columns=None):
    '''
    Return F-statistic of one (discrete) target column and many (discrete or continuous) numeric columns

    Args:
        table_id: (str) Unique ID of BigQuery table.
        target_column: (str) Name of target column.
        numeric_columns: (list) Optional. List of names of numeric columns. If not specified, all numeric columns are used.

    Returns:
        f: (pd.Series) F-statistic between each numeric column and the target column.
    '''

    if numeric_columns is None:
        numeric_columns = [column for column in get_numeric_columns(bq_client, table_id) if column != target_column]

    jinja_query = """
{% set COLUMNS = numeric_columns%}
SELECT 
  target_column,
  {% for COLUMN in COLUMNS -%}
  METRIC(CAST({{COLUMN}} AS FLOAT64)) AS {{COLUMN}}{% if not loop.last %},{% endif %}
  {% endfor -%}
FROM 
  table_id
GROUP BY 
  target_column
    """\
        .replace('table_id', table_id)\
        .replace('target_column', target_column)\
        .replace('numeric_columns', str(numeric_columns))

    avg = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'AVG')).render()
    ).to_dataframe().set_index(target_column, drop = True).astype(float)

    var = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'VAR_POP')).render()
    ).to_dataframe().set_index(target_column, drop = True).astype(float)

    n = bq_client.query(
        query=jinja2.Template(jinja_query.replace('METRIC', 'COUNT')).render()
    ).to_dataframe().set_index(target_column, drop = True).astype(float)

    f = groupstats2fstat(avg=avg, var=var, n=n)
    f.name = target_column

    return f


def correlation(bq_client, table_id, target_column, numeric_columns=None):
    '''
    Return (Pearson) correlation between one numeric target column and many numeric columnsx.

    Args:
        table_id: (str) Unique ID of BigQuery table.
        target_column: (str) Name of target column.
        numeric_columns: (list) Optional. List of names of numeric columns. If not specified, all numeric columns are used.

    Returns:
        corr: (pd.Series) Correlation between each column and the target column.
    '''

    if numeric_columns is None:
        numeric_columns = [column for column in get_numeric_columns(bq_client, table_id) if column != target_column]

    jinja_query = """
{% set COLUMNS = numeric_columns%}
SELECT
  {% for COLUMN in COLUMNS -%}
  CORR(target_column, {{COLUMN}}) AS {{COLUMN}}{% if not loop.last %},{% endif %}
  {% endfor -%}
FROM 
  table_id
        """ \
        .replace('table_id', table_id) \
        .replace('target_column', target_column) \
        .replace('numeric_columns', str(numeric_columns))

    corr = bq_client.query(query=jinja2.Template(jinja_query).render()).to_dataframe().iloc[0,:]
    corr.name = target_column

    return corr


def f_regression(bq_client, table_id, target_column, numeric_columns=None):

    if numeric_columns is None:
        numeric_columns = [column for column in get_numeric_columns(bq_client, table_id) if column != target_column]

    jinja_query = """
    {% set COLUMNS = numeric_columns%}
    SELECT
      {% for COLUMN in COLUMNS -%}
      COUNTIF(target_column IS NOT NULL AND {{COLUMN}} IS NOT NULL) AS {{COLUMN}}{% if not loop.last %},{% endif %}
      {% endfor -%}
    FROM 
      table_id
            """ \
        .replace('table_id', table_id) \
        .replace('target_column', target_column) \
        .replace('numeric_columns', str(numeric_columns))

    corr_coef = correlation(bq_client, table_id, target_column, numeric_columns)

    n = bq_client.query(query=jinja2.Template(jinja_query).render()).to_dataframe().iloc[0,:]
    n.name = target_column

    deg_of_freedom = n - 1
    corr_coef_squared = corr_coef ** 2
    f = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom

    return f


def mrmr_base(
    task,
    bq_client,
    table_id,
    target_column,
    K,
    numeric_columns=None,
    func_denominator='mean',
    only_same_domain=False
):

    FLOOR = .00001

    # compute f statistic for all columns
    if task == 'classif':
        f = f_classif(bq_client, table_id, target_column, numeric_columns)
    elif task == 'regression':
        f = f_regression(bq_client, table_id, target_column, numeric_columns)
    else:
        raise #

    if func_denominator == 'mean':
        func_denominator = np.mean
    elif func_denominator == 'max':
        func_denominator = np.max
    else:
        raise #

    # keep only features that have positive F-statistic
    numeric_columns = f[f.fillna(0) > 0].index.to_list()
    K = min(K, len(numeric_columns))
    f = f.loc[numeric_columns]

    # init
    corr = pd.DataFrame(FLOOR, index=numeric_columns, columns=numeric_columns)
    selected = []
    not_selected = numeric_columns.copy()
    score_denominator = pd.Series(1, index=not_selected)  # at the first iteration, only f will be considered (= denominator is 1.0)

    for i in tqdm(range(K)):

        score_numerator = f.loc[not_selected]

        if i > 0:
            last_selected = selected[-1]

            if only_same_domain:
                not_selected_subset = [c for c in not_selected if c.split('_')[0] == last_selected.split('_')[0]]
            else:
                not_selected_subset = not_selected

            # update correlation matrix (compute correlation coefficients between last selected feature and other features)
            if not_selected_subset:
                corr.loc[not_selected_subset, last_selected] = correlation(
                    bq_client=bq_client,
                    table_id=table_id,
                    target_column=last_selected,
                    numeric_columns=not_selected_subset
                ).fillna(FLOOR).abs().clip(FLOOR)

            # denominator is max correlation between feature and features that have been selected previously
            score_denominator = corr.loc[not_selected, selected].apply(func_denominator, axis=1).replace(1.0, float('Inf'))

        # score is basically f / max(correlation). then, select feature having highest score
        score = score_numerator / score_denominator
        best = score.idxmax()
        selected.append(best)
        not_selected.remove(best)

    return selected


def mrmr_classif(
    bq_client,
    table_id, 
    target_column, 
    K,
    numeric_columns=None,
    func_denominator='mean',
    only_same_domain=False
):
    '''
    Return selected (numeric) columns for a classification problem, using MRMR algorithm (https://arxiv.org/pdf/1908.05376.pdf).

    Args:
        bq_client:
        table_id:
        target_column:
        K:
        numeric_columns:
        func_denominator:
        only_same_domain:

    Returns:
        selected: (list) Ranked columns, up to K-th column.
    '''

    selected = mrmr_base(
        task='classif',
        bq_client=bq_client,
        table_id=table_id,
        target_column=target_column,
        K=K,
        numeric_columns=numeric_columns,
        func_denominator=func_denominator,
        only_same_domain=only_same_domain
    )
    return selected


def mrmr_regression(
    bq_client,
    table_id,
    target_column,
    K,
    numeric_columns=None,
    func_denominator='mean',
    only_same_domain=False
):
    '''
    Return selected (numeric) columns for a regression problem, using MRMR algorithm (https://arxiv.org/pdf/1908.05376.pdf).

    Args:
        bq_client:
        table_id:
        target_column:
        K:
        numeric_columns:
        func_denominator:
        only_same_domain:

    Returns:
        selected: (list) Ranked columns, up to K-th column.
    '''

    selected = mrmr_base(
        task='regression',
        bq_client=bq_client,
        table_id=table_id,
        target_column=target_column,
        K=K,
        numeric_columns=numeric_columns,
        func_denominator=func_denominator,
        only_same_domain=only_same_domain
    )
    return selected
