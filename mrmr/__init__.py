from importlib.metadata import version

from mrmr import bigquery
from mrmr import pandas
# trigger: zsh: illegal hardware instructions if used with wrong package
# from mrmr import polars
from mrmr import spark
from mrmr.pandas import mrmr_classif, mrmr_regression
from mrmr.main import mrmr_base

_version__ = version('mrmr_selection')