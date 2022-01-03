<p>
<img src="./docs/img/mrmr_logo_white_bck.png" alt="drawing" width="400"/>
</p>

## What is mRMR

*mRMR* *(minimum-Redundancy-Maximum-Relevance)* is a feature selection algorithm.

## Why is it unique

The peculiarity of *mRMR* is that it is a *minimal-optimal* feature selection algorithm. 
<br/>
This means that it is designed to find the smallest relevant subset of features for a given Machine Learning task.
<br/>
On the contrary, the majority of other methods (for instance, Boruta or Positive-Feature-Importance) are *all-relevant*, 
since they identify all the features that have some kind of relationship with the target variable.

## When to use mRMR

Due to its efficiency, *mRMR* is ideal for practical ML applications, 
where it is necessary to perform feature selection frequently and automatically, 
in a relatively small amount of time.

For instance, in **2019**, **Uber** engineers published a paper describing how they implemented 
mRMR in their marketing machine learning platform [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).

## Why this library
This library has been created to provide a **cross platform** go-to place for *mRMR*.
Currently, the following tools are supported:
- Pandas (in-memory)
- Spark
- Google BigQuery

Support for other tools will be added.

## How to install this library

You can install this library in your environment via pip:

<pre>
pip install git+https://github.com/smazzanti/mrmr
</pre>

Alternatively, if you want to add it to a "requirements.txt" file, you can paste this line into the txt file:
<pre>
git+https://github.com/smazzanti/mrmr@main#egg=mrmr
</pre>

## How to use this library

The library has a module for each supported tool.<br/>
Any module has *at least* these two functions:
- `mrmr_classif`, for feature selection when the target variable is categorical (binary or multiclass).
- `mrmr_regression`, for feature selection when the target variable is numeric.

Let's see some examples.

#### 1. Pandas example
You have a Pandas DataFrame (**X**) and a series which is your target variable (**y**).
You want to select the best **K** features to make predictions on **y**.

<pre>
# create some pandas data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
X = pd.DataFrame(X)
y = pd.Series(y)

# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X, y, K = 10)
</pre>

Note: the output of mrmr_classif is a list containing K selected features. This is a **ranking**, therefore, if you want to make a further selection, take the first elements of this list.

#### 2. Spark example

<pre>
# init spark session
import pyspark
session = pyspark.sql.SparkSession(pyspark.context.SparkContext())

# make a spark dataframe
data = [(1.0, 1.0, 1.0, 7.0, 1.5, -2.3), 
        (2.0, float('NaN'), 2.0, 7.0, 8.5, 6.7), 
        (2.0, float('NaN'), 3.0, 7.0, -2.3, 4.4),
        (3.0, 4.0, 3.0, 7.0, 0.0, 0.0),
        (4.0, 5.0, 4.0, 7.0, 12.1, -5.2)]
columns = ["target", "some_null", "feature", "constant", "other_feature", "another_feature"]
df_spark = session.createDataFrame(data=data, schema=columns)

# select top 2 features using mRMR
import mrmr
selected_features = mrmr.spark.mrmr_regression(df_spark, target_column="target", K=2)
</pre>


## Reference

For an easy-going introduction to MRMR, read my article on **Towards Data Science**: [“MRMR” Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b).

Also, this article describes an example of MRMR used on the world famous **MNIST** dataset: [Feature Selection: How To Throw Away 95% of Your Data and Get 95% Accuracy](https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877)

MRMR was born in **2003**, this is the original paper: [Minimum Redundancy Feature Selection From Microarray Gene Expression Data](https://www.researchgate.net/publication/4033100_Minimum_Redundancy_Feature_Selection_From_Microarray_Gene_Expression_Data).

Since then, it has been used in many practical applications, due to its simplicity and effectiveness.
For instance, in **2019**, **Uber** engineers published a paper describing how they implemented MRMR in their marketing machine learning platform [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).
