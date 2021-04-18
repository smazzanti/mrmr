# mrmr 
*(Minimum-Redundancy-Maximum-Relevance)*

**mrmr** is a "minimal optimal" *feature selection* algorithm, meaning that it seeks to find a feature set giving best possible
classification, given a (small) number of features.

## How to install

You can install **mrmr** in your environment via:

<pre>
pip install git+https://github.com/smazzanti/mrmr
</pre>

## How to use

You have a dataframe composed of numeric variables (**X**) and a series which is your (binary or multiclass) target variable (**y**).
You want to select **K** features such that they are maximally relevant, but also as little redundant as possible with each other.

<pre>
from mrmr import mrmr_classif
from sklearn.datasets import make_classification

# create some data
X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
X = pd.DataFrame(X)
y = pd.Series(y)

# use mrmr classification
selected_features = mrmr_classif(X, y, K = 10)
</pre>

Note: the output of mrmr_classif is a list containing K selected features. This is a **ranking**, therefore, if you want to make a further selection, take the first elements of this list.

## Reference

For an easy-going introduction to MRMR, read my article on **Towards Data Science**: [“MRMR” Explained Exactly How You Wished Someone Explained to You](https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b).

Also, this article describes an example of MRMR used on the world famous **MNIST** dataset: [Feature Selection: How To Throw Away 95% of Your Data and Get 95% Accuracy](https://towardsdatascience.com/feature-selection-how-to-throw-away-95-of-your-data-and-get-95-accuracy-ad41ca016877)

MRMR was born in **2003**, this is the original paper: [Minimum Redundancy Feature Selection From Microarray Gene Expression Data](https://www.researchgate.net/publication/4033100_Minimum_Redundancy_Feature_Selection_From_Microarray_Gene_Expression_Data).

Since then, it has been used in many practical applications, due to its simplicity and effectiveness.
For instance, in **2019**, **Uber** engineers published a paper describing how they implemented MRMR in their marketing machine learning platform [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/).
