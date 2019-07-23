import io
import os
import pickle

import requests
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkagg")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

# %matplotlib inline


# Load data
columns = [
    "age",
    "workClass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
    ]

train_data = pd.read_csv('data/adult.data',
                         names=columns,
                         sep=' *, *',
                         na_values='?')

test_data = pd.read_csv('data/adult.test',
                        names=columns,
                        sep=' *, *',
                        skiprows=1,
                        na_values='?')

# Exploratory analysis

train_data.info()

num_attributes = train_data.select_dtypes(include=['int'])
print(num_attributes.columns)
num_attributes.hist(figsize=(10, 10))
plt.show()


cat_attributes = train_data.select_dtypes(include=['object'])
print(cat_attributes.columns)

sns.countplot(y='workClass', hue='income', data=cat_attributes)
sns.countplot(y='occupation', hue='income', data=cat_attributes)

plt.show()


import ipdb; ipdb.set_trace()


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, type):
        self.type = type

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.select_dtypes(include=[self.type])


steps = [("num_attr_selector", ColumnsSelector(type='int')),
         ("scaler", StandardScaler())]

# If we call the fit and transform methods for the num_pipeline it internally
# calls the fit and transform methods for all the transformers defined in the
# pipeline. In this case, the ColumnsSelector and StandardScaler transformers.
num_pipeline = Pipeline(steps=steps)


class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, strategy='most_frequent'):
        self.columns = columns
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns

        if self.strategy == 'most_frequent':
            self.fill = {column: X[column].value_counts().index[0]
                         for column in self.columns}
        else:
            self.fill = {column: '0' for column in self.columns}

        return self

    def transform(self, X):

        X_copy = X.copy()

        for column in self.columns:
            X_copy[column] = X_copy[column].fillna(self.fill[column])

        return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropFirst=True):
        self.categories = dict()
        self.dropFirst = dropFirst

    def fit(self, X, y=None):
        join_df = pd.concat([train_data, test_data])
        join_df = join_df.select_dtypes(include=['object'])
        for column in join_df.columns:
            self.categories[column] = join_df[column].value_counts().index.tolist()
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.select_dtypes(include=['object'])

        for column in X_copy.columns:
            X_copy[column] = X_copy[column].astype(
                {
                    column: CategoricalDtype(self.categories[column])
                }
            )

        return pd.get_dummies(X_copy, drop_first=self.dropFirst)


steps = [
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=['workClass',
                                                'occupation',
                                                'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
]

cat_pipeline = Pipeline(steps=steps)


# Finish pipeline
full_pipeline = FeatureUnion([("num_pipe", num_pipeline),
                              ("cat_pipeline", cat_pipeline)]

drop_columns = ['fnlwgt', 'education']
train_data.drop(drop_columns, axis=1, inplace=True)
test_data.drop(drop_columns, axis=1, inplace=True)

# Training the model