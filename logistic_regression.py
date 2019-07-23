import itertools

from numpy import (
    log,
    exp
)
import pandas as pd




# Load data names from file
with open("data/adult.names", 'r') as infile:
    columns = [line.split(":", 1)[0]
               for line in itertools.islice(infile, 96, 110)]
columns.append("greater_than_50k")

# Add dummie cols for sex
df = pd.read_csv("data/adult.data", index_col=False, names=columns, header=None)

df = pd.merge(df,
              pd.get_dummies(df.sex, drop_first=True),
              left_index=True,
              right_index=True)
df = pd.merge(df,
              pd.get_dummies(df.greater_than_50k, drop_first=True),
              left_index=True,
              right_index=True)
df.columns = df.columns.str.replace(' ', '')

print(df.head())


Y = df[">50K"]
X = df[["Male"]]


def cost_logistic(theta, x, y):
    h = 1 / (1 + exp(- theta * x))
    return - y * log(h) - (1 - y) * log(1-h)


def gradient_logistic():

    return


import ipdb; ipdb.set_trace()


