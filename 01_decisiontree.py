import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
import seaborn as sns

data = np.load("MLProject2024/fashion_train.npy")
images = data[:, :-1]
labels = data[:, -1]


# functions for decision tree
# -gini group
# -gini split
# -split
# -best split
# -decision tree
#

# gini = 1- sum(p)
# p=p0**2+p1**2+p2**2+p3**2+p4**2+p5**2
# p0 = n0/n
# p1 = n1/n
# p2 = n2/n
# p3 = n3/n
# p4 = n4/n
# p5 = n5/n
# n0 = number of samples in class 0
# n1 = number of samples in class 1
# n2 = number of samples in class 2
# n3 = number of samples in class 3
# n4 = number of samples in class 4
# n = total number of samples in the group


def gini_group(dataset):
    """takes one group of data and calculates the gini impurity of that"""
    # labels = dataset[:, -1]
    counts = np.bincount(labels)  # count the number of samples in each class
    n0, n1, n2, n3, n4 = counts
    n = n0 + n1 + n2 + n3 + n4
    if n == 0:  # if the group is empty, gini impurity is 0
        return 0
    p0 = n0 / n
    p1 = n1 / n
    p2 = n2 / n
    p3 = n3 / n
    p4 = n4 / n

    p = p0**2 + p1**2 + p2**2 + p3**2 + p4**2
    gini = 1 - p
    return gini


def gini_split(left, right):
    """takes two groups of data and calculates the gini impurity of the split"""
    n_left = len(left)
    n_right = len(right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0
    gini_split = n_left / n_total * gini_group(left) + n_right / n_total * gini_group(
        right
    )

    return gini_split


def split(dataset, column, value):
    """takes a dataset, a column and a value and splits the dataset
    into two groups based on the value(threshold) of the column,
    returns two groups"""
    left = []
    right = []
    for row in dataset:
        if row[column] < value:
            left.append(row)
        else:
            right.append(row)
    return np.array(left), np.array(right)


## Min and max Values are too big to calcuate, check them


def best_split(data):
    """takes data, finds the best split for the data and returns the column, value and gini impurity of the split"""
    best_gini = 1
    best_column = None
    best_value = None
    for column in range(data.shape[1] - 1):  # exclude label column
        unique_values = np.unique(
            data[:, column]
        )  # find unique values in the column, not necessary but speeds up the process
        for row in unique_values:
            # row = int(row)
            left, right = split(data, column, row)
            gini = gini_split(left, right)
            if gini < best_gini:
                best_gini = gini
                best_column = column
                best_value = row
                best_left = left
                best_right = right
    return best_left, best_right, best_column, best_value, best_gini


def build_tree(
    min_samples, data
):  # limit the minimum number of samples in a group, !consider adding max depth or other limits
    if len(data) < min_samples:
        return data
    current_data = data
    best_left, best_right, best_column, best_value = best_split(current_data)

    # TBD


## Testing:

# testing_best_split = best_split(images)
# print(testing_best_split)


## Debuged till best_split, altough the fuction will neverfinish running, the nested loops not able to handle this much data - Ivett 28.10.
