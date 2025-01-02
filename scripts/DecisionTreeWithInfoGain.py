import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
import seaborn as sns
from collections import Counter

data = np.load("MLProject2024/fashion_train.npy")
images = data[:, :-1]
labels = data[:, -1]


class Node:
    """
    Represents a node in the decision tree.

    Parameters
    ----------
    feature : int, default=None
        The index of the feature used for splitting at this node.
        If None, this node is a leaf.

    threshold : float, default=None
        The threshold value for the feature at which the split occurs.
        If None, this node is a leaf.

    left : Node or None, default=None
        The left child node resulting from the split.

    right : Node or None, default=None
        The right child node resulting from the split.

    value : int, default=None
        The predicted class or value at a leaf node. If None, this node
        is not a leaf.

    Methods
    -------
    is_leaf_node() -> bool
        Returns True if the node is a leaf node, otherwise False.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the node is a leaf node.

        Returns
        -------
        bool
            True if this node is a leaf node (i.e., it has a prediction
            value instead of a split), otherwise False.
        """
        # print("Leafnode found")
        return self.value is not None


class DTfromScratch:
    """
    A decision tree classifier that uses information gain for splits.

    Parameters
    ----------
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.

    max_depth : int, default=50
        The maximum depth of the tree. If None, nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    number_of_features : int, default=None
        The number of features to consider when looking for the best split.

    Attributes
    ----------
    root : Node or None
        The root node of the decision tree.

    Methods
    -------
    fit(X, y)
        Trains the decision tree on the provided feature matrix X and target vector y.

    predict(X)
        Predicts the class labels for the input feature matrix X.

    _grow_tree(X, y, depth=0)
        Recursively builds the decision tree by splitting the data at each node.

    _best_split(X, y, feature_idx)
        Finds the best split for the data based on information gain.

    _information_gain(y, X_column, threshold)
        Calculates the information gain of a proposed split.

    _split(X_column, split_threshold)
        Splits the data into two groups based on the threshold value for a feature.

    _entropy(y)
        Calculates the entropy for the provided labels.

    _most_common_label(y)
        Determines the most common class label in the provided labels.

    _traverse_tree(x, node)
        Traverses the decision tree to make a prediction for a single sample x.
    """

    def __init__(self, min_samples_split=2, max_depth=50, number_of_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.number_of_features = number_of_features
        self.root = None

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.
        """
        self.number_of_features = (
            X.shape[1] if not self.number_of_features else min(X.shape[1], self.number_of_features)
        )
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively builds the decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        depth : int, default=0
            Current depth of the tree during construction.

        Returns
        -------
        Node
            The root node of the subtree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find best split
        feature_idx = np.random.choice(n_feats, self.number_of_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_idx)

        # create child nodes
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_idx):
        """
        Finds the best split for the data based on information gain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        y : array-like of shape (n_samples,)
            Target vector.

        feature_idx : array-like of shape (n_features,)
            Indices of features to consider for splitting.

        Returns
        -------
        tuple
            The index of the best feature and the best threshold value for the split.
        """
        best_info_gain = -1
        split_idx, split_threshold = None, None
        for i in feature_idx:
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                # calcualte the information gain
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_info_gain:
                    best_info_gain = gain
                    split_idx = i
                    split_threshold = threshold

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """
        Calculates the information gain of a proposed split.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target vector.

        X_column : array-like of shape (n_samples,)
            The feature column used for splitting.

        threshold : float
            The threshold value for the split.

        Returns
        -------
        float
            The information gain of the split.
        """

        # parent entropy
        parent_entropy = self._entropy(y)

        # creat children
        left_idx, right_idx = self._split(X_column, threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate the weighted avg. entroy of children
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        entropy_left, entropy_right = (
            self._entropy(y[left_idx]),
            self._entropy(y[right_idx]),
        )
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # calcualte IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        """
        Splits the data into two groups based on the threshold value for a feature.

        Parameters
        ----------
        X_column : array-like of shape (n_samples,)
            The feature column to split.

        split_threshold : float
            The threshold value for the split.

        Returns
        -------
        tuple
            The indices of samples in the left and right groups after the split.
        """

        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculates the entropy for the provided labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        float
            The calculated entropy.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        Determines the most common class label in the provided labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target vector.

        Returns
        -------
        int
            The most common label.
        """

        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predicts the class labels for the input feature matrix X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which predictions are to be made.

        Returns
        -------
        numpy.ndarray
            Predicted class labels for each sample in X.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverses the decision tree to make a prediction for a single sample x.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            A single input sample.

        node : Node
            The current node in the decision tree.

        Returns
        -------
        int or float
            The predicted class label or value for the input sample x.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node=None, depth=0):
        """Prints the structure of the decision tree.

        Parameters
        ----------
        node : Node, optional
            The current node in the decision tree. If None, starts from the root.

        depth : int, default=0
            The current depth in the tree used for indentation in output.
        """
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{'|   ' * depth}-> Leaf Node: Predict {node.value}")
        else:
            print(f"{'|   ' * depth}-> Node: Feature {node.feature} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            print(f"{'|   ' * depth}-> Node: Feature {node.feature} > {node.threshold}")
            self.print_tree(node.right, depth + 1)
