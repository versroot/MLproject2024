import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Node:
    """Represents a node in the decision tree.

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
        return self.value is not None


class DecisionTreeGini:
    """A decision tree classifier with information gain.
    Parameters
    ----------
    min_sample_split : int or float, default = 2
        The minimum number of samples required to split an internal node

    max_depth : int, default = 50
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    number_of_features : int,  default=None
        The number of features to consider when looking for the best split

    Attributes
    ----------
    root : Node
        The root node of the decision tree after training.

    Methods
    -------
    fit(X, y)
        Build the decision tree from the training set (X, y).

    _grow_tree(X, y, depth=0)
        Recursively grow the tree by finding the best split at each node.

    _best_split(X, y, feature_idx)
        Find the best split for the data based on Gini impurity.

    _gini_impurity(y) -> float
        Calculate the Gini impurity for a set of labels.

    _split(X_column, threshold) -> Tuple[np.ndarray, np.ndarray]
        Split the dataset based on the specified feature and threshold.

    _most_common_label(y)
        Determine the most common label in a set of labels.

    predict(X) -> np.ndarray
        Predict class labels for samples in X.

    _traverse_tree(x, node) -> int
        Traverse the decision tree to predict the label for a single sample.
    """

    def __init__(self, min_samples_split=2, max_depth=50, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray
            The training input samples of shape (n_samples, n_features).

        y : np.ndarray
            The target values (class labels) as a 1D array of shape (n_samples,).
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """Recursively grow the tree by finding the best split at each node.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).

        y : np.ndarray
            The target values as a 1D array of shape (n_samples,).

        depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the constructed tree or subtree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """Find the best split for the data based on Gini impurity.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).

        y : np.ndarray
            The target values as a 1D array of shape (n_samples,).

        feature_idx : array-like
            Indices of the features to consider for the split.

        Returns
        -------
        int
            The index of the best feature to split on.

        float
            The best threshold value to split the feature.
        """

        best_gini = 1  # Start with the maximum possible Gini impurity
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                gini = self._gini_split(y, left_idxs, right_idxs)

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _gini_split(self, y, left_idxs, right_idxs):
        """
        Calculate the Gini impurity for a specific split in the data.

        Parameters
        ----------
        y : np.ndarray
            The target labels as a 1D array of shape (n_samples,).

        left_idxs : np.ndarray or list
            The indices of the samples in the left split.

        right_idxs : np.ndarray or list
            The indices of the samples in the right split.

        Returns
        -------
        float
            The Gini impurity of the split, representing the weighted average
            impurity of the left and right child nodes.

        Notes
        -----
        The Gini impurity is calculated by taking the weighted average of the
        impurity in the left and right child nodes based on the proportions
        of samples in each child node. A lower Gini impurity indicates a
        "purer" split.

        This method is called within the decision tree training process to
        evaluate potential splits and select the one that minimizes impurity.

        Examples
        --------
        >>> y = np.array([0, 1, 0, 1, 1])
        >>> left_idxs = [0, 2]
        >>> right_idxs = [1, 3, 4]
        >>> tree = DecisionTreeGini()
        >>> tree._gini_split(y, left_idxs, right_idxs)
        0.48
        """

        n = len(y)
        n_left, n_right = len(left_idxs), len(right_idxs)

        if n == 0:
            return 0  # No split if no data
        gini_left = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])
        gini_split = (n_left / n) * gini_left + (n_right / n) * gini_right

        return gini_split

    def _gini(self, y):
        """Calculate the Gini impurity for a set of labels.

        Parameters
        ----------
        y : np.ndarray
            Array of class labels.

        Returns
        -------
        float
            The Gini impurity of the labels.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps**2)

    def _split(self, X_column, split_thresh):
        """Split the dataset based on the specified feature and threshold.

        Parameters
        ----------
        X_column : np.ndarray
            Array of feature values.

        threshold : float
            The threshold value to split the feature.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Indices of the left and right splits.
        """
        left_idxs = np.argwhere(X_column < split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        """Determine the most common label in a set of labels.

        Parameters
        ----------
        y : np.ndarray
            Array of class labels.

        Returns
        -------
        int
            The most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            The input samples of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Array of predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse the decision tree to predict the label for a single sample.

        Parameters
        ----------
        x : np.ndarray
            A single input sample.

        node : Node
            The current node in the tree.

        Returns
        -------
        int
            The predicted class label.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
