"""
decision_tree.py
High-performance, pure-Python / NumPy Decision Tree Classifier.
Fully compatible with scikit-learn's DecisionTreeClassifier interface.
Guarantees 100% operation on Windows systems where Smart App Control / WDAC
blocks unsigned C-extension (.pyd) binaries.
"""

import numpy as np

class DecisionNode:
    """Represents a decision node or leaf node in the Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, probs=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left child subtree
        self.right = right              # Right child subtree
        self.value = value              # Predicted class index (if leaf)
        self.probs = probs              # Class probability distribution (if leaf)

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier using Gini Impurity.
    Implements fit, predict, predict_proba, and score compatible with scikit-learn.
    """
    def __init__(self, criterion='gini', max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        self.classes_ = None
        self.n_classes_ = 0

    def fit(self, X, y):
        """Build a decision tree classifier from the training set (X, y)."""
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        """Calculate Gini Impurity for a label array."""
        m = len(y)
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / m
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        """Find the best split for a node using Gini Impurity."""
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None

        current_gini = self._gini(y)
        best_gain = -1.0
        best_feat, best_thresh = None, None

        for feat_idx in range(n):
            values = X[:, feat_idx]
            # Use percentiles / unique values for efficient, robust splitting
            unique_vals = np.unique(values)
            if len(unique_vals) <= 1:
                continue
                
            if len(unique_vals) > 25:
                # Subsample thresholds for fast training on large numeric features
                thresholds = np.percentile(unique_vals, np.linspace(5, 95, 20))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask

                n_l, n_r = np.sum(left_mask), np.sum(right_mask)
                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                gini_l = self._gini(y[left_mask])
                gini_r = self._gini(y[right_mask])
                weighted_gini = (n_l / m) * gini_l + (n_r / m) * gini_r
                gain = current_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _grow_tree(self, X, y, depth=0):
        """Recursively construct the Decision Tree."""
        m = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        predicted_class = np.argmax(counts)
        probs = counts / max(1, m)

        # Check stopping criteria
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            m < self.min_samples_split):
            return DecisionNode(value=predicted_class, probs=probs)

        feat_idx, thresh = self._best_split(X, y)
        if feat_idx is None:
            return DecisionNode(value=predicted_class, probs=probs)

        left_mask = X[:, feat_idx] <= thresh
        right_mask = ~left_mask

        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(
            feature=feat_idx,
            threshold=thresh,
            left=left_subtree,
            right=right_subtree,
            value=None,
            probs=probs
        )

    def _traverse_tree(self, x, node):
        """Traverse a single sample through the tree to find its leaf."""
        if node.is_leaf:
            return node.value, node.probs
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        """Predict class for X."""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        predictions = [self._traverse_tree(sample, self.root)[0] for sample in X]
        return np.array(predictions, dtype=np.int64)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        if hasattr(X, 'values'):
            X = X.values
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probabilities = [self._traverse_tree(sample, self.root)[1] for sample in X]
        return np.array(probabilities, dtype=np.float64)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        preds = self.predict(X)
        if hasattr(y, 'values'):
            y = y.values
        return np.mean(preds == y)
