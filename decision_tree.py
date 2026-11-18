import numpy as np

from tree import FEATURE, LABEL, LEFT, RIGHT, THRESHOLD, compact, is_leaf, make_leaf


def entropy(y):
    """ENTROPY. y is an array of 0s and 1s."""
   
    y = np.array(y)
    if len(y) == 0:
        return 0.0
    p = float(np.mean(y == 1))
    if p == 0.0 or p == 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def information_gain(y, y_left, y_right):
    """INFORMATION-GAIN."""

    n = len(y)
    if n == 0:
        return 0.0
    return float(entropy(y)
                - (len(y_left) / n) * entropy(y_left)
                - (len(y_right) / n) * entropy(y_right)
                )


def majority_label(y):
    """MAJORITY-LABEL. A tie goes to 0."""

    y = np.array(y)
    ones = int(np.sum(y == 1))
    nones = len(y) - ones
    if ones > nones:
        return 1.0
    return 0.0


def best_split(X, y, min_samples_leaf=1):
    """BEST-SPLIT. Returns (feature, threshold, gain), or None."""

    X = np.array(X)
    y = np.array(y)
    if len(y) == 0:
        return None

    best = None
    best_gain = 0.0

    for f in range(X.shape[1]):
        t = float(np.median(X[:, f]))
        left = X[:, f] <= t
        y_left = y[left]
        y_right = y[~left]
        if len(y_left) == 0 or len(y_right) == 0:
            continue
        if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
            continue
        g = information_gain(y, y_left, y_right)
        if g > best_gain:
            best = (f, t, g)
            best_gain = g
    return best


def build_tree(X, y, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_gain=0.0):
    """BUILD-TREE. Returns the tree as a NumPy array, in preorder."""

    X = np.array(X, dtype=float)
    y = np.array(y)
    nodes = []

    def grow(X, y, depth):
        at = len(nodes)
        nodes.append(make_leaf(majority_label(y)))

        if len(np.unique(y)) <= 1:
            return at
        if max_depth is not None and depth >= max_depth:
            return at
        if len(y) < min_samples_split:
            return at

        split = best_split(X, y, min_samples_leaf)
        if split is None or split[2] < min_gain:
            return at

        f, t, gain = split
        left = X[:, f] <= t
        nodes[at][FEATURE] = f
        nodes[at][THRESHOLD] = t
        nodes[at][LEFT] = grow(X[left], y[left], depth + 1)
        nodes[at][RIGHT] = grow(X[~left], y[~left], depth + 1)
        return at

    grow(X, y, 0)
    return np.array(nodes, dtype=float)


def predict_one(tree, row, i=0):
    """PREDICT-ONE. Starts at node i."""

    while not is_leaf(tree, i):
        f = int(tree[i, FEATURE])
        if row[f] <= tree[i, THRESHOLD]:
            i = int(tree[i, LEFT])
        else:
            i = int(tree[i, RIGHT])
    return int(tree[i, LABEL])


def predict(tree, X):
    """PREDICT. Returns an integer array with one label per row of X."""

    labels = []
    for row in np.array(X, dtype=float):
        labels.append(predict_one(tree, row, 0))
    return np.array(labels, dtype=int)


def f1_score(y_true, y_pred):
    """F1-SCORE, with 1 (gamma) as the positive class."""
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true != 1) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred != 1)))
    if TP == 0:
        return 0.0
    return 2 * TP / (2 * TP + FP + FN)


def prune(tree, X, y):
    """PRUNE. Returns a new, smaller tree; the tree passed in is left unchanged."""

    copy = np.array(tree, dtype=float, copy=True)
    X = np.array(X, dtype=float)
    y = np.array(y)

    def visit(i, X, y):
        if is_leaf(copy, i):
            return

        f = int(copy[i, FEATURE])
        left = X[:, f] <= copy[i, THRESHOLD]
        visit(int(copy[i, LEFT]), X[left], y[left])
        visit(int(copy[i, RIGHT]), X[~left], y[~left])

        subtree_errors = 0
        for row, label in zip(X, y):
            if predict_one(copy, row, i) != label:
                subtree_errors += 1
        leaf_errors = int(np.sum(y != copy[i, LABEL]))

        if leaf_errors <= subtree_errors:
            copy[i] = make_leaf(copy[i, LABEL])

    visit(0, X, y)
    return compact(copy)


# The hyperparameters used for the final model. These placeholder values are
# deliberately very poor. Replace all of them with values you have tuned.
HYPERPARAMETERS = {
    "max_depth": 12,
    "min_samples_split": 1000,
    "min_samples_leaf": 2000,
    "min_gain": 0.3,
}


def train_final_model(X, y):
    """Return the tree to be scored on the hidden test set."""
    return build_tree(X, y, **HYPERPARAMETERS)
