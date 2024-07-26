from typing import Tuple

import numpy as np

from sklearn.tree import DecisionTreeRegressor


# from metrics import mse, mae


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean squared error loss function and gradient."""
    # YOUR CODE HERE
    loss = np.mean((y_true - y_pred) ** 2)
    grad = y_pred - y_true
    return loss, grad


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
    """Mean absolute error loss function and gradient."""

    # YOUR CODE HERE
    def diff(x: float):
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    loss = np.mean(np.abs(y_true - y_pred))
    residuals = y_pred - y_true
    grad = np.vectorize(diff)(residuals)
    return loss, grad


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            subsample_size=1.0,
            replace=True,
            verbose=False,
    ):
        self.trees_ = [DecisionTreeRegressor(max_depth=max_depth,
                                             min_samples_split=min_samples_split) for _ in range(n_estimators)]
        self.learning_rate = learning_rate
        self.subsample_size = subsample_size
        self.replace = replace
        if loss == "mse":
            self.objective = mse
        elif loss == "mae":
            self.objective = mae
        elif callable(loss):
            self.objective = loss
        else:
            raise ValueError("Incorrect loss function name. Excepted: 'mse', 'mae'")

    def _mse(self, y_true, y_pred):
        return self.objective(y_true, y_pred)

    def _subsample(self, X, y):
        max_rows = X.shape[0]
        idx = np.random.choice(max_rows, size=round(self.subsample_size * max_rows), replace=self.replace)
        sub_X, sub_y = X[idx], y[idx]
        return sub_X, idx

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y)
        y_pred = np.array([self.base_pred_] * len(X))

        for tree in self.trees_:
            sub_X, idx = self._subsample(X, y)
            _, grad = self._mse(y, y_pred)
            tree.fit(sub_X, -grad[idx])
            grad_pred = tree.predict(X)
            y_pred += self.learning_rate * grad_pred

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """
        predictions = np.array([self.base_pred_] * len(X))

        for tree in self.trees_:
            grad_pred = tree.predict(X)
            predictions += self.learning_rate * grad_pred
        return predictions
