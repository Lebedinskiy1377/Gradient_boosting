from typing import Tuple
import numpy as np


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
