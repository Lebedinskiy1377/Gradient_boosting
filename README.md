# Gradient Boosting from Scratch

A minimal implementation of **Gradient Boosting for Regression** using pure Python, NumPy, and `sklearn.tree.DecisionTreeRegressor`.  
This project supports both **MSE** and **MAE** loss functions, as well as **stochastic gradient boosting** via subsampling.

---

## Overview

This project implements classic gradient boosting:

- Ensemble of weak learners (decision trees)
- Manual gradient computation for custom loss functions
- Optional subsampling (stochastic gradient boosting)
- sklearn-like interface: `fit()` and `predict()`

---

## Features

- Loss functions: `MSE`, `MAE`, or custom callable
- Base learner: `DecisionTreeRegressor`
- Configurable hyperparameters:
  - `n_estimators`
  - `learning_rate`
  - `max_depth`
  - `min_samples_split`
  - `subsample_size`
  - `replace`

---

## Example usage

```python
from src.gradient_boosting_regressor import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample regression dataset
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train custom Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=3, loss="mse"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
```
