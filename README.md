Gradient Boosting
A Python library implementing a custom gradient boosting regressor for regression tasks, supporting Mean Squared Error (MSE) and Mean Absolute Error (MAE) loss functions with decision trees as base learners.
Features

Custom Gradient Boosting: Implements gradient boosting with decision trees for regression tasks.
Flexible Loss Functions: Supports MSE and MAE loss functions, with the option to use custom loss functions via callable objects.
Stochastic Gradient Boosting: Includes subsampling of data for training efficiency, with configurable subsample size and replacement.
Base Predictor: Provides a simple mean-based predictor as a baseline (base_estimator.py).

Tech Stack

Python 3.9+
numpy
pandas
scikit-learn

Installation

Clone the repository:git clone https://github.com/Lebedinskiy1377/Gradient_boosting.git
cd Gradient_boosting


Install dependencies:pip install -r requirements.txt



Usage
Training a Gradient Boosting Regressor
import numpy as np
from src.gradient_boosting_regressor import GradientBoostingRegressor

# Sample data
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = np.random.rand(100)     # Target values

# Initialize and fit the model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    loss="mse",
    subsample_size=0.8,
    replace=True
)
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(predictions)

Using the Base Predictor
from src.base_estimator import GradientBoostingRegressor

# Sample data
X = np.random.rand(100, 5)
y = np.random.rand(100)

# Initialize and fit the base predictor
base_model = GradientBoostingRegressor()
base_model.fit(X, y)

# Predict (returns mean of training targets)
base_predictions = base_model.predict(X)
print(base_predictions)

Example Output
For gradient boosting predictions:
array([0.5123, 0.4987, 0.5342, ...])  # Predicted values

For base predictor:
array([0.5000, 0.5000, 0.5000, ...])  # Mean of training targets

Requirements
Install dependencies using:
pip install numpy pandas scikit-learn

Notes

The GradientBoostingRegressor in gradient_boosting_regressor.py supports MSE and MAE loss functions, with gradients computed for optimization.
The base_estimator.py provides a simple baseline predictor that returns the mean of the target values.
Subsampling in GradientBoostingRegressor enables stochastic gradient boosting for improved training efficiency.
The implementation is designed for regression tasks but can be extended for other applications by modifying the loss function.
