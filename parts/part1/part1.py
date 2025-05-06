# Import necessary libraries
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate sample data
np.random.seed(42)  # For reproducibility

# Parameters
n_samples = 100
true_slope = 2.5
true_intercept = 5
noise_level = 3

# Generate x values
X = np.random.uniform(0, 10, n_samples)

# Generate y values with some noise to mimic "real world" phenomenon
y = true_slope * X + true_intercept + np.random.normal(0, noise_level, n_samples)

# Split into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Initialize parameters. In training, these random values will be adjusted through gradient descent
weight = np.random.randn()  # coefficient/slope
bias = np.random.randn()  # intercept


# 3. Define model functions
def forward_pass(x: float, weight: float, bias: float) -> float:
    return x * weight + bias


def loss_calculation(y_pred: float, y_true: float) -> float:
    return np.mean((y_pred - y_true) ** 2)


def parameter_update(
    w: float, b: float, x: float, y: float, y_pred: float, learning_rate: float
) -> Tuple[int, int]:
    num_inputs = len(x)
    derivative_weight = -2 / num_inputs * np.sum(x * (y - y_pred))
    derivative_bias = -2 / num_inputs * np.sum(y - y_pred)

    weight = w - learning_rate * derivative_weight
    bias = b - learning_rate * derivative_bias

    return (weight, bias)


# 4. Training loop
# TODO: Implement iterations of forward pass, loss calculation, and parameter updates

# 5. Visualization
# TODO: Plot original data points and final regression line
# TODO: Plot loss over iterations
