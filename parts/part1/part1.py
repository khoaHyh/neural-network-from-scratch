# Import necessary libraries
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
# TODO: Implement forward pass function
# TODO: Implement loss calculation function
# TODO: Implement parameter update function

# 4. Training loop
# TODO: Implement iterations of forward pass, loss calculation, and parameter updates

# 5. Visualization
# TODO: Plot original data points and final regression line
# TODO: Plot loss over iterations
