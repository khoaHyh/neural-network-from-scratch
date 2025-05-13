# Import necessary libraries
import numpy as np
import numpy.typing as npt
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
def forward_pass(
    x: npt.NDArray[np.float64], weight: float, bias: float
) -> npt.NDArray[np.float64]:
    return x * weight + bias


def loss_calculation(
    y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
) -> np.floating:
    return np.mean((y_pred - y_true) ** 2)


def parameter_update(
    w: float,
    b: float,
    x: np.ndarray,
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    learning_rate: float,
) -> tuple[float, float]:
    num_inputs = x.size
    derivative_weight = -2 / num_inputs * np.sum(x * (y - y_pred))
    derivative_bias = -2 / num_inputs * np.sum(y - y_pred)

    weight = w - learning_rate * derivative_weight
    bias = b - learning_rate * derivative_bias

    return (weight, bias)


# 4. Training loop
# Hyperparameters
learning_rate = 0.01
iterations = 1000

loss_history = []

for i in range(iterations):
    y_pred = forward_pass(X_train, weight, bias)
    loss = loss_calculation(y_pred, y_train)
    loss_history.append(loss)

    weight, bias = parameter_update(
        weight, bias, X_train, y_train, y_pred, learning_rate
    )

    if (i + 1) % 100 == 0:
        print(
            f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}, Weight: {weight:.4f}, Bias: {bias:.4f}"
        )

print(f"Final parameters: Weight = {weight:.4f}, Bias = {bias:.4f}")
print(f"True parameters: Weight = {true_slope:.4f}, Bias = {true_intercept:.4f}")

# 5. Visualization
# NOTE: This part is done in the corresponding jupyter notebook in the current directory.
