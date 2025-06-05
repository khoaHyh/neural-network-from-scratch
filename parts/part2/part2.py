import numpy as np
import numpy.typing as npt

"""
btw, "wrt" stands for "with respect to"

[network architecture]
- 2 layers (1 hidden + 1 ouput)
- 1 neuron per layer (single weight/bias for each)
- parameter init with `np.random.randn()`

[Objectives]
[x] Implement ReLU activation function
[x] Generate non-linear data
[x] Define network architecture (# of layres, # of neurons, init weights)
[x] Extend forward pass to handle multiple layers
[x] Implement Backward pass
[x] Update training loop
7. Visualization and Experiement with Hyperparameters
NOTE: Part 7 is done in the corresponding jupyter notebook in the current directory.
"""

# 1. Generate sample data
np.random.seed(42)  # For reproducibility

n_samples = 100
noise_level = 3

# Generate x values
X = np.random.uniform(0, 10, n_samples)

# Generate y values with some noise to mimic "real world" phenomenon
# Non-linear relationship: quadratic function
y = 0.5 * X**2 - 2 * X + 5 + np.random.normal(0, noise_level, n_samples)

# Split into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Initialize parameters. In training, these random values will be adjusted through gradient descent
weight, weight2 = np.random.randn(), np.random.randn()  # coefficient/slope
bias, bias2 = np.random.randn(), np.random.randn()  # intercept


# 3. Define model functions
def ReLU(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, value)


def ReLU_derivative(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (value > 0).astype(float)


def forward_pass(
    x: npt.NDArray[np.float64], w: float, b: float, w2: float, b2: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    hidden_layer_raw_output = x * w + b
    hidden_layer_activated_output = ReLU(hidden_layer_raw_output)
    final_predictions = hidden_layer_activated_output * w2 + b2
    # return intermediate valeus for backprop
    return hidden_layer_raw_output, hidden_layer_activated_output, final_predictions


def loss_calculation(
    y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
) -> np.floating:
    return np.mean((y_pred - y_true) ** 2)


def calculate_gradients(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    hidden_layer_raw: npt.NDArray[np.float64],
    hidden_layer_activated: npt.NDArray[np.float64],
    output_weight: float,
) -> dict[str, float]:
    # Calculate all gradients via backpropagation
    num_inputs = x.size

    # Output layer gradients (calculated direct from loss func)
    output_weight_gradient = (
        -2 / num_inputs * np.sum(hidden_layer_activated * (y - y_pred))
    )
    output_bias_gradient = -2 / num_inputs * np.sum(y - y_pred)

    # Hidden layer gradients (using chain rule)
    loss_gradient_wrt_predictions = -2 * (y - y_pred) / num_inputs
    hidden_activation_gradient = ReLU_derivative(hidden_layer_raw)

    loss_gradient_wrt_hidden_activated = loss_gradient_wrt_predictions * output_weight
    loss_gradient_wrt_hidden_raw = (
        loss_gradient_wrt_hidden_activated * hidden_activation_gradient
    )

    hidden_weight_gradient = np.sum(loss_gradient_wrt_hidden_raw * x)
    hidden_bias_gradient = np.sum(loss_gradient_wrt_hidden_raw)

    return {
        "hidden_weight": hidden_weight_gradient,
        "hidden_bias": hidden_bias_gradient,
        "output_weight": output_weight_gradient,
        "output_bias": output_bias_gradient,
    }


def update_parameters(
    hidden_weight: float,
    hidden_bias: float,
    output_weight: float,
    output_bias: float,
    gradients: dict[str, float],
    learning_rate: float,
) -> tuple[float, float, float, float]:
    """Update all parameters using calculated gradients"""
    updated_hidden_weight = hidden_weight - learning_rate * gradients["hidden_weight"]
    updated_hidden_bias = hidden_bias - learning_rate * gradients["hidden_bias"]
    updated_output_weight = output_weight - learning_rate * gradients["output_weight"]
    updated_output_bias = output_bias - learning_rate * gradients["output_bias"]

    return (
        updated_hidden_weight,
        updated_hidden_bias,
        updated_output_weight,
        updated_output_bias,
    )


# 4. Training loop
# Hyperparameters
learning_rate = 0.01
iterations = 1000

loss_history = []

for i in range(iterations):
    hidden_layer_raw, hidden_layer_activated, y_pred = forward_pass(
        X_train, weight, bias, weight2, bias2
    )
    loss = loss_calculation(y_pred, y_train)
    loss_history.append(loss)

    gradients = calculate_gradients(
        X_train, y_train, y_pred, hidden_layer_raw, hidden_layer_activated, weight2
    )

    weight, bias, weight2, bias2 = update_parameters(
        weight, bias, weight2, bias2, gradients, learning_rate
    )

    if (i + 1) % 100 == 0:
        print(
            f"Iteration {i + 1}/{iterations}, Loss: {loss:.4f}, Weight: {weight:.4f}, Bias: {bias:.4f}"
        )

print(f"Final parameters: Weight = {weight:.4f}, Bias = {bias:.4f}")
print(f"Final output layer: Weight2 = {weight2:.4f}, Bias2 = {bias2:.4f}")
