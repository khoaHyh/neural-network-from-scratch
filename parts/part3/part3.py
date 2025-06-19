import numpy as np
import numpy.typing as npt

"""
## Part 3 TODO List

### 1. **Update Data Generation**
- [x] Modify data generation to create 2D datasets (multiple input features)
- [x] Generate batch data with shape `(batch_size, num_features)`
- [x] Ensure train/test split can handle matrix data

### 2. **Design Network Architecture**
- [ ] Define network structure: input_size → hidden_size → output_size
- [ ] Plan weight matrix dimensions (hint: `(input_size, hidden_size)` for first layer)
- [ ] Plan bias vector dimensions

### 3. **Initialize Parameters with Matrices**
- [ ] Replace scalar weights with weight matrices
- [ ] Replace scalar biases with bias vectors
- [ ] Use proper weight initialization (try `np.random.randn() * 0.1`)

### 4. **Implement Matrix Forward Pass**
- [ ] Rewrite `forward_pass()` to use matrix multiplication (`@` operator)
- [ ] Ensure ReLU works with matrices (should work automatically with numpy)
- [ ] Test that output shapes are correct

### 5. **Update Loss Function**
- [ ] Modify `loss_calculation()` to handle batch predictions
- [ ] Ensure MSE works across all examples in batch

### 6. **Implement Matrix Backpropagation**
- [ ] Rewrite gradient calculations using matrix operations
- [ ] Update `calculate_gradients()` for matrix weights/biases
- [ ] Verify gradient shapes match parameter shapes

### 7. **Create Neural Network Class**
- [ ] Wrap everything in a `NeuralNetwork` class
- [ ] Add methods: `__init__()`, `forward()`, `backward()`, `train()`
- [ ] Make layer sizes configurable parameters

### 8. **Test and Validate**
- [ ] Compare single-example results between Part 2 and Part 3
- [ ] Test with different batch sizes
- [ ] Verify training actually reduces loss
"""

# 1. Generate sample data
np.random.seed(42)  # For reproducibility

n_features = 3
n_samples = 100
noise_level = 3
coef = np.array([2.5, -1.5, -0.8])
true_intercept = 5

# Generate feature matrix
# Coming from a 1D array where we had 100 numbers (e.g., 100 heights)
# Now we have 100 examples where each example has 3 numbers aka features (e.g., height, weight, age)
X = np.random.uniform(0, 10, (n_samples, n_features))

# Generate y values with some noise to mimic "real world" phenomenon
y = (
    np.sum(coef * X, axis=1)
    + true_intercept
    + np.random.normal(0, noise_level, n_samples)
)

# Split into training and testing sets (80% train, 20% test)
split_idx = int(0.8 * n_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 2. Initialize parameters. In training, these random values will be adjusted through gradient descent
weight, weight2 = np.random.randn() * 0.5, np.random.randn() * 0.5  # coefficient/slope
bias, bias2 = 0.1, np.random.randn() * 0.1  # intercept

# NOTE: how would we organize weights in a matrix?
# Rows could represent input features (3 rows)
# Columns could represent hidden neurons (5 columns)
# We go FROM 3 inputs TO 5 hidden neurons
# So we'd have a (3, 5) matrix


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
