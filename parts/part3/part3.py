import numpy as np
import numpy.typing as npt

# 1. Generate sample data
np.random.seed(42)  # For reproducibility

n_features = 3
n_hidden_neurons = 5
n_samples = 100
noise_level = 3
coef = np.array([2.5, -1.5, -0.8])
true_intercept = 5

# Generate feature matrix
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

print(f"Data shapes: X_train {X_train.shape}, y_train {y_train.shape}")


# 2. Define model functions
def ReLU(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.maximum(0, value)


def ReLU_derivative(value: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (value > 0).astype(float)


# 3. Neural Network Class
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize a simple 2-layer neural network

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize parameters with proper matrix dimensions
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

        print(f"Initialized network: {input_size} -> {hidden_size} -> {output_size}")
        print(
            f"Parameter shapes: W1{self.W1.shape}, b1{self.b1.shape}, W2{self.W2.shape}, b2{self.b2.shape}"
        )

    def forward(
        self, X: npt.NDArray[np.float64]
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Forward pass through the network

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            hidden_raw: Raw hidden layer values (before activation)
            hidden_activated: Hidden layer values after ReLU
            output: Final network predictions
        """
        # Hidden layer
        hidden_raw = X @ self.W1 + self.b1  # Shape: (batch_size, hidden_size)
        hidden_activated = ReLU(hidden_raw)  # Shape: (batch_size, hidden_size)

        # Output layer
        output = (
            hidden_activated @ self.W2 + self.b2
        ).flatten()  # Shape: (batch_size,)

        return hidden_raw, hidden_activated, output

    def loss(
        self, y_pred: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> np.floating:
        return np.mean((y_pred - y_true) ** 2)

    def backward(
        self,
        X: npt.NDArray[np.float64],
        y_true: npt.NDArray[np.float64],
        y_pred: npt.NDArray[np.float64],
        hidden_raw: npt.NDArray[np.float64],
        hidden_activated: npt.NDArray[np.float64],
    ) -> dict[str, npt.NDArray[np.float64]]:
        batch_size = X.shape[0]

        loss_wrt_predictions = (
            -2 * (y_true - y_pred) / batch_size
        )  # Shape: (batch_size,)

        # Gradient for W2 (output weights)
        W2_gradient = hidden_activated.T @ loss_wrt_predictions.reshape(
            -1, 1
        )  # Shape: (hidden_size, output_size)

        # Gradient for b2 (output bias)
        b2_gradient = np.sum(loss_wrt_predictions)  # Scalar -> reshape
        b2_gradient = np.array([b2_gradient])  # Shape: (output_size,)

        # How much does loss change with respect to hidden activations?
        loss_wrt_hidden_activated = (
            loss_wrt_predictions.reshape(-1, 1) @ self.W2.T
        )  # Shape: (batch_size, hidden_size)

        hidden_activation_derivative = ReLU_derivative(
            hidden_raw
        )  # Shape: (batch_size, hidden_size)
        loss_wrt_hidden_raw = (
            loss_wrt_hidden_activated * hidden_activation_derivative
        )  # Shape: (batch_size, hidden_size)

        # Gradient for W1 (hidden weights)
        W1_gradient = X.T @ loss_wrt_hidden_raw  # Shape: (input_size, hidden_size)

        # Gradient for b1 (hidden bias)
        b1_gradient = np.sum(loss_wrt_hidden_raw, axis=0)  # Shape: (hidden_size,)

        return {
            "W1": W1_gradient,
            "b1": b1_gradient,
            "W2": W2_gradient,
            "b2": b2_gradient,
        }

    def update_parameters(
        self, gradients: dict[str, npt.NDArray[np.float64]], learning_rate: float
    ):
        """Update all parameters using gradient descent"""
        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

    def train(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        learning_rate: float = 0.01,
        epochs: int = 1000,
        verbose: bool = True,
    ):
        """
        Train the neural network

        Args:
            X: Training data
            y: Training labels
            learning_rate: Step size for gradient descent
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        loss_history = []

        for epoch in range(epochs):
            # Forward pass
            hidden_raw, hidden_activated, y_pred = self.forward(X)

            # Calculate loss
            current_loss = self.loss(y_pred, y)
            loss_history.append(current_loss)

            # Backward pass
            gradients = self.backward(X, y, y_pred, hidden_raw, hidden_activated)

            # Update parameters
            self.update_parameters(gradients, learning_rate)

            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.6f}")

        return loss_history

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Make predictions on new data"""
        _, _, predictions = self.forward(X)
        return predictions


# 4. Create and train the network
print("Creating neural network...")
nn = NeuralNetwork(input_size=n_features, hidden_size=n_hidden_neurons, output_size=1)

print("\nTraining network...")
loss_history = nn.train(X_train, y_train, learning_rate=0.01, epochs=1000)

print(f"\nFinal training loss: {loss_history[-1]:.6f}")

# 5. Test the network
print("\nTesting network...")
train_predictions = nn.predict(X_train)
test_predictions = nn.predict(X_test)

train_loss = nn.loss(train_predictions, y_train)
test_loss = nn.loss(test_predictions, y_test)

print(f"Training loss: {train_loss:.6f}")
print(f"Test loss: {test_loss:.6f}")

# 6. Show some example predictions
print("\nExample predictions vs actual (first 5 test examples):")
for i in range(5):
    print(f"Predicted: {test_predictions[i]:.2f}, Actual: {y_test[i]:.2f}")

print(f"\nNetwork architecture summary:")
print(f"Input features: {nn.input_size}")
print(f"Hidden neurons: {nn.hidden_size}")
print(f"Output neurons: {nn.output_size}")
print(f"Total parameters: {nn.W1.size + nn.b1.size + nn.W2.size + nn.b2.size}")
