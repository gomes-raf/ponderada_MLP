# implementação do MLP

import numpy as np

from mlp.activations import relu, relu_derivative, softmax
from mlp.losses import cross_entropy
from mlp.optimizers import SGD, Momentum


class MLP:
    def __init__(self, layer_sizes, learning_rate=0.01, optimizer="sgd", seed=42):
        np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.params = {}
        self.cache = {}
        self.grads = {}

        if optimizer.lower() == "momentum":
            self.optimizer = Momentum(
                learning_rate=learning_rate,
                beta=0.9
            )
        else:
            self.optimizer = SGD(
                learning_rate=learning_rate
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for i in range(1, len(self.layer_sizes)):
            input_size = self.layer_sizes[i - 1]
            output_size = self.layer_sizes[i]

            self.params[f"W{i}"] = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
            self.params[f"b{i}"] = np.zeros((1, output_size))

    def forward(self, X):
        self.cache["A0"] = X

        num_layers = len(self.layer_sizes) - 1

        for i in range(1, num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            Z = self.cache[f"A{i-1}"] @ W + b
            A = relu(Z)

            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = A

        W = self.params[f"W{num_layers}"]
        b = self.params[f"b{num_layers}"]

        Z = self.cache[f"A{num_layers-1}"] @ W + b
        A = softmax(Z)

        self.cache[f"Z{num_layers}"] = Z
        self.cache[f"A{num_layers}"] = A

        return A

    def backward(self, y_true):
        m = y_true.shape[0]
        num_layers = len(self.layer_sizes) - 1

        y_pred = self.cache[f"A{num_layers}"]

        dZ = y_pred - y_true

        for i in reversed(range(1, num_layers + 1)):
            A_prev = self.cache[f"A{i-1}"]
            W = self.params[f"W{i}"]

            self.grads[f"W{i}"] = (A_prev.T @ dZ) / m
            self.grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 1:
                dA_prev = dZ @ W.T
                dZ = dA_prev * relu_derivative(self.cache[f"Z{i-1}"])

    def update(self):
        self.optimizer.update(self.params, self.grads)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=64):
        history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            y_train = y_train[indices]

            epoch_loss = 0

            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size

                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred = self.forward(X_batch)
                loss = cross_entropy(y_batch, y_pred)

                self.backward(y_batch)
                self.update()

                epoch_loss += loss

            epoch_loss /= (X_train.shape[0] // batch_size)

            train_acc = self.evaluate(X_train, y_train)

            history["loss"].append(epoch_loss)
            history["accuracy"].append(train_acc)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)
                val_loss = cross_entropy(y_val, val_pred)
                val_acc = self.evaluate(X_val, y_val)

                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Acc: {train_acc:.4f}"
                )

        return history

    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        y_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_labels)