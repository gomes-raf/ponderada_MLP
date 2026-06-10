import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlp.network import MLP


def one_hot(y, num_classes=10):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return X_train, y_train, X_test, y_test


def plot_history(history):
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Curva de Loss")
    plt.legend()
    plt.savefig("results/loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Curva de Acurácia")
    plt.legend()
    plt.savefig("results/accuracy_curve.png")
    plt.close()


def plot_confusion_matrix(model, X_test, y_test):
    os.makedirs("results", exist_ok=True)

    y_true = np.argmax(y_test, axis=1)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Matriz de Confusão")
    plt.savefig("results/confusion_matrix.png")
    plt.close()


def main():
    X_train, y_train, X_test, y_test = load_data()

    model = MLP(
        layer_sizes=[784, 64, 10],
        learning_rate=0.01,
        seed=42
    )

    history = model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=20,
        batch_size=64
    )

    test_accuracy = model.evaluate(X_test, y_test)

    print(f"\nAcurácia final no teste: {test_accuracy:.4f}")

    plot_history(history)
    plot_confusion_matrix(model, X_test, y_test)


if __name__ == "__main__":
    main()