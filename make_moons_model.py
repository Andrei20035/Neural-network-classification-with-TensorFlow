# make_moons_model.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray) -> None:
    """Plot decision boundary for a 2D binary classifier."""
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    x_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(x_in, verbose=0)

    # Binary classification -> round predictions to {0,1}
    y_pred_grid = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    plt.contourf(xx, yy, y_pred_grid, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary (Make Moons)")
    plt.xlabel("feature_1")
    plt.ylabel("feature_2")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(6, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    tf.random.set_seed(42)
    np.random.seed(42)

    # Data
    features, labels = make_moons(n_samples=8000, random_state=42)
    df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
    df["labels"] = labels

    # Simple scaling from your notebook
    df["feature_1"] = df["feature_1"] / 2.0

    X = df[["feature_1", "feature_2"]]
    y = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = build_model()
    model.fit(X_train, y_train, epochs=10, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Make Moons - test_loss={loss:.6f}  test_accuracy={acc:.4f}")

    # Save decision boundary plot
    os.makedirs("images", exist_ok=True)
    plt.figure(figsize=(7, 5))
    plot_decision_boundary(model, X.values, y.values)
    plt.tight_layout()
    plt.savefig("images/decision_boundary.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
