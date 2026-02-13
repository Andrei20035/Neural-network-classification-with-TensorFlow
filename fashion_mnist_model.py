# fashion_mnist_model.py
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(28, activation="relu"),
            tf.keras.layers.Dense(23, activation="relu"),
            tf.keras.layers.Dense(23, activation="relu"),
            tf.keras.layers.Dense(18, activation="relu"),
            tf.keras.layers.Dense(18, activation="relu"),
            tf.keras.layers.Dense(15, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["accuracy"],
    )
    return model


def plot_confusion_matrix(cm: np.ndarray, cm_norm: np.ndarray) -> None:
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))

    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    labels = np.arange(n_classes)
    ax.set(
        title="Confusion Matrix (Fashion-MNIST)",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm.max() + cm.min()) / 2.0

    for i, j in itertools.product(range(n_classes), range(n_classes)):
        ax.text(
            j,
            i,
            f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=6,
        )

    plt.tight_layout()


def main() -> None:
    tf.random.set_seed(77)
    np.random.seed(77)

    # Data
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # One-hot labels
    train_labels_oh = tf.one_hot(train_labels, depth=10)
    test_labels_oh = tf.one_hot(test_labels, depth=10)

    # Model
    model = build_model()

    # Learning rate schedule (from your notebook)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 0.018 * 10 ** (epoch / 20)
    )

    model.fit(
        train_data,
        train_labels_oh,
        epochs=18,
        validation_data=(train_data, train_labels_oh),
        callbacks=[lr_scheduler],
        verbose=1,
    )

    loss, acc = model.evaluate(test_data, test_labels_oh, verbose=0)
    print(f"Fashion-MNIST - test_loss={loss:.4f}  test_accuracy={acc:.4f}")

    # Confusion matrix (use integer labels)
    pred_probs = model.predict(test_data, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)

    cm = confusion_matrix(test_labels, pred_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    os.makedirs("images", exist_ok=True)
    plot_confusion_matrix(cm, cm_norm)
    plt.savefig("images/confusion_matrix.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
