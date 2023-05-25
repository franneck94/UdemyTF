import numpy as np


def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    """OR dataset."""
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [1]])
    return x, y


def to_categorical(y: np.ndarray, num_classes: int) -> np.ndarray:
    y_categorical = np.zeros(shape=(len(y), num_classes))
    for i, yi in enumerate(y):
        y_categorical[i, yi] = 1
    return y_categorical


def softmax(y_pred: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(y_pred)
    for i in range(len(y_pred)):
        exps = np.exp(y_pred[i])
        probabilities[i] = exps / np.sum(exps)
    return probabilities


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num_samples = y_true.shape[0]
    loss = float(-np.sum(y_true * np.log(y_pred)) / num_samples)
    return loss


if __name__ == "__main__":
    x, y = get_dataset()
    print(y.shape)
    print(y)

    y_categorical = to_categorical(y, num_classes=2)
    print(y_categorical.shape)
    print(y_categorical)

    y_logits = np.array([[10.5, -2.3], [1.5, 3.3], [-22.5, 22.3], [1.5, 222.3]])
    y_pred = softmax(y_logits)
    print(y_pred.shape)
    print(y_pred)
    print(np.sum(y_pred, axis=1))

    loss = cross_entropy(y_categorical, y_pred)
    print(loss)
