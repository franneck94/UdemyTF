import numpy as np


def generate_data_or():
    x = [[0, 0], [1, 1], [1, 0], [0, 1]]
    y = [0, 1, 1, 1]
    return x, y


def to_one_hot(y, num_classes):
    y_one_hot = np.zeros(shape=(len(y), num_classes))  # 4x2
    for i, y_i in enumerate(y):
        y_oh = np.zeros(shape=num_classes)
        y_oh[y_i] = 1
        y_one_hot[i] = y_oh
    return y_one_hot


def softmax(y_pred):
    y_softmax = np.zeros(shape=y_pred.shape)
    for i in range(len(y_pred)):
        exps = np.exp(y_pred[i])
        y_softmax[i] = exps / np.sum(exps)
    return y_softmax


if __name__ == "__main__":
    x, y = generate_data_or()
    y = to_one_hot(y, num_classes=2)
    print(y)

    p1 = [0.223, 0.613]
    p2 = [-0.75, 0.5]
    p3 = [0.01, 0.2]
    p4 = [0.564, 0.234]
    y_pred = np.array([p1, p2, p3, p4])

    print(y_pred)
    y_pred = softmax(y_pred)
    print(y_pred)
