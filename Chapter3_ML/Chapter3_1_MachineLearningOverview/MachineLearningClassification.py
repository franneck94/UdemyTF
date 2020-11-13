import matplotlib.pyplot as plt
import numpy as np

from helper import classification_data


if __name__ == "__main__":
    x, y = classification_data()

    m = -4
    b = 7
    border = [m * xi + b for xi in x]

    colors = np.array(["red", "blue"])
    plt.scatter(x[:, 0], x[:, 1], color=colors[y[:]])
    plt.plot(x, border, color="black")
    plt.show()
