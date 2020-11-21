import matplotlib.pyplot as plt

from helper import regression_data


if __name__ == "__main__":
    x, y = regression_data()

    m = 2
    b = 5
    y_pred = [m * xi + b for xi in x]

    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()
