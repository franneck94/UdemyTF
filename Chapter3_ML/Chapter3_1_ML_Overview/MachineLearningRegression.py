import matplotlib.pyplot as plt

from tf_utils.dummyData import regression_data


def model(x):
    m = 2.0  # slope
    b = 5.0  # intercept

    return m * x + b


if __name__ == "__main__":
    x, y = regression_data()

    y_pred = model(x)

    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.show()
