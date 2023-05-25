import numpy as np


np.random.seed(0)


def f(x: np.ndarray) -> np.ndarray:
    return 2.0 * x + 5.0


def classification_data(N: int = 30) -> tuple[np.ndarray, np.ndarray]:
    N_class1 = N // 2
    N_class2 = N // 2
    x1 = np.random.multivariate_normal(
        mean=[5.0, 0.0], cov=[[3.0, 0.0], [0.0, 1.0]], size=N_class1
    )
    y1 = np.zeros(shape=(N_class1))
    x2 = np.random.multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, 0.0], [0.0, 3.0]], size=N_class2
    )
    y2 = np.ones(shape=(N_class1))
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y


def regression_data(N: int = 100) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(low=-10.0, high=10.0, size=N)
    y = f(x) + np.random.normal(scale=2.0, size=100)
    return x, y
