from typing import Tuple

import numpy as np


np.random.seed(42)


def f(x: np.ndarray) -> np.ndarray:
    return 2.0 * x + 5.0


def classification_data() -> Tuple[np.ndarray, np.ndarray]:
    x1 = np.random.multivariate_normal(mean=[5.0, 0.0], cov=[[3, 0], [0, 1]], size=15)
    y1 = np.array([0 for i in range(15)])
    x2 = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1, 0], [0, 3]], size=15)
    y2 = np.array([1 for i in range(15)])
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))
    return x, y


def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(low=-10.0, high=10.0, size=100)
    y = f(x) + np.random.normal(scale=2.0, size=100)  # y = f(x) + eps
    return x, y
