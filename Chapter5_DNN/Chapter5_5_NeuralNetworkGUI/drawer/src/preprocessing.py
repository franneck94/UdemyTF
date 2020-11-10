import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image


def resize(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (28, 28))
    return image


def normalize(image: np.ndarray) -> np.ndarray:
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image


def center(image: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image)
    rows, cols = image.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    image = cv2.warpAffine(image, M, (cols, rows))
    return image


def get_image(DrawingFrame: Any, debug: bool = False) -> np.ndarray:
    pixmap = DrawingFrame.grab()
    temp_image_path = os.path.join(PROJECT_PATH, "ressources", "imgs", "temp_image.jpg")
    pixmap.save(temp_image_path)
    image = load(temp_image_path).astype(np.float32)
    image = resize(image)
    if debug:
        plt.imshow(image, cmap="gray")
        plt.show()
    image = normalize(image)
    if debug:
        plt.imshow(image, cmap="gray")
        plt.show()
    image = center(image)
    if debug:
        plt.imshow(image, cmap="gray")
        plt.show()
    return image
