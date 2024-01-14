# type: ignore
import os
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


FILE_PATH = os.path.abspath(__file__)
PROJECT_DIR = os.path.dirname(os.path.dirname(FILE_PATH))


def load(image_path: str) -> np.ndarray:
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def resize(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (28, 28))


def normalize(image: np.ndarray) -> np.ndarray:
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return image


def center(image: np.ndarray) -> np.ndarray:
    cy, cx = center_of_mass(image)
    rows, cols = image.shape
    shiftx = np.round(cols / 2 - cx).astype(int)
    shifty = np.round(rows / 2 - cy).astype(int)
    m = np.array([[1, 0, shiftx], [0, 1, shifty]]).astype(np.float32)
    return cv2.warpAffine(image, m, (cols, rows))


def get_image(
    drawing_frame: Any,
) -> np.ndarray:
    pixmap = drawing_frame.grab()
    temp_image_file_path = os.path.join(
        PROJECT_DIR,
        "ressources",
        "imgs",
        "temp_image.jpg",
    )
    pixmap.save(temp_image_file_path)
    image = load(temp_image_file_path)
    image = resize(image)
    image = normalize(image)
    return center(image)
