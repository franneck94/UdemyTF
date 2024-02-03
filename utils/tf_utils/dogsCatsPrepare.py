import os

import cv2
import numpy as np
from skimage import transform


def remove_non_jpeg_files(
    dirs: list[str],
) -> None:
    for d in dirs:
        for f in os.listdir(d):
            if f.split(".")[-1] != "jpg":
                print(f"Removing file: {f}")
                os.remove(os.path.join(d, f))


def zero_init_arrays(
    cats_dir: str,
    dogs_dir: str,
    img_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    num_cats = len(os.listdir(cats_dir))
    num_dogs = len(os.listdir(dogs_dir))
    num_images = num_cats + num_dogs

    x = np.zeros(
        shape=(num_images, *img_shape),
        dtype=np.float32,
    )
    y = np.zeros(
        shape=(num_images,),
        dtype=np.float32,
    )
    return x, y


def load_image_files(
    dirs: list[str],
    class_names: list[str],
    img_shape: tuple[int, int, int],
    x: np.ndarray,
    y: np.ndarray,
) -> int:
    cnt = 0
    for d, class_name in zip(dirs, class_names, strict=False):
        for f in os.listdir(d):
            img_file_path = os.path.join(d, f)
            try:
                img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                x[cnt] = transform.resize(image=img, output_shape=img_shape)
                if class_name == "cat":
                    y[cnt] = 0
                elif class_name == "dog":
                    y[cnt] = 1
                else:
                    print("Invalid class name!")
                cnt += 1
            except:  # noqa: E722
                print(f"Image {f} cannt be read!")
                os.remove(img_file_path)
    return cnt


def extract_cats_vs_dogs(
    data_dir: str,
    img_shape: tuple[int, int, int],
) -> None:
    x_filepath = os.path.join(data_dir, "x.npy")
    y_filepath = os.path.join(data_dir, "y.npy")

    cats_dir = os.path.join(data_dir, "Cat")
    dogs_dir = os.path.join(data_dir, "Dog")

    dirs = [cats_dir, dogs_dir]
    class_names = ["Cat", "Dog"]

    remove_non_jpeg_files(dirs)
    x, y = zero_init_arrays(cats_dir, dogs_dir, img_shape)

    cnt = load_image_files(dirs, class_names, img_shape, x, y)

    # Dropping not readable image idxs
    x = x[:cnt]
    y = y[:cnt]

    np.save(x_filepath, x)
    np.save(y_filepath, y)


if __name__ == "__main__":
    data_dir = os.path.join("C:/Users/Jan/Documents/DogsAndCats")
    img_shape = (64, 64, 3)
    extract_cats_vs_dogs(data_dir, img_shape)
