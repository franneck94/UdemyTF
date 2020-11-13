import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # Black/White Image (1d)
    image = np.array(
        [0, 1, 1, 1, 1, 1, 0, 0, 0, ],
        dtype=np.uint8,
    )
    print(f"B/W (1D):\n{image}")
    plt.imshow(image.reshape((3, 3)), cmap="gray")
    plt.show()

    # Grayscale Image (2d)
    image = np.array(
        [[0, 100, 100],
         [255, 255, 255],
         [100, 100, 0]],
        dtype=np.uint8,
    )
    print(f"Gray (2D):\n{image}")
    plt.imshow(image, cmap="gray")
    plt.show()

    # RGB Image (3d)
    #                   R   G   B
    image = np.array([[[100, 42, 78],
                       [220, 47, 153]],
                      [[100, 42, 78],
                       [220, 47, 153]]], dtype=np.uint8)
    print(f"RGB (3D):\n{image}")
    plt.imshow(image)
    plt.show()
