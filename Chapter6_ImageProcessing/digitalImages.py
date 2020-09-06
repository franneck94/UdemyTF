import numpy as np
from matplotlib import pyplot as plt


# Schwarz Weiß Bilder (1d)
bild = np.array(
    [0, 1, 1, 1, 1, 1, 0, 0, 0, ],
    dtype=np.uint8,
)
print("\nB/W (1D):\n", bild)
plt.imshow(bild.reshape((3, 3)), cmap="gray")
plt.show()

# Grauwert Bild (2d)
bild = np.array(
    [
        [0, 100, 100],
        [255, 255, 255],
        [100, 100, 0],
    ],
    dtype=np.uint8,
)
print("\nGray (2D):\n", bild)
plt.imshow(bild, cmap="gray")
plt.show()

# RGB Bild (3d)
#                   R   G   B     R    G   B
bild = np.array([[[100, 42, 78], [220, 47, 153]],
                 [[100, 42, 78], [220, 47, 153]]], dtype=np.uint8)
print("\nRGB (3D):\n", bild)
plt.imshow(bild)
plt.show()
