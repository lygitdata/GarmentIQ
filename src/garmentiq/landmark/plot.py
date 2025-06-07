import matplotlib.pyplot as plt
import numpy as np

def plot(
    image_np: np.ndarray,
    coordinate: np.ndarray = None,
    figsize: tuple = (6, 6),
    color: str = 'red'
):
    """
    Display an image using matplotlib with optional overlay of coordinates.

    This function visualizes a NumPy image array using matplotlib. If the image is grayscale
    (2D array), it is displayed using a grayscale colormap. Optionally, a set of coordinates
    (e.g., landmarks) can be overlaid as scatter points, with a customizable color.

    :param image_np: The image to display. Should be a 2D (grayscale) or 3D (RGB) NumPy array.
    :type image_np: np.ndarray

    :param coordinate: Optional array of coordinates to overlay on the image.
                       Expected shape: (1, N, 2), where N is the number of points.
    :type coordinate: np.ndarray, optional

    :param figsize: Size of the displayed figure in inches (width, height).
    :type figsize: tuple, optional

    :param color: Color of the overlay points. Default is 'red'.
    :type color: str, optional

    :raises ValueError: If `image_np` is not a NumPy array.

    :return: None
    """
    if not isinstance(image_np, np.ndarray):
        raise ValueError("image_np must be a NumPy array")

    plt.figure(figsize=figsize)

    if image_np.ndim == 2:
        plt.imshow(image_np, cmap="gray")
    else:
        plt.imshow(image_np)

    if coordinate is not None:
        plt.scatter(coordinate[0][:, 0], coordinate[0][:, 1], c=color, s=10)

    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
