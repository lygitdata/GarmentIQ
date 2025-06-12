import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot(
    image_path: str,
    coordinate: np.ndarray = None,
    figsize: tuple = (6, 6),
    color: str = "red",
):
    """
    Display an image from a file path using matplotlib, with optional overlay of coordinates.

    This function loads an image from the given file path, displays it using matplotlib,
    and optionally overlays coordinate points on the image.

    Args:
        image_path (str): Path to the image file. The image will be loaded as RGB.
        coordinate (np.ndarray, optional): Optional array of coordinates to overlay on the image.
                                          Expected shape: (1, N, 2), where N is the number of points.
        figsize (tuple, optional): Size of the displayed figure in inches (width, height).
        color (str, optional): Color of the overlay points. Default is 'red'.

    Raises:
        ValueError: If image cannot be loaded or coordinate format is invalid.

    Returns:
        None
    """
    try:
        image_np = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        raise ValueError(f"Unable to load image from path: {image_path}. Error: {e}")

    plt.figure(figsize=figsize)

    if image_np.ndim == 2:
        plt.imshow(image_np, cmap="gray")
    else:
        plt.imshow(image_np)

    if coordinate is not None:
        try:
            plt.scatter(coordinate[0][:, 0], coordinate[0][:, 1], c=color, s=10)
        except Exception as e:
            raise ValueError(f"Invalid coordinate format: {coordinate}. Error: {e}")

    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
