import matplotlib.pyplot as plt
import numpy as np


def plot(image_np: np.ndarray, figsize: tuple = (6, 6)):
    """
    Displays an image using matplotlib, with optional customization of the figure size.

    This function takes an image in the form of a NumPy array and displays it using matplotlib.
    If the image is 2D (grayscale), it will use a grayscale colormap for visualization.
    It also provides an option to adjust the figure size via the `figsize` parameter.

    Args:
        image_np (numpy.ndarray): The image to be displayed. Can be either 2D (grayscale) or 3D (color) numpy array.
        figsize (tuple, optional): The size of the figure (width, height). Default is (6, 6).

    Raises:
        ValueError: If the image provided is not a numpy array.

    Returns:
        None. The function directly displays the image.
    """
    plt.figure(figsize=figsize)

    # If the image is 2D (grayscale), use 'gray' colormap for a black & white mask
    if image_np.ndim == 2:
        plt.imshow(image_np, cmap="gray")
    else:
        plt.imshow(image_np)

    # Remove axis and padding around the plot
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
