import matplotlib.pyplot as plt


def plot(image: np.ndarray, figsize: tuple = (6, 6)):
    """
    Displays an image using matplotlib, with optional customization of the figure size.

    This function takes an image in the form of a NumPy array and displays it using matplotlib.
    If the image is 2D (grayscale), it will use a grayscale colormap for visualization.
    It also provides an option to adjust the figure size via the `figsize` parameter.

    :param image: The image to be displayed. Can be either 2D (grayscale) or 3D (color) numpy array.
    :type image: numpy.ndarray
    :param figsize: The size of the figure (width, height). Default is (6, 6).
    :type figsize: tuple, optional

    :raises ValueError: If the image provided is not a numpy array.

    :returns: None. The function directly displays the image.
    :rtype: None
    """
    plt.figure(figsize=figsize)

    # If the image is 2D (grayscale), use 'gray' colormap for a black & white mask
    if image.ndim == 2:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    # Remove axis and padding around the plot
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
