import os


def check_unzipped_dir(folder):
    """
    Verifies the structure and contents of an unzipped dataset directory.

    This function checks that the specified folder contains:
    - A subdirectory named 'images' with at least one `.jpg`, `.png`, `.jpeg` file.
    - A file named 'metadata.csv'.

    If any of these components are missing, a `FileNotFoundError` is raised.

    Args:
        folder (str): Path to the root folder of the unzipped dataset.

    Raises:
        FileNotFoundError: If the 'images' folder is missing,
                           if no `.jpg`, `.png`, `.jpeg` files are found in the 'images' folder,
                           or if 'metadata.csv' is missing.

    Returns:
        None
    """
    image_dir = os.path.join(folder, "images")
    metadata_path = os.path.join(folder, "metadata.csv")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing 'images' folder in: {folder}")

    jpg_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg") or f.lower().endswith(".png") or f.lower().endswith(".jpeg")]
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg, .png, or .jpeg file found in: {image_dir}")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Missing 'metadata.csv' in: {folder}")
