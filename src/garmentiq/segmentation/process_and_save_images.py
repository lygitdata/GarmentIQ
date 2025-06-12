import os
from transformers import AutoModelForImageSegmentation
from tqdm.notebook import tqdm
import kornia
from PIL import Image
import numpy as np
from garmentiq.segmentation.extract import extract
from garmentiq.segmentation.change_background_color import change_background_color


def process_and_save_images(
    image_dir: str,
    output_dir: str,
    model: AutoModelForImageSegmentation,
    resize_dim: tuple[int, int],
    normalize_mean: list[float, float, float],
    normalize_std: list[float, float, float],
    background_color: tuple[int, int, int] = None,
    high_precision: bool = True,
):
    """
    Processes images from a directory by extracting segmentation masks and optionally modifying
    the background color, then saving the masks and modified images to specified output directories.

    This function applies the `extract` function to segment each image, generating a mask, and
    optionally modifies the background of each image using the `change_background_color` function
    before saving both the masks and the modified images to disk.

    Args:
        image_dir (str): The directory containing the input images to process.
        output_dir (str): The directory where the processed masks and modified images will be saved.
        model (transformers.AutoModelForImageSegmentation): The pre-trained model used for image segmentation.
        resize_dim (tuple[int, int]): The target dimensions to resize the images before processing (width, height).
        normalize_mean (list[float, float, float]): The mean values used for image normalization.
        normalize_std (list[float, float, float]): The standard deviation values used for image normalization.
        background_color (tuple[int, int, int], optional): The background color to apply to the image (RGB tuple),
                                                          or None to skip the modification. Default is None.
        high_precision (bool, optional): Whether to use high precision (32-bit) for image processing. Default is True.

    Raises:
        FileNotFoundError: If the input image directory does not exist.
        ValueError: If the `model` provided does not work correctly for image segmentation.

    Returns:
        None. The processed masks and modified images are saved to the specified output directory.
    """
    # Create output directories for masks and modified images
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    if background_color is not None:
        modified_image_dir = os.path.join(output_dir, "bg_modified")
        os.makedirs(modified_image_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Loop through all image files in the provided directory
    with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(image_dir, filename)

                # Extract image and mask using the extract function
                image_np, mask_np = extract(
                    model=model,
                    image_path=image_path,
                    resize_dim=resize_dim,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                    high_precision=high_precision,
                )

                # Save the mask image
                mask_pil = Image.fromarray(mask_np)
                mask_pil.save(
                    os.path.join(mask_dir, f"mask_{os.path.splitext(filename)[0]}.png")
                )

                if background_color is not None:
                    # Change the background color if provided
                    modified_image = change_background_color(
                        image_np, mask_np, background_color
                    )

                    # Save the modified image with the new background color
                    modified_image_pil = Image.fromarray(modified_image)
                    modified_image_pil.save(
                        os.path.join(
                            modified_image_dir,
                            f"bg_modified_{os.path.splitext(filename)[0]}.png",
                        )
                    )

                pbar.update(1)
