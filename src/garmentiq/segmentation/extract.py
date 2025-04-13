from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import kornia
import numpy as np


def extract(
    model: AutoModelForImageSegmentation,
    image_path: str,
    resize_dim: tuple[int, int],
    normalize_mean: list[float, float, float],
    normalize_std: list[float, float, float],
    high_precision: bool = True,
):
    """
    Extracts an image segmentation mask from a given image using a pretrained model.

    This function takes an image, applies the necessary transformations (resize, normalize),
    and then feeds it into the model to generate a segmentation mask. The result is a mask
    overlayed on the original image, which is then returned as a numpy array.

    :param model: The pretrained image segmentation model to use for predictions.
    :type model: AutoModelForImageSegmentation
    :param image_path: The path to the image file on which to perform segmentation.
    :type image_path: str
    :param resize_dim: The target size (height, width) to resize the image to before feeding it into the model.
    :type resize_dim: tuple[int, int]
    :param normalize_mean: A list of means for normalizing the input image, typically used for pretrained models.
                            Expected format: [R_mean, G_mean, B_mean].
    :type normalize_mean: list[float, float, float]
    :param normalize_std: A list of standard deviations for normalizing the input image.
                           Expected format: [R_std, G_std, B_std].
    :type normalize_std: list[float, float, float]
    :param high_precision: Flag indicating whether to use full precision (True) or half precision (False) for the model.
                            Default is True (full precision).
    :type high_precision: bool, optional

    :raises FileNotFoundError: If the image file at `image_path` does not exist.
    :raises ValueError: If the model is incompatible with the task or the image format is unsupported.

    :returns: The original image with the segmentation mask overlaid, and the mask as a numpy array.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )

    image = Image.open(image_path)
    input_image = transform(image).unsqueeze(0).to(device)

    if not high_precision:
        input_image = input_image.half()

    with torch.no_grad():
        preds = model(input_image)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)

    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask)

    del model
    torch.cuda.empty_cache()

    return image_np, mask_np
