from torchvision import transforms
import os
from PIL import Image
import torch
from garmentiq.classification.utils import (
    CachedDataset,
    seed_worker,
    train_epoch,
    validate_epoch,
    save_best_model,
    validate_train_param,
    validate_test_param,
)
from tqdm.notebook import tqdm


def load_data(
    df,
    img_dir,
    label_column,
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081],
):
    """
    Loads and preprocesses image data into memory from a DataFrame of filenames and labels.

    This function reads images from the specified directory, applies resizing, normalization,
    and tensor conversion, and encodes labels from a specified column. It returns tensors for
    images and labels, along with the transform pipeline used.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing at least a 'filename' column and a label column.
        img_dir (str): Path to the directory containing image files.
        label_column (str): Name of the column in `df` containing class labels.
        resize_dim (tuple[int, int]): Tuple indicating the dimensions (height, width) to resize each image to.
                                     Defaults to (120, 184).
        normalize_mean (list[float]): Mean values for normalization (per channel).
                                      Defaults to `[0.8047, 0.7808, 0.7769]`.
        normalize_std (list[float]): Standard deviation values for normalization (per channel).
                                     Defaults to `[0.2957, 0.3077, 0.3081]`.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torchvision.transforms.Compose]: A tuple containing:
            - cached_images (torch.Tensor): Tensor containing all preprocessed images.
            - cached_labels (torch.Tensor): Tensor containing all encoded labels.
            - transform (torchvision.transforms.Compose): The transformation pipeline used.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    classes = sorted(df[label_column].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    cached_images = []
    cached_labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading data into memory"):
        img_path = os.path.join(img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = transform(image)

        label = class_to_idx[row[label_column]]

        cached_images.append(image)
        cached_labels.append(label)

    cached_images = torch.stack(cached_images)
    cached_labels = torch.tensor(cached_labels)

    return cached_images, cached_labels, transform
