"""Single-image garment classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Type, List, Union
from garmentiq.utils.device import resolve_device, empty_cache


def predict(
    model: Type[nn.Module],
    image_path: str,
    classes: List[str],
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081],
    device: Union[str, torch.device] = "cpu",
    verbose=False,
):
    """
    Loads a trained PyTorch model and makes a prediction on a single image.

    This function processes a single image from disk, applies resizing and normalization,
    feeds it through a loaded model, and returns the predicted class along with the
    class probabilities. The model is expected to output logits over a fixed number of classes.

    The model and the preprocessed image tensor are placed on `device`, so the same value should
    be passed here as was used when loading the model. Any accelerator memory cached during
    inference is released before returning.

    Args:
        model (Type[nn.Module]): The loaded PyTorch model instance ready for inference.
        image_path (str): Path to the input image file (.jpg, .jpeg, .png).
        classes (List[str]): List of class names corresponding to model outputs. Will be sorted internally.
        resize_dim (tuple[int, int]): Tuple indicating the dimensions to resize the image to. Default is (120, 184).
        normalize_mean (list[float]): List of mean values for normalization. Default is [0.8047, 0.7808, 0.7769].
        normalize_std (list[float]): List of standard deviation values for normalization. Default is [0.2957, 0.3077, 0.3081].
        device (Union[str, torch.device], optional): The device to run inference on, e.g. `"cpu"`,
                                                     `"cuda"`, `"cuda:0"`, or `"mps"`. Hardware
                                                     acceleration is opt-in; pass it explicitly to
                                                     use a GPU or Apple Silicon. Defaults to `"cpu"`.
        verbose (bool): If True, prints the predicted label and class probabilities.

    Raises:
        ValueError: If the image file does not have a supported extension (.jpg, .jpeg, .png),
                    or if the requested `device` is invalid or unavailable on this machine.
        FileNotFoundError: If the model checkpoint file is not found or cannot be loaded.

    Returns:
        tuple[str, List[float]]: A tuple containing:
            - predicted label (str): The class label with the highest predicted probability.
            - prob_list (List[float]): The list of class probabilities in the same order as the sorted class list.
    """
    device = resolve_device(device)

    # Validate image extension
    if not any(
        image_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".JPG"]
    ):
        raise ValueError("Image file must end with .jpg, .jpeg, .png, or .JPG")

    # Sort the classes list to have a consistent order
    sorted_classes = sorted(classes)

    # Define the preprocessing transformation.
    transform = transforms.Compose(
        [
            transforms.Resize(resize_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std),
        ]
    )

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    model = model.to(device)
    image_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(image_tensor)
        # Compute probabilities using softmax
        probabilities = (
            F.softmax(outputs, dim=1).cpu().numpy()[0]
        )  # shape: (num_classes,)

    # Determine the predicted index and label
    pred_index = int(probabilities.argmax())
    pred_label = sorted_classes[pred_index]

    # Optionally, you might want to return probabilities as a list of floats:
    prob_list = probabilities.tolist()

    if verbose:
        print(f"Prediction: {pred_label}")
        print(f"Probabilities: {prob_list}")

    del image_tensor, outputs
    empty_cache(device)

    return pred_label, prob_list
