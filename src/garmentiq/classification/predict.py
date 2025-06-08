import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from typing import Type, List

def predict(
    model: Type[nn.Module],
    image_path: str,
    classes: List[str],
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=False,
):
    """
    Loads a trained PyTorch model and makes a prediction on a single image.

    This function processes a single image from disk, applies resizing and normalization,
    feeds it through a loaded model, and returns the predicted class along with the
    class probabilities. The model is expected to output logits over a fixed number of classes.

    :param model_path: Path to the saved model checkpoint file.
    :type model_path: str
    :param model_class: The class of the PyTorch model to instantiate. Must inherit from `torch.nn.Module`.
    :type model_class: Type[nn.Module]
    :param model_args: Dictionary of arguments used to initialize the model.
    :type model_args: dict
    :param image_path: Path to the input image file (.jpg, .jpeg, .png).
    :type image_path: str
    :param classes: List of class names corresponding to model outputs. Will be sorted internally.
    :type classes: List[str]
    :param resize_dim: Tuple indicating the dimensions to resize the image to. Default is (120, 184).
    :type resize_dim: tuple[int, int]
    :param normalize_mean: List of mean values for normalization. Default is [0.8047, 0.7808, 0.7769].
    :type normalize_mean: list[float]
    :param normalize_std: List of standard deviation values for normalization. Default is [0.2957, 0.3077, 0.3081].
    :type normalize_std: list[float]
    :param device: Device to run inference on. Defaults to CUDA if available, otherwise CPU.
    :type device: torch.device, optional
    :param verbose: If True, prints the predicted label and class probabilities.
    :type verbose: bool

    :raises ValueError: If the image file does not have a supported extension (.jpg, .jpeg, .png).
    :raises FileNotFoundError: If the model checkpoint file is not found or cannot be loaded.

    :returns: A tuple containing:
        - predicted label (str): The class label with the highest predicted probability.
        - prob_list (List[float]): The list of class probabilities in the same order as the sorted class list.
    :rtype: tuple[str, List[float]]
    """
    # Validate image extension
    if not any(image_path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".JPG"]):
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

    del model
    torch.cuda.empty_cache()

    return pred_label, prob_list
