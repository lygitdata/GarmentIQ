import json
import os
from typing import Type, Union
import torch
import requests
import numpy as np
from garmentiq.utils import validate_garment_class_dict
from garmentiq.landmark.utils import (
    find_instruction_landmark_index,
    fill_instruction_landmark_coordinate,
)
from garmentiq.landmark.detection.utils import (
    input_image_transform,
    get_final_preds,
    transform_preds,
)


def detect(
    class_name: str,
    class_dict: dict,
    image_path: Union[str, np.ndarray],
    model: Type[torch.nn.Module],
    scale_std: float = 200.0,
    resize_dim: list[int, int] = [288, 384],
    normalize_mean: list[float, float, float] = [0.485, 0.456, 0.406],
    normalize_std: list[float, float, float] = [0.229, 0.224, 0.225],
):
    """
    Detects predefined landmarks on a garment image using a specified model and class instructions.

    This function validates the input class dictionary and class name, loads the appropriate
    instruction schema (from local file or URL), preprocesses the image, runs it through
    the landmark detection model, and then transforms the detected heatmap predictions
    into image coordinates. The detected coordinates are then filled into the instruction data.

    Args:
        class_name (str): The name of the garment class (e.g., "vest dress", "trousers").
        class_dict (dict): A dictionary mapping class names to their properties, including
                           `num_predefined_points`, `index_range`, and `instruction` file path.
        image_path (Union[str, np.ndarray]): The path to the image file or a NumPy array of the image.
        model (Type[torch.nn.Module]): The loaded PyTorch landmark detection model.
        scale_std (float, optional): Standard scale for image transformation during preprocessing. Defaults to 200.0.
        resize_dim (list[int, int], optional): Target dimensions [width, height] for the transformed image.
                                               Defaults to [288, 384].
        normalize_mean (list[float, float, float], optional): Mean values for image normalization (RGB channels).
                                                              Defaults to [0.485, 0.456, 0.406].
        normalize_std (list[float, float, float], optional): Standard deviation values for image normalization (RGB channels).
                                                             Defaults to [0.229, 0.224, 0.225].

    Raises:
        ValueError: If `class_dict` is invalid or `class_name` is not found in `class_dict`.
        FileNotFoundError: If the instruction file is not found.
        ValueError: If loading instruction JSON from URL fails or `class_name` is not found in instruction file.

    Returns:
        tuple:
            - preds_all (np.array): All predicted landmark coordinates (including non-predefined).
            - maxvals (np.array): Confidence scores for the predefined landmark predictions.
            - instruction_data (dict): The instruction dictionary updated with detected landmark coordinates and confidences.
    """
    if not validate_garment_class_dict(class_dict):
        raise ValueError(
            "Provided class_dict is not in the expected garment_classes format."
        )

    if class_name not in class_dict:
        raise ValueError(
            f"Invalid class '{class_name}'. Must be one of: {list(class_dict.keys())}"
        )

    class_element = class_dict[class_name]

    instruction_path = class_element["instruction"]

    if instruction_path.startswith("http://") or instruction_path.startswith(
        "https://"
    ):
        try:
            response = requests.get(instruction_path)
            response.raise_for_status()
            instruction_data = response.json()
        except Exception as e:
            raise ValueError(
                f"Failed to load instruction JSON from URL: {instruction_path}\nError: {e}"
            )
    else:
        if not os.path.exists(instruction_path):
            raise FileNotFoundError(f"Instruction file not found: {instruction_path}")
        with open(instruction_path, "r") as f:
            instruction_data = json.load(f)

    if class_name not in instruction_data:
        raise ValueError(f"Class '{class_name}' not found in instruction file.")

    (input_tensor, image_np, center, scale,) = input_image_transform(
        image_path, scale_std, resize_dim, normalize_mean, normalize_std
    )

    with torch.no_grad():
        np_output_heatmap = model(input_tensor).detach().cpu().numpy()

    preds_heatmap, maxvals = get_final_preds(
        np_output_heatmap[
            :, class_element["index_range"][0] : class_element["index_range"][1], :, :
        ]
    )

    predefined_index = find_instruction_landmark_index(
        instruction_data[class_name]["landmarks"], predefined=True
    )
    preds_all = np.stack([transform_preds(p, center, scale) for p in preds_heatmap])
    preds = preds_all[:, predefined_index, :]

    instruction_data[class_name]["landmarks"] = fill_instruction_landmark_coordinate(
        instruction_landmarks=instruction_data[class_name]["landmarks"],
        index=predefined_index,
        fill_in_value=preds,
    )

    for idx in predefined_index:
        instruction_data[class_name]["landmarks"][str(idx + 1)]["conf"] = float(
            maxvals[0, idx, 0]
        )

    return preds_all, maxvals, instruction_data
