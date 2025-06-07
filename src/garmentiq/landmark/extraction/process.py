import json
import os
from typing import Type
import torch
import requests
import numpy as np
from garmentiq.utils import validate_garment_class_dict
from garmentiq.landmark.utils import (
    find_instruction_landmark_index,
    fill_instruction_landmark_coordinate,
)
from garmentiq.landmark.extraction.utils import (
    input_image_transform,
    get_final_preds,
    transform_preds,
)


def process(
    class_name: str,
    class_dict: dict,
    image_path: str,
    model: Type[torch.nn.Module],
):
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

    (
        input_tensor,
        image_np,
        center,
        scale,
    ) = input_image_transform(image_path)

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
    preds = np.stack(
        [
            transform_preds(p, center, scale)
            for p in preds_heatmap
        ]
    )[:, predefined_index, :]

    instruction_data[class_name]["landmarks"] = fill_instruction_landmark_coordinate(
        instruction_landmarks=instruction_data[class_name]["landmarks"],
        index=predefined_index,
        fill_in_value=preds,
    )

    return preds, instruction_data
