import numpy as np


def find_instruction_landmark_index(instruction_landmarks: dict, predefined: bool):
    """
    Finds the indices of predefined or non-predefined landmarks from an instruction landmarks dictionary.

    Args:
        instruction_landmarks (dict): A dictionary of landmark information from an instruction schema.
        predefined (bool): If True, returns indices of predefined landmarks.
                           If False, returns indices of non-predefined (derived) landmarks.

    Returns:
        list: A list of integer indices (0-based) for the requested type of landmarks.
    """
    if predefined:
        return [
            int(k) - 1
            for k, v in instruction_landmarks.items()
            if v.get("predefined") is True
        ]
    else:
        return [
            int(k) - 1
            for k, v in instruction_landmarks.items()
            if v.get("predefined") is False
        ]


def fill_instruction_landmark_coordinate(
    instruction_landmarks: dict, index: list, fill_in_value: np.array
):
    """
    Fills the 'x' and 'y' coordinates for specified landmarks in the instruction dictionary
    based on provided coordinate values.

    Args:
        instruction_landmarks (dict): The dictionary of landmark instructions to be updated.
        index (list): A list of integer indices (0-based) corresponding to the landmarks
                      in `instruction_landmarks` that should be updated.
        fill_in_value (np.array): A NumPy array containing the new coordinates.
                                  Expected shape: (1, N, 2), where N is the number of landmarks to fill.

    Returns:
        dict: The updated `instruction_landmarks` dictionary with filled coordinates.
    """
    for k in instruction_landmarks:
        idx = int(k) - 1
        if idx in index:
            preds_idx = index.index(idx)
            instruction_landmarks[k]["x"] = float(fill_in_value[0, preds_idx, 0])
            instruction_landmarks[k]["y"] = float(fill_in_value[0, preds_idx, 1])
    return instruction_landmarks
