def validate_garment_class_dict(class_dict: dict) -> bool:
    """
    Validates the structure and content of a garment class dictionary.

    Ensures that the dictionary adheres to the expected format, including the presence
    of required keys ("num_predefined_points", "index_range", "instruction") and
    correct data types and logical consistency for their values.

    Args:
        class_dict (dict): The dictionary to be validated.

    Returns:
        bool: True if the dictionary is valid, False otherwise.
    """
    required_keys = {"num_predefined_points", "index_range", "instruction"}

    if not isinstance(class_dict, dict):
        return False

    for class_name, class_info in class_dict.items():
        if not isinstance(class_name, str):
            return False
        if not isinstance(class_info, dict):
            return False
        if not required_keys.issubset(class_info.keys()):
            return False

        num_points = class_info["num_predefined_points"]
        index_range = class_info["index_range"]
        instruction = class_info["instruction"]

        # Check types
        if not isinstance(num_points, int):
            return False
        if (
            not isinstance(index_range, tuple)
            or len(index_range) != 2
            or not all(isinstance(i, int) for i in index_range)
        ):
            return False

        # Check logical consistency
        if index_range[1] - index_range[0] != num_points:
            return False

        # Validate instruction field (must be a string ending with .json)
        if not (isinstance(instruction, str) and instruction.endswith(".json")):
            return False

    return True
