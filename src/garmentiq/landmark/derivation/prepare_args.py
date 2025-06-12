import numpy as np
from .derivation_dict import derivation_dict


def prepare_args(
    entry: dict, derivation_dict: dict = derivation_dict, **extra_args
) -> dict:
    """
    Prepares arguments for a specific derivation function based on an entry from the
    landmark derivation configuration and extra arguments.

    This function ensures that all required arguments for a given derivation function
    are collected and validated against its schema defined in `derivation_dict`.
    It also adds any necessary `extra_args` (like `landmark_coords` or `np_mask`).

    Args:
        entry (dict): A dictionary representing a single landmark's derivation entry,
                      which must include a 'function' key and its specific parameters.
        derivation_dict (dict): The global dictionary defining schemas for all derivation functions.
                                Defaults to `derivation_dict`.
        **extra_args: Additional keyword arguments that might be required by the derivation function,
                      such as `landmark_coords` (NumPy array of landmark coordinates)
                      and `np_mask` (NumPy array of the segmentation mask).

    Raises:
        ValueError: If the 'function' key is missing in `entry`, or if the function name is unknown.
        TypeError: If an unsupported schema format is encountered for a key.
        ValueError: If a required `extra_arg` is missing for a specific function (e.g., `landmark_coords`).

    Returns:
        dict: A dictionary where the key is the function name and its value is a dictionary
              of arguments ready to be passed to that derivation function.
    """
    function_name = entry.get("function")
    if function_name is None:
        raise ValueError("Entry must include a 'function' key.")
    if function_name not in derivation_dict:
        raise ValueError(f"Unknown function: {function_name}")

    function_schema = derivation_dict[function_name]
    args = {}

    for key, value in entry.items():
        if key == "function":
            continue

        expected_type = function_schema.get(key)

        if expected_type is None:
            # Should be cast to int
            args[key] = int(value)
        elif isinstance(expected_type, list):
            # Should be one of the listed options
            if value not in expected_type:
                raise ValueError(
                    f"Invalid value '{value}' for {key}; expected one of {expected_type}"
                )
            args[key] = value
        else:
            raise TypeError(
                f"Unsupported schema format for key '{key}' in function '{function_name}'"
            )

    # Add function-specific extra arguments
    if function_name == "derive_keypoint_coord":
        if "landmark_coords" not in extra_args:
            raise ValueError(
                "'landmark_coords' is required for 'derive_keypoint_coord'"
            )
        elif not isinstance(extra_args["landmark_coords"], np.ndarray):
            raise ValueError("'landmark_coords' must be a 'np.ndarray'")

        if "np_mask" not in extra_args:
            raise ValueError("'np_mask' is required for 'derive_keypoint_coord'")
        elif not isinstance(extra_args["np_mask"], np.ndarray):
            raise ValueError("'np_mask' must be a 'np.ndarray'")
        args["landmark_coords"] = extra_args["landmark_coords"]
        args["np_mask"] = extra_args["np_mask"]
        return {"derive_keypoint_coord": args}
    # Add more if conditions if there are more derivation functions in the future
    # elif function_name == "another_function_1":
    #   if "arg_3" not in extra_args:
    #     raise ValueError("'arg_3' is required for 'another_function_1'")
    # else:
    #     args['mask_path'] = extra_args['mask_path']
    #     return args
