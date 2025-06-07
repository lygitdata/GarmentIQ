import numpy as np
from .derivation_dict import derivation_dict

def prepare_args(entry: dict, derivation_dict: dict = derivation_dict, **extra_args) -> dict:
    function_name = entry.get('function')
    if function_name is None:
        raise ValueError("Entry must include a 'function' key.")
    if function_name not in derivation_dict:
        raise ValueError(f"Unknown function: {function_name}")

    function_schema = derivation_dict[function_name]
    args = {}

    for key, value in entry.items():
        if key == 'function':
            continue

        expected_type = function_schema.get(key)

        if expected_type is None:
            # Should be cast to int
            args[key] = int(value)
        elif isinstance(expected_type, list):
            # Should be one of the listed options
            if value not in expected_type:
                raise ValueError(f"Invalid value '{value}' for {key}; expected one of {expected_type}")
            args[key] = value
        else:
            raise TypeError(f"Unsupported schema format for key '{key}' in function '{function_name}'")

    # Add function-specific extra arguments
    if function_name == "derive_keypoint_coord":
      if "landmark_coords" not in extra_args:
          raise ValueError("'landmark_coords' is required for 'derive_keypoint_coord'")
      elif not isinstance(extra_args['landmark_coords'], np.ndarray):
          raise ValueError("'landmark_coords' must be a 'np.ndarray'")

      if "np_mask" not in extra_args:
          raise ValueError("'np_mask' is required for 'derive_keypoint_coord'")
      elif not isinstance(extra_args['np_mask'], np.ndarray):
          raise ValueError("'np_mask' must be a 'np.ndarray'")
      args['landmark_coords'] = extra_args['landmark_coords']
      args['np_mask'] = extra_args['np_mask']
      return {"derive_keypoint_coord" : args}
    # Add more if conditions if there are more derivation functions in the future
    # elif function_name == "another_function_1":
    #   if "arg_3" not in extra_args:
    #     raise ValueError("'arg_3' is required for 'another_function_1'")
    # else:
    #     args['mask_path'] = extra_args['mask_path']
    #     return args