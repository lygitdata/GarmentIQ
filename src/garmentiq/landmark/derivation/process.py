from .derive_keypoint_coord import derive_keypoint_coord


def process(**args):
    """
    Dispatches to the appropriate derivation function based on the 'function' name
    provided in the arguments.

    This function acts as a router, taking a dictionary of arguments which
    includes the name of the derivation function to call (under a key
    that matches a function name in its internal dispatch table).

    Args:
        **args: Keyword arguments where one key's value is a dictionary
                containing the arguments for a specific derivation function,
                and that key's name corresponds to the derivation function.
                Example: `{'derive_keypoint_coord': {'p1_id': 1, ...}}`

    Raises:
        ValueError: If the function name specified in `args` is unknown.

    Returns:
        Any: The result returned by the dispatched derivation function.
    """
    dispatch = {
        "derive_keypoint_coord": derive_keypoint_coord,
    }
    for func_name, func_args in args.items():
        if func_name not in dispatch:
            raise ValueError(f"Unknown function: {func_name}")
        return dispatch[func_name](**func_args)
