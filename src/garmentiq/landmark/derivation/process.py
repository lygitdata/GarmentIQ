from .derive_keypoint_coord import derive_keypoint_coord

def process(**args):
    dispatch = {
        "derive_keypoint_coord": derive_keypoint_coord,
    }
    for func_name, func_args in args.items():
        if func_name not in dispatch:
            raise ValueError(f"Unknown function: {func_name}")
        return dispatch[func_name](**func_args)