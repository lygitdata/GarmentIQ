from garmentiq.landmark.derivation import (
  prepare_args,
  process,
)

def derive(
    class_name: str, detection_dict: dict, derivation_dict: dict, **extra_args
) -> dict:
    non_predefined_landmark = {
        k: detection_dict[class_name]["landmarks"][k]["derivation"]
        for k, v in detection_dict[class_name]["landmarks"].items()
        if v.get("predefined") is False
    }
    derived_coords = {}
    for k, v in non_predefined_landmark.items():
        args = prepare_args(
            non_predefined_landmark[k], derivation_dict, **extra_args
        )
        derived_coord = tuple(float(x) for x in process(**args))
        derived_coords[k] = derived_coord
        detection_dict[class_name]["landmarks"][k]["x"] = derived_coord[0]
        detection_dict[class_name]["landmarks"][k]["y"] = derived_coord[1]
    return derived_coords, detection_dict
