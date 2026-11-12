"""Calibrate image coordinates using the width and height of its physical frame."""
import math


def validate_calibrations(value, count):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Provide one calibration entry per uploaded image.")
    result = []
    for entry in value:
        if entry is None:
            result.append(None)
            continue
        if not isinstance(entry, dict) or set(entry) != {"width_cm", "height_cm"}:
            raise ValueError("Each image calibration needs both width_cm and height_cm.")
        normalized = {}
        for key, length in entry.items():
            if isinstance(length, bool) or not isinstance(length, (int, float)) or not math.isfinite(length) or length <= 0:
                raise ValueError("Image width and height in centimetres must be finite, positive numbers.")
            normalized[key] = float(length)
        result.append(normalized)
    return result


def calibrate_record(record, image_size, calibration):
    width, height = image_size
    record["image_size"] = {"width": width, "height": height}
    record["calibration"] = None
    for measurement in record["measurements"].values():
        measurement["distance_px"] = measurement["distance"]
    if calibration is None:
        return record
    sx, sy = calibration["width_cm"] / width, calibration["height_cm"] / height
    record["calibration"] = {**calibration, "cm_per_pixel_x": sx, "cm_per_pixel_y": sy}
    for point in record["landmarks"].values():
        point["x_cm"], point["y_cm"] = point["x"] * sx, point["y"] * sy
    for measurement in record["measurements"].values():
        start = record["landmarks"][measurement["landmarks"]["start"]]
        end = record["landmarks"][measurement["landmarks"]["end"]]
        measurement["distance_cm"] = math.hypot((end["x"] - start["x"]) * sx, (end["y"] - start["y"]) * sy)
    return record
