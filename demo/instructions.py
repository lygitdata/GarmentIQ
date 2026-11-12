"""Validate and persist the same measurement schemas used by the Python API."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import uuid

ROOT = Path(__file__).resolve().parent.parent
CATALOG = json.loads((Path(__file__).parent / "static" / "garments.json").read_text())
REFERENCE_KEYS = ("p1_id", "p2_id", "p3_id", "p4_id", "p5_id")


class RevisionConflict(ValueError):
    pass


def validate_instruction(category, instruction):
    if category not in CATALOG or not isinstance(instruction, dict) or set(instruction) != {category}:
        raise ValueError("Choose one supported garment category and supply only its instructions.")
    garment = instruction[category]
    if not isinstance(garment, dict):
        raise ValueError("Instructions must contain landmarks and measurements.")
    points, measurements = garment.get("landmarks"), garment.get("measurements")
    if not isinstance(points, dict) or not points or len(points) > 150:
        raise ValueError("Define between 1 and 150 landmarks.")
    if not isinstance(measurements, dict) or not measurements or len(measurements) > 200:
        raise ValueError("Define between 1 and 200 measurements.")
    base = CATALOG[category]["landmarks"]
    normalized = {category: {"landmarks": {}, "measurements": {}}}
    output = normalized[category]
    for key, point in points.items():
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or int(key) < 1 or not isinstance(point, dict):
            raise ValueError("Landmark IDs must be positive integers without leading zeros.")
        predefined = point.get("predefined")
        if type(predefined) is not bool or predefined != (key in base):
            raise ValueError(f"Landmark {key}: existing IDs must be detected; derived IDs must exceed {len(base)}.")
        description = point.get("description", base.get(key, {}).get("description", ""))
        if not isinstance(description, str) or not description.strip() or len(description) > 120:
            raise ValueError(f"Give landmark {key} a name of at most 120 characters.")
        entry = {"predefined": predefined, "description": description.strip()}
        for axis in ("x", "y"):
            value = point.get(axis, base.get(key, {}).get(axis, 0))
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Landmark {key}: diagram coordinates must be finite numbers.")
            entry[axis] = float(value)
        if not predefined:
            rule = point.get("derivation")
            required = {"function", "direction", *REFERENCE_KEYS}
            if not isinstance(rule, dict) or not required.issubset(rule) or set(rule) - required - {"intersection_mode"}:
                raise ValueError(f"Landmark {key}: provide a direction and all five reference landmarks.")
            if rule["function"] != "derive_keypoint_coord" or rule["direction"] not in ("parallel", "perpendicular"):
                raise ValueError(f"Landmark {key}: use the parallel or perpendicular contour-intersection rule.")
            mode = rule.get("intersection_mode", "nearest_edge")
            if mode not in ("exact", "nearest_edge"):
                raise ValueError(f"Landmark {key}: choose exact intersection or nearest edge.")
            entry["derivation"] = {"function": "derive_keypoint_coord", "direction": rule["direction"], "intersection_mode": mode}
            for ref in REFERENCE_KEYS:
                value = str(rule[ref])
                if value not in base:
                    raise ValueError(f"Landmark {key}: {ref} must refer to a detected landmark in this category.")
                entry["derivation"][ref] = value
            if str(rule["p2_id"]) == str(rule["p3_id"]) or str(rule["p4_id"]) == str(rule["p5_id"]):
                raise ValueError(f"Landmark {key}: each reference pair must contain two different points.")
        output["landmarks"][key] = entry
    # References can use any of HRNet's predefined points, even if the user only
    # selected their custom measurement endpoints in the editor.
    for point in list(output["landmarks"].values()):
        for ref in REFERENCE_KEYS if not point["predefined"] else ():
            key = point["derivation"][ref]
            output["landmarks"].setdefault(key, copy.deepcopy(base[key]))
    for name, measurement in measurements.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 120 or not isinstance(measurement, dict):
            raise ValueError("Each measurement needs a unique name of at most 120 characters.")
        endpoints = measurement.get("landmarks")
        if not isinstance(endpoints, dict) or set(endpoints) != {"start", "end"}:
            raise ValueError(f"Measurement {name}: choose a start and end point.")
        start, end = str(endpoints["start"]), str(endpoints["end"])
        if start == end or start not in output["landmarks"] or end not in output["landmarks"]:
            raise ValueError(f"Measurement {name}: choose two different, defined landmarks.")
        description = measurement.get("description", "")
        if not isinstance(description, str) or len(description) > 500:
            raise ValueError(f"Measurement {name}: description must be at most 500 characters.")
        clean_name = name.strip()
        if clean_name in output["measurements"]:
            raise ValueError("Measurement names must be unique.")
        output["measurements"][clean_name] = {"landmarks": {"start": start, "end": end}, "description": description}
    return normalized


class InstructionStore:
    def __init__(self, backup_dir, instruction_dir=None):
        self.directory = Path(instruction_dir or ROOT / "src" / "garmentiq" / "instruction").resolve()
        self.backup_dir = Path(backup_dir)

    def read(self, category):
        if category not in CATALOG:
            raise ValueError("Unknown garment category.")
        path = self.directory / f"{category}.json"
        raw = path.read_bytes()
        return {"category": category, "instruction": json.loads(raw),
                "revision": hashlib.sha256(raw).hexdigest(),
                "path": str(path), "diagram": CATALOG[category]}

    def save(self, category, instruction, revision):
        current = self.read(category)
        if revision != current["revision"]:
            raise RevisionConflict("Instructions changed since you opened them. Reload this category before saving.")
        normalized = validate_instruction(category, instruction)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backup = self.backup_dir / f"{category}-{uuid.uuid4().hex}.json"
        path = self.directory / f"{category}.json"
        backup.write_bytes(path.read_bytes())
        with tempfile.NamedTemporaryFile(mode="w", dir=self.directory, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump(normalized, file, indent=2, allow_nan=False)
            file.write("\n")
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return {**self.read(category), "backup": str(backup)}
