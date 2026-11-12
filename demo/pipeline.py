"""Adapt the public Python API to an isolated local web request."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .calibration import calibrate_record
from .instructions import ROOT


MODEL_URLS = {
    "tiny_vit_inditex_finetuned.pt": "https://huggingface.co/lygitdata/garmentiq/resolve/main/tiny_vit_inditex_finetuned.pt",
    "hrnet.pth": "https://huggingface.co/lygitdata/garmentiq/resolve/main/hrnet.pth",
    "model.safetensors": "https://huggingface.co/lygitdata/BiRefNet_garmentiq_backup/resolve/main/model.safetensors",
}


def runtime_status(model_dir):
    """Check setup without loading models or importing the ML stack."""
    missing_packages = [
        name for name in ("garmentiq", "torch", "transformers", "timm")
        if importlib.util.find_spec(name) is None
    ]
    missing_models = [
        name for name in MODEL_URLS
        if not (Path(model_dir) / name).is_file()
    ]
    return {
        "ready": not missing_packages and not missing_models,
        "missing_packages": missing_packages,
        "missing_models": missing_models,
    }


def instructions_for_run(class_dict, run_dir, derive):
    """Exclude measurements needing derived points when derivation is disabled.

    Bundled instructions contain illustrative coordinates. Leaving an underived
    point in the schema would accidentally measure from those template positions.
    """
    result = copy.deepcopy(class_dict)
    instruction_dir = Path(run_dir) / "instructions"
    instruction_dir.mkdir(parents=True, exist_ok=True)
    for name, info in result.items():
        data = json.loads(Path(info["instruction"]).read_text())
        garment = data[name]
        if not derive:
            garment["landmarks"] = {
                key: point for key, point in garment["landmarks"].items()
                if point.get("predefined", False)
            }
            garment["measurements"] = {
                key: measurement for key, measurement in garment["measurements"].items()
                if all(point in garment["landmarks"]
                       for point in measurement["landmarks"].values())
            }
        path = instruction_dir / f"{name}.json"
        path.write_text(json.dumps(data))
        info["instruction"] = str(path)
    return result


def render_measurement_image(base_path, output_path, record):
    """Use the actual detection background, with lines and calibrated labels."""
    with Image.open(base_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    radius = max(4, round(min(image.size) * 0.004))
    font = ImageFont.load_default(size=max(11, round(min(image.size) * 0.012)))
    for name, measurement in record["measurements"].items():
        a = record["landmarks"][measurement["landmarks"]["start"]]
        b = record["landmarks"][measurement["landmarks"]["end"]]
        line = (a["x"], a["y"], b["x"], b["y"])
        draw.line(line, fill="white", width=max(4, radius))
        draw.line(line, fill="#6b3fbc", width=max(2, radius // 2))
        value = f"{measurement['distance_cm']:.2f} cm" if "distance_cm" in measurement else f"{measurement['distance']:.1f} px"
        draw.text(((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2), f"{name}: {value}",
                  font=font, fill="#40205f", stroke_width=2, stroke_fill="white")
    for key, point in record["landmarks"].items():
        x, y = point["x"], point["y"]
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill="#2466c9" if point.get("predefined") else "#c26b00", outline="white", width=2)
        draw.text((x+radius+2, y-radius), key, font=font, fill="#172f28", stroke_width=2, stroke_fill="white")
    image.save(output_path)


class LocalPipeline:
    """Load weights once; the server serializes calls to this instance."""

    def __init__(self, model_dir, device="cpu", instruction_dir=None):
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        self.tailor = None
        self.instruction_dir = Path(instruction_dir or ROOT / "src" / "garmentiq" / "instruction")

    def __call__(self, run_dir, options):
        # Imports stay lazy so the UI and setup diagnostics work without weights.
        import garmentiq as giq
        from garmentiq.classification.model_definition import tinyViT
        from garmentiq.garment_classes import garment_classes
        from garmentiq.landmark.derivation.derivation_dict import derivation_dict
        from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
        from garmentiq.segmentation.model_definition.birefnet import (
            BiRefNet, load_birefnet_config,
        )

        run_dir = Path(run_dir)
        classes = copy.deepcopy(garment_classes)
        for name, info in classes.items():
            info["instruction"] = str(self.instruction_dir / f"{name}.json")
        class_dict = instructions_for_run(classes, run_dir, options["derive"])
        if self.tailor is None:
            self.tailor = giq.tailor(
                input_dir=str(run_dir / "input"),
                output_dir=str(run_dir / "output"),
                model_dir=str(self.model_dir),
                class_dict=class_dict,
                do_derive=options["derive"],
                do_refine=options["refine"],
                derivation_dict=derivation_dict,
                classification_model_path="tiny_vit_inditex_finetuned.pt",
                classification_model_class=tinyViT,
                classification_model_args={
                    "num_classes": len(garment_classes),
                    "img_size": (120, 184), "patch_size": 6,
                    "resize_dim": (120, 184),
                    "normalize_mean": [0.8047, 0.7808, 0.7769],
                    "normalize_std": [0.2957, 0.3077, 0.3081],
                },
                segmentation_model_path="model.safetensors",
                segmentation_model_class=BiRefNet,
                segmentation_model_args={
                    "model_config": load_birefnet_config(),
                    "resize_dim": (1024, 1024),
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                landmark_detection_model_path="hrnet.pth",
                # The landmark loader expects an instance, unlike the other loaders.
                landmark_detection_model_class=PoseHighResolutionNet(),
                landmark_detection_model_args={
                    "scale_std": 200.0, "resize_dim": [288, 384],
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                device=self.device,
            )

        agent = self.tailor
        agent.input_dir = str(run_dir / "input")
        agent.output_dir = str(run_dir / "output")
        agent.class_dict = class_dict
        agent.do_derive = options["derive"]
        agent.do_refine = options["refine"]
        agent.derivation_dict = derivation_dict if options["derive"] else None
        if options["background"] is None:
            agent.segmentation_model_args.pop("background_color", None)
        else:
            agent.segmentation_model_args["background_color"] = options["background"]
        metadata, _ = agent.measure(
            save_segmentation_image=True, save_measurement_image=True,
        )
        images = []
        for row in metadata.to_dict(orient="records"):
            record = json.loads(Path(row["measurement_json"]).read_text())[row["filename"]]
            artifacts = {"original": f"input/{row['filename']}"}
            for key in ("measurement_image", "mask_image", "bg_modified_image", "measurement_json"):
                value = row.get(key)
                if isinstance(value, str) and Path(value).is_file():
                    artifacts[key] = str(Path(value).relative_to(run_dir))
            with Image.open(run_dir / artifacts["original"]) as image:
                image_size = image.size
            index = int(row["filename"].split("_", 1)[0]) - 1
            calibrations = options.get("calibrations")
            calibrate_record(record, image_size, calibrations[index] if calibrations else None)
            base = artifacts.get("bg_modified_image", artifacts["original"])
            artifacts["measurement_base"] = base
            artifacts["instruction"] = str(Path(class_dict[record["class"]]["instruction"]).relative_to(run_dir))
            render_measurement_image(run_dir / base, row["measurement_image"], record)
            Path(row["measurement_json"]).write_text(json.dumps({row["filename"]: record}, indent=2, allow_nan=False))
            images.append({"filename": row["filename"], **record, "artifacts": artifacts})
        with (run_dir / "output" / "measurements.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["filename", "class", "measurement", "start", "end", "distance_px", "distance_cm", "image_width_cm", "image_height_cm"])
            for image in images:
                calibration = image.get("calibration") or {}
                for name, measurement in image["measurements"].items():
                    writer.writerow([image["filename"], image["class"], name,
                                     measurement["landmarks"]["start"], measurement["landmarks"]["end"],
                                     measurement["distance"], measurement.get("distance_cm", ""),
                                     calibration.get("width_cm", ""), calibration.get("height_cm", "")])
        return images
