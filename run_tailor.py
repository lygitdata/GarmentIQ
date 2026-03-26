import json
import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.landmark.detection.model_definition import PoseHighResolutionNet
from garmentiq.garment_classes import garment_classes
from garmentiq.landmark.derivation.derivation_dict import derivation_dict

base = r"E:\GarmentIQ"  # change if your folders are elsewhere
tailor = giq.tailor(
    input_dir=f"{base}\\test_image",
    model_dir=f"{base}\\models",
    output_dir=f"{base}\\output",
    class_dict=garment_classes,
    do_derive=True,
    derivation_dict=derivation_dict,
    do_refine=True,
    classification_model_path="tiny_vit_inditex_finetuned.pt",
    classification_model_class=tinyViT,
    classification_model_args={
        "num_classes": len(garment_classes.keys()),
        "img_size": (120, 184),
        "patch_size": 6,
        "resize_dim": (120, 184),
        "normalize_mean": [0.8047, 0.7808, 0.7769],
        "normalize_std": [0.2957, 0.3077, 0.3081],
    },
    segmentation_model_name="lygitdata/BiRefNet_garmentiq_backup",
    segmentation_model_args={
        "trust_remote_code": True,
        "resize_dim": (1024, 1024),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "high_precision": True,
        "background_color": [102, 255, 102],
    },
    landmark_detection_model_path="hrnet.pth",
    landmark_detection_model_class=PoseHighResolutionNet(),
    landmark_detection_model_args={
        "scale_std": 200.0,
        "resize_dim": [288, 384],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
)

tailor.summary()
metadata, outputs = tailor.measure(
    save_segmentation_image=True,
    save_measurement_image=True,
)
print(metadata)

for path in metadata.get("measurement_json", []):
    with open(path, "r", encoding="utf-8") as f:
        print(json.dumps(json.load(f), indent=2))