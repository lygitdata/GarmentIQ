import os
from typing import List, Dict, Type, Any, Optional, Union
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import textwrap
from PIL import Image, ImageDraw, ImageFont
from . import classification
from . import segmentation
from . import landmark
from . import utils


class tailor:
    def __init__(
        self,
        input_dir: str,
        model_dir: str,
        output_dir: str,
        class_dict: dict,
        do_derive: bool,
        do_refine: bool,
        classification_model_path: str,
        classification_model_class: Type[nn.Module],
        classification_model_args: Dict,
        segmentation_model_name: str,
        segmentation_model_args: Dict,
        landmark_detection_model_path: str,
        landmark_detection_model_class: Type[nn.Module],
        landmark_detection_model_args: Dict,
        refinement_args: Optional[Dict] = None,
        derivation_dict: Optional[Dict] = None,
    ):
        # Directories
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.output_dir = output_dir

        # Classes
        self.class_dict = class_dict
        self.classes = sorted(list(class_dict.keys()))

        # Derivation
        self.do_derive = do_derive
        if self.do_derive:
            if derivation_dict is None:
                raise ValueError(
                    "`derivation_dict` must be provided if `do_derive=True`."
                )
            self.derivation_dict = derivation_dict
        else:
            self.derivation_dict = None

        # Refinement setup
        self.do_refine = do_refine
        self.do_refine = do_refine
        if self.do_refine:
            if refinement_args is None:
                self.refinement_args = {}
            self.refinement_args = refinement_args
        else:
            self.refinement_args = None

        # Classification model setup
        self.classification_model_path = classification_model_path
        self.classification_model_args = classification_model_args
        self.classification_model_class = classification_model_class
        filtered_model_args = {
            k: v
            for k, v in self.classification_model_args.items()
            if k not in ("resize_dim", "normalize_mean", "normalize_std")
        }

        # Load the model using the filtered arguments
        self.classification_model = classification.load_model(
            model_path=f"{self.model_dir}/{self.classification_model_path}",
            model_class=self.classification_model_class,
            model_args=filtered_model_args,
        )

        # Segmentation model setup
        self.segmentation_model_name = segmentation_model_name
        self.segmentation_model_args = segmentation_model_args
        self.segmentation_has_bg_color = "background_color" in segmentation_model_args
        self.segmentation_model = segmentation.load_model(
            pretrained_model=self.segmentation_model_name,
            pretrained_model_args={
                "trust_remote_code": segmentation_model_args["trust_remote_code"]
            },
            high_precision=segmentation_model_args["high_precision"],
        )

        # Landmark detection model setup
        self.landmark_detection_model_path = landmark_detection_model_path
        self.landmark_detection_model_class = landmark_detection_model_class
        self.landmark_detection_model_args = landmark_detection_model_args
        self.landmark_detection_model = landmark.detection.load_model(
            model_path=f"{self.model_dir}/{self.landmark_detection_model_path}",
            model_class=self.landmark_detection_model_class,
        )

    def summary(self):
        width = 80
        sep = "=" * width

        print(sep)
        print("TAILOR AGENT SUMMARY".center(width))
        print(sep)

        # Directories
        print("DIRECTORY PATHS".center(width, "-"))
        print(f"{'Input directory:':25} {self.input_dir}")
        print(f"{'Model directory:':25} {self.model_dir}")
        print(f"{'Output directory:':25} {self.output_dir}")
        print()

        # Classes
        print("CLASSES".center(width, "-"))
        print(f"{'Class Index':<11} | Class Name")
        print(f"{'-'*11} | {'-'*66}")
        for i, cls in enumerate(self.classes):
            print(f"{i:<11} | {cls}")
        print()

        # Flags
        print("OPTIONS".center(width, "-"))
        print(f"{'Do refine?:':25} {self.do_refine}")
        print(f"{'Do derive?:':25} {self.do_derive}")
        print()

        # Models
        print("MODELS".center(width, "-"))
        print(
            f"{'Classification Model:':25} {self.classification_model_class.__name__}"
        )
        print(f"{'Segmentation Model:':25} {self.segmentation_model_name}")
        print(f"{'  └─ Change BG color?:':25} {self.segmentation_has_bg_color}")
        print(
            f"{'Landmark Detection Model:':25} {self.landmark_detection_model_class.__class__.__name__}"
        )
        print(sep)

    def classify(self, image: str, verbose=False):
        label, probablities = classification.predict(
            model=self.classification_model,
            image_path=f"{self.input_dir}/{image}",
            classes=self.classes,
            resize_dim=self.classification_model_args.get("resize_dim"),
            normalize_mean=self.classification_model_args.get("normalize_mean"),
            normalize_std=self.classification_model_args.get("normalize_std"),
            verbose=verbose,
        )
        return label, probablities

    def segment(self, image: str):
        original_img, mask = segmentation.extract(
            model=self.segmentation_model,
            image_path=f"{self.input_dir}/{image}",
            resize_dim=self.segmentation_model_args.get("resize_dim"),
            normalize_mean=self.segmentation_model_args.get("normalize_mean"),
            normalize_std=self.segmentation_model_args.get("normalize_std"),
            high_precision=self.segmentation_model_args.get("high_precision"),
        )

        background_color = self.segmentation_model_args.get("background_color")

        if background_color is None:
            return original_img, mask
        else:
            bg_modified_img = segmentation.change_background_color(
                image_np=original_img, mask_np=mask, background_color=background_color
            )
            return original_img, mask, bg_modified_img

    def detect(self, class_name: str, image: Union[str, np.ndarray]):

        if isinstance(image, str):
            image = f"{self.input_dir}/{image}"

        coords, maxval, detection_dict = landmark.detect(
            class_name=class_name,
            class_dict=self.class_dict,
            image_path=image,
            model=self.landmark_detection_model,
            scale_std=self.landmark_detection_model_args.get("scale_std"),
            resize_dim=self.landmark_detection_model_args.get("resize_dim"),
            normalize_mean=self.landmark_detection_model_args.get("normalize_mean"),
            normalize_std=self.landmark_detection_model_args.get("normalize_std"),
        )
        return coords, maxval, detection_dict

    def derive(
        self,
        class_name: str,
        detection_dict: dict,
        derivation_dict: dict,
        landmark_coords: np.array,
        np_mask: np.array,
    ):
        derived_coords, updated_detection_dict = landmark.derive(
            class_name=class_name,
            detection_dict=detection_dict,
            derivation_dict=derivation_dict,
            landmark_coords=landmark_coords,
            np_mask=np_mask,
        )
        return derived_coords, updated_detection_dict

    def refine(
        self,
        class_name: str,
        detection_np: np.array,
        detection_conf: np.array,
        detection_dict: dict,
        mask: np.array,
        window_size: int = 5,
        ksize: tuple = (11, 11),
        sigmaX: float = 0.0,
    ):
        if self.refinement_args:
            if self.refinement_args.get("window_size") is not None:
                window_size = self.refinement_args["window_size"]
            if self.refinement_args.get("ksize") is not None:
                ksize = self.refinement_args["ksize"]
            if self.refinement_args.get("sigmaX") is not None:
                sigmaX = self.refinement_args["sigmaX"]

        refined_detection_np, refined_detection_dict = landmark.refine(
            class_name=class_name,
            detection_np=detection_np,
            detection_conf=detection_conf,
            detection_dict=detection_dict,
            mask=mask,
            window_size=window_size,
            ksize=ksize,
            sigmaX=sigmaX,
        )

        return refined_detection_np, refined_detection_dict

    def measure(
        self,
        save_segmentation_image: bool = False,
        save_measurement_image: bool = False,
    ):
        # Some helper variables
        use_bg_color = self.segmentation_model_args.get("background_color") is not None
        outputs = {}

        # Step 1: Create the output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/measurement_json").mkdir(parents=True, exist_ok=True)

        if save_segmentation_image and (use_bg_color or self.do_derive or self.do_refine):
            Path(f"{self.output_dir}/mask_image").mkdir(parents=True, exist_ok=True)
            if use_bg_color:
                Path(f"{self.output_dir}/bg_modified_image").mkdir(
                    parents=True, exist_ok=True
                )

        if save_measurement_image:
            Path(f"{self.output_dir}/measurement_image").mkdir(
                parents=True, exist_ok=True
            )

        # Step 2: Collect image filenames from input_dir
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        input_path = Path(self.input_dir)

        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(ext))

        # Step 3: Determine column structure
        columns = [
            "filename",
            "class",
            "mask_image" if use_bg_color or self.do_derive or self.do_refine else None,
            "bg_modified_image" if use_bg_color else None,
            "measurement_image",
            "measurement_json",
        ]
        columns = [col for col in columns if col is not None]

        metadata = pd.DataFrame(columns=columns)
        metadata["filename"] = [img.name for img in image_files]

        # Step 4: Print start message and information
        print(f"Start measuring {len(metadata['filename'])} garment images ...")

        if self.do_derive and self.do_refine:
            message = (
                "There are 5 measurement steps: classification, segmentation, "
                "landmark detection, landmark refinement, and landmark derivation."
            )
        elif self.do_derive:
            message = (
                "There are 4 measurement steps: classification, segmentation, "
                "landmark detection, and landmark derivation."
            )
        elif self.do_refine:
            message = (
                "There are 4 measurement steps: classification, segmentation, "
                "landmark detection, and landmark refinement."
            )
        elif use_bg_color:
            message = (
                "There are 3 measurement steps: classification, segmentation, "
                "and landmark detection."
            )
        else:
            message = (
                "There are 2 measurement steps: classification and landmark detection."
            )

        print(textwrap.fill(message, width=80))

        # Step 5: Classification
        for idx, image in tqdm(
            enumerate(metadata["filename"]), total=len(metadata), desc="Classification"
        ):
            label, _ = self.classify(image=image, verbose=False)
            metadata.at[idx, "class"] = label
            outputs[image] = {}

        # Step 6: Segmentation
        if use_bg_color or (self.do_derive or self.do_refine):
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Segmentation",
            ):
                if use_bg_color:
                    original_img, mask, bg_modified_image = self.segment(image=image)
                    outputs[image] = {
                        "mask": mask,
                        "bg_modified_image": bg_modified_image,
                    }
                else:
                    original_img, mask = self.segment(image=image)
                    outputs[image] = {
                        "mask": mask,
                    }

        # Step 7: Landmark detection
        for idx, image in tqdm(
            enumerate(metadata["filename"]),
            total=len(metadata),
            desc="Landmark detection",
        ):
            label = metadata.loc[metadata["filename"] == image, "class"].values[0]
            if use_bg_color:
                coords, maxvals, detection_dict = self.detect(
                    class_name=label, image=outputs[image]["bg_modified_image"]
                )
                outputs[image]["detection_dict"] = detection_dict
                if self.do_derive or self.do_refine:
                    outputs[image]["coords"] = coords
                    outputs[image]["maxvals"] = maxvals
            else:
                coords, maxvals, detection_dict = self.detect(
                    class_name=label, image=image
                )
                outputs[image]["detection_dict"] = detection_dict
                if self.do_derive or self.do_refine:
                    outputs[image]["coords"] = coords
                    outputs[image]["maxvals"] = maxvals

        # Step 8: Landmark refinement
        if self.do_refine:
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Landmark refinement",
            ):
                label = metadata.loc[metadata["filename"] == image, "class"].values[0]
                updated_coords, updated_detection_dict = self.refine(
                    class_name=label,
                    detection_np=outputs[image]["coords"],
                    detection_conf=outputs[image]["maxvals"],
                    detection_dict=outputs[image]["detection_dict"],
                    mask=outputs[image]["mask"],
                )
                outputs[image]["coords"] = updated_coords
                outputs[image]["detection_dict"] = updated_detection_dict

        # Step 9: Landmark derivation
        if self.do_derive:
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Landmark derivation",
            ):
                label = metadata.loc[metadata["filename"] == image, "class"].values[0]
                derived_coords, updated_detection_dict = self.derive(
                    class_name=label,
                    detection_dict=outputs[image]["detection_dict"],
                    derivation_dict=self.derivation_dict,
                    landmark_coords=outputs[image]["coords"],
                    np_mask=outputs[image]["mask"],
                )
                outputs[image]["detection_dict"] = updated_detection_dict

        # Step 10: Save segmentation image
        if save_segmentation_image and (use_bg_color or self.do_derive or self.do_refine):
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Save segmentation image",
            ):
                transformed_name = os.path.splitext(image)[0]
                Image.fromarray(outputs[image]["mask"]).save(
                    f"{self.output_dir}/mask_image/{transformed_name}_mask.png"
                )
                metadata.at[
                    idx, "mask_image"
                ] = f"{self.output_dir}/mask_image/{transformed_name}_mask.png"
                if use_bg_color:
                    Image.fromarray(outputs[image]["bg_modified_image"]).save(
                        f"{self.output_dir}/bg_modified_image/{transformed_name}_bg_modified.png"
                    )
                    metadata.at[
                        idx, "bg_modified_image"
                    ] = f"{self.output_dir}/bg_modified_image/{transformed_name}_bg_modified.png"

        # Step 10: Save measurement image
        if save_measurement_image:
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Save measurement image",
            ):
                label = metadata.loc[metadata["filename"] == image, "class"].values[0]
                transformed_name = os.path.splitext(image)[0]

                image_to_save = Image.open(f"{self.input_dir}/{image}").convert("RGB")
                draw = ImageDraw.Draw(image_to_save)
                font = ImageFont.load_default()
                landmarks = outputs[image]["detection_dict"][label]["landmarks"]

                for lm_id, lm_data in landmarks.items():
                    x, y = lm_data["x"], lm_data["y"]
                    radius = 5
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius), fill="green"
                    )
                    draw.text((x + 8, y - 8), lm_id, fill="green", font=font)

                image_to_save.save(
                    f"{self.output_dir}/measurement_image/{transformed_name}_measurement.png"
                )
                metadata.at[
                    idx, "measurement_image"
                ] = f"{self.output_dir}/measurement_image/{transformed_name}_measurement.png"

        # Step 11: Save measurement json
        for idx, image in tqdm(
            enumerate(metadata["filename"]),
            total=len(metadata),
            desc="Save measurement json",
        ):
            label = metadata.loc[metadata["filename"] == image, "class"].values[0]
            transformed_name = os.path.splitext(image)[0]

            # Clean the detection dictionary
            final_dict = utils.clean_detection_dict(
                class_name=label, 
                image_name=image, 
                detection_dict=outputs[image]["detection_dict"]
            )

            # Export JSON
            utils.export_dict_to_json(
                data=final_dict,
                filename=f"{self.output_dir}/measurement_json/{transformed_name}_measurement.json",
            )

            metadata.at[
                idx, "measurement_json"
            ] = f"{self.output_dir}/measurement_json/{transformed_name}_measurement.json"

        # Step 12: Save metadata as a CSV
        metadata.to_csv(f"{self.output_dir}/metadata.csv", index=False)

        return metadata, outputs
