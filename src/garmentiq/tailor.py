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
    """
    The `tailor` class acts as a central agent for the GarmentIQ pipeline,
    orchestrating garment measurement from classification to landmark derivation.

    It integrates functionalities from other modules (classification, segmentation, landmark)
    to provide a smooth end-to-end process for automated garment measurement from images.

    Attributes:
        input_dir (str): Directory containing input images.
        model_dir (str): Directory where models are stored.
        output_dir (str): Directory to save processed outputs.
        class_dict (dict): Dictionary defining garment classes and their properties.
        do_derive (bool): Flag to enable landmark derivation.
        do_refine (bool): Flag to enable landmark refinement.
        classification_model_path (str): Path to the classification model.
        classification_model_class (Type[nn.Module]): Class definition for the classification model.
        classification_model_args (Dict): Arguments for the classification model.
        segmentation_model_name (str): Name or path for the segmentation model.
        segmentation_model_args (Dict): Arguments for the segmentation model.
        landmark_detection_model_path (str): Path to the landmark detection model.
        landmark_detection_model_class (Type[nn.Module]): Class definition for the landmark detection model.
        landmark_detection_model_args (Dict): Arguments for the landmark detection model.
        refinement_args (Optional[Dict]): Arguments for landmark refinement.
        derivation_dict (Optional[Dict]): Dictionary for landmark derivation rules.
    """

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
        """
        Initializes the `tailor` agent with paths, model configurations, and processing flags.

        Args:
            input_dir (str): Path to the directory containing input images.
            model_dir (str): Path to the directory where all required models are stored.
            output_dir (str): Path to the directory where all processed outputs will be saved.
            class_dict (dict): A dictionary defining the garment classes, their predefined points,
                                index ranges, and instruction JSON file paths.
            do_derive (bool): If True, enables the landmark derivation step.
            do_refine (bool): If True, enables the landmark refinement step.
            classification_model_path (str): The filename or relative path to the classification model.
            classification_model_class (Type[nn.Module]): The Python class of the classification model.
            classification_model_args (Dict): A dictionary of arguments to initialize the classification model.
            segmentation_model_name (str): The name or path of the pretrained segmentation model.
            segmentation_model_args (Dict): A dictionary of arguments for the segmentation model.
            landmark_detection_model_path (str): The filename or relative path to the landmark detection model.
            landmark_detection_model_class (Type[nn.Module]): The Python class of the landmark detection model.
            landmark_detection_model_args (Dict): A dictionary of arguments for the landmark detection model.
            refinement_args (Optional[Dict]): Optional arguments for the refinement process,
                                              e.g., `window_size`, `ksize`, `sigmaX`. Defaults to None.
            derivation_dict (Optional[Dict]): A dictionary defining derivation rules for non-predefined landmarks.
                                               Required if `do_derive` is True.

        Raises:
            ValueError: If `do_derive` is True but `derivation_dict` is None.
        """
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
        """
        Prints a summary of the `tailor` agent's configuration, including directory paths,
        defined classes, processing options (refine, derive), and loaded models.
        """
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
        """
        Classifies a single garment image using the configured classification model.

        Args:
            image (str): The filename of the image to classify, located in `self.input_dir`.
            verbose (bool): If True, prints detailed classification output. Defaults to False.

        Returns:
            tuple:
                - label (str): The predicted class label of the garment.
                - probabilities (List[float]): A list of probabilities for each class.
        """
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
        """
        Segments a single garment image to extract its mask and optionally modifies the background color.

        Args:
            image (str): The filename of the image to segment, located in `self.input_dir`.

        Returns:
            tuple:
                - original_img (np.ndarray): The original image with the mask overlaid.
                - mask (np.ndarray): The binary segmentation mask.
                - bg_modified_img (np.ndarray, optional): The image with the background color changed,
                                                         returned only if `background_color` is specified
                                                         in `segmentation_model_args`.
        """
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
        """
        Detects predefined landmarks on a garment image based on its classified class.

        Args:
            class_name (str): The classified name of the garment.
            image (Union[str, np.ndarray]): The path to the image file or a NumPy array of the image.

        Returns:
            tuple:
                - coords (np.array): Detected landmark coordinates.
                - maxval (np.array): Confidence scores for detected landmarks.
                - detection_dict (dict): A dictionary containing detailed landmark detection data.
        """
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
        """
        Derives non-predefined landmark coordinates based on predefined landmarks and a mask.

        Args:
            class_name (str): The name of the garment class.
            detection_dict (dict): The dictionary containing detected landmarks.
            derivation_dict (dict): The dictionary defining derivation rules.
            landmark_coords (np.array): NumPy array of initial landmark coordinates.
            np_mask (np.array): NumPy array of the segmentation mask.

        Returns:
            tuple:
                - derived_coords (dict): A dictionary of the newly derived landmark coordinates.
                - updated_detection_dict (dict): The detection dictionary updated with derived landmarks.
        """
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
        """
        Refines detected landmark coordinates using a blurred segmentation mask.

        Args:
            class_name (str): The name of the garment class.
            detection_np (np.array): NumPy array of initial landmark predictions.
            detection_conf (np.array): NumPy array of confidence scores for each predicted landmark.
            detection_dict (dict): Dictionary containing landmark data for each class.
            mask (np.array): Grayscale mask image used to guide refinement.
            window_size (int, optional): Size of the window used in the refinement algorithm. Defaults to 5.
            ksize (tuple, optional): Kernel size for Gaussian blur. Must be odd integers. Defaults to (11, 11).
            sigmaX (float, optional): Gaussian kernel standard deviation in the X direction. Defaults to 0.0.

        Returns:
            tuple:
                - refined_detection_np (np.array): Array of the same shape as `detection_np` with refined coordinates.
                - detection_dict (dict): Updated detection dictionary with refined landmark coordinates.
        """
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
        """
        Executes the full garment measurement pipeline for all images in the input directory.
    
        This method processes each image through a multi-stage pipeline that includes garment classification, 
        segmentation, landmark detection, optional refinement, and measurement derivation. During classification, 
        the system identifies the type of garment (e.g., shirt, dress, pants). Segmentation follows, producing 
        binary or instance masks that separate the garment from the background. Landmark detection is then 
        performed to locate anatomical or garment-specific keypoints such as shoulders or waist positions. If 
        enabled, an optional refinement step applies post-processing or model-based corrections to improve the 
        accuracy of detected keypoints. Finally, the system calculates key garment dimensions - such as chest width, 
        waist width, and full length - based on the detected landmarks. In addition to this processing pipeline, 
        the method also manages data and visual output exports. For each input image, a cleaned JSON file is 
        generated containing the predicted garment class, landmark coordinates, and the resulting measurements. 
        Optionally, visual outputs such as segmentation masks and images annotated with landmarks and measurements 
        can be saved to assist in inspection or debugging.
    
        Args:
            save_segmentation_image (bool): If True, saves segmentation masks and background-modified images.
                                            Defaults to False.
            save_measurement_image (bool): If True, saves images overlaid with detected landmarks and measurements.
                                           Defaults to False.
    
        Returns:
            tuple:
                - metadata (pd.DataFrame): A DataFrame containing metadata for each processed image, such as:
                    - Original image path
                    - Paths to any saved segmentation or annotated images
                    - Class and measurement results
                - outputs (dict): A dictionary mapping image filenames to their detailed processing results, including:
                    - Predicted class
                    - Detected landmarks with coordinates and confidence scores
                    - Calculated measurements
                    - File paths to any saved images (if applicable)
    
        Example of exported JSON:
            ```
            {
                "cloth_3.jpg": {
                    "class": "vest dress",
                    "landmarks": {
                        "10": {
                            "conf": 0.7269417643547058,
                            "x": 611.0,
                            "y": 861.0
                        },
                        "16": {
                            "conf": 0.6769524812698364,
                            "x": 1226.0,
                            "y": 838.0
                        },
                        "17": {
                            "conf": 0.7472652196884155,
                            "x": 1213.0,
                            "y": 726.0
                        },
                        "18": {
                            "conf": 0.7360446453094482,
                            "x": 1238.0,
                            "y": 613.0
                        },
                        "2": {
                            "conf": 0.9256571531295776,
                            "x": 703.0,
                            "y": 264.0
                        },
                        "20": {
                            "x": 700.936,
                            "y": 2070.0
                        },
                        "8": {
                            "conf": 0.7129100561141968,
                            "x": 563.0,
                            "y": 613.0
                        },
                        "9": {
                            "conf": 0.8203497529029846,
                            "x": 598.0,
                            "y": 726.0
                        }
                    },
                    "measurements": {
                        "chest": {
                            "distance": 675.0,
                            "landmarks": {
                                "end": "18",
                                "start": "8"
                            }
                        },
                        "full length": {
                            "distance": 1806.0011794281863,
                            "landmarks": {
                                "end": "20",
                                "start": "2"
                            }
                        },
                        "hips": {
                            "distance": 615.4299310238331,
                            "landmarks": {
                                "end": "16",
                                "start": "10"
                            }
                        },
                        "waist": {
                            "distance": 615.0,
                            "landmarks": {
                                "end": "17",
                                "start": "9"
                            }
                        }
                    }
                }
            }
            ```
        """
        # Some helper variables
        use_bg_color = self.segmentation_model_args.get("background_color") is not None
        outputs = {}

        # Step 1: Create the output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.output_dir}/measurement_json").mkdir(parents=True, exist_ok=True)

        if save_segmentation_image and (
            use_bg_color or self.do_derive or self.do_refine
        ):
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
        if save_segmentation_image and (
            use_bg_color or self.do_derive or self.do_refine
        ):
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

        # Step 11: Save measurement image
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

        # Step 12: Save measurement json
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
                detection_dict=outputs[image]["detection_dict"],
            )

            # Export JSON
            utils.export_dict_to_json(
                data=final_dict,
                filename=f"{self.output_dir}/measurement_json/{transformed_name}_measurement.json",
            )

            metadata.at[
                idx, "measurement_json"
            ] = f"{self.output_dir}/measurement_json/{transformed_name}_measurement.json"

        # Step 13: Save metadata as a CSV
        metadata.to_csv(f"{self.output_dir}/metadata.csv", index=False)

        return metadata, outputs
