"""The tailor agent, which runs the whole GarmentIQ pipeline.

Processes a folder of images end to end: classification, segmentation, optional alpha
matting, landmark detection, refinement, derivation, and measurement. Results are
written to an output directory and summarised in a metadata table.
"""
import os
from typing import List, Dict, Type, Any, Optional, Union
import torch
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
from . import matting
from . import utils


# Background colour used to composite the alpha matte for landmark detection when neither
# the matting nor the segmentation stage specifies one. Detection needs an RGB image, and a
# plain neutral background measurably helps the pose model compared with the raw photo.
DEFAULT_MATTE_DETECTION_BACKGROUND = (255, 255, 255)


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
        segmentation_model_path (str): Name or path for the segmentation model.
        segmentation_model_class (Type[nn.Module]): Class definition for the segmentation model.
        segmentation_model_args (Dict): Arguments for the segmentation model.
        landmark_detection_model_path (str): Path to the landmark detection model.
        landmark_detection_model_class (Type[nn.Module]): Class definition for the landmark detection model.
        landmark_detection_model_args (Dict): Arguments for the landmark detection model.
        refinement_args (Optional[Dict]): Arguments for landmark refinement.
        derivation_dict (Optional[Dict]): Dictionary for landmark derivation rules.
        device (torch.device): The device all models are loaded onto and run on.
        do_matte (bool): Flag to enable the alpha matting stage.
        matting_model_args (Optional[Dict]): Arguments for the matting model.
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
        segmentation_model_path: str,
        segmentation_model_class: Type[nn.Module],
        segmentation_model_args: Dict,
        landmark_detection_model_path: str,
        landmark_detection_model_class: Type[nn.Module],
        landmark_detection_model_args: Dict,
        refinement_args: Optional[Dict] = None,
        derivation_dict: Optional[Dict] = None,
        device: Union[str, torch.device] = "cpu",
        do_matte: bool = False,
        matting_model_path: Optional[str] = None,
        matting_model_class: Optional[Type[nn.Module]] = None,
        matting_model_args: Optional[Dict] = None,
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
            segmentation_model_path (str): The filename or relative path of the segmentation model.
            segmentation_model_class (Type[nn.Module]): The Python class of the segmentation model.
            segmentation_model_args (Dict): A dictionary of arguments for the segmentation model.
                                            For SAM this typically holds `model_config`, `processor`,
                                            and `prompt` (with `"points"`, `"labels"`, `"boxes"`,
                                            and/or `"text"`), plus optional `grounding_model` and
                                            `grounding_processor` when using a text prompt with
                                            SAM 1 or SAM 2. An optional `background_color` triggers
                                            background replacement.
            landmark_detection_model_path (str): The filename or relative path to the landmark detection model.
            landmark_detection_model_class (Type[nn.Module]): The Python class of the landmark detection model.
            landmark_detection_model_args (Dict): A dictionary of arguments for the landmark detection model.
            refinement_args (Optional[Dict]): Optional arguments for the refinement process,
                                              e.g., `window_size`, `ksize`, `sigmaX`. Defaults to None.
            derivation_dict (Optional[Dict]): A dictionary defining derivation rules for non-predefined landmarks.
                                               Required if `do_derive` is True.
            device (Union[str, torch.device], optional): The device that every model in the pipeline
                                                         is loaded onto and run on, e.g. `"cpu"`,
                                                         `"cuda"`, `"cuda:0"`, or `"mps"`. Hardware
                                                         acceleration is opt-in; pass it explicitly
                                                         to use a GPU or Apple Silicon.
                                                         Defaults to `"cpu"`.
            do_matte (bool, optional): If True, enables the alpha matting stage, which refines the
                                       hard segmentation mask into a soft alpha matte. Matting in
                                       the pipeline is deliberately built on top of segmentation:
                                       the segmentation mask supplies the trimap (ViTMatte) or the
                                       guidance mask (Matting Anything), so enabling this forces the
                                       segmentation stage to run. Defaults to False.
            matting_model_path (str, optional): The filename or relative path to the matting model
                                                weights, relative to `model_dir`. Required when
                                                `do_matte` is True and the model is loaded by
                                                GarmentIQ. Defaults to None.
            matting_model_class (Type[nn.Module], optional): The Python class of the matting model,
                                                             e.g. `VitMatteForImageMatting`.
                                                             Required when `do_matte` is True.
                                                             Defaults to None.
            matting_model_args (Dict, optional): Arguments for the matting model. For ViTMatte this
                                                 holds `model_config` (e.g.
                                                 `{"config": load_vitmatte_config(...)}`),
                                                 `processor`, and optional `trimap_args` and
                                                 `background_color`. Alternatively pass a
                                                 preconstructed model as `model`, which is how
                                                 Matting Anything is supplied since it pairs a
                                                 decoder with a SAM instance. Defaults to None.

        Raises:
            ValueError: If `do_derive` is True but `derivation_dict` is None, if `do_matte` is True
                        but no matting model is provided, or if the requested `device` is invalid
                        or unavailable on this machine.
        """
        # Device (resolved once and reused by every stage of the pipeline)
        self.device = utils.resolve_device(device)

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
            if k not in ("pretrained", "resize_dim", "normalize_mean", "normalize_std")
        }

        # Load the model using the filtered arguments
        self.classification_model = classification.load_model(
            model_path=f"{self.model_dir}/{self.classification_model_path}",
            model_class=self.classification_model_class,
            model_args=filtered_model_args,
            device=self.device,
        )

        # Segmentation model setup
        self.segmentation_model_path = segmentation_model_path
        self.segmentation_model_class = segmentation_model_class
        self.segmentation_model_args = segmentation_model_args
        self.segmentation_has_bg_color = "background_color" in segmentation_model_args
        self.segmentation_model = segmentation.load_model(
            model_path=f"{self.model_dir}/{self.segmentation_model_path}",
            model_class=self.segmentation_model_class,
            model_args=self.segmentation_model_args.get("model_config"),
            device=self.device,
        )

        # Landmark detection model setup
        self.landmark_detection_model_path = landmark_detection_model_path
        self.landmark_detection_model_class = landmark_detection_model_class
        self.landmark_detection_model_args = landmark_detection_model_args
        self.landmark_detection_model = landmark.detection.load_model(
            model_path=f"{self.model_dir}/{self.landmark_detection_model_path}",
            model_class=self.landmark_detection_model_class,
            device=self.device,
        )

        # Matting setup (optional, and always layered on top of segmentation)
        self.do_matte = do_matte
        self.matting_model_path = matting_model_path
        self.matting_model_class = matting_model_class
        self.matting_model_args = matting_model_args or {}
        self.matting_model = None

        if self.do_matte:
            # Matting refines a segmentation mask, so the pipeline cannot run it
            # without a working segmentation stage.
            if self.segmentation_model is None:
                raise ValueError(
                    "`do_matte=True` requires segmentation, because the segmentation mask "
                    "supplies the trimap (ViTMatte) or guidance mask (Matting Anything). "
                    "Configure the segmentation model, or set `do_matte=False`."
                )

            preloaded = self.matting_model_args.get("model")
            if preloaded is not None:
                # Matting Anything is assembled by the caller because it pairs a
                # decoder with an existing SAM instance.
                self.matting_model = preloaded.to(self.device)
            elif matting_model_class is not None and matting_model_path is not None:
                self.matting_model = matting.load_model(
                    model_class=self.matting_model_class,
                    model_path=f"{self.model_dir}/{self.matting_model_path}",
                    model_args=self.matting_model_args.get("model_config"),
                    device=self.device,
                )
            else:
                missing = []
                if matting_model_class is None:
                    missing.append("`matting_model_class`")
                if matting_model_path is None:
                    missing.append("`matting_model_path`")
                raise ValueError(
                    f"`do_matte=True` requires a matting model, but {' and '.join(missing)} "
                    f"{'was' if len(missing) == 1 else 'were'} not provided. Either pass "
                    f"`matting_model_class` and `matting_model_path` (for ViTMatte), or pass "
                    f"an already constructed model as `matting_model_args={{'model': ...}}` "
                    f"(for Matting Anything)."
                )

            # ViTMatte consumes a stacked image+trimap tensor built by its processor,
            # so a missing processor would only fail deep inside inference.
            model_names = {c.__name__ for c in type(self.matting_model).__mro__}
            is_mam = "MattingAnything" in model_names
            if not is_mam and self.matting_model_args.get("processor") is None:
                raise ValueError(
                    "`do_matte=True` with a trimap-based model such as ViTMatte requires a "
                    "processor. Pass it as "
                    "`matting_model_args={'processor': load_vitmatte_processor(...)}`."
                )
            if is_mam and not self.matting_model_args.get("prompt"):
                raise ValueError(
                    "`do_matte=True` with Matting Anything requires a prompt for its internal "
                    "SAM. Pass it as "
                    "`matting_model_args={'prompt': {'boxes': [[[x0, y0, x1, y1]]]}}`."
                )

    def summary(self):
        """
        Prints a summary of the `tailor` agent's configuration, including directory paths,
        defined classes, processing options (refine, derive, device), and loaded models.
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
        print(f"{'Do matte?:':25} {self.do_matte}")
        print(f"{'Device:':25} {self.device}")
        print()

        # Models
        print("MODELS".center(width, "-"))
        print(
            f"{'Classification Model:':25} {self.classification_model_class.__name__}"
        )
        print(f"{'Segmentation Model:':25} {self.segmentation_model_class.__name__}")
        print(f"{'  └─ Change BG color?:':25} {self.segmentation_has_bg_color}")
        print(
            f"{'Landmark Detection Model:':25} {self.landmark_detection_model_class.__class__.__name__}"
        )
        if self.do_matte:
            print(f"{'Matting Model:':25} {type(self.matting_model).__name__}")
            matte_bg = self.matting_model_args.get("background_color")
            print(f"{'  └─ Composite BG color?:':25} {matte_bg is not None}")
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
            device=self.device,
            verbose=verbose,
        )
        return label, probablities

    def segment(self, image: str):
        """
        Segments a single garment image to extract its mask and optionally modifies the background color.

        This method acts as an intelligent router for your segmentation arguments. It automatically 
        filters out initialization keys (e.g., `model_config`) and post-processing keys 
        (e.g., `background_color`) from `self.segmentation_model_args`. The remaining arguments 
        (such as `processor` and `prompt` for SAM or `resize_dim` for standard models such as BiRefNet) 
        are dynamically passed into the extraction pipeline.

        For Segment Anything models the prompt is supplied via the `prompt` dictionary, which accepts
        `"points"`, `"labels"`, `"boxes"`, and/or `"text"`. When a text prompt is used with SAM 1 or
        SAM 2, also provide `grounding_model` and `grounding_processor` in the segmentation arguments,
        because those families have no text encoder and need the phrase grounded into boxes first.

        Args:
            image (str): The filename of the image to segment, located in `self.input_dir`.

        Raises:
            ValueError: If a SAM model is configured without any prompt, or if a text prompt is used
                        with SAM 1 or SAM 2 without a grounding model.

        Returns:
            tuple:
                - original_img (np.ndarray): The original input image converted to a numpy array.
                - mask (np.ndarray): The extracted binary segmentation mask as a numpy array.
                - bg_modified_img (np.ndarray, optional): The image with the background color replaced. 
                                                          This third element is only returned if 
                                                          `background_color` is provided in the 
                                                          segmentation arguments.
        """
        # 1. Filter out initialization and post-processing arguments
        extraction_kwargs = {
            k: v for k, v in self.segmentation_model_args.items()
            if k not in ["model_config", "background_color"]
        }

        # 2. Extract using the unified function and unpacked kwargs
        original_img, mask = segmentation.extract(
            model=self.segmentation_model,
            image_path=f"{self.input_dir}/{image}",
            device=self.device,
            **extraction_kwargs
        )

        # 3. Handle optional background color modification
        background_color = self.segmentation_model_args.get("background_color")

        if background_color is None:
            return original_img, mask
        else:
            bg_modified_img = segmentation.change_background_color(
                image_np=original_img, mask_np=mask, background_color=background_color
            )
            return original_img, mask, bg_modified_img

    def matte(self, image: str, mask: np.ndarray):
        """
        Refines a segmentation mask into a soft alpha matte for a single image.

        Matting in the pipeline is intentionally built on top of segmentation rather than
        run standalone: the segmentation mask is what supplies the trimap for ViTMatte or
        the guidance mask for Matting Anything. Calling this without a mask therefore
        raises, which is why `do_matte=True` forces the segmentation stage to run.

        Standalone matting has no such requirement; call `garmentiq.matting.matte` directly
        with whatever image and trimap you already have.

        Args:
            image (str): The filename of the image to matte, located in `self.input_dir`.
            mask (numpy.ndarray): The segmentation mask produced for the same image.

        Raises:
            ValueError: If matting is not configured, or if `mask` is None.

        Returns:
            numpy.ndarray: The alpha matte as `uint8` in `[0, 255]`, matching the image size.
        """
        if not self.do_matte or self.matting_model is None:
            raise ValueError(
                "Matting is not configured on this tailor agent. Construct it with "
                "`do_matte=True` and a matting model."
            )
        if mask is None:
            raise ValueError(
                "Matting in the tailor pipeline requires a segmentation mask. Run the "
                "segmentation stage first, then pass its mask here."
            )

        _, alpha = matting.matte(
            model=self.matting_model,
            image_path=f"{self.input_dir}/{image}",
            processor=self.matting_model_args.get("processor"),
            mask=mask,
            prompt=self.matting_model_args.get("prompt"),
            trimap_args=self.matting_model_args.get("trimap_args"),
            device=self.device,
        )
        return alpha

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
            device=self.device,
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
        save_matting_image: bool = False,
    ):
        """
        Executes the full garment measurement pipeline for all images in the input directory.
    
        This method processes each image through a multi-stage pipeline that includes garment classification, 
        segmentation, landmark detection, optional refinement, and measurement derivation. During classification, 
        the system identifies the type of garment (e.g., shirt, dress, pants). Segmentation follows, producing 
        binary or instance masks that separate the garment from the background. When matting is
        enabled, the mask is then refined into a soft alpha matte, and the resulting alpha-composited
        image replaces the hard background-modified image as the input to landmark detection.
        Landmark detection is then 
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
            save_matting_image (bool): If True, saves the alpha mattes produced by the matting stage,
                                       and the alpha-composited images when a `background_color` is
                                       set in `matting_model_args`. Only has an effect when the agent
                                       was constructed with `do_matte=True`. Defaults to False.

        Raises:
            ValueError: If `save_matting_image` is True but the agent was not constructed with
                        `do_matte=True`, or if the matting stage cannot find a segmentation mask.
    
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
        use_matte_bg = (
            self.do_matte
            and self.matting_model_args.get("background_color") is not None
        )
        outputs = {}

        if save_matting_image and not self.do_matte:
            raise ValueError(
                "`save_matting_image=True` but this tailor agent was not constructed with "
                "`do_matte=True`, so there is no matting stage to save output from."
            )

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

        if save_matting_image and self.do_matte:
            Path(f"{self.output_dir}/matte_image").mkdir(parents=True, exist_ok=True)
            if use_matte_bg:
                Path(f"{self.output_dir}/matte_composite_image").mkdir(
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
            "matte_image" if save_matting_image and self.do_matte else None,
            "matte_composite_image"
            if save_matting_image and use_matte_bg
            else None,
            "measurement_image",
            "measurement_json",
        ]
        columns = [col for col in columns if col is not None]

        metadata = pd.DataFrame(columns=columns)
        metadata["filename"] = [img.name for img in image_files]

        # Step 4: Print start message and information
        print(f"Start measuring {len(metadata['filename'])} garment images ...")

        # Build the step list dynamically so every enabled stage is reported.
        steps = ["classification"]
        if use_bg_color or self.do_derive or self.do_refine or self.do_matte:
            steps.append("segmentation")
        if self.do_matte:
            steps.append("matting")
        steps.append("landmark detection")
        if self.do_refine:
            steps.append("landmark refinement")
        if self.do_derive:
            steps.append("landmark derivation")

        if len(steps) == 1:
            listed = steps[0]
        elif len(steps) == 2:
            listed = f"{steps[0]} and {steps[1]}"
        else:
            listed = ", ".join(steps[:-1]) + f", and {steps[-1]}"
        message = f"There are {len(steps)} measurement steps: {listed}."

        print(textwrap.fill(message, width=80))

        # Step 5: Classification
        for idx, image in tqdm(
            enumerate(metadata["filename"]), total=len(metadata), desc="Classification"
        ):
            label, _ = self.classify(image=image, verbose=False)
            metadata.at[idx, "class"] = label
            outputs[image] = {}

        # Step 6: Segmentation
        # Matting consumes the segmentation mask, so enabling it forces this stage.
        if use_bg_color or self.do_derive or self.do_refine or self.do_matte:
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

        # Step 6b: Matting (always after segmentation, using its mask as guidance)
        if self.do_matte:
            matte_bg_color = self.matting_model_args.get("background_color")
            # Landmark detection needs an RGB image, so the alpha matte is always
            # composited onto some background. The colour is chosen in order of
            # specificity: the matting colour, then the segmentation colour, then a
            # neutral default. Compositing always happens when matting is enabled, so
            # that the matte genuinely drives detection even when neither stage was
            # asked to replace the background in its saved output.
            detect_bg_color = matte_bg_color
            if detect_bg_color is None:
                detect_bg_color = self.segmentation_model_args.get("background_color")
            if detect_bg_color is None:
                detect_bg_color = DEFAULT_MATTE_DETECTION_BACKGROUND

            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Matting",
            ):
                if outputs[image].get("mask") is None:
                    raise ValueError(
                        f"Matting requires a segmentation mask, but none was produced for "
                        f"{image!r}. The segmentation stage must run before matting."
                    )
                alpha = self.matte(image=image, mask=outputs[image]["mask"])
                outputs[image]["alpha"] = alpha

                composited = matting.composite(
                    image_np=np.array(
                        Image.open(f"{self.input_dir}/{image}").convert("RGB")
                    ),
                    alpha_np=alpha,
                    background_color=detect_bg_color,
                )
                outputs[image]["matte_detect_image"] = composited
                # The composited image is only offered as a saved output when the user
                # explicitly asked for a matting background colour, keeping that output
                # optional in the same way segmentation's background replacement is.
                if matte_bg_color is not None:
                    outputs[image]["matte_composite"] = composited

        # Step 7: Landmark detection
        # Detection runs on a background-replaced image when one was requested, because
        # a clean background helps the pose model. When matting is enabled its softer,
        # more accurate composite is used in place of the hard segmentation composite.
        for idx, image in tqdm(
            enumerate(metadata["filename"]),
            total=len(metadata),
            desc="Landmark detection",
        ):
            label = metadata.loc[metadata["filename"] == image, "class"].values[0]

            if self.do_matte and outputs[image].get("matte_detect_image") is not None:
                detect_input = outputs[image]["matte_detect_image"]
            elif use_bg_color:
                detect_input = outputs[image]["bg_modified_image"]
            else:
                detect_input = image

            coords, maxvals, detection_dict = self.detect(
                class_name=label, image=detect_input
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

        # Step 10b: Save matting image
        if save_matting_image and self.do_matte:
            for idx, image in tqdm(
                enumerate(metadata["filename"]),
                total=len(metadata),
                desc="Save matting image",
            ):
                transformed_name = os.path.splitext(image)[0]
                alpha_path = (
                    f"{self.output_dir}/matte_image/{transformed_name}_matte.png"
                )
                Image.fromarray(outputs[image]["alpha"]).save(alpha_path)
                metadata.at[idx, "matte_image"] = alpha_path

                if use_matte_bg:
                    composite_path = (
                        f"{self.output_dir}/matte_composite_image/"
                        f"{transformed_name}_matte_composite.png"
                    )
                    Image.fromarray(outputs[image]["matte_composite"]).save(
                        composite_path
                    )
                    metadata.at[idx, "matte_composite_image"] = composite_path

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
