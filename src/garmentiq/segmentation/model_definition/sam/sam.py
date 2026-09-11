"""The SAM registry, loaders, and capability table.

Each supported variant is registered with the family it belongs to, the prompt types it
accepts, and the nesting depth its processor expects. SAM 3 is a special case: one
checkpoint contains both an open-vocabulary **detector**, prompted by text or boxes, and
a **tracker** carrying the classic SAM prompt encoder, which accepts points and labels.
`load_sam3_tracker` reassembles the tracker from the same weights file.
"""
import os
import json
import re
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from transformers import (
        SamModel,
        SamConfig,
        SamProcessor,
        Sam2Model,
        Sam2Config,
        Sam2Processor,
        Sam2ImageProcessor,
        Sam3Model,
        Sam3Config,
        Sam3Processor,
    )

# Which transformers release each class needs. Resolved lazily so that importing
# GarmentIQ works on older transformers, and only using a newer SAM family raises.
_LAZY_CLASSES = {
    "SamModel": "SAM 1",
    "SamConfig": "SAM 1",
    "SamProcessor": "SAM 1",
    "Sam2Model": "SAM 2",
    "Sam2Config": "SAM 2",
    "Sam2Processor": "SAM 2",
    "Sam2ImageProcessor": "SAM 2",
    "Sam3Model": "SAM 3",
    "Sam3Config": "SAM 3",
    "Sam3Processor": "SAM 3",
    "Sam3TrackerModel": "SAM 3",
    "Sam3TrackerConfig": "SAM 3",
    "Sam3TrackerProcessor": "SAM 3",
    "Sam3ImageProcessor": "SAM 3",
}


def _resolve(name: str):
    """Fetches a transformers class on demand with an actionable error message."""
    import transformers

    try:
        return getattr(transformers, name)
    except AttributeError as e:
        raise ImportError(
            f"'{name}' is not available in transformers "
            f"{getattr(transformers, '__version__', 'unknown')}. "
            f"{_LAZY_CLASSES.get(name, 'This feature')} support requires a newer "
            f"transformers release; upgrade with `pip install -U transformers`."
        ) from e


def __getattr__(name):
    """Module-level lazy attribute access (PEP 562)."""
    if name in _LAZY_CLASSES:
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Segment Anything 1 variants. Configs are bundled with the package.
SAM1_MODELS = ["sam-vit-b", "sam-vit-l", "sam-vit-h"]

# Segment Anything 2.1 variants. Configs are bundled with the package.
SAM2_MODELS = [
    "sam2.1-hiera-tiny",
    "sam2.1-hiera-small",
    "sam2.1-hiera-base-plus",
    "sam2.1-hiera-large",
]

# Segment Anything 3. Meta distributes SAM 3 under a gated licence, so the weights must be
# downloaded manually after accepting it, but the configuration files are bundled here so
# that no Hugging Face access is needed once you have the checkpoint.
SAM3_MODELS = ["sam3"]

# Backwards-compatible alias: prior releases only supported SAM 1.
VALID_SAM_MODELS = SAM1_MODELS

SAM_FAMILIES = {
    "sam1": {
        "models": SAM1_MODELS,
        "model_class": "SamModel",
        "config_class": "SamConfig",
        "processor_class": "SamProcessor",
        "bundled": True,
    },
    "sam2": {
        "models": SAM2_MODELS,
        "model_class": "Sam2Model",
        "config_class": "Sam2Config",
        "processor_class": "Sam2Processor",
        "bundled": True,
    },
    "sam3": {
        "models": SAM3_MODELS,
        "model_class": "Sam3Model",
        "config_class": "Sam3Config",
        "processor_class": "Sam3Processor",
        "bundled": True,
    },
}

# Which prompt types each family accepts natively.
#   points / labels / boxes : geometric prompts understood by the prompt encoder
#   text                    : natural-language prompt understood by the model itself
# SAM 1 and SAM 2 have no text encoder at all, so a text prompt for those
# families is resolved into boxes by a separate grounding model before inference.
SAM_CAPABILITIES = {
    "sam1": {"points": True, "labels": True, "boxes": True, "text": False},
    "sam2": {"points": True, "labels": True, "boxes": True, "text": False},
    "sam3": {"points": False, "labels": False, "boxes": True, "text": True},
    # SAM 3's tracker carries the SAM-style prompt encoder, so it accepts clicks and
    # boxes on still images. It has no text encoder of its own.
    "sam3-tracker": {"points": True, "labels": True, "boxes": True, "text": False},
}

# Nesting depth each family's processor expects for geometric prompts.
# SAM 2 strictly requires one extra level over SAM 1 and raises otherwise.
SAM_PROMPT_DEPTHS = {
    "sam1": {"points": 3, "labels": 2, "boxes": 3},
    "sam2": {"points": 4, "labels": 3, "boxes": 3},
    "sam3": {"boxes": 3},
    # Verified against Sam3TrackerProcessor: it rejects depth-3 points like SAM 2 does.
    "sam3-tracker": {"points": 4, "labels": 3, "boxes": 3},
}

ALL_SAM_MODELS = SAM1_MODELS + SAM2_MODELS + SAM3_MODELS


def sam_family(model_type: str) -> str:
    """
    Returns the Segment Anything family a variant belongs to.

    Args:
        model_type (str): A SAM variant identifier, e.g. `"sam-vit-b"`,
                          `"sam2.1-hiera-tiny"`, or `"sam3"`.

    Raises:
        ValueError: If `model_type` is not a recognised SAM variant.

    Returns:
        str: One of `"sam1"`, `"sam2"`, or `"sam3"`.
    """
    for family, spec in SAM_FAMILIES.items():
        if model_type in spec["models"]:
            return family
    raise ValueError(
        f"Invalid model_type '{model_type}'. Choose from: {ALL_SAM_MODELS}"
    )


def sam_capabilities(model_type: str) -> dict:
    """
    Reports which prompt types a SAM variant accepts natively.

    Args:
        model_type (str): A SAM variant identifier, e.g. `"sam-vit-b"` or `"sam3"`.

    Raises:
        ValueError: If `model_type` is not a recognised SAM variant.

    Returns:
        dict: Mapping of `"points"`, `"labels"`, `"boxes"`, and `"text"` to booleans.
              SAM 1 and SAM 2 report `text=False` because they contain no text encoder,
              and SAM 3 reports `points=False` because it accepts only text and box prompts.
    """
    return dict(SAM_CAPABILITIES[sam_family(model_type)])


def sam_model_class(model_type: str):
    """
    Returns the Hugging Face model class matching a SAM variant.

    Args:
        model_type (str): A SAM variant identifier, e.g. `"sam2.1-hiera-tiny"`.

    Raises:
        ValueError: If `model_type` is not a recognised SAM variant.

    Returns:
        type: `SamModel`, `Sam2Model`, or `Sam3Model`.
    """
    return _resolve(SAM_FAMILIES[sam_family(model_type)]["model_class"])


# Prefix carried by SAM 3 image-model weights inside the released video checkpoint.
SAM3_DETECTOR_PREFIX = "detector_model."

# Prefix carried by SAM 3 tracker weights inside the same checkpoint. The tracker shares
# the detector's vision encoder backbone, which is stored only once under the detector
# prefix, but it has its own FPN neck stored under `tracker_neck.`.
SAM3_TRACKER_PREFIX = "tracker_model."
SAM3_TRACKER_NECK_PREFIX = "tracker_neck."


def _unwrap_detector_config(config_dict: dict) -> dict:
    """
    Returns the image-level SAM 3 configuration from a released SAM 3 config.

    Meta distributes SAM 3 as a video model whose `config.json` describes `sam3_video`
    and nests the image model under `detector_config`. GarmentIQ segments still images,
    so the nested detector configuration is the one that matches `Sam3Model`. Configs
    that already describe the image model are returned unchanged.

    Args:
        config_dict (dict): The raw configuration dictionary read from `config.json`.

    Returns:
        dict: The configuration describing the image-level model.
    """
    detector = config_dict.get("detector_config")
    if isinstance(detector, dict) and detector.get("model_type") == "sam3":
        return detector
    return config_dict


def _variant_dir(model_type: str, model_dir: Optional[str]) -> str:
    """Resolves the directory holding a variant's offline configuration files."""
    if model_dir is not None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Provided model_dir '{model_dir}' is not an existing directory."
            )
        return model_dir

    family = sam_family(model_type)
    if not SAM_FAMILIES[family]["bundled"]:
        raise FileNotFoundError(
            f"'{model_type}' is distributed under a gated licence, so its configuration "
            f"is not bundled with GarmentIQ. Accept the licence, download the model files "
            f"once (config.json, processor/tokenizer files and the weights), then pass "
            f"model_dir='/path/to/{model_type}' to load it fully offline."
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, model_type)


def load_sam_config(model_type: str = "sam-vit-b", model_dir: Optional[str] = None):
    """
    Reads and loads the configuration for a specified Segment Anything Model variant.

    This function facilitates strictly offline initialization by dynamically locating the
    `config.json` file associated with the chosen SAM variant within the local package
    directory. It reads the JSON file securely and converts it into the Hugging Face
    configuration object matching the variant's family, entirely bypassing external
    network requests.

    Configurations for SAM 1 (`sam-vit-*`), SAM 2.1 (`sam2.1-hiera-*`) and SAM 3 (`sam3`)
    all ship with the package, so no Hugging Face access is needed. SAM 3 weights are still
    gated, so download the checkpoint once and point `model_path` at it when loading the
    model; `model_dir` here only overrides the bundled configuration.

    Released SAM 3 configurations describe the video model and nest the image model under
    `detector_config`. GarmentIQ segments still images, so that nested section is unwrapped
    automatically.

    Args:
        model_type (str, optional): The identifier for the desired SAM variant. Must be one
                                    of `["sam-vit-b", "sam-vit-l", "sam-vit-h",
                                    "sam2.1-hiera-tiny", "sam2.1-hiera-small",
                                    "sam2.1-hiera-base-plus", "sam2.1-hiera-large", "sam3"]`.
                                    Default is `"sam-vit-b"`.
        model_dir (str, optional): Path to a local directory containing the variant's
                                   `config.json`, used to override the bundled configuration.
                                   Default is None.

    Raises:
        ValueError: If the provided `model_type` is not within the supported variants list.
        FileNotFoundError: If the corresponding offline `config.json` file cannot be located.

    Returns:
        Union[SamConfig, Sam2Config, Sam3Config]: The loaded configuration object ready to be
                                                  passed into the matching SAM model class.
    """
    family = sam_family(model_type)
    config_path = os.path.join(_variant_dir(model_type, model_dir), "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Offline config missing at {config_path}.")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config_dict = _unwrap_detector_config(config_dict)

    return _resolve(SAM_FAMILIES[family]["config_class"]).from_dict(config_dict)


def _supports_backend_kwarg() -> bool:
    """
    Reports whether the installed transformers accepts the `backend` image-processor kwarg.

    `backend` replaced the deprecated `use_fast` flag in transformers 5.4.0. Older releases
    do not recognise it, so the two must be selected by version.
    """
    import transformers

    version = getattr(transformers, "__version__", "0")
    parts = re.findall(r"\d+", version)[:2]
    try:
        return tuple(int(p) for p in parts) >= (5, 4)
    except ValueError:  # pragma: no cover - unparseable version string
        return False


def load_sam_processor(
    model_type: str = "sam-vit-b",
    use_fast: Optional[bool] = None,
    model_dir: Optional[str] = None,
    backend: Optional[str] = None,
):
    """
    Loads the offline processor for a specified Segment Anything Model variant.

    This function instantiates the processor matching the variant's family by reading
    bundled preprocessor (and, for SAM 3, tokenizer) configuration files from the local
    variant directory. Loading from a local path ensures the package remains completely
    air-gapped. The image resizing backend can be selected to balance speed against
    strict backward compatibility.

    Args:
        model_type (str, optional): The identifier for the desired SAM variant.
                                    Default is `"sam-vit-b"`.
        use_fast (bool, optional): Deprecated alias for `backend`, kept for backwards
                                   compatibility. `True` maps to `"torchvision"` and `False`
                                   maps to `"pil"`. Only applies to SAM 1. Default is None.
        model_dir (str, optional): Path to a local directory containing the variant's
                                   processor files, used to override the bundled ones.
                                   Default is None.
        backend (str, optional): Image resizing backend, either `"pil"` or `"torchvision"`.
                                 `"torchvision"` is faster, while `"pil"` matches the
                                 historical GarmentIQ default. Only applies to SAM 1.
                                 Default is None, which resolves to `"pil"`.

    Raises:
        ValueError: If the provided `model_type` is not within the supported variants list,
                    or if `backend` is neither `"pil"` nor `"torchvision"`.
        FileNotFoundError: If the corresponding offline processor configuration cannot be
                           located.

    Returns:
        Union[SamProcessor, Sam2Processor, Sam3Processor]: The instantiated processor ready
                                                           for image and prompt transformations.
    """
    family = sam_family(model_type)
    variant_dir = _variant_dir(model_type, model_dir)

    if backend is None and use_fast is not None:
        backend = "torchvision" if use_fast else "pil"
    if backend is None:
        # Historical GarmentIQ default. Selected explicitly so results stay identical
        # across transformers releases, which have shifted their own default over time.
        backend = "pil"
    if backend not in ("pil", "torchvision"):
        raise ValueError(
            f"Invalid backend {backend!r}. Choose either 'pil' or 'torchvision'."
        )

    if family == "sam1":
        processor_path = os.path.join(variant_dir, "preprocessor_config.json")
        if not os.path.exists(processor_path):
            raise FileNotFoundError(
                f"Offline processor config missing at {processor_path}."
            )
        # transformers 5.4 replaced `use_fast` with `backend`; passing the deprecated
        # flag to a newer release emits a warning, so select the kwarg by version.
        backend_kwargs = (
            {"backend": backend}
            if _supports_backend_kwarg()
            else {"use_fast": backend == "torchvision"}
        )
        # Loading from a local directory disables Hugging Face network calls
        return _resolve("SamProcessor").from_pretrained(
            processor_path, **backend_kwargs
        )

    if family == "sam2":
        processor_path = os.path.join(variant_dir, "preprocessor_config.json")
        if not os.path.exists(processor_path):
            raise FileNotFoundError(
                f"Offline processor config missing at {processor_path}."
            )
        with open(processor_path, "r") as f:
            image_processor_dict = json.load(f)
        # The bundled config names the deprecated `*Fast` class; build the image
        # processor explicitly so loading stays offline and warning-free.
        image_processor_dict.pop("image_processor_type", None)
        image_processor = _resolve("Sam2ImageProcessor")(**image_processor_dict)
        return _resolve("Sam2Processor")(image_processor=image_processor)

    # SAM 3 bundles a tokenizer alongside the image processor, so the whole
    # directory is handed to the processor loader.
    return _resolve("Sam3Processor").from_pretrained(variant_dir)


def load_sam3_tracker_config(model_dir: Optional[str] = None):
    """
    Reads and loads the bundled configuration for SAM 3's point-promptable tracker.

    SAM 3's image model is an open-vocabulary detector prompted by text or boxes, and does
    not accept points. The release also contains a tracker, whose prompt encoder is the
    SAM-style one that understands points, boxes and masks. This function loads that
    tracker's configuration so still images can be segmented from a click.

    Args:
        model_dir (str, optional): Path to a local directory containing
                                   `tracker_config.json`, used to override the bundled
                                   configuration. Default is None.

    Raises:
        FileNotFoundError: If the offline `tracker_config.json` cannot be located.
        ImportError: If the installed transformers release does not provide SAM 3.

    Returns:
        Sam3TrackerConfig: The configuration for `Sam3TrackerModel`.
    """
    variant_dir = _variant_dir("sam3", model_dir)
    config_path = os.path.join(variant_dir, "tracker_config.json")

    if not os.path.exists(config_path):
        # A directly downloaded SAM 3 release nests it inside the video config.
        fallback = os.path.join(variant_dir, "config.json")
        if os.path.exists(fallback):
            with open(fallback, "r") as f:
                nested = json.load(f).get("tracker_config")
            if isinstance(nested, dict):
                return _resolve("Sam3TrackerConfig").from_dict(nested)
        raise FileNotFoundError(f"Offline tracker config missing at {config_path}.")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return _resolve("Sam3TrackerConfig").from_dict(config_dict)


def load_sam3_tracker_processor():
    """
    Builds the processor for SAM 3's point-promptable tracker.

    The tracker takes only geometric prompts, so unlike `load_sam_processor("sam3")` no
    tokenizer is involved. The image processor is constructed from its defaults, which
    keeps loading fully offline.

    Raises:
        ImportError: If the installed transformers release does not provide SAM 3.

    Returns:
        Sam3TrackerProcessor: The processor for `Sam3TrackerModel`.
    """
    image_processor = _resolve("Sam3ImageProcessor")()
    return _resolve("Sam3TrackerProcessor")(image_processor=image_processor)


def load_sam3_tracker(
    model_path: str,
    model_dir: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
):
    """
    Loads SAM 3's point-promptable tracker from a released SAM 3 checkpoint.

    SAM 3 ships as a single checkpoint holding both the detector and the tracker, and the
    two share one vision encoder that is stored only once under the detector's prefix. This
    function reassembles the tracker by pairing `tracker_model.*` weights with the shared
    `detector_model.vision_encoder.*` backbone, then verifies every expected tensor was
    found so a mismatched checkpoint fails loudly instead of yielding a random model.

    Use this when you want to prompt SAM 3 with points on a still image. For text or box
    prompts use the detector via `Sam3Model` and `load_sam_config("sam3")` instead.

    Args:
        model_path (str): Local path to the SAM 3 `model.safetensors` checkpoint.
        model_dir (str, optional): Directory holding `tracker_config.json`, used to
                                   override the bundled configuration. Default is None.
        device (Union[str, torch.device], optional): The device to load onto, e.g. `"cpu"`,
                                                     `"cuda"`, or `"mps"`. Default is `"cpu"`.

    Raises:
        FileNotFoundError: If `model_path` does not exist.
        ValueError: If the checkpoint does not contain the tensors the tracker expects, or
                    if the requested `device` is unavailable.
        ImportError: If the installed transformers release does not provide SAM 3.

    Returns:
        Sam3TrackerModel: The assembled tracker in evaluation mode, placed on `device`.
    """
    from safetensors.torch import load_file

    from garmentiq.utils.device import resolve_device

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SAM 3 checkpoint not found at {model_path}.")

    device = resolve_device(device)

    model = _resolve("Sam3TrackerModel")(load_sam3_tracker_config(model_dir))
    expected = model.state_dict()

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device="cpu")
    else:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    shared_prefix = SAM3_DETECTOR_PREFIX + "vision_encoder."
    neck_prefix = shared_prefix + "neck."
    assembled = {}
    for key, value in state_dict.items():
        if key.startswith(neck_prefix):
            # The detector has its own neck with the same layer names as the tracker's.
            # Skip it: the tracker's neck is stored separately under `tracker_neck.`,
            # and using the detector's would silently produce a garbage mask.
            continue
        if key.startswith(shared_prefix):
            # The vision encoder backbone is shared, and stored once under the detector.
            assembled[key[len(SAM3_DETECTOR_PREFIX) :]] = value
        elif key.startswith(SAM3_TRACKER_NECK_PREFIX):
            assembled[
                "vision_encoder.neck." + key[len(SAM3_TRACKER_NECK_PREFIX) :]
            ] = value
        elif key.startswith(SAM3_TRACKER_PREFIX):
            stripped = key[len(SAM3_TRACKER_PREFIX) :]
            if stripped in expected:
                assembled[stripped] = value

    missing = set(expected) - set(assembled)
    if missing:
        raise ValueError(
            f"SAM 3 tracker weights are incomplete: {len(missing)} of {len(expected)} "
            f"tensors were not found in {model_path!r} "
            f"(first missing: {sorted(missing)[:3]}). Check that the checkpoint is a "
            f"full SAM 3 release."
        )

    model.load_state_dict(assembled, strict=True)
    model = model.to(device)
    model.eval()
    return model


__all__ = [
    "SamModel",
    "Sam2Model",
    "Sam3Model",
    "load_sam_config",
    "load_sam_processor",
    "load_sam3_tracker",
    "load_sam3_tracker_config",
    "load_sam3_tracker_processor",
    "sam_family",
    "sam_capabilities",
    "sam_model_class",
    "SAM1_MODELS",
    "SAM2_MODELS",
    "SAM3_MODELS",
    "VALID_SAM_MODELS",
    "ALL_SAM_MODELS",
    "SAM_CAPABILITIES",
    "SAM_PROMPT_DEPTHS",
]
