"""The Grounding DINO text-to-region detector.

Loads the model fully offline from a local directory and converts a natural-language
phrase into bounding boxes. A dedicated loader is used because Grounding DINO ties
several decoder heads to one shared set of weights, which the generic loader would
leave randomly initialised.
"""
import os
import json
from typing import TYPE_CHECKING, Optional, Union

import torch
from PIL import Image

from garmentiq.utils.device import resolve_device, inputs_to_device

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from transformers import (
        GroundingDinoForObjectDetection,
        GroundingDinoConfig,
        GroundingDinoProcessor,
    )

# Transformers classes are resolved lazily so that importing GarmentIQ never
# depends on a specific transformers release. The helpful error is raised only
# when a grounding feature is actually used.
_LAZY_CLASSES = {
    "GroundingDinoForObjectDetection": "Text-prompted grounding",
    "GroundingDinoConfig": "Text-prompted grounding",
    "GroundingDinoProcessor": "Text-prompted grounding",
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
            f"{_LAZY_CLASSES.get(name, 'This feature')} requires a newer transformers "
            f"release; upgrade with `pip install -U transformers`."
        ) from e


def __getattr__(name):
    """Module-level lazy attribute access (PEP 562)."""
    if name in _LAZY_CLASSES:
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_grounding_config(model_dir: str):
    """
    Reads and loads a local Grounding DINO configuration for offline use.

    Grounding DINO turns a natural-language phrase into bounding boxes, which SAM 1 and
    SAM 2 accept natively as prompts. Its configuration and tokenizer files are not
    bundled with GarmentIQ because they belong to a separate upstream model, so download
    the model once and point `model_dir` at the resulting directory.

    Args:
        model_dir (str): Path to a local directory containing the Grounding DINO
                         `config.json`, e.g. a download of `IDEA-Research/grounding-dino-tiny`.

    Raises:
        FileNotFoundError: If `model_dir` is not a directory or does not contain `config.json`.
        ImportError: If the installed transformers release does not provide Grounding DINO.

    Returns:
        GroundingDinoConfig: The loaded configuration object, ready to be passed into
                             `GroundingDinoForObjectDetection`.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Provided model_dir '{model_dir}' is not an existing directory."
        )

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Offline config missing at {config_path}.")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return _resolve("GroundingDinoConfig").from_dict(config_dict)


def load_grounding_processor(model_dir: str):
    """
    Loads a local Grounding DINO processor for offline use.

    The processor pairs an image processor with a text tokenizer, so the whole local
    directory is required rather than a single JSON file.

    Args:
        model_dir (str): Path to a local directory containing the Grounding DINO processor
                         and tokenizer files.

    Raises:
        FileNotFoundError: If `model_dir` is not an existing directory.
        ImportError: If the installed transformers release does not provide Grounding DINO.

    Returns:
        GroundingDinoProcessor: The instantiated processor ready for image and text inputs.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Provided model_dir '{model_dir}' is not an existing directory."
        )

    return _resolve("GroundingDinoProcessor").from_pretrained(model_dir)


def load_grounding_model(
    model_dir: str,
    device: Union[str, torch.device] = "cpu",
):
    """
    Loads a Grounding DINO model from a local directory, fully offline.

    Grounding DINO ties several decoder heads to a shared set of weights, so its checkpoint
    stores only one copy of them. Constructing the model from a config and calling
    `load_state_dict` therefore leaves those tied tensors randomly initialised, which
    silently degrades grounding quality. `from_pretrained` performs the tying correctly,
    so it is used here rather than the generic loader used for other GarmentIQ models.

    Args:
        model_dir (str): Path to a local directory holding the Grounding DINO `config.json`,
                         weights, and tokenizer files, e.g. a download of
                         `IDEA-Research/grounding-dino-tiny`.
        device (Union[str, torch.device], optional): The device to load the model onto, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`.
                                                     Default is `"cpu"`.

    Raises:
        FileNotFoundError: If `model_dir` is not an existing directory.
        ValueError: If the requested `device` is invalid or unavailable on this machine.
        ImportError: If the installed transformers release does not provide Grounding DINO.

    Returns:
        GroundingDinoForObjectDetection: The loaded model in evaluation mode, on `device`.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Provided model_dir '{model_dir}' is not an existing directory."
        )

    device = resolve_device(device)
    model = _resolve("GroundingDinoForObjectDetection").from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    return model


def ground_text_to_boxes(
    model,
    processor,
    image: Image.Image,
    text: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.3,
    max_boxes: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
):
    """
    Converts a natural-language phrase into bounding boxes for an image.

    This is the bridge that gives SAM 1 and SAM 2 text-prompted segmentation. Neither model
    contains a text encoder, so the phrase is first grounded into boxes by Grounding DINO,
    and those boxes are then used as ordinary geometric prompts for SAM.

    Grounding DINO expects lowercase phrases terminated by a period; the text is normalised
    to that convention automatically.

    Args:
        model (GroundingDinoForObjectDetection): The loaded Grounding DINO model.
        processor (GroundingDinoProcessor): The matching processor with its tokenizer.
        image (PIL.Image.Image): The image to ground the phrase against.
        text (str): The natural-language prompt, e.g. `"a shirt"` or `"trousers"`.
        box_threshold (float, optional): Minimum box confidence to keep a detection.
                                         Default is 0.3.
        text_threshold (float, optional): Minimum text-matching score to keep a detection.
                                          Default is 0.3.
        max_boxes (int, optional): If given, keep at most this many highest-scoring boxes.
                                   Default is None, meaning keep all boxes above threshold.
        device (Union[str, torch.device], optional): The device to run grounding on, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`.
                                                     Default is `"cpu"`.

    Raises:
        ValueError: If `text` is empty, if the requested `device` is unavailable, or if no
                    region of the image matches the phrase at the given thresholds.

    Returns:
        list[list[float]]: Bounding boxes in `[x_min, y_min, x_max, y_max]` pixel coordinates,
                           ordered by descending confidence.
    """
    if not text or not text.strip():
        raise ValueError("A non-empty text prompt is required for grounding.")

    device = resolve_device(device)
    model = model.to(device)

    # Grounding DINO is trained on lowercase phrases ending with a period.
    caption = text.strip().lower()
    if not caption.endswith("."):
        caption = caption + "."

    inputs = inputs_to_device(
        processor(images=image, text=caption, return_tensors="pt"), device
    )

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()

    if boxes.numel() == 0:
        raise ValueError(
            f"Grounding model found no region matching text prompt {text!r} at "
            f"box_threshold={box_threshold} and text_threshold={text_threshold}. "
            f"Try lowering the thresholds or rephrasing the prompt."
        )

    order = torch.argsort(scores, descending=True)
    boxes = boxes[order]
    if max_boxes is not None:
        boxes = boxes[:max_boxes]

    return [[float(v) for v in box] for box in boxes]


__all__ = [
    "GroundingDinoForObjectDetection",
    "GroundingDinoConfig",
    "GroundingDinoProcessor",
    "load_grounding_config",
    "load_grounding_processor",
    "load_grounding_model",
    "ground_text_to_boxes",
]
