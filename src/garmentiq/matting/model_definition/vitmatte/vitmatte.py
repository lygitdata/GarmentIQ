"""ViTMatte loading and its bundled configurations.

Configurations and processors for the `small` and `base` variants ship with the
package, so only the weights are read from disk. The processor rescales the trimap
itself, so it must be handed the raw 0-255 trimap rather than a pre-normalised one.
"""
import os
import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from transformers import (
        VitMatteForImageMatting,
        VitMatteConfig,
        VitMatteImageProcessor,
    )

# ViTMatte variants whose configurations ship with GarmentIQ.
VITMATTE_MODELS = [
    "vitmatte-small-composition-1k",
    "vitmatte-base-composition-1k",
]

_LAZY_CLASSES = {
    "VitMatteForImageMatting": "ViTMatte",
    "VitMatteConfig": "ViTMatte",
    "VitMatteImageProcessor": "ViTMatte",
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


def _variant_dir(model_type: str, model_dir: Optional[str]) -> str:
    """Resolves the directory holding a variant's offline configuration files."""
    if model_dir is not None:
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Provided model_dir '{model_dir}' is not an existing directory."
            )
        return model_dir

    if model_type not in VITMATTE_MODELS:
        raise ValueError(
            f"Invalid model_type '{model_type}'. Choose from: {VITMATTE_MODELS}"
        )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, model_type)


def load_vitmatte_config(
    model_type: str = "vitmatte-small-composition-1k",
    model_dir: Optional[str] = None,
):
    """
    Reads and loads the bundled configuration for a specified ViTMatte variant.

    This function facilitates strictly offline initialization by locating the `config.json`
    associated with the chosen ViTMatte variant within the local package directory and
    converting it into a Hugging Face `VitMatteConfig`, entirely bypassing network requests.

    Args:
        model_type (str, optional): The identifier for the desired ViTMatte variant. Must be
                                    one of `["vitmatte-small-composition-1k",
                                    "vitmatte-base-composition-1k"]`.
                                    Default is `"vitmatte-small-composition-1k"`.
        model_dir (str, optional): Path to a local directory containing `config.json`, used
                                   to override the bundled configuration. Default is None.

    Raises:
        ValueError: If `model_type` is not a supported variant.
        FileNotFoundError: If the offline `config.json` cannot be located.
        ImportError: If the installed transformers release does not provide ViTMatte.

    Returns:
        VitMatteConfig: The loaded configuration, ready for `VitMatteForImageMatting`.
    """
    config_path = os.path.join(_variant_dir(model_type, model_dir), "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Offline config missing at {config_path}.")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return _resolve("VitMatteConfig").from_dict(config_dict)


def load_vitmatte_processor(
    model_type: str = "vitmatte-small-composition-1k",
    model_dir: Optional[str] = None,
):
    """
    Loads the offline image processor for a specified ViTMatte variant.

    The ViTMatte processor stacks the image and its trimap into a single four-channel
    tensor, which is the input the model expects.

    Args:
        model_type (str, optional): The identifier for the desired ViTMatte variant.
                                    Default is `"vitmatte-small-composition-1k"`.
        model_dir (str, optional): Path to a local directory containing
                                   `preprocessor_config.json`. Default is None.

    Raises:
        ValueError: If `model_type` is not a supported variant.
        FileNotFoundError: If the offline processor configuration cannot be located.
        ImportError: If the installed transformers release does not provide ViTMatte.

    Returns:
        VitMatteImageProcessor: The instantiated processor.
    """
    processor_path = os.path.join(
        _variant_dir(model_type, model_dir), "preprocessor_config.json"
    )

    if not os.path.exists(processor_path):
        raise FileNotFoundError(f"Offline processor config missing at {processor_path}.")

    with open(processor_path, "r") as f:
        processor_dict = json.load(f)

    processor_dict.pop("image_processor_type", None)
    return _resolve("VitMatteImageProcessor")(**processor_dict)


__all__ = [
    "VitMatteForImageMatting",
    "VitMatteConfig",
    "VitMatteImageProcessor",
    "load_vitmatte_config",
    "load_vitmatte_processor",
    "VITMATTE_MODELS",
]
