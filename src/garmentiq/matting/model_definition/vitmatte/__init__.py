# garmentiq/matting/model_definition/vitmatte/__init__.py
"""The ViTMatte image matting model."""
from .vitmatte import (
    load_vitmatte_config,
    load_vitmatte_processor,
    VITMATTE_MODELS,
)

_LAZY = {"VitMatteForImageMatting", "VitMatteConfig", "VitMatteImageProcessor"}


def __getattr__(name):
    """Defer transformers imports until a ViTMatte class is actually requested."""
    if name in _LAZY:
        from . import vitmatte

        return getattr(vitmatte, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VitMatteForImageMatting",
    "VitMatteConfig",
    "VitMatteImageProcessor",
    "load_vitmatte_config",
    "load_vitmatte_processor",
    "VITMATTE_MODELS",
]
