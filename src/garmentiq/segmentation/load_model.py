"""Loading a segmentation model onto a device."""
import inspect
import torch
import torch.nn as nn
from typing import Type, Union
from safetensors.torch import load_file
from garmentiq.utils.device import resolve_device
from garmentiq.utils.checkpoint import load_state_dict_checked


def load_model(
    model_class: Type[nn.Module],
    model_path: str,
    model_args: dict = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs,
):
    """
    Loads a PyTorch model from a local checkpoint and prepares it for inference.

    This function instantiates the provided model class using safely filtered configuration
    arguments, loads the weights from a local `.pth` or `.safetensors` file, moves the model
    to the requested device, and sets it to evaluation mode. It automatically
    strips common weight prefixes (e.g., "module.", "model.") to ensure compatibility.

    Args:
        model_class (Type[nn.Module]): The uninstantiated PyTorch model class to be used.
        model_path (str): The local file path to the model checkpoint weights, typically
                          ending in `.pth` or `.safetensors`.
        model_args (dict, optional): A dictionary of configuration arguments for initializing
                                     the model. Incompatible arguments are safely ignored.
                                     Default is None.
        device (Union[str, torch.device], optional): The device to load the model onto, e.g.
                                                     `"cpu"`, `"cuda"`, `"cuda:0"`, or `"mps"`.
                                                     Hardware acceleration is opt-in; pass it
                                                     explicitly to use a GPU or Apple Silicon.
                                                     Default is `"cpu"`.
        **kwargs: Additional arbitrary keyword arguments.

    Raises:
        ValueError: If the requested `device` is invalid or unavailable on this machine.
        Exception: If the model weights cannot be loaded from the specified local path or if
                   the file format is unsupported.

    Returns:
        nn.Module: The loaded and prepared PyTorch model instance, placed on `device`.
    """
    model_args = model_args or {}
    
    # --- THE SMART CONVERTER ---
    # If the user passes a Config object instead of a dictionary, safely convert it.
    if not isinstance(model_args, dict):
        if hasattr(model_args, "to_dict"):
            model_args = model_args.to_dict()  # Hugging Face standard
        elif hasattr(model_args, "__dict__"):
            model_args = vars(model_args)      # Standard Python objects
        else:
            raise TypeError("model_args must be a dictionary or a configuration object.")
    # ---------------------------

    device = resolve_device(device)

    sig = inspect.signature(model_class.__init__)
    valid_params = set(sig.parameters.keys())

    has_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if not has_kwargs:
        filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
    else:
        filtered_args = model_args

    # ... [Keep the rest of your loading logic exactly the same] ...
    model = model_class(**filtered_args).to(device)

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device=str(device))
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # SAM 3 is published as a video checkpoint that nests the image ("detector") model
    # alongside tracker weights. Keep only the detector half when that layout is seen,
    # otherwise the image model would silently load nothing.
    from garmentiq.segmentation.model_definition.sam.sam import SAM3_DETECTOR_PREFIX

    if any(k.startswith(SAM3_DETECTOR_PREFIX) for k in state_dict):
        state_dict = {
            k[len(SAM3_DETECTOR_PREFIX) :]: v
            for k, v in state_dict.items()
            if k.startswith(SAM3_DETECTOR_PREFIX)
        }

    new_state_dict = {
        k.removeprefix("module.").removeprefix("model."): v
        for k, v in state_dict.items()
    }

    load_state_dict_checked(model, new_state_dict, model_path)
    model.eval()

    return model
