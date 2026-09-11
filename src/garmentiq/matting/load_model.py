"""Loading a matting model onto a device."""
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
    Loads a matting model from a local checkpoint and prepares it for inference.

    This mirrors `garmentiq.segmentation.load_model`: the model class is instantiated with
    safely filtered configuration arguments, weights are read from a local `.pth` or
    `.safetensors` file, the model is moved to the requested device and set to evaluation
    mode, and common weight prefixes are stripped for compatibility.

    Matting Anything checkpoints are not loaded here because they pair a decoder with a
    separate SAM model; use `garmentiq.matting.model_definition.mam.load_mam` for those.

    Args:
        model_class (Type[nn.Module]): The uninstantiated model class, typically
                                       `VitMatteForImageMatting`.
        model_path (str): Local path to the checkpoint weights, ending in `.pth` or
                          `.safetensors`.
        model_args (dict, optional): Configuration arguments for initializing the model,
                                     e.g. `{"config": load_vitmatte_config(...)}`.
                                     Incompatible arguments are safely ignored.
                                     Default is None.
        device (Union[str, torch.device], optional): The device to load the model onto, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`. Hardware
                                                     acceleration is opt-in. Default is `"cpu"`.
        **kwargs: Additional arbitrary keyword arguments.

    Raises:
        ValueError: If the requested `device` is invalid or unavailable on this machine.
        Exception: If the weights cannot be loaded or the file format is unsupported.

    Returns:
        nn.Module: The loaded model in evaluation mode, placed on `device`.
    """
    model_args = model_args or {}

    if not isinstance(model_args, dict):
        if hasattr(model_args, "to_dict"):
            model_args = model_args.to_dict()
        elif hasattr(model_args, "__dict__"):
            model_args = vars(model_args)
        else:
            raise TypeError(
                "model_args must be a dictionary or a configuration object."
            )

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

    model = model_class(**filtered_args).to(device)

    if model_path.endswith(".safetensors"):
        state_dict = load_file(model_path, device=str(device))
    else:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

    new_state_dict = {
        k.removeprefix("module.").removeprefix("model."): v
        for k, v in state_dict.items()
    }

    load_state_dict_checked(model, new_state_dict, model_path)
    model.eval()

    return model
