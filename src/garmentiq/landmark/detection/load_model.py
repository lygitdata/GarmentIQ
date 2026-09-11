"""Loading the HRNet landmark detection model onto a device."""
import torch
from typing import Callable, Type, Union
from garmentiq.utils.device import resolve_device
from garmentiq.utils.checkpoint import load_state_dict_checked


def load_model(
    model_path: str,
    model_class: Type[torch.nn.Module],
    device: Union[str, torch.device] = "cpu",
):
    """
    Load a PyTorch model from a checkpoint and prepare it for inference.

    This function initializes a model from the provided `model_class`, loads its weights from
    the given file path, moves it to the requested device, and sets it to evaluation mode.
    When multiple CUDA GPUs are available and a CUDA device is requested, the model is
    additionally wrapped with `DataParallel` for multi-GPU inference.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth or .pt file).
        model_class (Type[torch.nn.Module]): The class definition of the model to be instantiated.
                                           This must be a subclass of `torch.nn.Module`.
        device (Union[str, torch.device], optional): The device to load the model onto, e.g.
                                                     `"cpu"`, `"cuda"`, `"cuda:0"`, or `"mps"`.
                                                     Hardware acceleration is opt-in; pass it
                                                     explicitly to use a GPU or Apple Silicon.
                                                     Default is `"cpu"`.

    Raises:
        ValueError: If the requested `device` is invalid or unavailable on this machine.
        RuntimeError: If the model checkpoint cannot be loaded.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model, placed on `device`.
    """
    device = resolve_device(device)

    model = model_class
    load_state_dict_checked(
        model, torch.load(model_path, map_location=device), model_path
    )
    model = model.to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    return model
