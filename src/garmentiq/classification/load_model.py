"""Loading a trained classification model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List, Union
from garmentiq.utils.device import resolve_device
from garmentiq.utils.checkpoint import load_state_dict_checked


def load_model(
    model_path: str,
    model_class: Type[nn.Module],
    model_args: dict,
    device: Union[str, torch.device] = "cpu",
):
    """
    Loads a PyTorch model from a checkpoint and prepares it for inference.

    This function initializes a model from the provided `model_class`, loads its weights from
    the given file path, moves it to the requested device, and sets it to evaluation mode.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth or .pt file).
        model_class (Type[nn.Module]): The class definition of the model to be instantiated.
                                       This must be a subclass of `torch.nn.Module`.
        model_args (dict): A dictionary of arguments used to initialize the model class.
        device (Union[str, torch.device], optional): The device to load the model onto, e.g.
                                                     `"cpu"`, `"cuda"`, `"cuda:0"`, or `"mps"`.
                                                     Hardware acceleration is opt-in; pass it
                                                     explicitly to use a GPU or Apple Silicon.
                                                     Default is `"cpu"`.

    Raises:
        ValueError: If the requested `device` is invalid or unavailable on this machine.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model, placed on `device`.
    """
    device = resolve_device(device)

    model = model_class(**model_args).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    load_state_dict_checked(model, new_state_dict, model_path)
    model.eval()

    return model
