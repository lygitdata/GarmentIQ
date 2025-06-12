import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List


def load_model(model_path: str, model_class: Type[nn.Module], model_args: dict):
    """
    Loads a PyTorch model from a checkpoint and prepares it for inference.

    This function initializes a model from the provided `model_class`, loads its weights from
    the given file path, moves it to the appropriate device (GPU if available, otherwise CPU),
    and sets it to evaluation mode.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth or .pt file).
        model_class (Type[nn.Module]): The class definition of the model to be instantiated.
                                       This must be a subclass of `torch.nn.Module`.
        model_args (dict): A dictionary of arguments used to initialize the model class.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class(**model_args).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model
