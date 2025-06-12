import torch
from typing import Callable, Type


def load_model(model_path: str, model_class: Type[torch.nn.Module]):
    """
    Load a PyTorch model from a checkpoint and prepare it for inference.

    This function initializes a model from the provided `model_class`, loads its weights from
    the given file path, wraps it with `DataParallel` for multi-GPU support, moves it to the
    appropriate device (GPU if available, otherwise CPU), and sets it to evaluation mode.

    Args:
        model_path (str): Path to the saved model checkpoint (.pth or .pt file).
        model_class (Type[torch.nn.Module]): The class definition of the model to be instantiated.
                                           This must be a subclass of `torch.nn.Module`.

    Raises:
        RuntimeError: If the model checkpoint cannot be loaded.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(device)), strict=False
    )
    model = torch.nn.DataParallel(model)
    model.eval()

    return model
