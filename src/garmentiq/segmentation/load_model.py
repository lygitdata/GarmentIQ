import torch
from transformers import AutoModelForImageSegmentation
import kornia


def load_model(
    pretrained_model: str,
    pretrained_model_args: dict = {"trust_remote_code": True},
    high_precision: bool = True,
):
    """
    Loads a pretrained image segmentation model and prepares it for inference.

    This function loads the model from a specified pretrained model checkpoint,
    moves the model to the appropriate device (GPU or CPU), and sets it to evaluation mode.
    Optionally, the model can be loaded in half-precision (FP16) for faster inference.

    Args:
        pretrained_model (str): The identifier of the pretrained model, e.g., from Hugging Face model hub.
        pretrained_model_args (dict, optional): Additional arguments for loading the pretrained model.
                                               Default includes `trust_remote_code` as True for trusting external code.
        high_precision (bool, optional): Flag indicating whether to use full precision (True) or half precision (False) for the model.
                                         Default is True (full precision).

    Raises:
        ValueError: If the model cannot be loaded or if the model type is incompatible with the task.

    Returns:
        AutoModelForImageSegmentation: The loaded and prepared model.
    """
    model = AutoModelForImageSegmentation.from_pretrained(
        pretrained_model, **pretrained_model_args
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    if not high_precision:
        model.half()

    return model
