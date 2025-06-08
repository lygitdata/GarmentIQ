import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, List

def load_model(
    model_path: str,
    model_class: Type[nn.Module],
    model_args: dict
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model_class(**model_args).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model
