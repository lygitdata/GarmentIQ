"""Checkpoint loading with a mismatch guard.

Weights are loaded leniently so that partial checkpoints keep working, but a checkpoint
that does not match the model class would otherwise load nothing and leave a silently
random model. This module warns when few or none of the expected tensors are found.
"""
import warnings


def load_state_dict_checked(model, state_dict, model_path: str = ""):
    """
    Loads weights into a model and warns when the checkpoint does not match it.

    GarmentIQ loads every checkpoint with `strict=False`, because legitimate checkpoints
    routinely carry extra tensors (a bundled tracker, an optimiser state, a frozen
    backbone) or omit a head that is re-initialised. The cost of that leniency is that a
    checkpoint meant for a different architecture loads *nothing* and silently yields a
    randomly initialised model, which then produces plausible-looking but meaningless
    output.

    This helper keeps the lenient behaviour but makes that failure visible: if no tensor
    in the checkpoint matches the model, or almost none do, a warning is raised naming the
    file. It never blocks a load, so existing code keeps working.

    Args:
        model (torch.nn.Module): The model to load weights into.
        state_dict (dict): The checkpoint tensors, already stripped of any prefixes.
        model_path (str, optional): Path to the checkpoint, used in the warning message.
                                    Default is "".

    Returns:
        torch.nn.modules.module._IncompatibleKeys: The result of `load_state_dict`, listing
                                                   missing and unexpected keys.
    """
    expected = set(model.state_dict().keys())
    provided = set(state_dict.keys())
    matched = expected & provided

    where = f" from {model_path!r}" if model_path else ""

    if expected and not matched:
        warnings.warn(
            f"No weights{where} matched {type(model).__name__}: none of the "
            f"{len(provided)} tensor(s) in the checkpoint correspond to the "
            f"{len(expected)} the model expects. The model is left randomly initialised "
            f"and its output will be meaningless. Check that the checkpoint matches the "
            f"model class.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif expected and len(matched) < 0.5 * len(expected):
        warnings.warn(
            f"Only {len(matched)} of {len(expected)} weights expected by "
            f"{type(model).__name__} were found{where}. The remaining "
            f"{len(expected) - len(matched)} are randomly initialised, which usually "
            f"means the checkpoint does not match the model class.",
            RuntimeWarning,
            stacklevel=3,
        )

    return model.load_state_dict(state_dict, strict=False)


__all__ = ["load_state_dict_checked"]
