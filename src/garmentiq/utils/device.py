"""Device resolution and accelerator memory management.

GarmentIQ runs on CPU by default and never auto-detects an accelerator. These helpers
validate a requested device, fail fast with an actionable message when it is
unavailable, move processor output onto it, and release its cached memory afterwards.
"""
import torch
from typing import Union

DeviceLike = Union[str, torch.device, None]


def resolve_device(device: DeviceLike = "cpu") -> torch.device:
    """
    Resolves and validates a user supplied device specification.

    GarmentIQ runs on CPU by default. Hardware acceleration is opt-in: the caller
    explicitly requests it by passing `device="cuda"`, `device="cuda:1"`, or
    `device="mps"` (Apple Silicon). This function normalises the request into a
    `torch.device` and fails fast with an actionable message when the requested
    backend is unavailable, instead of silently degrading to CPU.

    Args:
        device (Union[str, torch.device], optional): The requested device, e.g. `"cpu"`,
                                                     `"cuda"`, `"cuda:0"`, or `"mps"`.
                                                     `None` is treated as `"cpu"`.
                                                     Default is `"cpu"`.

    Raises:
        ValueError: If `device` is not a valid device specification, or if the requested
                    accelerator (CUDA or MPS) is not available on the current machine.

    Returns:
        torch.device: The validated device to run computation on.
    """
    if device is None:
        return torch.device("cpu")

    if isinstance(device, torch.device):
        resolved = device
    else:
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid device specification {device!r}. Expected values such as "
                f'"cpu", "cuda", "cuda:0", or "mps". Original error: {e}'
            )

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            f"Requested device {str(resolved)!r} but CUDA is not available in this "
            f"PyTorch installation. Use device='cpu', or device='mps' on Apple Silicon."
        )

    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError(
            f"Requested device {str(resolved)!r} but the MPS backend is not available. "
            f"MPS requires macOS on Apple Silicon with a compatible PyTorch build. "
            f"Use device='cpu' instead."
        )

    return resolved


def inputs_to_device(inputs, device: DeviceLike = "cpu"):
    """
    Moves a processor's batched output onto `device`, narrowing dtypes MPS cannot hold.

    Hugging Face image processors emit some tensors as float64 — SAM's `input_points`
    and `input_boxes` are built from plain Python floats, for example. The MPS backend
    has no float64 support at all, so moving such a batch onto an Apple Silicon GPU
    raises `TypeError: Cannot convert a MPS Tensor to float64 dtype`. Narrowing those
    tensors to float32 is lossless in practice, because they carry pixel coordinates
    rather than values that need double precision.

    The narrowing is applied **only** when the target is MPS. CPU and CUDA keep the
    processor's original dtypes, so their numerical results are unchanged.

    Args:
        inputs (BatchFeature | BatchEncoding | dict): The processor output to move.
        device (Union[str, torch.device], optional): The device to move onto, e.g.
                                                     `"cpu"`, `"cuda"`, or `"mps"`.
                                                     Default is `"cpu"`.

    Raises:
        ValueError: If the requested `device` is invalid or unavailable on this machine.

    Returns:
        The same mapping, with its tensors placed on `device`.
    """
    resolved = resolve_device(device)

    if resolved.type == "mps":
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
                inputs[key] = value.to(torch.float32)

    return inputs.to(resolved)


def empty_cache(device: DeviceLike = "cpu") -> None:
    """
    Releases cached accelerator memory held by the allocator after inference or training.

    PyTorch caching allocators keep freed blocks reserved for reuse, which inflates
    reported memory usage and can starve other processes on the same accelerator.
    This helper clears that cache for the backend actually in use, and is a no-op on
    CPU (where no such cache exists) and for accelerators that are not available.

    Args:
        device (Union[str, torch.device], optional): The device whose cache should be
                                                     released. Default is `"cpu"`.

    Returns:
        None
    """
    if device is None:
        return

    if isinstance(device, torch.device):
        resolved = device
    else:
        try:
            resolved = torch.device(device)
        except (RuntimeError, TypeError, ValueError):
            return

    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
