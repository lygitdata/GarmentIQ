"""Running segmentation to obtain a garment mask.

Dispatches on the model family, so BiRefNet and every SAM generation share one entry
point. SAM prompts arrive in a unified `prompt` dictionary accepting `points`,
`labels`, `boxes`, and `text`; nesting differences between SAM versions are normalised
here, and a text prompt for SAM 1 or SAM 2 is routed through a grounding model first.
"""
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from typing import Optional, Union
from garmentiq.utils.device import resolve_device, empty_cache, inputs_to_device
from garmentiq.segmentation.model_definition.sam import (
    SAM_CAPABILITIES,
    SAM_PROMPT_DEPTHS,
)

PROMPT_KEYS = ("points", "labels", "boxes", "text")

# Legacy keyword arguments accepted for backwards compatibility, mapped onto the
# unified prompt dictionary.
LEGACY_PROMPT_KWARGS = {
    "input_points": "points",
    "input_labels": "labels",
    "input_boxes": "boxes",
}


def _nesting_depth(value):
    """Returns how many list/tuple levels wrap the first scalar in `value`."""
    depth = 0
    current = value
    while isinstance(current, (list, tuple)):
        depth += 1
        if len(current) == 0:
            break
        current = current[0]
    return depth


def _to_depth(value, target: int, name: str):
    """
    Reshapes a nested prompt list to the nesting depth a processor expects.

    SAM 1 and SAM 2 disagree on prompt nesting (SAM 2 requires exactly one extra
    level and raises otherwise), so prompts are normalised here to let callers use
    a single consistent format across every SAM family.
    """
    if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
        return value

    depth = _nesting_depth(value)
    while depth < target:
        value = [value]
        depth += 1
    while depth > target:
        if len(value) != 1:
            raise ValueError(
                f"Prompt '{name}' is nested {depth} levels deep but this SAM family "
                f"expects {target}, and it cannot be unwrapped because the outermost "
                f"level holds {len(value)} items. Provide '{name}' with {target} levels."
            )
        value = value[0]
        depth -= 1
    return value


def _detect_sam_family(model):
    """
    Identifies which SAM family a loaded model belongs to, if any.

    Detection walks the class hierarchy by name rather than importing the transformers
    classes, so a model from any SAM generation is recognised without requiring every
    generation to exist in the installed transformers release.
    """
    names = {cls.__name__ for cls in type(model).__mro__}
    # Checked before Sam3Model: the tracker is a separate class that carries the SAM-style
    # prompt encoder, so it accepts points where the SAM 3 detector does not.
    if {"Sam3TrackerModel", "Sam3TrackerPreTrainedModel"} & names:
        return "sam3-tracker"
    if {"Sam3Model", "Sam3PreTrainedModel"} & names:
        return "sam3"
    if {"Sam2Model", "Sam2PreTrainedModel"} & names:
        return "sam2"
    if {"SamModel", "SamPreTrainedModel"} & names:
        return "sam1"
    return None


def _build_prompt(prompt: Optional[dict], kwargs: dict):
    """Merges the `prompt` dict with legacy `input_*` keyword arguments."""
    merged = {}

    if prompt is not None:
        if not isinstance(prompt, dict):
            raise ValueError(
                f"`prompt` must be a dictionary with any of {list(PROMPT_KEYS)}, "
                f"got {type(prompt).__name__}."
            )
        unknown = set(prompt) - set(PROMPT_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown prompt key(s) {sorted(unknown)}. "
                f"Valid keys are {list(PROMPT_KEYS)}."
            )
        merged.update({k: v for k, v in prompt.items() if v is not None})

    for legacy_key, new_key in LEGACY_PROMPT_KWARGS.items():
        if kwargs.get(legacy_key) is not None:
            merged.setdefault(new_key, kwargs[legacy_key])

    return merged


def extract(
    model: torch.nn.Module,
    image_path: str,
    processor=None,
    prompt: Optional[dict] = None,
    device: Union[str, torch.device] = "cpu",
    grounding_model=None,
    grounding_processor=None,
    grounding_args: Optional[dict] = None,
    **kwargs,
):
    """
    Intelligently extracts an image segmentation mask from a given image using either a standard
    PyTorch model or a Processor-based foundation model.

    This function takes an image and processes it based on the model strategy. If a processor is supplied,
    it delegates preprocessing (e.g., resizing, scaling, prompt-handling) to the processor. Otherwise,
    it applies standard manual transformations based on provided kwargs. It then feeds the input into
    the model to generate a segmentation mask. The original image and the mask are returned as numpy arrays.

    For Segment Anything models the prompt is supplied through the unified `prompt` dictionary,
    which may carry geometric prompts (`"points"`, `"labels"`, `"boxes"`) and/or a natural-language
    prompt (`"text"`). At least one prompt is required, because an unprompted SAM silently returns
    a meaningless mask rather than raising. Prompt nesting is normalised per family, so the same
    `prompt` works across SAM 1, SAM 2, and SAM 3.

    Text prompts are handled differently per family, because only SAM 3 has a text encoder:
        - SAM 3 consumes the text directly.
        - SAM 1 and SAM 2 have no text encoder, so `grounding_model` and `grounding_processor`
          must be supplied. The phrase is grounded into boxes first, and those boxes are then
          used as ordinary SAM prompts.

    The model and its inputs are placed on `device`, so the same value should be passed here as was
    used when loading the model. Any accelerator memory cached during inference is released before
    returning.

    Args:
        model (torch.nn.Module): The pretrained PyTorch model to use for segmentation predictions.
        image_path (str): The path to the image file on which to perform segmentation.
        processor (Any, optional): The model-specific processor (e.g., from Hugging Face) used for preprocessing
                                   inputs. If None, standard PyTorch manual transformations are applied.
                                   Default is None.
        prompt (dict, optional): The unified prompt for Segment Anything models. Recognised keys:
                                 - `"points"` (list): Point coordinates, e.g. `[[[x, y]]]`.
                                   Not supported by SAM 3.
                                 - `"labels"` (list): Point labels, `1` for foreground and `0` for
                                   background, e.g. `[[1]]`. Not supported by SAM 3.
                                 - `"boxes"` (list): Boxes as `[[[x_min, y_min, x_max, y_max]]]`.
                                 - `"text"` (str): Natural-language prompt, e.g. `"a shirt"`.
                                 Default is None.
        device (Union[str, torch.device], optional): The device to run inference on, e.g. `"cpu"`,
                                                     `"cuda"`, `"cuda:0"`, or `"mps"`. Hardware
                                                     acceleration is opt-in; pass it explicitly to
                                                     use a GPU or Apple Silicon. Default is `"cpu"`.
        grounding_model (GroundingDinoForObjectDetection, optional): Grounding model used to turn a
                                                                     text prompt into boxes for SAM 1
                                                                     and SAM 2. Default is None.
        grounding_processor (GroundingDinoProcessor, optional): Processor matching `grounding_model`.
                                                                Default is None.
        grounding_args (dict, optional): Extra grounding options, accepting `"box_threshold"`,
                                         `"text_threshold"`, and `"max_boxes"`. Default is None.
        **kwargs: Additional arbitrary keyword arguments for model-specific configurations.
                  For standard models (e.g., BiRefNet): `resize_dim`, `normalize_mean`, `normalize_std`.
                  The legacy `input_points`, `input_labels`, and `input_boxes` arguments are still
                  accepted and are folded into `prompt`.

    Raises:
        FileNotFoundError: If the image file at `image_path` does not exist.
        ValueError: If the requested `device` is invalid or unavailable, if no prompt is supplied for
                    a SAM model, if a prompt type is unsupported by the chosen SAM family, if a text
                    prompt is given for SAM 1 or SAM 2 without a grounding model, or if the processor
                    output format is unrecognized.

    Returns:
        tuple (numpy.ndarray, numpy.ndarray): The original image converted to a numpy array,
                                              and the extracted segmentation mask as a numpy array.
    """
    device = resolve_device(device)
    model = model.to(device)
    image = Image.open(image_path).convert("RGB")

    # Processor-Based Models (SAM 1 / SAM 2 / SAM 3)
    if processor is not None:
        family = _detect_sam_family(model)
        prompt_dict = _build_prompt(prompt, kwargs)

        if family is not None:
            capabilities = SAM_CAPABILITIES[family]

            if not prompt_dict:
                options = []
                if capabilities["points"]:
                    options.append("prompt={'points': [[[x, y]]]}")
                if capabilities["boxes"]:
                    options.append("prompt={'boxes': [[[x0, y0, x1, y1]]]}")
                if capabilities["text"]:
                    options.append("prompt={'text': 'a shirt'}")
                else:
                    options.append(
                        "prompt={'text': 'a shirt'} together with a grounding model"
                    )
                raise ValueError(
                    f"A prompt is required for {family}. Pass one of: {', '.join(options)}. "
                    f"Without a prompt SAM returns an arbitrary mask instead of failing."
                )

            # `text` is intentionally excluded here: it is always acceptable, either
            # natively (SAM 3) or by grounding it into boxes first (SAM 1 / SAM 2).
            unsupported = [
                key
                for key in prompt_dict
                if key != "text" and key in capabilities and not capabilities[key]
            ]
            if unsupported:
                detail = ""
                if family == "sam3" and {"points", "labels"} & set(unsupported):
                    # SAM 3's release does contain a point-promptable model, but it is
                    # the tracker rather than the detector loaded here.
                    detail = (
                        " SAM 3's image model is an open-vocabulary detector, prompted by "
                        "description rather than by clicking. For point prompts on a still "
                        "image load SAM 3's tracker with "
                        "garmentiq.segmentation.model_definition.sam.load_sam3_tracker(), "
                        "or use SAM 1 / SAM 2."
                    )
                raise ValueError(
                    f"Prompt key(s) {sorted(unsupported)} are not supported by {family}. "
                    f"{family} supports: "
                    f"{sorted(k for k, v in capabilities.items() if v)}.{detail}"
                )

            # Resolve a text prompt into boxes for the families without a text encoder.
            text = prompt_dict.pop("text", None)
            if text is not None and not capabilities["text"]:
                if grounding_model is None or grounding_processor is None:
                    raise ValueError(
                        f"{family} has no text encoder, so a text prompt requires a grounding "
                        f"model. Pass grounding_model and grounding_processor (see "
                        f"garmentiq.grounding), or use SAM 3 which understands text natively."
                    )
                # Imported here so the grounding stack is only required when used.
                from garmentiq.grounding import ground_text_to_boxes

                grounded = ground_text_to_boxes(
                    model=grounding_model,
                    processor=grounding_processor,
                    image=image,
                    text=text,
                    device=device,
                    **(grounding_args or {}),
                )
                # Grounded boxes replace any caller-supplied boxes for this prompt.
                prompt_dict["boxes"] = [grounded]
                text = None

            depths = SAM_PROMPT_DEPTHS[family]
            processor_kwargs = {}
            if "points" in prompt_dict:
                processor_kwargs["input_points"] = _to_depth(
                    prompt_dict["points"], depths["points"], "points"
                )
            if "labels" in prompt_dict:
                processor_kwargs["input_labels"] = _to_depth(
                    prompt_dict["labels"], depths["labels"], "labels"
                )
            if "boxes" in prompt_dict:
                processor_kwargs["input_boxes"] = _to_depth(
                    prompt_dict["boxes"], depths["boxes"], "boxes"
                )
            if text is not None:
                processor_kwargs["text"] = text

            inputs = inputs_to_device(
                processor(images=image, return_tensors="pt", **processor_kwargs),
                device,
            )

            with torch.no_grad():
                outputs = model(**inputs)

            if family == "sam3":
                results = processor.post_process_instance_segmentation(
                    outputs, target_sizes=[(image.height, image.width)]
                )[0]
                instance_masks = results["masks"]
                if instance_masks.shape[0] == 0:
                    raise ValueError(
                        f"SAM 3 found no instance matching the prompt in {image_path!r}. "
                        f"Try rephrasing the text prompt or lowering the threshold."
                    )
                # Keep the highest-scoring instance, mirroring the single-mask contract.
                best = int(torch.argmax(results["scores"]))
                best_mask = instance_masks[best].detach().cpu().numpy()
            else:
                post_kwargs = {}
                if family == "sam1":
                    post_kwargs["reshaped_input_sizes"] = inputs[
                        "reshaped_input_sizes"
                    ].cpu()
                masks = processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    **post_kwargs,
                )
                if family == "sam3-tracker":
                    # The tracker returns three candidate masks per prompt with an IoU
                    # score for each, so pick the highest-scoring one rather than the
                    # first, which is often the smallest sub-part.
                    per_prompt_masks = masks[0]
                    iou = outputs.iou_scores.cpu()
                    selected = []
                    for i in range(per_prompt_masks.shape[0]):
                        scores = iou[0, i] if iou.ndim == 3 else iou[i]
                        selected.append(per_prompt_masks[i, int(torch.argmax(scores))])
                    per_prompt = torch.stack(selected, dim=0)
                else:
                    # masks[0] has shape (num_prompts, num_masks, H, W). Keep the first
                    # mask of each prompt and union them, which is identical to the
                    # previous single-prompt behaviour when only one prompt is given.
                    per_prompt = masks[0][:, 0]
                best_mask = (
                    torch.any(per_prompt.bool(), dim=0).detach().cpu().numpy()
                )

            mask_np = (best_mask.astype(np.float32) * 255).astype(np.uint8)
            del inputs, outputs

        else:
            # Unknown processor-based model: fall back to the generic SAM-like contract.
            inputs = inputs_to_device(
                processor(image, return_tensors="pt", **kwargs), device
            )

            with torch.no_grad():
                outputs = model(**inputs)

            if hasattr(outputs, "pred_masks"):
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                best_mask = masks[0][0][0].numpy()
                mask_np = (best_mask * 255).astype(np.uint8)
            else:
                raise ValueError("Unrecognized processor output format.")

            del inputs, outputs, masks

    # Standard Models (BiRefNet)
    else:
        # Extract BiRefNet-specific kwargs with safe defaults
        resize_dim = kwargs.get("resize_dim", (1024, 1024))
        normalize_mean = kwargs.get("normalize_mean", [0.485, 0.456, 0.406])
        normalize_std = kwargs.get("normalize_std", [0.229, 0.224, 0.225])

        transform = transforms.Compose(
            [
                transforms.Resize(resize_dim),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(input_tensor)

            # BiRefNet returns a tuple/list of tensors; we want the last one
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]

            preds = preds.sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)

        del input_tensor, preds

    # Clean up and Return
    image_np = np.array(image)
    empty_cache(device)

    return image_np, mask_np
