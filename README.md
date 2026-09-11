# GarmentIQ: Automated Garment Measurement for Fashion Retail

[🌐 Official website](https://garmentiq.ly.gd.edu.kg/) | [📖 Documentation](https://garmentiq.ly.gd.edu.kg/documentation/) | [⚙️ Web pipeline](https://garmentiq.ly.gd.edu.kg/application/) | [🪄 MagicBox](https://garmentiq.ly.gd.edu.kg/documentation/magicbox/) | [📄 Paper](https://archive.gd.edu.kg/abs/20250525121523/)

**Free & Open Source. Precise and flexible garment measurements from images - no tape measures, no delays, just fashion - forward automation.**

> 🎉 **Update (09/11/2026):** GarmentIQ now supports **SAM 1, SAM 2, and SAM 3**, plus **natural-language (text) prompts** for [segmentation](#tutorial-notebooks), a new **[matting](#tutorial-notebooks) module** (ViTMatte and Matting Anything) for soft alpha edges, and an explicit `device` argument (`cpu`, `cuda`, `mps`) on every model function! Cheers! 🥂

<img src="https://github.com/user-attachments/assets/b816c16b-bd33-4370-80b1-acc5df81cfcd" alt="GarmentIQ" width="600px"/>

---

**Content**:

1. [What Are the Key Features of GarmentIQ?](#what-are-the-key-features-of-garmentiq)
2. [Overview of GarmentIQ Python Package](#overview-of-garmentiq-python-package)
3. [Tutorials](#tutorials)
    - [Installation](#installation)
    - [Tutorial notebooks](#tutorial-notebooks)
4. [Advanced Usage](#advanced-usage)
5. [Trained Models for Classification](#trained-models-for-classification)
6. [Issues & Feedback](#issues--feedback)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

---

## What Are the Key Features of GarmentIQ?

GarmentIQ uses computer vision and models like DeiT, BiRefNet, SAM, and HRNet to classify garments, remove backgrounds, and detect key features with precision. It turns expert know-how into an intuitive measurement system - no intensive coding required. Fully modular and customizable, it adapts to your workflows while delivering fast, accurate results out of the box.

| Feature | Web Demo | Video guide |
|---------|----------|----------|
| **Tailor (the whole pipeline)** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/guide.mp4) |
| **Garment measurement instruction generation** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/instruction-generation/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/demo/instruction-generation/guide.mp4) |
| **Garment image classification** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-classification/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/demo/image-classification/guide.mp4) |
| **Garment image segmentation** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-segmentation/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/demo/image-segmentation/guide.mp4) |
| **Garment landmark detection** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-detection/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-detection/guide.mp4) |
| **Garment landmark adjustment** | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-adjustment/) | [Watch video guide](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-adjustment/guide.mp4) |

## Overview of GarmentIQ Python Package

The `garmentiq` package provides an automated solution for garment measurement from images, utilizing computer vision techniques for classification, segmentation, and landmark extraction.

- `tailor`: This module acts as the central agent for the entire pipeline, orchestrating the different stages of garment measurement from classification to landmark derivation. It integrates the functionalities of other modules to provide a smooth end-to-end process.

- `classification`: This module is responsible for identifying the type of garment in an image. Its key functions include: `fine_tune_pytorch_nn`, `load_data`, `load_model`, `predict`, `test_pytorch_nn`, `train_pytorch_nn`, and `train_test_split`

- `segmentation`: This module focuses on isolating garment features from the background for improved measurement accuracy. Its key functions include: `change_background_color`, `extract`, `load_model`, and `process_and_save_images`.

- `landmark`: This module handles the detection, derivation, and refinement of key points on garments. Its key functions include: `derive`, `detect`, and `refine`.

- `matting`: This module refines a hard segmentation mask into a soft alpha matte, so edges and semi-transparent detail composite naturally. Its key functions include: `generate_trimap`, `load_model`, `matte`, and `composite`.

- `grounding`: This module turns a natural-language phrase into bounding boxes, which is what gives SAM 1 and SAM 2 text-prompted segmentation. Its key functions include: `load_grounding_model`, `load_grounding_processor`, and `ground_text_to_boxes`.

### Common conventions

Every model-backed module follows the same two-step shape, so moving between them requires no
relearning:

```python
model = giq.<module>.load_model(...)        # 1. load weights onto a device
result = giq.<module>.<run>(model=model, image_path=..., device=...)   # 2. run on an image
```

where `<run>` is `predict` for classification, `extract` for segmentation, `matte` for matting,
and `detect` for landmarks. Across all of them:

- **`device` is opt-in and identical everywhere.** Every `load_model` and every inference
  function takes `device`, defaulting to `"cpu"`. Pass `"cuda"` or `"mps"` explicitly to use an
  accelerator; nothing is auto-detected.
- **Accelerator memory is released automatically** after each inference call.
- **Configurations are bundled.** BiRefNet, every SAM variant and ViTMatte load their configs
  and processors from inside the package, so only weights need downloading.
- **Arguments are keyword-style.** The examples throughout this README pass every argument by
  name, which is the supported way to call these functions.
- **A mismatched checkpoint warns.** Weights are loaded leniently so partial checkpoints keep
  working, but if a checkpoint does not match the model class GarmentIQ raises a
  `RuntimeWarning` instead of silently returning a randomly initialised model.

- Instruction Schemas: The `instruction/` folder contains 9 predefined measurement schemas in `.json` format, which are utilized by the `garment_classes.py` file `garment_classes` dictionary to define different garment types and their predefined measurement properties. Users can also define their own custom measurement instructions by creating new dictionaries formatted similarly to the existing garment classes.

## Tutorials

Each tutorial is a self-contained Colab notebook covering one part of the `garmentiq` Python API, from a single module up to the whole pipeline. Open one to read the explanation and run the code. For the full API reference, see our [documentation](https://garmentiq.ly.gd.edu.kg/documentation/).

<img src="https://github.com/user-attachments/assets/2de8ee34-166e-42db-8428-6e376848a9ec" width="600px" alt="GarmentIQ Example"/>

> ⚠️ Note: If you encounter errors while running these notebooks in Colab, they are likely due to Python package version conflicts specific to the Colab environment. We recommend installing [MagicBox](https://garmentiq.ly.gd.edu.kg/documentation/magicbox/) on your local machine, where you can find and run these examples under `working/examples/`.

### Installation

Please install from PyPI using the following command.

```bash
pip install garmentiq -q
```

### Tutorial notebooks

| Tutorial | What it covers | Open in Colab |
|---|---|---|
| **Tailor (the whole pipeline)** | Run every stage end to end on a folder of images, then read the masks, annotated images, and measurements it writes out. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_tailor.ipynb) |
| **Classification** | Identify the garment type, which decides the landmarks and measurement instructions used by every later stage. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_classification.ipynb) |
| **Segmentation** | Separate the garment from its background with BiRefNet or SAM 1, 2, and 3, using point, box, label, and text prompts. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_segmentation.ipynb) |
| **Grounding** | Turn a natural-language phrase into bounding boxes, which is what gives SAM 1 and SAM 2 text-prompted segmentation. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_grounding.ipynb) |
| **Matting** | Refine a hard mask into a soft alpha matte with ViTMatte or Matting Anything, so edges composite naturally. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_matting.ipynb) |
| **Landmark detection** | Locate the key points a measurement runs between, such as shoulders, sleeve ends, and hems. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_landmark_detection.ipynb) |
| **Landmark refinement and derivation** | Snap detected landmarks onto the true garment edge, and compute extra points the model does not predict at all. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/tutorial_landmark_refinement_and_derivation.ipynb) |

## Advanced Usage

These notebooks go beyond everyday use: defining your own measurements, and training or fine-tuning the classification model on your own data.

> ⚠️ Note: If you encounter errors while running these notebooks in Colab, they are likely due to Python package version conflicts specific to the Colab environment. We recommend installing [MagicBox](https://garmentiq.ly.gd.edu.kg/documentation/magicbox/) on your local machine, where you can find and run these examples under `working/examples/`.

| Notebook | What it covers | Open in Colab |
|---|---|---|
| **Custom measurement instruction** | Define your own landmarks and measurements, then register them in a garment class dictionary. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/adv_usage_custom_measurement_instruction.ipynb) |
| **Classification model training & evaluation** | Train the built-in CNN3 or a model you wrote yourself from scratch, then compare them on a held-out test set. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/adv_usage_classification_model_training_evaluation.ipynb) |
| **Classification model fine-tuning** | Adapt the pretrained tinyViT classifier to your own catalog by freezing the backbone and retraining the head. | [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/test/adv_usage_classification_model_fine_tuning.ipynb) |

## Trained Models for Classification

We release the following models trained as part of this project. Models having `_inditex_finetuned` in their names means that they were finetuned on a small set of garment data from Inditex - Zara.

| Model | Test Accuracy | Test F1 Score | Fine-tune Accuracy | Fine-tune F1 Score | Link |
|---------|----------|----------|----------|----------|----------|
| `cnn_3.pt` | 0.9458 | 0.9459 | / | / | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/cnn_3.pt) |
| `cnn_4.pt` | 0.9533 | 0.9533 | / | / | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/cnn_4.pt) |
| `tiny_vit.pt` | 0.9576 | 0.9576 | / | / | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/tiny_vit.pt) |
| `cnn_3_inditex_finetuned.pt` | 0.9074 | 0.9068 | 0.9197 | 0.9216 | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/cnn_3_inditex_finetuned.pt) |
| `cnn_4_inditex_finetuned.pt` | 0.9132 | 0.9137 | 0.9592 | 0.9585 | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/cnn_4_inditex_finetuned.pt) |
| `tiny_vit_inditex_finetuned.pt` | 0.9484 | 0.9483 | 0.9916 | 0.9917 | [See the model](https://huggingface.co/lygitdata/garmentiq/blob/main/tiny_vit_inditex_finetuned.pt) |

## Issues & Feedback

Found a bug or have a feature request? Please open an issue on our [GitHub Issues page](https://github.com/lygitdata/GarmentIQ/issues).

## License

GarmentIQ's Python API code is licensed under the MIT License.

## Acknowledgements

We sincerely thank Adrian Gonzalez-Sieira and Laura Rodriguez-Barreiro from INDITEX for their invaluable suggestions and continuous support throughout this research. We are also grateful to everyone at ETH Zurich and the ETH AI Center for their coordination and collaborative efforts.

We gratefully acknowledge the use and adaptation of the following open-source resources:

- https://github.com/facebookresearch/deit
- https://github.com/ZhengPeng7/BiRefNet
- https://github.com/facebookresearch/segment-anything
- https://github.com/facebookresearch/sam2
- https://github.com/facebookresearch/sam3
- https://github.com/SHI-Labs/Matting-Anything
- https://github.com/hustvl/ViTMatte
- https://github.com/svip-lab/HRNet-for-Fashion-Landmark-Estimation.PyTorch
- https://github.com/switchablenorms/DeepFashion2
- https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
- https://www.kaggle.com/datasets/lygitdata/garmentiq-classification-set-nordstrom-and-myntra
- https://www.kaggle.com/datasets/lygitdata/zara-clothes-image-data
- All Python packages listed in `requirements.txt`

---

© 2025 - 2026 GarmentIQ

"GarmentIQ" is a registered trademark. Unauthorized use, reproduction, or distribution is strictly prohibited and may result in legal action.
