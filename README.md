# GarmentIQ: Automated Garment Measurement for Fashion Retail

[🌐 Official website](https://garmentiq.ly.gd.edu.kg/) | [📖 Documentation](https://garmentiq.ly.gd.edu.kg/documentation/) | [⚙️ Pipeline web interface](https://garmentiq.ly.gd.edu.kg/application/) | [📄 Paper](https://archive.gd.edu.kg/abs/20250525121523/)

**Precise and flexible garment measurements from images - no tape measures, no delays, just fashion - forward automation.**

<img src="https://github.com/user-attachments/assets/49e176e2-59c7-4e8b-b7e1-f834de965760" alt="GarmentIQ" width="600px"/>

**Content**: 

1. [What Are the Key Features of GarmentIQ?](#what-are-the-key-features-of-garmentiq)
2. [Overview of QarmentIQ Python Package](#overview-of-qarmentiq-python-package)
3. [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Classification](#classification)
    - [Segmentation](#segmentation)
    - [Landmark detection, derivation, and refinement](#landmark-detection-derivation-and-refinement)

---

## What Are the Key Features of GarmentIQ?

GarmentIQ uses computer vision and models like tinyViT, BiRefNet, and HRNet to classify garments, remove backgrounds, and detect key features with precision. It turns expert know-how into an intuitive measurement system - no intensive coding required. Fully modular and customizable, it adapts to your workflows while delivering fast, accurate results out of the box.

| Feature | Web Demo |
|---------|----------|
| **1. Garment image classification**<br/>Our system accurately classifies garments into categories like tops, trousers, and skirts, ensuring seamless organization. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-classification/) |
| **2. Garment image segmentation**<br/>We use advanced segmentation models to isolate garment features from the background for better measurement accuracy. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-segmentation/) |
| **3. Garment measurement instruction generation**<br/>Our system generates detailed measurement instructions automatically, tailored to the specific garment type and its characteristics. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/instruction-generation/) |
| **4. Garment landmark extraction**<br/>Key landmarks are extracted from garment images, enabling precise measurement locations for consistent results. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-extraction/) |
| **5. Garment landmark adjustment**<br/>Landmarks can be refined and adjusted manually to ensure they align perfectly, improving the accuracy of garment measurements. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-adjustment/) |

## Overview of QarmentIQ Python Package

The `gqrmentiq` package provides an automated solution for garment measurement from images, utilizing computer vision techniques for classification, segmentation, and landmark extraction.

- `tailor`: This module acts as the central agent for the entire pipeline, orchestrating the different stages of garment measurement from classification to landmark derivation. It integrates the functionalities of other modules to provide a smooth end-to-end process.

- `classification`: This module is responsible for identifying the type of garment in an image. Its key functions include: `fine_tune_pytorch_nn`, `load_data`, `load_model`, `predict`, `test_pytorch_nn`, `train_pytorch_nn`, and `train_test_split`

- `segmentation`: This module focuses on isolating garment features from the background for improved measurement accuracy. Its key functions include: `change_background_color`, `extract`, `load_model`, and `process_and_save_images`.

- `landmark`: This module handles the detection, derivation, and refinement of key points on garments. Its key functions include: `derive`, `detect`, and `refine`.

- Instruction Schemas: The `instruction/` folder contains 9 predefined measurement schemas in `.json` format, which are utilized by the `garment_classes.py` file `garment_classes` dictionary to define different garment types and their predefined measurement properties. Users can also define their own custom measurement instructions by creating new dictionaries formatted similarly to the existing garment classes.

## Quick Start

<img src="https://github.com/user-attachments/assets/2de8ee34-166e-42db-8428-6e376848a9ec" width="600px" alt="GarmentIQ Example"/>

### Installation

Please install from PyPI using the following command.

```python
!pip install -r garmentiq -q
```

### Classification

[Open in Colab](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/quick_start/classification_quick_start.ipynb)

```python
import garmentiq as giq
from garmentiq.classification.model_definition import tinyViT
from garmentiq.classification.utils import CachedDataset

# Download test data and a pretrained model
!mkdir -p models

!curl -L -o /content/garmentiq-classification-set-nordstrom-and-myntra.zip \
  https://www.kaggle.com/api/v1/datasets/download/lygitdata/garmentiq-classification-set-nordstrom-and-myntra

!wget -q -O /content/models/tiny_vit_inditex_finetuned.pt \
    https://huggingface.co/lygitdata/garmentiq/resolve/main/tiny_vit_inditex_finetuned.pt

# Prepare test data
DATA = giq.classification.train_test_split(
    output_dir="data",
    metadata_csv="metadata.csv",
    label_column="garment",
    train_zip_dir="garmentiq-classification-set-nordstrom-and-myntra.zip",
    test_size=0.15,
    verbose=True
)

test_images, test_labels, _ = giq.classification.load_data(
    df=DATA["test_metadata"],
    img_dir=DATA["test_images"],
    label_column="garment",
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081]
)

# Load the pretrained model
classifier = giq.classification.load_model(
    model_path="/content/models/tiny_vit_inditex_finetuned.pt",
    model_class=tinyViT,
    model_args={"num_classes": 9, "img_size": (120, 184), "patch_size": 6}
)

# Fit the model on the whole test data
giq.classification.test_pytorch_nn(
    model_path="/content/models/tiny_vit_inditex_finetuned.pt",
    model_class=tinyViT,
    model_args={"num_classes": 9, "img_size": (120, 184), "patch_size": 6},
    dataset_class=CachedDataset,
    dataset_args={
        "raw_labels": DATA["test_metadata"]["garment"],
        "cached_images": test_images,
        "cached_labels": test_labels,
    },
    param={"batch_size": 64},
)

# Fit the model on a single image
img_to_test = DATA['test_metadata']['filename'][88]

pred_label, pred_prob = giq.classification.predict(
    model=classifier,
    image_path=f"data/test/images/{img_to_test}",
    classes=DATA['test_metadata']['garment'].unique().tolist(),
    resize_dim=(120, 184),
    normalize_mean=[0.8047, 0.7808, 0.7769],
    normalize_std=[0.2957, 0.3077, 0.3081]
)

print(
    "True label: ", img_to_test,
    "\nPredicted label: ", pred_label,
    "\nPredicted Probabilities: ", pred_prob
)
```

### Segmentation

[Open in Colab](https://colab.research.google.com/github/lygitdata/GarmentIQ/blob/main/quick_start/segmentation_quick_start.ipynb)

```python
import garmentiq as giq

# Download a test image
!mkdir -p test_image
!wget -q -O /content/test_image/cloth_1.jpg \
    https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/asset/img/cloth_1.jpg

# Load the pretrained model from Hugging Face
BiRefNet = giq.segmentation.load_model(
    pretrained_model='lygitdata/BiRefNet_garmentiq_backup',
    pretrained_model_args={'trust_remote_code': True},
    high_precision=True
)

# Extract the mask
original_img, mask = giq.segmentation.extract(
    model=BiRefNet,
    image_path='/content/test_image/cloth_1.jpg',
    resize_dim=(1024, 1024),
    normalize_mean=[0.485, 0.456, 0.406],
    normalize_std=[0.229, 0.224, 0.225],
    high_precision=True
)

# Change background color
bg_modified_img = giq.segmentation.change_background_color(
    image_np=original_img,
    mask_np=mask,
    background_color=[102, 255, 102]
)

# Plot the original image, mask, and background modified image
giq.segmentation.plot(image_np=original_img, figsize=(3, 3))
giq.segmentation.plot(image_np=mask, figsize=(3, 3))
giq.segmentation.plot(image_np=bg_modified_img, figsize=(3, 3))
```

### Landmark detection, derivation, and refinement
