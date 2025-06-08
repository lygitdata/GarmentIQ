# GarmentIQ: Automated Garment Measurement for Fashion Retail

[![PyPI](https://img.shields.io/pypi/v/garmentiq)](https://pypi.org/project/garmentiq/) ![MIT](https://img.shields.io/github/license/lygitdata/GarmentIQ)

Try the full pipeline - use the [Web](https://garmentiq.ly.gd.edu.kg/application/) or [Python](https://pypi.org/project/garmentiq/) interface.

**Precise and flexible garment measurements from images - no tape measures, no delays, just fashion - forward automation.**

<img src="https://raw.githubusercontent.com/lygitdata/GarmentIQ/refs/heads/gh-pages/asset/img/bg.jpg" alt="GarmentIQ Background Image" width="300px"/>

## What Are the Key Features of GarmentIQ?

GarmentIQ uses computer vision and models like tinyViT, BiRefNet, and HRNet to classify garments, remove backgrounds, and detect key features with precision. It turns expert know-how into an intuitive measurement system—no coding required. Fully modular and customizable, it adapts to your workflows while delivering fast, accurate results out of the box.

| Feature | Web Demo | Python Demo |
|---------|----------|-------------|
| **1. Garment image classification**<br/>Our system accurately classifies garments into categories like tops, trousers, and skirts, ensuring seamless organization. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-classification/) | [Try Python demo](#) |
| **2. Garment image segmentation**<br/>We use advanced segmentation models to isolate garment features from the background for better measurement accuracy. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/image-segmentation/) | [Try Python demo](#) |
| **3. Garment measurement instruction generation**<br/>Our system generates detailed measurement instructions automatically, tailored to the specific garment type and its characteristics. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/instruction-generation/) | / |
| **4. Garment landmark extraction**<br/>Key landmarks are extracted from garment images, enabling precise measurement locations for consistent results. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-extraction/) | [Try Python demo](#) |
| **5. Garment landmark adjustment**<br/>Landmarks can be refined and adjusted manually to ensure they align perfectly, improving the accuracy of garment measurements. | [Try web demo](https://garmentiq.ly.gd.edu.kg/application/demo/landmark-adjustment/) | [Try Python demo](#) |

## Overview of QarmentIQ Python package

The `gqrmentiq` package provides an automated solution for garment measurement from images, utilizing computer vision techniques for classification, segmentation, and landmark extraction.

### Package Structure Overview

- `tailor`: This module acts as the central agent for the entire pipeline, orchestrating the different stages of garment measurement from classification to landmark derivation. It integrates the functionalities of other modules to provide a smooth end-to-end process.

- `classification`: This module is responsible for identifying the type of garment in an image. Its key functions include:
  - train_test_split: For preparing and splitting image datasets into training and testing sets.
load_data: For loading and preprocessing image data into memory.
load_model: For loading pre-trained classification models.
train_pytorch_nn: For training PyTorch neural networks for classification.
fine_tune_pytorch_nn: For fine-tuning pre-trained models on new datasets.
test_pytorch_nn: For evaluating the performance of trained models.
predict: For making class predictions on new images.
segmentation: This module focuses on isolating garment features from the background for improved measurement accuracy. Its key functions include:

load_model: For loading pre-trained segmentation models.
extract: For extracting segmentation masks from images.
change_background_color: For modifying the background color of segmented images.
process_and_save_images: For processing multiple images through segmentation and saving the results.
landmark: This module handles the detection, derivation, and refinement of key points on garments. Its key functions include:

detect: For identifying initial landmark coordinates on a garment image.
derive: For calculating the coordinates of new, derived landmarks based on existing ones.
refine: For adjusting and improving the precision of detected landmark locations.
Instruction Schemas
The instruction/ folder contains 9 predefined measurement schemas, which are utilized by the garment_classes.py file to define different garment types and their measurement properties. Users can also define their own custom measurement instructions by creating new dictionaries formatted similarly to the existing garment classes.
