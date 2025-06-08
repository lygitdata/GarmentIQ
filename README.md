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

## Overview of QarmentIQ

The GarmentIQ package is designed for automated garment measurement from images, leveraging computer vision models. It provides a modular and customizable framework for classifying, segmenting, and extracting key features from garment images to generate precise measurements.

The package structure is organized into several key modules, each handling a specific aspect of the garment measurement pipeline:

tailor: This is the main class that orchestrates the entire measurement pipeline, integrating functionalities from other modules. It handles classification, segmentation, landmark detection, refinement, and derivation, and manages input/output operations.
utils: Contains various utility functions used across the package, such as unzipping files, checking directory structures, validating metadata, computing measurement distances, and exporting data to JSON.
classification: This module focuses on garment image classification. It includes functionalities for:
Training and fine-tuning PyTorch neural networks (train_pytorch_nn.py, fine_tune_pytorch_nn.py).
Loading data and models (load_data.py, load_model.py).
Making predictions (predict.py).
Defining various CNN and transformer models (model_definition.py).
Splitting data into training and testing sets (train_test_split.py).
segmentation: Handles image segmentation tasks, including:
Loading segmentation models (load_model.py).
Extracting segmentation masks (extract.py).
Changing background colors of images (change_background_color.py).
Processing and saving segmented images (process_and_save_images.py).
landmark: Dedicated to garment landmark detection, derivation, and refinement:
detection: Manages landmark detection, including model loading (load_model.py) and utility functions for image transformation and prediction processing (utils.py).
derivation: Focuses on deriving new keypoint coordinates based on existing landmarks and mask intersections (derive_keypoint_coord.py, process.py).
refinement: Contains functions for refining landmark locations using blurred masks (refine_landmark_with_blur.py).
Utilities for finding and filling landmark coordinates within instruction dictionaries (utils.py).
The garment_classes.py file defines the different garment types supported by the system, along with their respective landmark properties and instructional JSON file paths. The instruction JSON files (e.g., long sleeve dress.json) provide detailed information about predefined and custom landmarks, and the measurements associated with each garment type, including the landmarks used for each measurement.
