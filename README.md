# GarmentIQ: Automated Garment Measurement for Fashion Retail

[![PyPI](https://img.shields.io/pypi/v/garmentiq)](https://pypi.org/project/garmentiq/) ![MIT](https://img.shields.io/github/license/lygitdata/GarmentIQ)

[Official website](https://garmentiq.ly.gd.edu.kg/) [Documentation](https://garmentiq.ly.gd.edu.kg/documentation/)

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

- `tailor`: This module acts as the central agent for the entire pipeline, orchestrating the different stages of garment measurement from classification to landmark derivation. It integrates the functionalities of other modules to provide a smooth end-to-end process.

- `classification`: This module is responsible for identifying the type of garment in an image. Its key functions include: `fine_tune_pytorch_nn`, `load_data`, `load_model`, `predict`, `test_pytorch_nn`, `train_pytorch_nn`, and `train_test_split`

- `segmentation`: This module focuses on isolating garment features from the background for improved measurement accuracy. Its key functions include: `change_background_color`, `extract`, `load_model`, and `process_and_save_images`.

- `landmark`: This module handles the detection, derivation, and refinement of key points on garments. Its key functions include: `derive`, `detect`, and `refine`.

- Instruction Schemas: The `instruction/` folder contains 9 predefined measurement schemas in `.json` format, which are utilized by the `garment_classes.py` file `garment_classes` dictionary to define different garment types and their predefined measurement properties. Users can also define their own custom measurement instructions by creating new dictionaries formatted similarly to the existing garment classes.
