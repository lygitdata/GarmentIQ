# A simple CNN model for garment classification

Last update: 03/12/2025

WARNING: This demonstration uses your local machine's computing power. We are not in charge for any consequence of using this demonstration.

## Demonstration 

**Link**: https://garmentiq.ly.gd.edu.kg/lygitdata/classification/demo.html

**Setup guide video**:

https://github.com/user-attachments/assets/918d2a4b-6099-43ad-994b-e2ee392de6c6

## Reproduce the result

**Download link**: https://drive.google.com/file/d/1zHclZ1TJKcnrCo5Ln1ZC43ouQbl7tsAQ/view?usp=drive_link

Download the zip file, then unzip it. You will see a folder named `dsl`, please upload the whole folder to your Google Drive root directory. Then open the notebook under `classification/main..pynb`.

## Technical detail

- 5 types of garments: Kurtas, Pants, Shirts, Shorts, Tops

- Raw dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

- Cleaned dataset: 14,874 images in total (13,362 images in train set; 1,485 images in test set)

  `R` was used to clean the dataset, please see code below:

  ```r
  
  ```

- Model size in .pth format: 16.9 MB

- Training time: ~ 1 hour on Colab with free plan GPU

- Test accuracy: 93.74%

- Test F1 score: 0.9336

- Test cross entropy: 0.1785
