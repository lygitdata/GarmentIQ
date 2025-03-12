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

- Model size in .pth format: 16.9 MB

- Training time: ~ 1 hour on Colab with free plan GPU

- Test accuracy: 93.74%

- Test F1 score: 0.9336

- Test cross entropy: 0.1785

- Raw dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

- Cleaned dataset: 14,874 images in total (13,362 images in train set; 1,485 images in test set)

  `R` was used to clean the dataset, please see code below. Make sure you have the images in the folder `images` and `styles.csv` downloaded and put in the same directory as the R script before running the code.

  ```r
  rm(list = ls())
  library("tidyverse")
  
  # Clean data
  apparel = read.csv("styles.csv")[, c("id", "masterCategory", "subCategory", "articleType")] %>%
    filter(masterCategory == "Apparel" & (subCategory == "Topwear" | subCategory == "Bottomwear")) %>%
    filter(masterCategory == "Apparel" & subCategory %in% c("Topwear", "Bottomwear")) %>%
    filter(complete.cases(.) & !apply(., 1, function(x) any(x == ""))) %>%
    filter(articleType %in% names(table(articleType)[table(articleType) > 454])) %>%
    mutate(articleType = recode(articleType,
                                "Tshirts" = "Shirts", 
                                "Shirts" = "Shirts", 
                                "Jeans" = "Pants", 
                                "Trousers" = "Pants")) %>%
    select(id, articleType)
  
  # Copy the valid images to a new directory
  new_folder = "valid_images/"
  dir.create(new_folder, showWarnings = FALSE)
  valid_ids = as.character(apparel$id)
  image_files = list.files("images/", pattern = "\\.jpg$", full.names = TRUE)
  file_ids = tools::file_path_sans_ext(basename(image_files))
  
  # Track missing files
  missing_ids = character(0)
  copied_ids = character(0)
  
  # Copy valid images
  for (img_path in image_files) {
    file_id = tools::file_path_sans_ext(basename(img_path))
    if (file_id %in% valid_ids) {
      dest_path = file.path(new_folder, basename(img_path))
      if (file.copy(img_path, dest_path)) {
        copied_ids = c(copied_ids, file_id)
      }
    } else {
      missing_ids = c(missing_ids, file_id)
    }
  }
  
  # Find IDs that SHOULD have been copied but had no images
  expected_but_missing = setdiff(valid_ids, file_ids)
  
  # Console report
  cat("===== Copying Summary =====\n")
  cat("Successfully copied:", length(copied_ids), "images\n")
  cat("Images in folder not copied (invalid IDs):", length(missing_ids), "\n")
  cat("IDs in dataframe missing images:", length(expected_but_missing), "\n\n")
  
  if (length(expected_but_missing) > 0) {
    cat("These valid IDs had no corresponding images:\n")
    print(expected_but_missing)
  }
  
  if (length(missing_ids) > 0) {
    cat("\nThese image IDs were ignored (not in dataframe):\n")
    print(missing_ids)
  }
  
  # Delete those unmatched records
  apparel = apparel[apparel$id != "39403",]
  apparel = apparel[apparel$id != "39410",]
  apparel = apparel[apparel$id != "39401",]
  apparel = apparel[apparel$id != "39425",]
  
  # Verify the image names and id's are exactly the same
  valid_image_files = sub("\\.jpg$", "", basename(list.files("valid_images/", pattern = "\\.jpg$", full.names = TRUE)))
  length(intersect(valid_image_files, apparel$id))
  
  # Output the cleaned styles as apparel.csv
  write.csv(apparel, "apparel.csv", row.names = FALSE)
  ```
