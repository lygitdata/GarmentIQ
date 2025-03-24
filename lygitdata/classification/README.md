# A simple CNN model for garment classification

Last update: 03/24/2025

WARNING: This demonstration uses your local machine's computing power. We are not in charge for any consequence of using this demonstration.

## Demonstration 

**Link**: https://garmentiq.ly.gd.edu.kg/lygitdata/classification/demo.html

**Setup guide video**:

https://github.com/user-attachments/assets/918d2a4b-6099-43ad-994b-e2ee392de6c6

## Reproduce the result

**Download link**: https://drive.google.com/file/d/1zHclZ1TJKcnrCo5Ln1ZC43ouQbl7tsAQ/view?usp=drive_link

Download the zip file, then unzip it. You will see a folder named `dsl`, please upload the whole folder to your Google Drive root directory. Then open the notebook under `classification/main.ipynb`.

## Technical detail

- 5 types of garments: `Dress` (sample size 449), `Pants` (sample size 2,257), `Skirt` (sample size 125), `Sleeveless top` (sample size 37), `Top` (sample size 11,709)

- Model size in .pth format: 16.9 MB

- Training time: ~ 4 hour on Colab with free plan GPU

- **Issues**:
    1. Cannot find requested garment type for overalls in the raw data.
    2. Small sample size for `Sleeveless top`, some sweaters can be included in this type, but manually inspection and modification on more than 200 sweater images are required.
    3. A few misleading images in `Skirt`.

## Model Metrics

- Test accuracy: TBD

- Test F1 score: TBD

- Test cross entropy: TBD

## Data Information

- Raw dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

- Cleaned dataset: 14,577 images in total (13,119 images in train set; 1,458 images in test set)

  You do not need to run the following code to run the demo.

  `R` was used to clean the dataset, please see code below. Make sure you have the images in the folder `images` and `styles.csv` downloaded and put in the same directory as the R script before running the code.

  ```r
  rm(list = ls())
  library("tidyverse")
  
  # Clean data
  apparel = read.csv("styles.csv") %>%
    filter(masterCategory == "Apparel") %>%
    filter(subCategory %in% c("Bottomwear", "Dress", "Topwear")) %>%
    mutate(articleType = ifelse(
      grepl("sleeveless", productDisplayName, ignore.case = TRUE),
      "sleeveless",
      articleType
    )) %>%
    select(id, articleType)%>%
    filter(complete.cases(.) &
             !apply(., 1, function(x)
               any(x == ""))) %>%
    filter(
      articleType %in% c(
        "Skirts",
        "sleeveless",
        "Dresses",
        "Jumpsuit",
        "Jackets",
        "Shirts",
        "Tops",
        "Tshirts",
        "Jeans",
        "Capris",
        "Jeggings",
        "Leggings",
        "Shorts",
        "Track Pants",
        "Trousers"
      )
    ) %>%
    mutate(
      articleType = case_when(
        articleType == "Skirts" ~ "Skirt",
        articleType == "sleeveless" ~ "Sleeveless top",
        articleType %in% c("Dresses", "Jumpsuit") ~ "Dress",
        articleType %in% c("Jackets", "Shirts", "Tops", "Tshirts") ~ "Top",
        articleType %in% c(
          "Jeans",
          "Capris",
          "Jeggings",
          "Leggings",
          "Shorts",
          "Track Pants",
          "Trousers"
        ) ~ "Pants",
      )
    )
    
  
  # See statistics summary
  summary(as.factor(apparel$articleType))
  
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
  
  # Delete those unmatched records
  if (length(expected_but_missing) > 0) {
    for (i in expected_but_missing) {
      apparel = apparel[apparel$id != i,]
    }
  }
  
  # Verify the image names and id's are exactly the same
  valid_image_files = sub("\\.jpg$", "", basename(
    list.files("valid_images/", pattern = "\\.jpg$", full.names = TRUE)
  ))
  length(intersect(valid_image_files, apparel$id))
  
  # Output the cleaned styles as apparel.csv
  write.csv(apparel, "apparel.csv", row.names = FALSE)
  
  # See statistics summary
  summary(as.factor(apparel$articleType))
  ```
