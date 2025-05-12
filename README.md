# Image Scraping Tool

*Last update*: 04/10/2025

*Navigation*:

1. [Nordstrom](#nordstrom)
  - [Why Do We Need Images from Nordstrom?](#why-do-we-need-images-from-nordstrom)
  - [Why Choose Nordstrom?](#why-choose-nordstrom)
  - [Why Use This Tool?](#why-use-this-tool)
  - [How Does the Scraping Process Work?](#how-does-the-scraping-process-work)

2. [Zara](#zara)

---

# Nordstrom

## Why Do We Need Images from Nordstrom?

We need more image data to train our classification model and prepare for a landmark detection task. The [Fashion Product Images Dataset](https://doi.org/10.34740/kaggle/ds/139630) lacks several garment categories we require. Therefore, it's essential to supplement our dataset with additional images from external sources. 

Our scraped data has already been uploaded to Kaggle for better accessibility - [Nordstrom & Myntra Clothes Image Data - GarmentIQ](https://doi.org/10.34740/kaggle/ds/7099732).

## Why Choose Nordstrom?

Nordstrom is a well-known fashion retailer with a long-standing reputation. The product listings on their site feature high-quality images and are consistently well-labeled, making them an excellent source for training data.

## Why Use This Tool?

Nordstrom's website includes protection mechanisms that may restrict access based on IP address, user-agent, and other request headers. To work around these restrictions, we developed this semi-automated scraping tool.

The scraping process involves two parts:
1. **Manual**: Search and collect image URLs using the browser's developer tools.
2. **Automated**: Use a PowerShell script to organize metadata and download the images locally.

## How Does the Scraping Process Work?

To use the PowerShell script, you’ll need a Windows PC with PowerShell installed (run as Administrator if needed).

### Video Guide:

https://github.com/user-attachments/assets/7a161694-e4a5-4624-aec3-22a659f02494

### Steps:

- **Step 1**: Download the PowerShell script `nordstrom_scrape.ps1` to your local machine.

- **Step 2**: Visit [Nordstrom's website](https://www.nordstrom.com/) and open your browser’s Developer Tools. Go to the **Network** tab.

- **Step 3**: Use the website’s search feature to find your desired garment. Scroll through the results and paginate to load more items.

- **Step 4**: Once enough images have loaded (typically ~70 per page), go to the **Img** tab and enter `?h=368&w=240&dpr=2` in the filter bar to isolate the relevant image requests.

- **Step 5**: Click the small download button above the filter bar and save the file as `<garment>_raw.json`, replacing `<garment>` with the actual garment name (e.g., `shirt_raw.json`).

- **Step 6**: Open PowerShell and navigate to the folder containing `nordstrom_scrape.ps1`. Run the script using the following command:

  ```powershell
  .\nordstrom_scrape.ps1 `
    -WorkingDir "<working_directory>" `
    -GarmentType "<garment>" `
    -InputJsonFile "<garment>_raw.json" `
    -OutputJsonFile "<garment>.json" `
    -ImageDir "images"
  ```

  Replace `<working_directory>` with the directory path where the script resides, and `<garment>` with the actual garment name.

- **Step 7**: The script will automatically download all the images and save them in the `images` folder inside your working directory. It will also generate a cleaned and organized JSON metadata file and delete the original raw JSON file.

# Zara

Please simply run the PowerShell script `zara_scrape.ps1` with the [`zara_measures_data.csv`](https://garmentiq.ly.gd.edu.kg/asset/csv/zara_measures_data.csv) at the same directory. All the images will be downloaded in a folder named `downloaded_images`.
