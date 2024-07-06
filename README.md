# Art Attributor

## Overview
This project aims to develop an art attribution algorithm that can guess the artist when provided with a painting. The dataset used is from Kaggle, titled Best Artworks of All Time, which contains paintings from the top 50 artists, totaling 8446 images. Each artist has between 24 to 877 images, and the dataset includes a CSV file with general information about each artist such as genre, nationality, basic biography, and the number of images included.

## Dataset
Source: [Kaggle - Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/code?datasetId=130081&sortBy=voteCount)
Total Images: 8446
Artists: 50
Images per Artist: 24 to 877
Additional Data: CSV file with artist information (genre, nationality, biography, number of images)

## Approach
1. Data Filtering:
   * Filtered for artists with over 200 paintings to ensure sufficient data for training.
   * Final set included top 11 artists with a varied number of images (239 to 877).
2. Class Weights:
   * Introduced class weights to address class imbalance.
3. Data Splitting:
   * Used an 80/20 train-test split, resulting in approximately 3400 training images and 850 test images.
4. Data Augmentation:
   * Applied data augmentation using ImageDataGenerator with the following parameters: rescale, rotation, width & height shift, shear, zoom, horizontal & vertical flips, and nearest fill mode

## Model Architecture
* Base Model: ResNet50 with pretrained weights on ImageNet.
* Custom Layers:
    * GlobalAveragePooling2D.
    * Dense layer with 1024 neurons.
    * Dense layer with 512 neurons.
    * 50% dropout rate between each custom layer and the output layer.

## Usage
1. Install Dependencies:
   * Install dependencies listed in the [requirements.txt](./requirements.txt) file.
2. Download the Dataset:
   * Download the dataset from the Kaggle link listed in 'Source' above and save it to a folder called 'Data' within your working directory.
3. Run the Notebook:
   * Save and run the [ArtAttributor.ipynb](./ArtAttributor.ipynb) notebook within your working directory to train the model and generate the art_attribution_model.h5 file.
4. Run the Streamlit App:
   * Once the 'art_attribution_model.h5' file is downloaded (mine was too large to be uploaded here), you can test the model via a local Streamlit app.
   * Save the app.py script to your working directory.
   * Navigate to the shell within your working directory and run the following command:
      ```
      streamlit run app.py
      ```
   * You should be directed to a webpage for the model where you can upload image files or URLs of one of the 11 artists represented in the model.
  
## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Contributing
Contributions are welcome! If you have suggestions, feature requests, or find bugs, please [open an issue](https://github.com/hodubena/Art_Attributor/issues) or submit a pull request.
