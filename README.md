# Art Attributor

## Overview
This project aims to develop an art attribution algorithm that can guess the artist when provided with a painting. The dataset used is from Kaggle, titled Best Artworks of All Time, which contains paintings from the top 50 artists, totaling 8446 images. Each artist has between 24 to 877 images, and the dataset includes a CSV file with general information about each artist such as genre, nationality, basic biography, and the number of images included.

## Dataset
Source: https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/code?datasetId=130081&sortBy=voteCount
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
1. **Option 1: Streamlit UI**
   - Access a deployed version of this project via Streamlit. Check it out [here](https://hodubena.shinyapps.io/EPLStandingsPro/).

2. **Option 2: Run Locally**
   - Install Streamlit: `pip install streamlit`
   - Run the app: `streamlit run app.py`
  
## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

## Contributing
Contributions are welcome! If you have suggestions, feature requests, or find bugs, please [open an issue](https://github.com/hodubena/Art_Attributor/issues) or submit a pull request.
