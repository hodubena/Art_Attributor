import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests
from io import BytesIO

# Function to normalize artist names to match folder names
def normalize_artist_name(name):
    return name.replace(' ', '_')

# Load the pre-trained model
model = load_model('art_attribution_model.h5')

# Load class labels and normalize artist names
artists_df = pd.read_csv('Data/artists.csv')
filtered_artists = artists_df[artists_df["paintings"] >= 200]
sorted_artists_df = filtered_artists.sort_values(by="name", ascending=True)
artist_names = sorted_artists_df["name"].apply(normalize_artist_name).tolist()
artist_labels = {artist: idx for idx, artist in enumerate(artist_names)}

# Define preprocessing function
def preprocess_image(image):
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize to [0, 1]
    return image

st.title('Art Attributor')
st.write("Please enter the URL of an image or upload an image of a painting from one of these artists:")

# Format artist names in the desired format
artist_list_formatted = ", ".join(artist.replace('_', ' ') for artist in artist_names[:-1])
artist_list_formatted += f", and {artist_names[-1].replace('_', ' ')}"

st.write(artist_list_formatted)

option = st.selectbox('Choose input type:', ('Upload Image', 'Image URL'))

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        prediction_probability = np.amax(prediction)
        prediction_idx = np.argmax(prediction)
        st.write(f"Predicted artist: {artist_names[prediction_idx].replace('_', ' ')}")
        st.write(f"Prediction probability: {prediction_probability * 100:.2f}%")
else:
    url = st.text_input("Enter image URL:")
    if url:
        response = requests.get(url)
        image = load_img(BytesIO(response.content), target_size=(224, 224))
        st.image(image, caption='Image from URL.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        prediction_probability = np.amax(prediction)
        prediction_idx = np.argmax(prediction)
        st.write(f"Predicted artist: {artist_names[prediction_idx].replace('_', ' ')}")
        st.write(f"Prediction probability: {prediction_probability * 100:.2f}%")
