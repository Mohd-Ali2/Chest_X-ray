import streamlit as st 
from keras.preprocessing import image 
import numpy as np 
from keras.models import model_from_json
from PIL import Image

# Load the model
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_weights.weights.h5")

# Function to make predictions
def predict(image_file):
    # Open the uploaded image
    test_image = Image.open(image_file)
    # Resize the image to the desired size (64x64) using bilinear interpolation
    test_image = test_image.resize((64, 64), resample=Image.BILINEAR)
    # Convert the image to numpy array and preprocess if necessary
    # (e.g., converting to float32 and normalizing pixel values)
    test_image = np.array(test_image)
    # Perform any necessary preprocessing here

    # If the model expects input images with 3 channels (RGB), expand the dimensions
    if test_image.ndim == 2:
        # If the image is grayscale, convert it to RGB by duplicating the channels
        test_image = np.stack((test_image,) * 3, axis=-1)
    elif test_image.shape[2] == 1:
        # If the image has only one channel, duplicate it to create 3 channels
        test_image = np.concatenate([test_image] * 3, axis=2)

    # Reshape the image to match the model's input shape
    test_image = np.expand_dims(test_image, axis=0)
    
    # Make predictions using the loaded model
    result = loaded_model.predict(test_image)
    return result

# Streamlit UI
st.sidebar.title('Author :memo:')
st.sidebar.subheader('Mohammad Ali :name_badge:')
st.title("Deep Learning Model for X-Ray Image Classification")
st.sidebar.title('Description :scroll:')
st.sidebar.write("Medical imaging plays a crucial role in diagnosing various diseases and conditions, with X-ray imaging being one of the most widely used modalities. However, interpreting X-ray images can be challenging and time-consuming for healthcare professionals. To assist in this process, deep learning models based on Convolutional Neural Networks (CNNs) have shown promising results in automating X-ray image analysis")
st.sidebar.title('Connect :link:')
st.sidebar.link_button('Linkdin :large_blue_diamond:', url='https://www.linkedin.com/in/mohdali02/')
st.sidebar.link_button('Github  :black_large_square:', url='github.com/Mohd-Ali2')
# File uploader
uploaded_file = st.file_uploader('Choose an image ..', type=['jpg', 'jpeg', 'png', 'webp'])
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
    
# Button to trigger prediction
if st.button('Click Here to Check'):
    result = predict(uploaded_file)
    if result == 0:
        st.write('NORMAL:white_check_mark:')
    else:
        st.write('PNEUMONIA:heavy_exclamation_mark:')
