import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import model_from_json, load_model
from PIL import Image

# Load model architecture from config.json
with open('/mnt/data/config.json', 'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load weights into the model
model.load_weights('/mnt/data/model.weights.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Streamlit app
st.title('Intelligent Virtual Shopping Assistant')

# User input for budget
user_budget = st.number_input('Enter your budget:', min_value=0, step=1)

# Placeholder for recommendations
recommendations = st.empty()

# Function to preprocess input and get recommendations
def get_recommendations(budget):
    # Example preprocessing - you will need to adjust this based on your model's input requirements
    input_data = np.array([[budget]])
    predictions = model.predict(input_data)
    return predictions

if st.button('Get Recommendations'):
    preds = get_recommendations(user_budget)
    # Process predictions to get human-readable recommendations
    # This is an example; adjust it according to your actual labels and recommendation logic
    recommended_goods = ['Good 1', 'Good 2', 'Good 3']  # Placeholder for actual recommended items
    images = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']  # Placeholder for actual images
    
    for good, img_path in zip(recommended_goods, images):
        st.write(f"Recommended: {good}")
        st.image(Image.open(img_path), caption=good)


