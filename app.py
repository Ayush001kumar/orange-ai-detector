
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page settings
st.set_page_config(
    page_title="Orange AI Detector",
    page_icon="🍊",
    layout="centered"
)

# Title
st.title("🍊 Orange AI Detector")
st.write("Upload an orange image and the AI will classify it.")

# Load trained model
model = tf.keras.models.load_model("orange_classifier.keras")

# Class names — keep this exact order
class_names = [
    "blackspotted_orange",
    "cankered_orange",
    "fresh_orange",
    "green_orange",
    "rotten_orange"
]

# Upload image
uploaded_file = st.file_uploader(
    "Choose an orange image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prepare image according to model input size
    input_height = model.input_shape[1]
    input_width = model.input_shape[2]

    img = image.resize((input_width, input_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # Result
    st.success(f"🍊 Prediction: {predicted_class}")
    st.info(f"🎯 Confidence: {confidence:.2f}%")

    # Show all probabilities
    st.subheader("Class Probabilities")

    for i, class_name in enumerate(class_names):
        probability = predictions[0][i] * 100
        st.write(f"{class_name}: {probability:.2f}%")