import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os 

print("Current Folder:", os.getcwd())
print(os.listdir())

# Load pickle model
import tensorflow as tf
model = tf.keras.models.load_model(r"C:\Users\HP\Downloads\intership\crop_disease_app\crop_model.h5")

st.title("🌿 Crop Disease Detection")

# Upload image
uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224,224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.success("Healthy Leaf ✅")
        st.info("💡 Keep up the good work!")
    else:
        st.error("Diseased Leaf ❌")
        st.info("💡 Apply pesticides and remove infected leaves.")
