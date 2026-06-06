import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os 
import gdown

print("Current Folder:", os.getcwd())
print(os.listdir())

FILE_ID = "1ja0p0NVFKvDWwk0XYiT3fBuHG3tU8vR5"

if not os.path.exists("crop_model.h5"):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, "crop_model.h5", quiet=False)

model = tf.keras.models.load_model("crop_model.h5")

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
