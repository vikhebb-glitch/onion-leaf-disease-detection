import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import urllib.request
import os

class_names = ['downy_mildew', 'healthy', 'leaf_blight', 'purple_blotch']

# 🔥 Download model from GitHub raw (safe method)
model_url = "https://github.com/vikhebb-glitch/onion-leaf-disease-detection/raw/main/onion_leaf_cnn_model_quant.tflite"
model_path = "model.tflite"

urllib.request.urlretrieve(model_url, model_path)

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("Onion Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = class_names[np.argmax(prediction)]
    conf = np.max(prediction)*100

    st.success(result)
    st.info(f"{conf:.2f}%")
