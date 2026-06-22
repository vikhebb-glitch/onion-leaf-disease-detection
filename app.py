import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Class names
class_names = ['downy_mildew', 'healthy', 'leaf_blight', 'purple_blotch']

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="onion_leaf_cnn_model_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# UI
st.title("Onion Leaf Disease Detection System")

st.markdown("""
### Research Scholar
**Bhausaheb Bhaskar Vikhe**

### Under the Guidance of
**Dr. Purushottam Patil**

### Sandip University Campus, Nashik
""")

# Upload image
uploaded_file = st.file_uploader("Upload Onion Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Read image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)   # (1,224,224,3)

    st.write("Input Shape:", img.shape)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Output
    st.subheader("Prediction Result")
    st.success(f"Disease Detected: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
