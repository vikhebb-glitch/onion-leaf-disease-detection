import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Class Names

class_names = [
'downy_mildew',
'healthy',
'leaf_blight',
'purple_blotch'
]

# Load TFLite Model

interpreter = tflite.Interpreter(
model_path="onion_leaf_cnn_model_quant.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Title

st.title("Onion Leaf Disease Detection")

st.markdown("""

### Research Scholar

**Bhausaheb Bhaskar Vikhe**

### Under the Guidance of

**Dr. Purushottam Patil**

### Sandip University Campus, Nashik

""")

st.write("Upload Onion Leaf Image")

# Upload Image

uploaded_file = st.file_uploader(
"Choose onion leaf image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

```
img = Image.open(uploaded_file).convert("RGB")

st.image(img, caption="Uploaded Image")

img = img.resize((224, 224))

img_array = np.array(img, dtype=np.float32)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)

# Prediction
interpreter.set_tensor(
    input_details[0]['index'],
    img_array
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_details[0]['index']
)

predicted_class = np.argmax(prediction)

confidence = float(np.max(prediction)) * 100

result = class_names[predicted_class]

st.subheader("Prediction Result")

st.success(
    f"Detected Disease: {result}"
)

st.info(
    f"Confidence: {confidence:.2f}%"
)
```
