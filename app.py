import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🔢 Handwritten Digit Recognition")
st.write("Draw a digit from **0 to 9** in the box below!")

# -----------------------------
# Drawing Canvas
# -----------------------------
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Digit", use_container_width=True):

    if canvas_result.image_data is not None:

        # Convert canvas image to PIL
        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        # Convert to grayscale
        image = image.convert("L")

        # Invert colors: black background + white digit
        image = ImageOps.invert(image)

        # Resize to MNIST size
        image = image.resize((28, 28))

        # Convert to numpy
        image_array = np.array(image).astype("float32") / 255.0

        # Add channel and batch dimensions
        image_array = image_array.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(image_array, verbose=0)

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # -----------------------------
        # Display Result
        # -----------------------------
        st.success(f"🎉 Predicted Digit: **{predicted_digit}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Show probabilities
        st.subheader("Prediction Probabilities")

        probabilities = prediction[0] * 100

        for digit, probability in enumerate(probabilities):
            st.write(f"**{digit}** — {probability:.2f}%")
