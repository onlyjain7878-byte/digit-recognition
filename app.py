import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_model.h5")

model = load_model()

st.title("🔢 Handwritten Digit Recognition")
st.write("Draw a digit from 0 to 9 below!")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=18,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if st.button("🔮 Predict Digit", use_container_width=True):

    if canvas_result.image_data is not None:

        image = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        image = image.convert("L")
        image = ImageOps.invert(image)
        image = image.resize((28, 28))

        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(
            f"🎉 Predicted Digit: **{predicted_digit}**"
        )

        st.info(
            f"Confidence: **{confidence:.2f}%**"
        )

        st.subheader("Prediction Probabilities")

        for digit, probability in enumerate(prediction):
            st.write(
                f"**{digit}** — {probability * 100:.2f}%"
            )
