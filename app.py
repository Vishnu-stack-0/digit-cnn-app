import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="CNN Digit Recognizer",
    page_icon="🔢"
)

st.title("🔢 Handwritten Digit Recognizer")
st.write("Draw a digit from 0 to 9 and let the CNN predict it.")

model = tf.keras.models.load_model("digit_cnn.keras")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=18,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas"
)

if st.button("Predict digit"):
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        image = canvas_result.image_data

        # Convert RGBA image to grayscale
        image = Image.fromarray(image.astype("uint8")).convert("L")

        # Resize to the MNIST input size
        image = image.resize((28, 28))

        # Convert to NumPy array and normalize
        image_array = np.array(image).astype("float32") / 255.0

        # Add batch and channel dimensions
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict probabilities
        probabilities = model.predict(image_array, verbose=0)[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities)) * 100

        st.success(f"Predicted digit: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("Probabilities")
        st.bar_chart(probabilities)