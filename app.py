import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CNN Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🔢 Handwritten Digit Recognizer")
st.write("Draw a digit from 0 to 9 and let the CNN predict it.")


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_cnn.keras")


model = load_model()


# --------------------------------------------------
# Canvas
# --------------------------------------------------

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    return_image_data=True
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict digit"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")

    else:

        # --------------------------------------------------
        # 1. Get canvas image
        # --------------------------------------------------

        rgba = canvas_result.image_data.astype("uint8")

        # RGBA -> grayscale
        gray = Image.fromarray(rgba).convert("L")

        # NumPy array
        img = np.array(gray)

        # --------------------------------------------------
        # 2. Remove very low-intensity pixels
        # --------------------------------------------------

        img[img < 20] = 0

        # Find digit pixels
        coords = np.argwhere(img > 0)

        if coords.size == 0:
            st.warning("Please draw a digit first.")
            st.stop()

        # --------------------------------------------------
        # 3. Find bounding box
        # --------------------------------------------------

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped = img[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

        cropped_image = Image.fromarray(cropped)


        # --------------------------------------------------
        # 4. Resize while preserving aspect ratio
        #
        # MNIST digits usually occupy roughly
        # 20 x 20 pixels inside the 28 x 28 image.
        # --------------------------------------------------

        target_size = 20

        width, height = cropped_image.size

        scale = target_size / max(width, height)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cropped_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )


        # --------------------------------------------------
        # 5. Create 28 x 28 black image
        # --------------------------------------------------

        final_image = Image.new(
            "L",
            (28, 28),
            0
        )


        # --------------------------------------------------
        # 6. Center the digit
        # --------------------------------------------------

        x_offset = (28 - new_width) // 2
        y_offset = (28 - new_height) // 2

        final_image.paste(
            resized,
            (x_offset, y_offset)
        )


        # --------------------------------------------------
        # 7. Convert to NumPy
        # --------------------------------------------------

        image_array = np.array(
            final_image
        ).astype("float32")


        # --------------------------------------------------
        # 8. Normalize
        # --------------------------------------------------

        image_array = image_array / 255.0


        # --------------------------------------------------
        # 9. Add batch + channel dimensions
        #
        # (28, 28)
        #     ↓
        # (1, 28, 28, 1)
        # --------------------------------------------------

        image_array = image_array.reshape(
            1,
            28,
            28,
            1
        )


        # --------------------------------------------------
        # 10. Display processed image
        # --------------------------------------------------

        st.subheader("What the CNN sees")

        st.image(
            image_array[0],
            width=200
        )


        # --------------------------------------------------
        # 11. Predict
        # --------------------------------------------------

        probabilities = model.predict(
            image_array,
            verbose=0
        )[0]


        # --------------------------------------------------
        # 12. Get prediction
        # --------------------------------------------------

        predicted_digit = int(
            np.argmax(probabilities)
        )


        # --------------------------------------------------
        # 13. Confidence
        # --------------------------------------------------

        confidence = (
            float(np.max(probabilities)) * 100
        )


        # --------------------------------------------------
        # 14. Display result
        # --------------------------------------------------

        st.success(
            f"Predicted digit: {predicted_digit}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )


        # --------------------------------------------------
        # 15. Probability chart
        # --------------------------------------------------

        st.subheader("Digit Probabilities")

        st.bar_chart(
            probabilities
        )