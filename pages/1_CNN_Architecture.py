import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

st.set_page_config(
    page_title="CNN Internal Operations",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 How the CNN Works")
st.write(
    "This page explains the internal operations used by the handwritten "
    "digit recognition CNN."
)

st.info(
    "The model receives a 28 × 28 grayscale image and predicts one of "
    "ten classes: digits 0 through 9."
)

# --------------------------------------------------
# 1. Architecture diagram
# --------------------------------------------------

st.header("1. CNN Architecture")

st.code(
    """
Input Image
28 × 28 × 1
    │
    ▼
Conv2D
16 filters, 3 × 3 kernel, ReLU
Output: 26 × 26 × 16
    │
    ▼
MaxPooling2D
2 × 2 pool
Output: 13 × 13 × 16
    │
    ▼
Conv2D
32 filters, 3 × 3 kernel, ReLU
Output: 11 × 11 × 32
    │
    ▼
MaxPooling2D
2 × 2 pool
Output: 5 × 5 × 32
    │
    ▼
Flatten
5 × 5 × 32 = 800 values
    │
    ▼
Dense
64 neurons, ReLU
    │
    ▼
Dropout
30% randomly disabled during training
    │
    ▼
Output Dense Layer
10 neurons, Softmax
    │
    ▼
Prediction: 0, 1, 2, ..., or 9
""",
    language="text"
)

st.caption(
    "The output sizes assume the default valid padding used by Conv2D "
    "and the default stride of 1."
)

# --------------------------------------------------
# 2. Input image
# --------------------------------------------------

st.header("2. Input Image")

st.write(
    """
    The CNN expects one grayscale image with the shape 28 × 28 × 1.

    - 28: image height
    - 28: image width
    - 1: grayscale channel
    - Pixel values are normalized from 0–255 to 0–1
    """
)

st.latex(r"\text{Input shape} = 28 \times 28 \times 1")

# --------------------------------------------------
# 3. First convolution
# --------------------------------------------------

st.header("3. First Convolution Layer")

st.code(
    """
layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation="relu"
)
""",
    language="python"
)

st.write(
    """
    This layer contains 16 learnable filters. Each filter is a 3 × 3
    matrix that moves across the image and searches for a pattern.

    Early filters may learn simple features such as:

    - Horizontal edges
    - Vertical edges
    - Curves
    - Corners
    - Thick or thin strokes
    """
)

st.latex(
    r"\text{Number of filters} = 16"
)

st.latex(
    r"\text{Kernel size} = 3 \times 3"
)

st.latex(
    r"\text{Output shape} = 26 \times 26 \times 16"
)

st.write(
    """
    The output contains 16 feature maps, one feature map for each filter.
    A feature map shows where a particular pattern was detected.
    """
)

# --------------------------------------------------
# 4. ReLU
# --------------------------------------------------

st.header("4. ReLU Activation Function")

st.code(
    """
activation="relu"
""",
    language="python"
)

st.write(
    """
    ReLU means Rectified Linear Unit. It replaces negative values with
    zero and keeps positive values.
    """
)

st.latex(r"\operatorname{ReLU}(x) = \max(0, x)")

st.write(
    """
    Examples:
    """
)

st.code(
    """
Input:   [-3, -1, 0, 2, 5]
Output:  [ 0,  0, 0, 2, 5]
""",
    language="text"
)

st.write(
    """
    ReLU introduces nonlinearity. Without an activation function, multiple
    layers would behave too much like one linear operation, making it
    difficult to learn complex digit shapes.
    """
)

# --------------------------------------------------
# 5. First max pooling
# --------------------------------------------------

st.header("5. First Max-Pooling Layer")

st.code(
    """
layers.MaxPooling2D(
    pool_size=(2, 2)
)
""",
    language="python"
)

st.write(
    """
    Max pooling examines each 2 × 2 region and keeps only the largest
    value. This reduces the spatial size of the feature maps while
    preserving strong detected features.
    """
)

st.latex(
    r"\text{Input shape} = 26 \times 26 \times 16"
)

st.latex(
    r"\text{Output shape} = 13 \times 13 \times 16"
)

st.write(
    """
    Pooling reduces computation and makes the model less sensitive to the
    exact position of a feature.
    """
)

# --------------------------------------------------
# 6. Second convolution
# --------------------------------------------------

st.header("6. Second Convolution Layer")

st.code(
    """
layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu"
)
""",
    language="python"
)

st.write(
    """
    The second convolution layer has 32 filters. It receives the first
    layer's feature maps and learns more complex patterns.

    For example, it can combine edges and curves into parts of digits such
    as loops, diagonal strokes, and intersections.
    """
)

st.latex(r"\text{Number of filters} = 32")
st.latex(r"\text{Kernel size} = 3 \times 3")
st.latex(r"\text{Output shape} = 11 \times 11 \times 32")

# --------------------------------------------------
# 7. Second max pooling
# --------------------------------------------------

st.header("7. Second Max-Pooling Layer")

st.write(
    """
    The second pooling layer again uses a 2 × 2 window to reduce the
    feature-map dimensions.
    """
)

st.latex(
    r"\text{Input shape} = 11 \times 11 \times 32"
)

st.latex(
    r"\text{Output shape} = 5 \times 5 \times 32"
)

# --------------------------------------------------
# 8. Flatten
# --------------------------------------------------

st.header("8. Flatten Layer")

st.code(
    """
layers.Flatten()
""",
    language="python"
)

st.write(
    """
    Flatten converts the three-dimensional feature maps into one
    one-dimensional vector so that it can be passed to the dense neurons.
    """
)

st.latex(
    r"5 \times 5 \times 32 = 800 \text{ values}"
)

st.write(
    """
    Therefore, the Flatten layer produces a vector containing 800 values.
    """
)

# --------------------------------------------------
# 9. Dense layer
# --------------------------------------------------

st.header("9. Fully Connected Dense Layer")

st.code(
    """
layers.Dense(
    64,
    activation="relu"
)
""",
    language="python"
)

st.write(
    """
    This layer contains 64 neurons. Every neuron receives the values from
    the Flatten layer and learns a combination of the detected features.

    These neurons help answer questions such as:

    - Does the image contain a loop?
    - Is there a vertical stroke?
    - Is there a diagonal line?
    - Which combination of features looks like a 3 or an 8?
    """
)

st.latex(r"\text{Number of dense neurons} = 64")

# --------------------------------------------------
# 10. Dropout
# --------------------------------------------------

st.header("10. Dropout Regularization")

st.code(
    """
layers.Dropout(0.3)
""",
    language="python"
)

st.write(
    """
    A dropout rate of 0.3 means that approximately 30% of the outputs
    from the previous layer are randomly ignored during each training
    step.

    Dropout is active during training but disabled during prediction.
    It helps reduce overfitting, where the model performs well on training
    images but poorly on unseen images.
    """
)

st.latex(
    r"\text{Dropout rate} = 0.3 = 30\%"
)

# --------------------------------------------------
# 11. Output layer
# --------------------------------------------------

st.header("11. Output Layer")

st.code(
    """
layers.Dense(
    10,
    activation="softmax"
)
""",
    language="python"
)

st.write(
    """
    The output layer has 10 neurons because there are 10 possible classes:
    digits 0 through 9.

    Each neuron represents one digit:
    """
)

st.code(
    """
Neuron 0 → probability that the image is digit 0
Neuron 1 → probability that the image is digit 1
...
Neuron 9 → probability that the image is digit 9
""",
    language="text"
)

st.latex(
    r"\text{Number of output neurons} = 10"
)

st.write(
    """
    Softmax converts the output values into probabilities. The probabilities
    add up to approximately 1. The class with the highest probability is
    selected as the prediction.
    """
)

st.latex(
    r"\operatorname{softmax}(z_i) =
    \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}"
)

# --------------------------------------------------
# 12. Complete model summary
# --------------------------------------------------

st.header("12. Complete Model Summary")

model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

summary_lines = []
model.summary(print_fn=lambda line: summary_lines.append(line))

st.code(
    "\n".join(summary_lines),
    language="text"
)

# --------------------------------------------------
# 13. Parameter explanation
# --------------------------------------------------

st.header("13. What Are Trainable Parameters?")

st.write(
    """
    Trainable parameters are values that the CNN learns during training.
    They include filter weights, biases, and dense-layer weights.
    """
)

st.write(
    "The number of trainable parameters in this model is:"
)

st.metric(
    label="Trainable parameters",
    value=f"{model.count_params():,}"
)

st.write(
    """
    During training, the optimizer changes these parameters so that the
    predicted digit becomes closer to the correct digit.
    """
)

# --------------------------------------------------
# 14. Training flow
# --------------------------------------------------

st.header("14. Training Process")

st.code(
    """
1. Give the CNN a training image.
2. CNN calculates probabilities for digits 0–9.
3. Compare the prediction with the correct label.
4. Calculate the loss.
5. Backpropagate the error.
6. Adam updates the weights.
7. Repeat for many images and epochs.
""",
    language="text"
)

st.write(
    """
    In your code, sparse categorical cross-entropy measures the prediction
    error, while Adam adjusts the model's trainable parameters.
    """
)

st.code(
    """
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
""",
    language="python"
)

st.success(
    "The complete flow is: input → convolution → ReLU → pooling → "
    "convolution → ReLU → pooling → flatten → dense neurons → dropout → "
    "softmax prediction."
)