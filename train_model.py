import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------
# 1. Load MNIST data
# --------------------------------------------------

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# --------------------------------------------------
# 2. Preprocess images
# --------------------------------------------------

# Add channel dimension
x_train = x_train.reshape(
    -1, 28, 28, 1
).astype("float32")

x_test = x_test.reshape(
    -1, 28, 28, 1
).astype("float32")

# Normalize pixels: 0-255 -> 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0


# --------------------------------------------------
# 3. Create CNN model
# --------------------------------------------------

model = models.Sequential([

    # Input
    layers.Input(shape=(28, 28, 1)),

    # First convolution
    layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Reduce image size
    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Second convolution
    layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    # Reduce image size again
    layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    # Convert feature maps to 1D
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(
        64,
        activation="relu"
    ),

    # Dropout helps reduce overfitting
    layers.Dropout(0.3),

    # Output layer: digits 0-9
    layers.Dense(
        10,
        activation="softmax"
    )
])


# --------------------------------------------------
# 4. Configure training
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 5. Display model architecture
# --------------------------------------------------

model.summary()


# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)


# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print()
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------

model.save("digit_cnn.keras")

print()
print("Model saved as digit_cnn.keras")

