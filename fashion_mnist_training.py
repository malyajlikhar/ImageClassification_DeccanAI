import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# Load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to add a channel dimension for grayscale images
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Resize images to 32x32 for MobileNetV2
x_train = tf.image.resize(x_train, (32, 32)).numpy()  # Convert to NumPy array
x_test = tf.image.resize(x_test, (32, 32)).numpy()    # Convert to NumPy array

# Convert grayscale (1 channel) to RGB (3 channels) for MobileNetV2
x_train = np.repeat(x_train, 3, axis=-1)  # Repeat across the last dimension to make it (32, 32, 3)
x_test = np.repeat(x_test, 3, axis=-1)

# Split the training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.3
)
datagen.fit(x_train)

# Define the input
input_layer = Input(shape=(32, 32, 3))

# Load MobileNetV2 as base model
base_model = MobileNet(weights="imagenet", include_top=False, input_tensor=input_layer)

# Add Dense layer without regularization
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)  # Remove L2 regularization
output = Dense(10, activation="softmax")(x)  # Output layer without L2 regularization

# Create the full model
model = Model(inputs=input_layer, outputs=output)

# Set a custom learning rate
learning_rate = 0.001  # You can adjust this as needed
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

# Early stopping callback
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    datagen.flow(np.array(x_train), y_train, batch_size=32),
    validation_data=(np.array(x_val), y_val),
    epochs=20,
    callbacks=[early_stop]
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(np.array(x_test), y_test)
print(f"Test Accuracy: {test_acc}")

# Save the model
model.save("fashion_mnist_mobilenet.h5")

# Save x_test and y_test for evaluation
np.save("x_test.npy", x_test)
np.save("y_test.npy", y_test)
