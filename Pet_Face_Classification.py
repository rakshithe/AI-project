

# Pet Face Classification using TensorFlow and Keras

## 1. Install Required Libraries
# pip install tensorflow keras numpy pandas matplotlib opencv-python

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset path (replace with your dataset location)
dataset_path = "path_to_pet_faces_dataset"  # e.g., 'cats_and_dogs_filtered'

# Image Data Generator for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(150, 150), batch_size=32, class_mode="binary", subset="training"
)
validation_generator = train_datagen.flow_from_directory(
    dataset_path, target_size=(150, 150), batch_size=32, class_mode="binary", subset="validation"
)

# Define a simple CNN model for pet face classification
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")  # Binary classification (Cat or Dog)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Plot training accuracy and loss
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Save the model
model.save("pet_face_classifier.h5")

