

# Text Classification using TensorFlow and Keras

## 1. Install Required Libraries
# pip install tensorflow numpy pandas scikit-learn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample dataset
texts = ["I love this product", "This is the worst experience", "Absolutely fantastic!", "Not good at all"]
labels = [1, 0, 1, 0]  # 1 = Positive, 0 = Negative

# Tokenization and Padding
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10, padding="post")

# Define a Simple Neural Network Model
model = keras.Sequential([
    keras.layers.Embedding(1000, 16, input_length=10),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Compile the Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the Model (For demonstration, using a small dataset)
model.fit(padded_sequences, np.array(labels), epochs=10, verbose=1)

# Predict Sentiment
test_texts = ["I really enjoyed this", "Worst thing ever"]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_padded = pad_sequences(test_sequences, maxlen=10, padding="post")

predictions = model.predict(test_padded)
print("Predictions:", predictions)

