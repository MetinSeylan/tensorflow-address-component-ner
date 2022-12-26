import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers.legacy import Adam

def split_label_string(label_string):
  return list(map(int, label_string.split(",")))
def split_input_string(input_string):
  return list(input_string.split(","))

label_values = {"city": 0, "district": 1, "neighborhood": 2, "space": 3, "neighborhood_suffix": 4, "road": 5, "road_suffix": 6, "building_number_prefix": 7, "building_number": 8, "floor_suffix": 9, "floor": 10, "door_number_prefix": 11, "door_number": 12}

# Read the input data and labels from the CSV files
data = pd.read_csv("address.csv", delimiter=']', header=None, names=["input"], dtype=str)
labels = pd.read_csv("labels.csv", delimiter=']', header=None, names=["label"])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Split the label strings into a list of labels
y_train = y_train["label"].apply(split_label_string)
y_test = y_test["label"].apply(split_label_string)

# Split the input strings into a list of input
X_train = X_train["input"].apply(split_input_string)
X_test = X_test["input"].apply(split_input_string)


# Define the maximum input length and the number of unique labels
max_input_length = 128
num_labels = 128

# Use a tokenizer to preprocess the input data
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X_train)
word_count = len(tokenizer.word_index)
X_train = [np.pad(tokenizer.texts_to_matrix(y), [[0, 0], [0, 127 - word_count]]) for y in X_train]
X_test = [np.pad(tokenizer.texts_to_matrix(y), [[0, 0], [0, 127 - word_count]]) for y in X_test]

# Convert the labels to a one-hot encoding
y_train = [tf.keras.utils.to_categorical(y, 128) for y in y_train]
y_test = [tf.keras.utils.to_categorical(y, 128) for y in y_test]

y_train = np.array(y_train)
y_test = np.array(y_test)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Define the size of the vocabulary and the maximum input length
vocab_size = len(tokenizer.word_index) + 1

# Define the embedding layer
embedding_layer = tf.keras.layers.Embedding(vocab_size, 128, input_length=128)

# Define the LSTM layer
lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))

# Define the model
model = tf.keras.Sequential([
  lstm,
  tf.keras.layers.Dense(2048, activation="relu"),
  tf.keras.layers.Dense(1024, activation="relu"),
  tf.keras.layers.Dense(512, activation="relu"),
  tf.keras.layers.Dense(256, activation="relu"),
  tf.keras.layers.Dense(128, activation="softmax")
])

# Add a TensorBoard callback to the model
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Compile the model with a categorical cross-entropy loss function and an Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])


# Fit the model on the input data and labels
model.fit(X_train, y_train, validation_data=(X_test, y_test))
print('fit sonrasi')

# Calculate the model's performance on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

model.save('./trained')

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
